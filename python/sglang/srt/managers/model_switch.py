"""
Multi-model manager: model hot-switching with D2D staging + bump allocator.

Structured as five ordered phases (see `do_model_switch_bump`):
  1. Save old model caches + tear down GPU state
  2. Load new weights (D2D fast path or full H2D fallback)
  3. Rebuild KV pool (placeholder restore + full rebuild — true KV restore is
     tracked as future work; see `_kv_restore_placeholder`)
  4. Rebuild attention backend + CUDA graphs
  5. Propagate references back to scheduler / worker

All cross-phase state is carried in `_SwitchContext`. Per-phase helpers never
reach into `scheduler._*` private fields; they go through the public accessors
(`host_model_mgr`, `active_preload_manager`, `wait_for_preload`).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.flashinfer_backend import reset_global_workspace_buffer
from sglang.srt.layers.rotary_embedding.factory import clear_rope_cache
from sglang.srt.mem_cache.vram_manager import (
    ModelResourceCache,
    _align_up,
    _set_graph_pool,
)
from sglang.srt.model_executor.input_buffers import clear_forward_input_buffer_pool

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)
_SWITCH_DIAG = os.environ.get("SGLANG_SWITCH_DIAG", "0") == "1"

# Pool reference ownership map. Scheduler and TpWorker don't hold
# `token_to_kv_pool` — only model_runner does. Keep these tuples in sync with
# the real owners; otherwise setattr(None) will inject phantom attributes.
_POOL_ATTRS_RUNNER = (
    "req_to_token_pool",
    "token_to_kv_pool",
    "token_to_kv_pool_allocator",
)
_POOL_ATTRS_SCHED_WORKER = ("req_to_token_pool", "token_to_kv_pool_allocator")

# Global resource cache (shared across switches)
_resource_cache: Optional[ModelResourceCache] = None


def _get_resource_cache() -> ModelResourceCache:
    global _resource_cache
    if _resource_cache is None:
        _resource_cache = ModelResourceCache()
    return _resource_cache


# ==========================================================================
# Context
# ==========================================================================


@dataclass
class _SwitchContext:
    """State carried between phases of `do_model_switch_bump`.

    Fields populated at construction time are invariants (runner / bump / rc
    etc.). Fields with None defaults are outputs of a specific phase and
    should only be read after that phase runs.
    """

    # static references
    scheduler: "Scheduler"
    runner: Any
    bump: Any
    target_model_name: str
    target_model_path: str
    resource_cache: ModelResourceCache
    memory_saver: Optional[Any]
    tp_size: int
    tp_cpu_group: Optional[Any]
    prev_model_name: str

    # timing
    t_total_start: float = field(default_factory=time.perf_counter)
    timings: Dict[str, float] = field(default_factory=dict)

    # populated by phase 2
    new_config: Optional[ModelConfig] = None
    has_staging: bool = False

    # populated by phase 3
    kv_hit: bool = False


def _build_context(
    scheduler: "Scheduler", target_model_path: str, target_model_name: Optional[str]
) -> _SwitchContext:
    runner = scheduler.tp_worker.model_runner
    memory_saver = (
        scheduler.memory_saver_adapter
        if scheduler.memory_saver_adapter.enabled
        else None
    )
    return _SwitchContext(
        scheduler=scheduler,
        runner=runner,
        bump=runner.vram_mgr,
        target_model_name=target_model_name,
        target_model_path=target_model_path,
        resource_cache=_get_resource_cache(),
        memory_saver=memory_saver,
        tp_size=scheduler.tp_size,
        tp_cpu_group=getattr(scheduler, "tp_cpu_group", None),
        prev_model_name=scheduler.active_model_name,
    )


# ==========================================================================
# Small shared helpers
# ==========================================================================


def _force_flush_cache(scheduler: "Scheduler") -> None:
    """Drop tree cache / KV allocator / grammar state (bypass the idle check)."""
    # Wrap in tms region so the new tree_cache big tensor is per-model tagged.
    msa = getattr(scheduler, "memory_saver_adapter", None)
    if msa is not None and msa.enabled:
        with msa.region("flush"):
            scheduler.tree_cache.reset()
            if scheduler.token_to_kv_pool_allocator is not None:
                scheduler.token_to_kv_pool_allocator.clear()
            scheduler.grammar_manager.clear()
    else:
        scheduler.tree_cache.reset()
        if scheduler.token_to_kv_pool_allocator is not None:
            scheduler.token_to_kv_pool_allocator.clear()
        scheduler.grammar_manager.clear()
    logger.info("Cache force-flushed for model switch")


def _detach_pool_refs(ctx: _SwitchContext) -> None:
    """Null out stale pool references before a full KV rebuild.

    Fan-out across runner / scheduler / worker is unavoidable until the
    authoritative-owner refactor (see issue E4) lands. Keep the attr tuples
    scoped per-holder so we never inject phantom attributes.
    """
    for attr in _POOL_ATTRS_RUNNER:
        setattr(ctx.runner, attr, None)
    for holder in (ctx.scheduler, ctx.scheduler.tp_worker):
        for attr in _POOL_ATTRS_SCHED_WORKER:
            setattr(holder, attr, None)


def _tp_consensus_bool(value: bool, group) -> bool:
    """All-reduce MIN across the TP CPU group so every rank agrees on a bool.

    Returns True only if every rank voted True. When no TP group is active the
    input value is returned unchanged.
    """
    if group is None:
        return value
    vote = torch.tensor([int(value)], dtype=torch.int32)
    torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN, group=group)
    return bool(vote.item())


def _sync_worker(ctx: _SwitchContext) -> None:
    runner = ctx.runner
    worker = ctx.scheduler.tp_worker
    worker.model_config = ctx.new_config
    worker.max_total_num_tokens = runner.max_total_num_tokens
    worker.max_running_requests = runner.max_running_requests
    worker.max_req_len = min(
        ctx.new_config.context_len - 1, runner.max_token_pool_size - 1
    )
    worker.max_req_input_len = worker.max_req_len - 5


def _sync_scheduler(ctx: _SwitchContext) -> None:
    runner = ctx.runner
    scheduler = ctx.scheduler
    worker = scheduler.tp_worker
    scheduler.req_to_token_pool = runner.req_to_token_pool
    scheduler.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
    scheduler.max_total_num_tokens = runner.max_total_num_tokens
    scheduler.max_running_requests = runner.max_running_requests
    scheduler.max_req_len = worker.max_req_len
    scheduler.max_req_input_len = worker.max_req_input_len
    scheduler.model_config = ctx.new_config


def _sync_tree_cache(ctx: _SwitchContext) -> None:
    tree_cache = ctx.scheduler.tree_cache
    if tree_cache is None:
        return
    tree_cache.req_to_token_pool = ctx.runner.req_to_token_pool
    tree_cache.token_to_kv_pool_allocator = ctx.runner.token_to_kv_pool_allocator


def _update_server_args(ctx: _SwitchContext) -> None:
    server_args = ctx.scheduler.server_args
    server_args.model_path = ctx.target_model_path
    server_args.served_model_name = ctx.target_model_name
    ctx.scheduler.offload_tags.clear()


def _propagate_refs(ctx: _SwitchContext) -> None:
    """Sync runner state to scheduler and worker after a successful switch."""
    _sync_worker(ctx)
    _sync_scheduler(ctx)
    _sync_tree_cache(ctx)
    _update_server_args(ctx)


# ==========================================================================
# DIAG helpers (only run when SGLANG_SWITCH_DIAG=1)
# ==========================================================================


def _diag_log_d2d_layout(
    preload_mgr, bump, weights_start, total_target, needed_bytes, current_cap
) -> None:
    si = preload_mgr.scatter_info
    bump_addr = bump.buffer.data_ptr()
    logger.debug(
        f"  DIAG weights_region: start={weights_start} "
        f"cap={bump.regions['weights'].capacity} "
        f"bump.left_offset={bump.left_offset}"
    )
    logger.debug(
        f"  DIAG total_target={total_target} needed_bytes={needed_bytes} "
        f"current_cap={current_cap}"
    )
    logger.debug(
        f"  DIAG staging_info: staging_start={si.staging_start} "
        f"staging_rows={si.staging_rows} staged_per_block={si.staged_per_block} "
        f"row_size={si.row_size} pool_size={si.pool_size} "
        f"num_blocks={si.num_blocks} total_staged_bytes={si.total_staged_bytes}"
    )
    weights_region_end = weights_start + total_target
    logger.debug(
        f"  DIAG addresses: bump_base=0x{bump_addr:x} "
        f"weights_region=[{weights_start}, {weights_region_end}) "
        f"kv_pool_phys_start={bump.left_offset}"
    )
    for bi in range(min(5, len(preload_mgr.block_tensors))):
        blk = preload_mgr.block_tensors[bi]
        blk_start = blk.data_ptr() - bump_addr
        blk_end = blk_start + blk.numel() * blk.element_size()
        staging_off = blk_start + si.staging_start * si.row_size
        staging_end = blk_start + si.pool_size * si.row_size
        logger.info(
            f"  DIAG block[{bi}]: phys=[{blk_start}, {blk_end}) "
            f"staging_phys=[{staging_off}, {staging_end}) "
            f"overlap_with_weights={staging_off < weights_region_end} "
            f"gap={staging_off - weights_region_end}"
        )


def _diag_verify_d2d_chunks(preload_mgr, bump, weights_start) -> None:
    """Byte-equal staging source vs weights region for the first 3 blocks / valid chunks."""
    torch.cuda.synchronize()
    si = preload_mgr.scatter_info
    blocks = preload_mgr.block_tensors
    ok = bad = 0
    for bi in range(min(3, si.num_blocks)):
        blk_flat = blocks[bi].view(-1).view(torch.uint8)
        for ci in range(si.chunk_rows and len(si.chunk_valid or [])):
            if not (si.chunk_valid or [])[ci]:
                continue
            row_s = si.staging_start + ci * si.chunk_rows
            row_e = min(row_s + si.chunk_rows, si.pool_size)
            byte_off = ci * si.chunk_rows * si.row_size
            nbytes = min((row_e - row_s) * si.row_size, si.staged_per_block - byte_off)
            if nbytes <= 0:
                break
            src_off = row_s * si.row_size
            staging_bytes = blk_flat[src_off : src_off + nbytes]
            dst_off = weights_start + bi * si.staged_per_block + byte_off
            bump_bytes = bump.buffer[dst_off : dst_off + nbytes]
            if torch.equal(staging_bytes, bump_bytes):
                ok += 1
            else:
                ndiff = (staging_bytes != bump_bytes).sum().item()
                logger.error(
                    f"  D2D CHUNK MISMATCH: block={bi} chunk={ci} "
                    f"{ndiff}/{nbytes} bytes differ"
                )
                bad += 1
    logger.info(f"  D2D chunk verify: {ok} OK, {bad} BAD (checked blocks 0-2)")


def _diag_verify_params(runner, h2d_src) -> None:
    """Byte-exact equality check for every GPU param against the CPU source.

    Runs only under SGLANG_SWITCH_DIAG=1. Intentionally does a full H2D copy
    per param — cheaper alternatives (sampling / checksum) would miss mid-tensor
    corruption from chunk-layout bugs (see weight_staging §10 C2/C3/C4) and
    isolated NaN/Inf clusters, so the whole point of this DIAG would be
    defeated. Slow switch latency under DIAG is acceptable.
    """
    torch.cuda.synchronize()
    ok = 0
    bad = []
    for name, cpu_t in h2d_src.items():
        parts = name.split(".")
        obj = runner.model
        try:
            for p in parts:
                obj = getattr(obj, p)
            gpu_t = obj.data
            cpu_ref = cpu_t.to(gpu_t.dtype).to(gpu_t.device)
            if torch.equal(gpu_t, cpu_ref):
                ok += 1
            else:
                diff = (gpu_t.float() - cpu_ref.float()).abs()
                bad.append(
                    f"{name}: max={diff.max().item():.4f} "
                    f"nan={gpu_t.isnan().any().item()}"
                )
        except Exception as e:
            logger.warning(f"  PARAM CHECK SKIP: {name}: {e}")
    logger.info(f"  D2D PARAM CHECK: {ok} OK, {len(bad)} BAD")
    for entry in bad[:10]:
        logger.error(f"  D2D PARAM MISMATCH: {entry}")


# ==========================================================================
# Weight load helpers (Phase 2 sub-strategies)
# ==========================================================================


def _load_via_d2d(ctx: _SwitchContext, cpu_model, cpu_sd) -> None:
    """Fast path: D2D scatter-gather from KV block tails into the weights region."""
    bump = ctx.bump
    runner = ctx.runner
    preload_mgr = ctx.scheduler.active_preload_manager

    weights_start = bump.regions["weights"].start
    needed_bytes = (
        sum(_align_up(t.numel() * t.element_size()) for t in cpu_sd.values())
        if cpu_sd
        else 0
    )
    current_cap = bump.regions["weights"].capacity
    if needed_bytes > 0 and needed_bytes != current_cap:
        t_rel = time.perf_counter()
        bump.release_region("kv_cache")
        logger.info(f"  kv_release: {(time.perf_counter()-t_rel)*1000:.1f}ms")
        t_resize = time.perf_counter()
        bump.reset_region("weights", needed_bytes)
        logger.info(
            f"  Weights region resized: {current_cap/1024**2:.1f}MB -> "
            f"{needed_bytes/1024**2:.1f}MB "
            f"time={(time.perf_counter()-t_resize)*1000:.1f}ms"
        )
        weights_start = bump.regions["weights"].start

    t_h2d_src = time.perf_counter()
    h2d_src = cpu_sd
    if h2d_src is None and cpu_model is not None:
        h2d_src = {
            k: v.detach().cpu().clone() for k, v in cpu_model.state_dict().items()
        }
    logger.info(f"  h2d_src_prep: {(time.perf_counter()-t_h2d_src)*1000:.1f}ms")

    total_target = needed_bytes or current_cap
    if _SWITCH_DIAG:
        t_diag = time.perf_counter()
        _diag_log_d2d_layout(
            preload_mgr, bump, weights_start, total_target, needed_bytes, current_cap
        )
        logger.info(f"  diag_pre_logging: {(time.perf_counter()-t_diag)*1000:.1f}ms")

    t_d2d = time.perf_counter()
    d2d_bytes, h2d_bytes = preload_mgr.gather_to_bump(
        bump=bump,
        weights_start=weights_start,
        cpu_sd=h2d_src,
        total_weight_target=total_target,
    )
    logger.info(
        f"  D2D scatter-gather: d2d={d2d_bytes/1024**2:.1f}MB "
        f"h2d={h2d_bytes/1024**2:.1f}MB "
        f"time={(time.perf_counter()-t_d2d)*1000:.1f}ms"
    )
    if _SWITCH_DIAG:
        _diag_verify_d2d_chunks(preload_mgr, bump, weights_start)

    t_assign = time.perf_counter()
    runner.model = cpu_model
    runner.model.eval()
    bump.regions["weights"].mark_full()
    logger.info(f"  model_assign+eval: {(time.perf_counter()-t_assign)*1000:.1f}ms")

    t_buf = time.perf_counter()
    runner._finalize_model_on_gpu()
    logger.info(f"  finalize_model_on_gpu: {(time.perf_counter()-t_buf)*1000:.1f}ms")

    if _SWITCH_DIAG and h2d_src is not None:
        _diag_verify_params(runner, h2d_src)


def _load_via_h2d_fallback(ctx: _SwitchContext, cpu_model, cpu_sd) -> None:
    """Fallback path: no valid staging — resize weights region and do full H2D."""
    bump = ctx.bump
    runner = ctx.runner

    logger.info(
        f"  FALLBACK: full H2D load for {ctx.target_model_name}, "
        f"cpu_model={cpu_model is not None}, staging={ctx.has_staging}"
    )
    # Must match _load_via_d2d's `needed_bytes = sum(cpu_sd.values())`.
    # Otherwise tied-weight models (OPT-125m's lm_head) get a smaller H2D
    # region than D2D, `save_kv_pool(weight_bytes)` records the smaller
    # number, and the next D2D round-trip's weights_match check fails,
    # dropping into _kv_rebuild + evict_graph whose empty_cache then
    # trips on stale pytorch blocks.
    if cpu_sd is not None:
        total_weight_bytes = sum(
            _align_up(v.numel() * v.element_size()) for v in cpu_sd.values()
        )
    elif cpu_model is not None:
        total_weight_bytes = sum(
            _align_up(v.numel() * v.element_size())
            for v in cpu_model.state_dict().values()
        )
    else:
        total_weight_bytes = 0

    _diag_cuda_mem("p2_fb_start", ctx.prev_model_name, ctx.target_model_name)
    bump.release_region("kv_cache")
    _diag_cuda_mem("p2_fb_after_release_kv", ctx.prev_model_name, ctx.target_model_name)
    if "weights" in bump.regions:
        bump.reset_region("weights", total_weight_bytes)
    else:
        bump.allocate_region("weights", total_weight_bytes)
    _diag_cuda_mem("p2_fb_after_reset_weights", ctx.prev_model_name, ctx.target_model_name)
    runner._load_model_bump(model_name=ctx.target_model_name, skip_region_alloc=True)
    _diag_cuda_mem("p2_fb_after_load_bump", ctx.prev_model_name, ctx.target_model_name)


# ==========================================================================
# KV cache helpers (Phase 3 sub-strategies)
# ==========================================================================


def _kv_restore_placeholder(ctx: _SwitchContext, kv_cached) -> None:
    """Placeholder KV "restore" path.

    Recycles cached allocator / pool *shells* to skip pool re-init, but
    immediately clears their contents — it does NOT restore the token-level
    KV data that was live on the old model. A real restore depends on an
    async per-token D2H offload + H2D reload scheme and a redesigned radix
    structure; that work is tracked for the next iteration.
    """
    scheduler = ctx.scheduler
    runner = ctx.runner
    bump = ctx.bump

    if "kv_cache" in bump.regions:
        bump.release_region("kv_cache")
    runner._init_runtime_region()

    # Resume per-model KV metadata physical pages before reusing cached pool
    # objects. Phase 1 paused tag=f"kv_cache:{prev}" when switching away;
    # the cached req_to_token tensor's VA was unmapped, so we must remap it
    # before any kernel touches it.
    if ctx.memory_saver is not None and ctx.memory_saver.enabled:
        for _t in ("kv_cache",):
            try:
                ctx.memory_saver.resume(f"{_t}:{ctx.target_model_name}")
            except Exception as e:
                logger.warning(f"  resume {_t}:{ctx.target_model_name} failed: {e}")

    # Allocate with exact cached capacity. _phase_3_kv_cache already verified
    # that this fits in the remaining bump space — using min() here would let
    # a smaller region be allocated while the cached pool's tensor views still
    # span the full old capacity, causing out-of-bounds decode reads.
    bump.allocate_region("kv_cache", kv_cached["kv_region_capacity"])

    runner.req_to_token_pool = kv_cached["req_to_token_pool"]
    runner.token_to_kv_pool = kv_cached["token_to_kv_pool"]
    runner.token_to_kv_pool_allocator = kv_cached["token_to_kv_pool_allocator"]
    runner.max_running_requests = kv_cached["max_running_requests"]
    runner.token_to_kv_pool_allocator.clear()
    # Recompute max_total_num_tokens from allocator's actual size
    # (KV region may be smaller than cached due to runtime size changes)
    runner.max_total_num_tokens = runner.token_to_kv_pool_allocator.available_size()

    _force_flush_cache(scheduler)
    logger.debug(f"  KV pool cache hit: {ctx.target_model_name}")


def _kv_rebuild(ctx: _SwitchContext) -> None:
    """Full KV rebuild: evict stale cache, detach old refs, reinit memory pool."""
    scheduler = ctx.scheduler
    runner = ctx.runner
    bump = ctx.bump

    _diag_cuda_mem("p3_rebuild_start", ctx.prev_model_name, ctx.target_model_name)
    ctx.resource_cache.evict_kv_pool(ctx.target_model_name)
    if "kv_cache" in bump.regions:
        bump.release_region("kv_cache")
    _diag_cuda_mem("p3_after_evict_release", ctx.prev_model_name, ctx.target_model_name)

    _force_flush_cache(scheduler)
    _diag_cuda_mem("p3_after_flush", ctx.prev_model_name, ctx.target_model_name)

    _detach_pool_refs(ctx)
    _diag_cuda_mem("p3_after_detach", ctx.prev_model_name, ctx.target_model_name)

    # Allocate runtime BEFORE KV so init_memory_pool doesn't consume all space.
    runner._init_runtime_region()
    _diag_cuda_mem("p3_after_runtime", ctx.prev_model_name, ctx.target_model_name)
    logger.info(
        f"  KV pool init: num_heads={runner.model_config.num_attention_heads}, "
        f"num_kv_heads={runner.model_config.get_num_kv_heads(ctx.tp_size)}, "
        f"head_dim={runner.model_config.head_dim}, "
        f"num_layers={runner.model_config.num_hidden_layers}"
    )
    runner.init_memory_pool(0)
    _diag_cuda_mem("p3_after_init_mem_pool", ctx.prev_model_name, ctx.target_model_name)


# ==========================================================================
# Graph helper (Phase 4 sub-strategy)
# ==========================================================================


def _recapture_graphs(ctx: _SwitchContext) -> None:
    """Restore a cached CUDA graph if possible, else recapture it.

    KV pool was rebuilt → graph captured with stale KV addresses → must recapture.
    TP > 1 → all ranks must agree on HIT / MISS (NCCL in graph capture).
    """
    scheduler = ctx.scheduler
    rc = ctx.resource_cache

    if ctx.kv_hit:
        graph_hit = rc.restore_graph(
            ctx.runner, ctx.target_model_name, memory_saver_adapter=ctx.memory_saver
        )
    else:
        graph_hit = False
        rc.evict_graph(ctx.target_model_name, memory_saver_adapter=ctx.memory_saver)

    if ctx.tp_size > 1:
        consensus_hit = _tp_consensus_bool(graph_hit, ctx.tp_cpu_group)
        if not consensus_hit and graph_hit:
            # Local HIT but some remote MISS: invalidate local, recapture together.
            rc.evict_graph(ctx.target_model_name, memory_saver_adapter=ctx.memory_saver)
        graph_hit = consensus_hit

    if graph_hit:
        return

    if ctx.tp_size > 1 and ctx.tp_cpu_group is not None:
        torch.distributed.barrier(group=ctx.tp_cpu_group)
    _diag_cuda_mem("p4_before_piecewise", ctx.prev_model_name, ctx.target_model_name)
    ctx.runner.init_piecewise_cuda_graphs()
    _diag_cuda_mem("p4_after_piecewise", ctx.prev_model_name, ctx.target_model_name)
    _set_graph_pool(None)
    ctx.runner.init_device_graphs()
    _diag_cuda_mem("p4_after_device_graphs", ctx.prev_model_name, ctx.target_model_name)
    if ctx.tp_size > 1 and ctx.tp_cpu_group is not None:
        torch.distributed.barrier(group=ctx.tp_cpu_group)


# ==========================================================================
# Phases
# ==========================================================================


def _phase_1_save_and_teardown(ctx: _SwitchContext) -> None:
    """Snapshot old model state + release GPU resources owned by it."""
    t0 = time.perf_counter()
    _tlast = [t0]
    def _tick(label):
        now = time.perf_counter()
        dt = (now - _tlast[0]) * 1000
        _tlast[0] = now
        logger.info(f"  P1_TIMING {label}: {dt:.1f}ms")

    _diag_cuda_mem("p1_start", ctx.prev_model_name, ctx.target_model_name)
    ctx.resource_cache.save_graph(
        ctx.runner, ctx.prev_model_name, memory_saver_adapter=ctx.memory_saver
    )
    _tick("save_graph")
    _diag_cuda_mem("p1_after_save_graph", ctx.prev_model_name, ctx.target_model_name)
    ctx.resource_cache.save_kv_pool(ctx.prev_model_name, ctx.runner, ctx.bump)
    _tick("save_kv_pool")
    _diag_cuda_mem("p1_after_save_kv", ctx.prev_model_name, ctx.target_model_name)

    ctx.runner.graph_runner = None
    _tick("clear_graph_runner")
    ctx.bump.release_region("runtime")
    _tick("release_runtime")
    _diag_cuda_mem("p1_after_release_runtime", ctx.prev_model_name, ctx.target_model_name)
    ctx.runner.attn_backend = None
    _tick("clear_attn_backend")
    clear_rope_cache()
    _tick("clear_rope_cache")
    _diag_cuda_mem("p1_after_clear_rope", ctx.prev_model_name, ctx.target_model_name)
    reset_global_workspace_buffer()
    _tick("reset_workspace_buffer")
    _diag_cuda_mem("p1_after_reset_ws", ctx.prev_model_name, ctx.target_model_name)

    ctx.timings["save_teardown"] = time.perf_counter() - t0
    logger.info(f"  save_teardown: {ctx.timings['save_teardown']*1000:.1f}ms")


def _phase_2_load_weights(ctx: _SwitchContext) -> None:
    """Resolve staging vs fallback, update model config, load weights."""
    t0 = time.perf_counter()

    # 2a. Decide staging availability (must agree across TP ranks)
    ctx.scheduler.wait_for_preload()
    preload_mgr = ctx.scheduler.active_preload_manager
    local_has_staging = (
        preload_mgr is not None
        and preload_mgr.model_name == ctx.target_model_name
        and preload_mgr.is_valid
    )
    if ctx.tp_size > 1:
        local_has_staging = _tp_consensus_bool(local_has_staging, ctx.tp_cpu_group)
    ctx.has_staging = local_has_staging
    if ctx.has_staging:
        t_wait = time.perf_counter()
        preload_mgr.wait_complete()
        logger.info(f"  wait_complete: {(time.perf_counter()-t_wait)*1000:.1f}ms")
        t_verify = time.perf_counter()
        preload_mgr.verify_integrity()
        logger.info(f"  verify_integrity: {(time.perf_counter()-t_verify)*1000:.1f}ms")

    # 2b. Update runner's model config.
    #
    # served_model_name must be updated here (not in Phase 5 finalize):
    # `memory_saver_adapter.cuda_graph(tag=f"cuda_graph:{served_model_name}")`
    # runs in Phase 4 and tags every captured allocation. If we wait until
    # finalize, new graph allocs land under the *previous* model's tag,
    # mixing PAUSED and ACTIVE entries for that tag and crashing the next
    # save_graph/resume cycle with "Cannot resume allocation that is not
    # paused".
    t_config = time.perf_counter()
    ctx.runner.server_args.model_path = ctx.target_model_path
    ctx.scheduler.server_args.served_model_name = ctx.target_model_name
    new_config = ModelConfig.from_server_args(
        ctx.runner.server_args, model_path=ctx.target_model_path
    )
    ctx.runner.model_config = new_config
    ctx.runner.start_layer = 0
    ctx.runner.end_layer = new_config.num_hidden_layers
    ctx.runner.num_effective_layers = new_config.num_hidden_layers
    ctx.runner.dtype = new_config.dtype
    ctx.runner.kv_cache_dtype = new_config.dtype
    ctx.new_config = new_config
    logger.info(f"  config_update: {(time.perf_counter()-t_config)*1000:.1f}ms")

    # 2c. Resolve CPU-side model / state_dict
    t_cpu = time.perf_counter()
    cpu_model = None
    cpu_sd = None
    host_mgr = ctx.scheduler.host_model_mgr
    if host_mgr is not None:
        cpu_model = host_mgr.get_cpu_model(ctx.target_model_name)
        entry = host_mgr.get_entry(ctx.target_model_name)
        if entry:
            cpu_sd = entry.cpu_state_dict
    logger.info(f"  get_cpu_model: {(time.perf_counter()-t_cpu)*1000:.1f}ms")

    # 2d. Load weights via D2D or H2D fallback
    if _SWITCH_DIAG:
        logger.debug(f"DIAG: cached={cpu_model is not None}, staging={ctx.has_staging}")
    if cpu_model is not None and ctx.has_staging:
        _load_via_d2d(ctx, cpu_model, cpu_sd)
    else:
        _load_via_h2d_fallback(ctx, cpu_model, cpu_sd)

    ctx.timings["load"] = time.perf_counter() - t0


def _phase_3_kv_cache(ctx: _SwitchContext) -> None:
    """Pick placeholder KV restore (cache-shell reuse) or full rebuild.

    Hit criteria:
      1. Weights region size matches cached — ensures new kv_cache region
         starts at the same offset as when it was cached (so cached pool
         tensor views still point to the right bump.buffer slice).
      2. Cached kv_region_capacity fits within the space available after we
         release the current kv_cache and re-allocate the runtime region —
         otherwise the cached pool's tensor views would span more memory
         than we allocate, and decode would read/write past the region.
    """
    t0 = time.perf_counter()

    bump = ctx.bump
    kv_cached = ctx.resource_cache.get_kv_pool(ctx.target_model_name)
    weights_match = bool(
        kv_cached and kv_cached["weight_bytes"] == bump.regions["weights"].capacity
    )
    cap_match = False
    if weights_match:
        current_kv_cap = (
            bump.regions["kv_cache"].capacity if "kv_cache" in bump.regions else 0
        )
        ws_size, buf_size = ctx.runner._estimate_runtime_bytes()
        space_for_kv = bump.get_available_bytes() + current_kv_cap - (ws_size + buf_size)
        cap_match = kv_cached["kv_region_capacity"] <= space_for_kv

    ctx.kv_hit = weights_match and cap_match
    if ctx.kv_hit:
        _kv_restore_placeholder(ctx, kv_cached)
    else:
        _kv_rebuild(ctx)

    ctx.timings["kv_cache"] = time.perf_counter() - t0
    logger.info(f"  kv_cache: {ctx.timings['kv_cache']*1000:.1f}ms")


def _phase_4_runtime_and_graph(ctx: _SwitchContext) -> None:
    """Rebuild attention backend; restore-or-recapture CUDA graphs.

    When the KV pool hit AND the graph cache holds the target model, skip
    the ~140ms `init_attention_backend` (B200 CUTLASS FlashInfer wrapper
    construction) and restore attn_backend + graph_runner straight from
    the cache in a single pass.
    """
    t0 = time.perf_counter()

    graph_cache_has = (
        not ctx.scheduler.server_args.disable_cuda_graph
        and ctx.kv_hit
        and ctx.target_model_name in ctx.resource_cache.graph_cache
    )

    if graph_cache_has:
        graph_hit = ctx.resource_cache.restore_graph(
            ctx.runner,
            ctx.target_model_name,
            memory_saver_adapter=ctx.memory_saver,
        )
        _diag_cuda_mem("p4_after_restore_graph", ctx.prev_model_name, ctx.target_model_name)
        if not graph_hit:
            if ctx.memory_saver is not None and ctx.memory_saver.enabled:
                with ctx.memory_saver.region("attn"):
                    ctx.runner.init_attention_backend()
            else:
                ctx.runner.init_attention_backend()
            _diag_cuda_mem("p4_after_init_attn", ctx.prev_model_name, ctx.target_model_name)
            clear_forward_input_buffer_pool()
            _recapture_graphs(ctx)
            _diag_cuda_mem("p4_after_recapture", ctx.prev_model_name, ctx.target_model_name)
    else:
        if ctx.memory_saver is not None and ctx.memory_saver.enabled:
            with ctx.memory_saver.region("attn"):
                ctx.runner.init_attention_backend()
        else:
            ctx.runner.init_attention_backend()
        _diag_cuda_mem("p4_after_init_attn", ctx.prev_model_name, ctx.target_model_name)
        clear_forward_input_buffer_pool()
        if not ctx.scheduler.server_args.disable_cuda_graph:
            _recapture_graphs(ctx)
            _diag_cuda_mem("p4_after_recapture", ctx.prev_model_name, ctx.target_model_name)

    ctx.timings["runtime_graph"] = time.perf_counter() - t0
    logger.info(f"  runtime_graph: {ctx.timings['runtime_graph']*1000:.1f}ms")


def _phase_5_finalize(ctx: _SwitchContext) -> None:
    """Propagate refs back to scheduler / worker and record total time."""
    _propagate_refs(ctx)

    if ctx.memory_saver is not None and ctx.memory_saver.enabled and ctx.prev_model_name:
        _diag_cuda_mem("p5_before_pause", ctx.prev_model_name, ctx.target_model_name)
        for _t in ("kv_cache",):
            try:
                ctx.memory_saver.pause(f"{_t}:{ctx.prev_model_name}")
            except Exception as e:
                logger.warning(f"  pause {_t}:{ctx.prev_model_name} failed: {e}")
        _diag_cuda_mem("p5_after_pause", ctx.prev_model_name, ctx.target_model_name)

    ctx.timings["total"] = time.perf_counter() - ctx.t_total_start
    logger.info(
        f"  SWITCH {ctx.prev_model_name} -> {ctx.target_model_name}: "
        f"save={ctx.timings['save_teardown']*1000:.1f}ms "
        f"load={ctx.timings['load']*1000:.1f}ms "
        f"kv={ctx.timings['kv_cache']*1000:.1f}ms "
        f"rt+graph={ctx.timings['runtime_graph']*1000:.1f}ms "
        f"TOTAL={ctx.timings['total']*1000:.1f}ms"
    )


# ==========================================================================
# Entry
# ==========================================================================


def _diag_cuda_mem(tag: str, prev_model: str, target_model: str) -> None:
    """[DIAG] Print torch.cuda.memory_stats key fields for OOM root-cause analysis.

    SGLANG_SWITCH_DIAG levels:
      unset / '0': skip all probes (zero overhead, fastest switch)
      'bounds'   : only pre_switch / post_switch (measure fresh 显存净增, minimal latency impact)
      '1' / 'full': all probes (heavy gc+sync per probe, breaks D2D latency)
    """
    import os
    lvl = os.environ.get('SGLANG_SWITCH_DIAG', '0')
    if lvl == '0':
        return
    if lvl == 'bounds' and tag not in ('pre_switch', 'post_switch'):
        return
    try:
        import torch
        st = torch.cuda.memory_stats(0)
        # Pick informative fields
        fields = [
            ("allocated_bytes.all.current", "allocated"),
            ("reserved_bytes.all.current", "reserved"),
            ("active_bytes.all.current", "active"),
            ("inactive_split_bytes.all.current", "inactive_split"),
            ("segment.all.current", "segments"),
            ("num_alloc_retries", "alloc_retries"),
            ("num_ooms", "ooms"),
        ]
        parts = []
        for k, label in fields:
            v = st.get(k, 0)
            if k.endswith("_bytes.all.current"):
                parts.append(f"{label}={v/1024**2:.0f}MB")
            else:
                parts.append(f"{label}={v}")
        import gc
        gc.collect()
        torch.cuda.synchronize(0)
        free_b, tot_b = torch.cuda.mem_get_info(0)
        parts.append(f"driver_free={free_b/1024**2:.0f}MB")
        parts.append(f"driver_used={(tot_b-free_b)/1024**2:.0f}MB")
        logger.info(f"[DIAG {tag}] {prev_model}->{target_model} | " + " ".join(parts))
    except Exception as e:
        logger.warning(f"[DIAG {tag}] failed: {e}")


def do_model_switch_bump(
    scheduler: "Scheduler",
    target_model_path: str,
    target_model_name: Optional[str] = None,
) -> Dict[str, float]:
    """Switch model without releasing bump regions.

    D2D staging + in-place weights overwrite. See the 5-phase docstring at the
    top of the module for the overall contract.
    """
    logger.info(
        f"Bump switch: {scheduler.server_args.model_path} -> {target_model_path}"
    )
    ctx = _build_context(scheduler, target_model_path, target_model_name)
    _diag_cuda_mem("pre_switch", ctx.prev_model_name, ctx.target_model_name)
    # Reset the total timer AFTER pre_switch probe so DIAG overhead (gc.collect +
    # cuda.synchronize + mem_get_info ~50-100ms) is not counted in switch latency.
    import time as _t
    ctx.t_total_start = _t.perf_counter()
    _phase_1_save_and_teardown(ctx)
    _diag_cuda_mem("after_phase1", ctx.prev_model_name, ctx.target_model_name)
    _phase_2_load_weights(ctx)
    from sglang.srt.utils.torch_memory_saver_adapter import set_per_model_tag_suffix as _set_suffix
    _set_suffix(ctx.target_model_name)
    try:
        _phase_3_kv_cache(ctx)
        _diag_cuda_mem("after_phase3", ctx.prev_model_name, ctx.target_model_name)
        _phase_4_runtime_and_graph(ctx)
        _diag_cuda_mem("after_phase4", ctx.prev_model_name, ctx.target_model_name)
    finally:
        _set_suffix("")
    _phase_5_finalize(ctx)
    _diag_cuda_mem("post_switch", ctx.prev_model_name, ctx.target_model_name)
    return ctx.timings
