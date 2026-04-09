"""
Multi-model manager: model hot-switching with D2D staging + bump allocator.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import ModelConfig

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Per-model CUDA graph + attn_backend cache
_graph_cache = {}


def _get_graph_pool():
    from sglang.srt.model_executor.cuda_graph_runner import get_global_graph_memory_pool

    return get_global_graph_memory_pool()


def _set_graph_pool(pool):
    from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool

    set_global_graph_memory_pool(pool)


_kv_pool_cache = {}
_model_cache = {}
_runtime_cache = {}  # model_name -> runtime region capacity


def _save_graph_cache(runner, model_name, memory_saver_adapter=None):
    """Cache CUDA graph runner + attn_backend, and pause VMM graph pool."""
    if runner.graph_runner is None:
        return
    try:
        from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool

        # Save CUDA buffers (cos_sin_cache etc.) for address-stable restore
        saved_buffers = {}
        if runner.model is not None:
            for n, b in runner.model.named_buffers():
                if b is not None and b.device.type == "cuda" and b.numel() > 0:
                    saved_buffers[n] = b
        vmm_tag = f"cuda_graph:{model_name}"
        _graph_cache[model_name] = {
            "vmm_graph_tag": vmm_tag,
            "graph_runner": runner.graph_runner,
            "attn_backend": runner.attn_backend,
            "input_buffer_pool": dict(_forward_input_buffer_pool),
            "graph_pool_handle": _get_graph_pool(),
            "model_buffers": saved_buffers,
        }
        # Pause VMM physical pages for this model's graph pool
        if memory_saver_adapter is not None and memory_saver_adapter.enabled:
            memory_saver_adapter.pause(vmm_tag)
            logger.debug(f"  VMM: paused graph pool tag={vmm_tag}")
        logger.info(f"  Graph cache saved for {model_name}")
    except Exception as e:
        logger.warning(f"  Graph cache save failed: {e}")


def _get_module_by_path(root, dotted_name):
    """Walk dotted path, return (parent_module, last_attr_name)."""
    parts = dotted_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


def _restore_graph_cache(runner, model_name, memory_saver_adapter=None):
    """Restore cached CUDA graph runner + attn_backend, and resume VMM graph pool."""
    if model_name not in _graph_cache:
        return False
    try:
        cached = _graph_cache[model_name]
        # Resume VMM physical pages before restoring graph
        if memory_saver_adapter is not None and memory_saver_adapter.enabled:
            resume_tag = cached.get("vmm_graph_tag", f"cuda_graph:{model_name}")
            memory_saver_adapter.resume(resume_tag)
            logger.debug(f"  VMM: resumed graph pool tag={resume_tag}")
        runner.graph_runner = cached["graph_runner"]
        # Restore the graph memory pool (each model must use its own pool)
        saved_pool = cached.get("graph_pool_handle")
        if saved_pool is not None:
            _set_graph_pool(saved_pool)
        runner.attn_backend = cached["attn_backend"]

        # Update stale internal references in restored attn_backend
        ab = runner.attn_backend
        new_req_to_token = runner.req_to_token_pool.req_to_token
        new_allocator = runner.token_to_kv_pool_allocator
        # indices_updater_decode holds req_to_token and allocator refs from old init
        if hasattr(ab, "indices_updater_decode"):
            ab.indices_updater_decode.req_to_token = new_req_to_token
            ab.indices_updater_decode.token_to_kv_pool_allocator = new_allocator
        # indices_updater_prefill too
        if hasattr(ab, "indices_updater_prefill"):
            if hasattr(ab.indices_updater_prefill, "req_to_token"):
                ab.indices_updater_prefill.req_to_token = new_req_to_token
            if hasattr(ab.indices_updater_prefill, "token_to_kv_pool_allocator"):
                ab.indices_updater_prefill.token_to_kv_pool_allocator = new_allocator

        from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool

        _forward_input_buffer_pool.clear()
        _forward_input_buffer_pool.update(cached["input_buffer_pool"])

        # Restore model buffers - skip copy if model cache hit (same addresses)
        _is_model_cached = model_name in _model_cache
        saved_buffers = cached.get("model_buffers", {})
        if saved_buffers and runner.model is not None:
            if _is_model_cached:
                # Model cache hit: copy current buffer data to graph-known addresses.
                # D2D + recompute may have updated data; copy it to saved tensor
                # (whose address was captured by CUDA graph) then replace ref.
                for buf_name, saved_buf in saved_buffers.items():
                    try:
                        module, attr = _get_module_by_path(runner.model, buf_name)
                        current_buf = module._buffers.get(attr)
                        if current_buf is not None and current_buf.shape == saved_buf.shape:
                            saved_buf.copy_(current_buf)
                        module._buffers[attr] = saved_buf
                    except (AttributeError, KeyError):
                        pass
            else:
                for buf_name, saved_buf in saved_buffers.items():
                    try:
                        module, attr = _get_module_by_path(runner.model, buf_name)
                        new_buf = module._buffers.get(attr)
                        if (
                            new_buf is not None
                            and new_buf.shape == saved_buf.shape
                            and new_buf.dtype == saved_buf.dtype
                        ):
                            saved_buf.copy_(new_buf)
                        module._buffers[attr] = saved_buf
                    except (AttributeError, KeyError):
                        pass

        # Kept for reuse on next switch
        # Diagnostic: log key tensor addresses after restore
        logger.info(f"  Graph cache restored for {model_name}")
        return True
    except Exception as e:
        logger.warning(f"  Graph cache restore failed: {e}")
        if model_name in _graph_cache:
            del _graph_cache[model_name]
        return False


def _propagate_refs(scheduler, runner, new_config, target_model_path):
    """Sync runner state to scheduler and worker after model switch."""
    worker = scheduler.tp_worker
    worker.model_config = new_config
    worker.max_total_num_tokens = runner.max_total_num_tokens
    worker.max_running_requests = runner.max_running_requests
    worker.max_req_len = min(new_config.context_len - 1, runner.max_token_pool_size - 1)
    worker.max_req_input_len = worker.max_req_len - 5
    scheduler.req_to_token_pool = runner.req_to_token_pool
    scheduler.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
    scheduler.max_total_num_tokens = runner.max_total_num_tokens
    scheduler.max_running_requests = runner.max_running_requests
    scheduler.max_req_len = worker.max_req_len
    scheduler.max_req_input_len = worker.max_req_input_len
    if scheduler.tree_cache is not None:
        scheduler.tree_cache.req_to_token_pool = runner.req_to_token_pool
        scheduler.tree_cache.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator
    scheduler.model_config = new_config
    scheduler.server_args.model_path = target_model_path
    scheduler.offload_tags.clear()


def _serialize_cpu_state_dict(cpu_sd, padded_len):
    """Serialize CPU state_dict into a padded contiguous byte tensor."""
    from sglang.srt.mem_cache.weight_staging import PreloadManager
    return PreloadManager._serialize_params(cpu_sd, padded_len, 1, padded_len)


def _scatter_gather_weights(si, blocks, bump, weights_start, cpu_sd, total_weight_target):
    """D2D gather valid chunks from staging, H2D corrupted chunks from CPU.

    Args:
        si: ScatterStagingInfo from preload
        blocks: list of KV layer block tensors
        bump: BumpVramAllocator
        weights_start: byte offset of weights region in bump.buffer
        cpu_sd: CPU state_dict for H2D fallback (may be None)
        total_weight_target: total weight bytes needed (for unstaged tail)
    Returns:
        (d2d_bytes, h2d_bytes) tuple
    """
    d2d_stream = torch.cuda.Stream()
    h2d_stream = torch.cuda.Stream()

    chunk_valid = si.chunk_valid or []
    chunk_rows = si.chunk_rows
    num_chunks = len(chunk_valid)
    d2d_bytes = 0
    h2d_bytes = 0

    # Prepare padded CPU stream for H2D fallback (only if corrupted chunks exist)
    has_corrupted = any(not v for v in chunk_valid)
    cpu_stream = None
    if has_corrupted and cpu_sd is not None:
        padded_len = si.num_blocks * si.staged_per_block
        cpu_stream = _serialize_cpu_state_dict(cpu_sd, padded_len)

    # Per-chunk: valid → D2D from staging, corrupted → H2D from CPU
    # Byte layout: lowest staging rows = first bytes (reversed during H2D)
    for ci in range(num_chunks):
        row_start = si.staging_start + ci * chunk_rows
        row_end = min(row_start + chunk_rows, si.pool_size)
        byte_off = ci * chunk_rows * si.row_size
        nbytes = min((row_end - row_start) * si.row_size,
                     si.staged_per_block - byte_off)
        if nbytes <= 0:
            break

        src_off = row_start * si.row_size

        if chunk_valid[ci]:
            with torch.cuda.stream(d2d_stream):
                for bi in range(si.num_blocks):
                    src = blocks[bi].view(-1).view(torch.uint8)[src_off : src_off + nbytes]
                    dst = weights_start + bi * si.staged_per_block + byte_off
                    bump.buffer[dst : dst + nbytes].copy_(src, non_blocking=True)
            d2d_bytes += nbytes * si.num_blocks
        elif cpu_stream is not None:
            with torch.cuda.stream(h2d_stream):
                for bi in range(si.num_blocks):
                    cpu_off = bi * si.staged_per_block + byte_off
                    dst = weights_start + cpu_off
                    bump.buffer[dst : dst + nbytes].copy_(
                        cpu_stream[cpu_off : cpu_off + nbytes], non_blocking=True
                    )
            h2d_bytes += nbytes * si.num_blocks

    # H2D for unstaged params (didn't fit in staging)
    if cpu_sd is not None and si.total_staged_bytes < total_weight_target:
        with torch.cuda.stream(h2d_stream):
            _h2d_remaining_params(
                cpu_sd, bump,
                weights_start + si.total_staged_bytes,
                si.total_staged_bytes, total_weight_target,
            )
        h2d_bytes += total_weight_target - si.total_staged_bytes

    d2d_stream.synchronize()
    h2d_stream.synchronize()
    return d2d_bytes, h2d_bytes


def _h2d_remaining_params(cpu_sd: dict, bump, dst_offset: int, skip_bytes: int, total_bytes: int):
    """H2D params that weren't staged (the tail of the byte stream)."""
    pos = 0
    for name, tensor in cpu_sd.items():
        nbytes = tensor.numel() * tensor.element_size()
        end = pos + nbytes
        if end <= skip_bytes:
            pos = end
            continue
        # This param has some bytes beyond skip_bytes
        start_in_param = max(0, skip_bytes - pos)
        take = min(nbytes - start_in_param, total_bytes - max(pos, skip_bytes))
        if take <= 0:
            break
        cpu_bytes = tensor.contiguous().view(-1).view(torch.uint8)[start_in_param : start_in_param + take]
        write_off = dst_offset + max(pos, skip_bytes) - skip_bytes
        bump.buffer[write_off : write_off + take].copy_(cpu_bytes, non_blocking=True)
        pos = end
        if pos >= total_bytes:
            break


def do_model_switch_bump(scheduler, target_model_path, target_model_name=None):
    """Switch model without releasing bump regions. D2D staging + in-place overwrite.

    5 phases:
      1. Save & teardown — snapshot old model state, release runtime/graph/attn
      2. Load weights — verify staging integrity, D2D/H2D weights
      3. KV cache — restore or rebuild KV pool
      4. Runtime & CUDA graph — allocate runtime, restore or recapture graphs
      5. Finalize — propagate refs to scheduler/worker
    """
    runner = scheduler.tp_worker.model_runner
    bump = runner.bump_vram_manager
    timings = {}
    t_total = time.perf_counter()
    current_model_name = scheduler.active_model_name
    logger.info(f"Bump switch: {scheduler.server_args.model_path} -> {target_model_path}")

    # === Phase 1: Save & teardown ===
    t0 = time.perf_counter()

    # 1a. Save caches for fast switch-back
    if current_model_name not in _graph_cache:
        _msa = scheduler.memory_saver_adapter if scheduler.memory_saver_adapter.enabled else None
        _save_graph_cache(runner, current_model_name, memory_saver_adapter=_msa)
    _kv_pool_cache[current_model_name] = {
        "req_to_token_pool": runner.req_to_token_pool,
        "token_to_kv_pool": runner.token_to_kv_pool,
        "token_to_kv_pool_allocator": runner.token_to_kv_pool_allocator,
        "max_total_num_tokens": runner.max_total_num_tokens,
        "max_running_requests": runner.max_running_requests,
        "weight_bytes": bump.regions["weights"].capacity,
        "kv_region_capacity": bump.regions["kv_cache"].capacity,
    }
    _model_cache[current_model_name] = runner.model
    _runtime_cache[current_model_name] = bump.regions["runtime"].capacity

    # 1b. Teardown old model state
    runner.graph_runner = None
    bump.release_region("runtime")
    runner.attn_backend = None
    from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
    _ROPE_DICT.clear()
    import sglang.srt.layers.attention.flashinfer_backend as _fi_mod
    _fi_mod.global_workspace_buffer = None

    timings["save_teardown"] = time.perf_counter() - t0
    logger.debug(f"  save_teardown: {timings['save_teardown']*1000:.1f}ms")

    # === Phase 2: Load weights ===
    t0 = time.perf_counter()

    # 2a. Verify staging integrity (before clearing allocator)
    _pt = scheduler._preload_thread
    if _pt is not None and _pt.is_alive():
        _pt.join()
    _preload_mgr = scheduler._preload_manager
    _has_staging = (
        _preload_mgr is not None
        and _preload_mgr.model_name == target_model_name
        and _preload_mgr.is_valid
    )
    if _has_staging:
        _preload_mgr.wait_complete()
        _preload_mgr.verify_integrity(scheduler.token_to_kv_pool_allocator)
        _si = _preload_mgr.scatter_info
        _n_corrupted = sum(1 for v in (_si.chunk_valid or []) if not v)
        logger.debug(
            f"  Staging: {_si.total_staged_bytes / 1024**2:.1f}MB across {_si.num_blocks} blocks, "
            f"corrupted_chunks={_n_corrupted}/{len(_si.chunk_valid or [])}"
        )

    # 2b. Update config for target model
    runner.server_args.model_path = target_model_path
    new_config = ModelConfig.from_server_args(runner.server_args, model_path=target_model_path)
    runner.model_config = new_config
    runner.start_layer = 0
    runner.end_layer = new_config.num_hidden_layers
    runner.num_effective_layers = new_config.num_hidden_layers
    runner.dtype = new_config.dtype
    runner.kv_cache_dtype = new_config.dtype
    bump._current_model = target_model_name

    # 2c. Get CPU model data
    _cpu_model = None
    _cpu_sd = None
    if scheduler._cpu_model_cache is not None:
        _cpu_model, _ = scheduler._cpu_model_cache.get_cpu_model(target_model_name)
        _reg_info = scheduler._cpu_model_cache.get_entry(target_model_name)
        if _reg_info:
            _cpu_sd = _reg_info.cpu_state_dict

    # 2d. Load weights: D2D from staging or full H2D
    _cached_model = _model_cache.get(target_model_name)
    if _cached_model is not None and _has_staging:
        # D2D scatter-gather from KV block tails to weights region
        layer_map = bump.get_layer_map(target_model_name)
        weights_start = bump.regions["weights"].start
        if layer_map:
            needed_bytes = max(s.offset + s.nbytes for s in layer_map) - weights_start
        else:
            needed_bytes = 0
        current_cap = bump.regions["weights"].capacity
        if needed_bytes > current_cap:
            bump.release_region("kv_cache")
            bump.reset_region("weights", needed_bytes)
            logger.info(f"  Weights region resized: {current_cap/1024**2:.1f}MB -> {needed_bytes/1024**2:.1f}MB")
            weights_start = bump.regions["weights"].start

        h2d_src = _cpu_sd
        if h2d_src is None and _cpu_model is not None:
            h2d_src = {k: v.detach().cpu().clone() for k, v in _cpu_model.state_dict().items()}

        total_target = needed_bytes or current_cap
        # DEBUG: check staging data before D2D
        _si_dbg = _preload_mgr.scatter_info
        _blk0 = _preload_mgr.block_tensors[0]
        _stg_start = _si_dbg.staging_start * _si_dbg.row_size
        _stg_data = _blk0.view(-1).view(torch.uint8)[_stg_start:_stg_start+32]
        logger.info(f"  DEBUG staging pre-D2D: block0[{_stg_start}:{_stg_start+32}] = {_stg_data[:16].tolist()}")
        logger.info(f"  DEBUG staging nonzero: {(_stg_data != 0).sum().item()}/32")
        d2d_bytes, h2d_bytes = _scatter_gather_weights(
            si=_preload_mgr.scatter_info,
            blocks=_preload_mgr.block_tensors,
            bump=bump,
            weights_start=weights_start,
            cpu_sd=h2d_src,
            total_weight_target=total_target,
        )

        runner.model = _cached_model
        runner.model.eval()
        bump.regions["weights"]._sub_offset = bump.regions["weights"].capacity

        # Move non-persistent buffers (cos_sin_cache etc.) to GPU
        for name, buf in runner.model.named_buffers():
            if buf is not None and buf.device.type == "cpu":
                parts = name.split(".")
                module = runner.model
                for part in parts[:-1]:
                    module = getattr(module, part)
                module._buffers[parts[-1]] = buf.to(runner.device)

        # Recompute RoPE cache for new model
        from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
        _ROPE_DICT.clear()
        from sglang.srt.utils.common import reserve_rope_cache_for_long_sequences
        reserve_rope_cache_for_long_sequences(runner.model, runner.server_args, runner.model_config)
        bump._current_model = target_model_name
        logger.info(f"  Scatter D2D: {d2d_bytes / 1024**2:.1f}MB, H2D fallback: {h2d_bytes / 1024**2:.1f}MB")

        # DEBUG: compare param ordering
        if h2d_src is not None:
            sd_keys = list(h2d_src.keys())[:5]
            np_keys = [n for n, _ in runner.model.named_parameters()][:5]
            logger.info(f"  DEBUG key order: sd={sd_keys}")
            logger.info(f"  DEBUG key order: np={np_keys}")
            logger.info(f"  DEBUG keys match: {sd_keys == np_keys}")

        # DEBUG: compare raw bump bytes with CPU serialized stream
        if h2d_src is not None:
            torch.cuda.synchronize()
            from sglang.srt.mem_cache.weight_staging import PreloadManager
            padded_len = _preload_mgr.scatter_info.num_blocks * _preload_mgr.scatter_info.staged_per_block
            cpu_ref = PreloadManager._serialize_params(h2d_src, padded_len, 1, padded_len)
            gpu_bytes = bump.buffer[weights_start : weights_start + min(padded_len, len(cpu_ref))].cpu()
            n_compare = len(cpu_ref)  # compare ALL bytes  # compare first 1MB
            match = torch.equal(gpu_bytes[:n_compare], cpu_ref[:n_compare])
            if not match:
                diff_mask = (gpu_bytes[:n_compare] != cpu_ref[:n_compare])
                n_diff = diff_mask.sum().item()
                first_diff = diff_mask.nonzero(as_tuple=True)[0][0].item() if n_diff > 0 else -1
                logger.error(f"  RAW BYTE MISMATCH: {n_diff}/{n_compare} bytes differ, first at offset {first_diff}")
                logger.error(f"    GPU[{first_diff}:{first_diff+8}] = {gpu_bytes[first_diff:first_diff+8].tolist()}")
                logger.error(f"    CPU[{first_diff}:{first_diff+8}] = {cpu_ref[first_diff:first_diff+8].tolist()}")
            else:
                logger.info(f"  RAW BYTE VERIFY: first {n_compare} bytes match!")

        # DEBUG: verify weights after D2D scatter gather
        # First, check if D2D wrote to correct location
        logger.info(f"  DEBUG D2D: weights_start={weights_start}, staged_per_block={_preload_mgr.scatter_info.staged_per_block}, num_blocks={_preload_mgr.scatter_info.num_blocks}")
        # Check model params' actual bump offsets
        bump_base = bump.buffer.data_ptr()
        for i, (name, p) in enumerate(runner.model.named_parameters()):
            if i >= 3: break
            param_ptr = p.data.data_ptr()
            param_offset = param_ptr - bump_base
            param_bytes = p.data.numel() * p.data.element_size()
            logger.info(f"  DEBUG param {name}: offset={param_offset}, nbytes={param_bytes}, shape={p.data.shape}")
        if h2d_src is not None:
            torch.cuda.synchronize()
            n_checked = 0
            n_mismatch = 0
            for name, cpu_t in list(h2d_src.items())[:5]:  # check first 5 params
                # Find GPU tensor in model
                parts = name.split(".")
                obj = runner.model
                try:
                    for p in parts:
                        obj = getattr(obj, p)
                    gpu_t = obj.data
                    cpu_ref = cpu_t.to(gpu_t.dtype).to(gpu_t.device)
                    if not torch.equal(gpu_t, cpu_ref):
                        max_diff = (gpu_t.float() - cpu_ref.float()).abs().max().item()
                        logger.warning(f"  WEIGHT MISMATCH: {name} max_diff={max_diff:.6f} shape={gpu_t.shape}")
                        n_mismatch += 1
                    n_checked += 1
                except (AttributeError, RuntimeError) as e:
                    pass
            if n_mismatch > 0:
                logger.error(f"  WEIGHT VERIFY: {n_mismatch}/{n_checked} params MISMATCHED!")
            else:
                logger.info(f"  WEIGHT VERIFY: {n_checked} params checked, all match")
    else:
        # Full load via _load_model_bump from CPU or disk
        if _cpu_model is not None:
            param_bytes = sum(p.numel() * p.element_size() for p in _cpu_model.parameters())
            buf_bytes = sum(b.numel() * b.element_size() for b in _cpu_model.buffers()
                           if b is not None and b.numel() > 0)
            total_weight_bytes = param_bytes + buf_bytes
        elif _cpu_sd is not None:
            total_weight_bytes = sum(v.numel() * v.element_size() for v in _cpu_sd.values())
        else:
            total_weight_bytes = 0
        bump.release_region("kv_cache")
        if "weights" in bump.regions:
            bump.reset_region("weights", total_weight_bytes)
        else:
            bump.allocate_region("weights", total_weight_bytes)
        runner._load_model_bump(model_name=target_model_name, staging_info=None, skip_region_alloc=True)

    timings["load"] = time.perf_counter() - t0
    logger.info(f'  load: {timings["load"]:.3f}s')

    # === Phase 3: KV cache rebuild ===
    t0 = time.perf_counter()

    _kv_cached = _kv_pool_cache.get(target_model_name)
    if _kv_cached:
        if "kv_cache" in bump.regions:
            bump.release_region("kv_cache")
        # Estimate runtime size from _init_runtime_region logic
        from sglang.srt.environ import envs
        _ws = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
        archs = getattr(runner.model_config.hf_config, "architectures", []) or []
        if any(a.startswith(("Qwen2", "Qwen3", "MiMo")) for a in archs):
            _ws = max(_ws, 512 * 1024 * 1024)
        _kv_cap = min(_kv_cached["kv_region_capacity"], bump.get_available_bytes() - _ws)
        _kv_cap = max(_kv_cap, 0)
        bump.allocate_region("kv_cache", _kv_cap)
        runner.req_to_token_pool = _kv_cached["req_to_token_pool"]
        runner.token_to_kv_pool = _kv_cached["token_to_kv_pool"]
        runner.token_to_kv_pool_allocator = _kv_cached["token_to_kv_pool_allocator"]
        runner.max_total_num_tokens = _kv_cached["max_total_num_tokens"]
        runner.max_running_requests = _kv_cached["max_running_requests"]
        runner.token_to_kv_pool_allocator.clear()
        # Force flush: tree_cache.reset() + allocator.clear() (bypass idle check)
        scheduler.tree_cache.reset()
        if scheduler.token_to_kv_pool_allocator is not None:
            scheduler.token_to_kv_pool_allocator.clear()
        scheduler.grammar_manager.clear()
        logger.info("Cache force-flushed for model switch")
        logger.debug(f"  KV pool cache hit: {target_model_name}")
    else:
        if "kv_cache" in bump.regions:
            bump.release_region("kv_cache")
        # Force flush: tree_cache.reset() + allocator.clear() (bypass idle check)
        scheduler.tree_cache.reset()
        if scheduler.token_to_kv_pool_allocator is not None:
            scheduler.token_to_kv_pool_allocator.clear()
        scheduler.grammar_manager.clear()
        logger.info("Cache force-flushed for model switch")
        for attr in ["req_to_token_pool", "token_to_kv_pool", "token_to_kv_pool_allocator"]:
            for obj in [runner, scheduler, scheduler.tp_worker]:
                setattr(obj, attr, None)
        # Allocate runtime BEFORE KV so init_memory_pool doesn't consume all space
        # Always use _init_runtime_region for correct workspace sizing
        runner._init_runtime_region()
        logger.info(f"  KV pool init: num_heads={runner.model_config.num_attention_heads}, "
                    f"num_kv_heads={runner.model_config.get_num_kv_heads(1)}, "
                    f"head_dim={runner.model_config.head_dim}, "
                    f"num_layers={runner.model_config.num_hidden_layers}")
        runner.init_memory_pool(0)

    timings["kv_cache"] = time.perf_counter() - t0
    logger.debug(f"  kv_cache: {timings['kv_cache']*1000:.1f}ms")

    # === Phase 4: Attention backend + CUDA graph ===
    t0 = time.perf_counter()
    runner.init_attention_backend()
    # Update piecewise graph runner's cached attention/moe layers for new model
    runner.init_piecewise_cuda_graphs()

    from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
    _forward_input_buffer_pool.clear()

    if not scheduler.server_args.disable_cuda_graph:
        _msa = scheduler.memory_saver_adapter if scheduler.memory_saver_adapter.enabled else None
        if not _restore_graph_cache(runner, target_model_name, memory_saver_adapter=_msa):
            _set_graph_pool(None)
            runner.init_device_graphs()

    timings["runtime_graph"] = time.perf_counter() - t0
    logger.debug(f"  runtime_graph: {timings['runtime_graph']*1000:.1f}ms")

    # === Phase 5: Finalize ===
    _propagate_refs(scheduler, runner, new_config, target_model_path)

    # DEBUG: verify model attention head config after switch
    for i, layer in enumerate(runner.model.modules()):
        cls_name = type(layer).__name__
        if cls_name == "RadixAttention":
            logger.info(f"  DEBUG RadixAttention layer {i}: tp_q_head_num={layer.tp_q_head_num}, head_dim={layer.head_dim}")
            break

    timings["total"] = time.perf_counter() - t_total
    logger.info(
        f"  SWITCH {current_model_name} -> {target_model_name}: "
        f"save={timings.get('save_teardown',0)*1000:.1f}ms "
        f"load={timings.get('load',0)*1000:.1f}ms "
        f"kv={timings.get('kv_cache',0)*1000:.1f}ms "
        f"rt+graph={timings.get('runtime_graph',0)*1000:.1f}ms "
        f"TOTAL={timings['total']*1000:.1f}ms"
    )

    return timings
