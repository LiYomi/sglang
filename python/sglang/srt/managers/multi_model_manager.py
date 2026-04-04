"""
Multi-model manager: model hot-switching with D2D staging + bump allocator.
"""

from __future__ import annotations

import gc
import logging
import time
from typing import TYPE_CHECKING

import torch

from sglang.srt.configs.model_config import ModelConfig

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Per-model KV cache snapshots for offload/restore across switches
_kv_snapshots = {}

# Per-model CUDA graph + attn_backend cache
_graph_cache = {}


def _save_kv_snapshot(tree_cache, allocator, model_name):
    """Save radix tree state + KV cache data to CPU before model switch."""
    # Access inner RadixCache through SessionAwareCache
    inner = getattr(tree_cache, 'inner', tree_cache)

    if inner.disable or inner.evictable_size() == 0:
        return

    try:
        all_indices = tree_cache.all_values_flatten()
        if len(all_indices) == 0:
            return

        pool = allocator.get_kvcache()
        kv_data_cpu = pool.get_cpu_copy(all_indices)

        saved_state = {
            'root_node': inner.root_node,
            'evictable_size': inner.evictable_size_,
            'protected_size': inner.protected_size_,
            'evictable_leaves': set(inner.evictable_leaves),
        }

        _kv_snapshots[model_name] = (saved_state, all_indices.cpu(), kv_data_cpu)
        logger.info(f"  KV snapshot saved: {len(all_indices)} indices, "
                    f"evictable={inner.evictable_size_}, "
                    f"allocator_available={allocator.available_size()} for {model_name}")
    except Exception as e:
        logger.warning(f"  KV snapshot save failed: {e}")



def _get_graph_pool():
    from sglang.srt.model_executor.cuda_graph_runner import get_global_graph_memory_pool
    return get_global_graph_memory_pool()

def _set_graph_pool(pool):
    from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool
    set_global_graph_memory_pool(pool)


_kv_pool_cache = {}
_model_cache = {}
_runtime_cache = {}  # model_name -> runtime region capacity  # model_name → runner.model (nn.Module with bump-pointed params)

def _save_graph_cache(runner, model_name):
    """Cache CUDA graph runner + attn_backend for later restoration."""
    if runner.graph_runner is None:
        return
    try:
        from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
        # Save CUDA buffers (cos_sin_cache etc.) for address-stable restore
        saved_buffers = {}
        if hasattr(runner, 'model') and runner.model is not None:
            for n, b in runner.model.named_buffers():
                if b is not None and b.device.type == 'cuda' and b.numel() > 0:
                    saved_buffers[n] = b
        _graph_cache[model_name] = {
            'vmm_graph_tag': runner.vmm_graph_tag,
            'graph_runner': runner.graph_runner,
            'attn_backend': runner.attn_backend,
            'input_buffer_pool': dict(_forward_input_buffer_pool),
            'graph_pool_handle': _get_graph_pool(),  # pool id logged below
            'model_buffers': saved_buffers,
        }
        logger.info(f"  Graph cache saved for {model_name}")
    except Exception as e:
        logger.warning(f"  Graph cache save failed: {e}")


def _restore_graph_cache(runner, model_name):
    """Restore cached CUDA graph runner + attn_backend."""
    if model_name not in _graph_cache:
        return False
    try:
        cached = _graph_cache[model_name]
        runner.graph_runner = cached['graph_runner']
        # Restore the graph memory pool (each model must use its own pool)
        saved_pool = cached.get('graph_pool_handle')
        if saved_pool is not None:
            _set_graph_pool(saved_pool)
        runner.attn_backend = cached['attn_backend']

        # Update stale internal references in restored attn_backend
        ab = runner.attn_backend
        new_req_to_token = runner.req_to_token_pool.req_to_token
        new_allocator = runner.token_to_kv_pool_allocator
        # indices_updater_decode holds req_to_token and allocator refs from old init
        if hasattr(ab, 'indices_updater_decode'):
            ab.indices_updater_decode.req_to_token = new_req_to_token
            ab.indices_updater_decode.token_to_kv_pool_allocator = new_allocator
        # indices_updater_prefill too
        if hasattr(ab, 'indices_updater_prefill'):
            if hasattr(ab.indices_updater_prefill, 'req_to_token'):
                ab.indices_updater_prefill.req_to_token = new_req_to_token
            if hasattr(ab.indices_updater_prefill, 'token_to_kv_pool_allocator'):
                ab.indices_updater_prefill.token_to_kv_pool_allocator = new_allocator

        from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
        _forward_input_buffer_pool.clear()
        _forward_input_buffer_pool.update(cached['input_buffer_pool'])

        # Restore model buffers - skip copy if model cache hit (same addresses)
        _is_model_cached = model_name in _model_cache
        saved_buffers = cached.get('model_buffers', {})
        if saved_buffers and hasattr(runner, 'model') and runner.model is not None:
            if _is_model_cached:
                # Model cache hit: buffer addresses unchanged, D2D already restored data
                # Just ensure model refs point to saved (graph-known) addresses
                for buf_name, saved_buf in saved_buffers.items():
                    parts = buf_name.split(".")
                    module = runner.model
                    try:
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        module._buffers[parts[-1]] = saved_buf
                    except (AttributeError, KeyError):
                        pass
            else:
                for buf_name, saved_buf in saved_buffers.items():
                    parts = buf_name.split(".")
                    module = runner.model
                    try:
                        for part in parts[:-1]:
                            module = getattr(module, part)
                        new_buf = module._buffers.get(parts[-1])
                        if new_buf is not None and new_buf.shape == saved_buf.shape and new_buf.dtype == saved_buf.dtype:
                            saved_buf.copy_(new_buf)
                        module._buffers[parts[-1]] = saved_buf
                    except (AttributeError, KeyError):
                        pass

        del _graph_cache[model_name]
        # Diagnostic: log key tensor addresses after restore
        logger.info(f"  Graph cache restored for {model_name}")
        return True
    except Exception as e:
        logger.warning(f"  Graph cache restore failed: {e}")
        if model_name in _graph_cache:
            del _graph_cache[model_name]
        return False



def _restore_kv_from_staging(runner, preload_mgr, bump, scheduler):
    """D2D KV staging data to KV pool slots, then insert into radix tree."""
    from sglang.srt.mem_cache.preload_manager import StagedKVEntry
    
    if not hasattr(preload_mgr, 'kv_staged') or not preload_mgr.kv_staged:
        return
    
    preload_mgr.wait_complete()
    
    tree_cache = scheduler.tree_cache
    allocator = runner.token_to_kv_pool_allocator
    kv_pool = runner.token_to_kv_pool
    num_layers = runner.num_effective_layers
    
    total_restored = 0
    total_tokens = 0
    
    for idx, staged_entry in preload_mgr.kv_staged.items():
        if staged_entry.status != "staged":
            continue
        
        n_tokens = staged_entry.num_tokens
        
        # Allocate new slots in KV pool
        if allocator.available_size() < n_tokens:
            logger.info(f"  KV restore: not enough slots ({allocator.available_size()} < {n_tokens}), skipping")
            continue
        
        new_slots = allocator.alloc(n_tokens)
        if new_slots is None:
            logger.info(f"  KV restore: alloc failed for {n_tokens} tokens")
            continue
        
        # D2D: staging buffer -> KV pool k_buffer/v_buffer
        d2d_stream = torch.cuda.Stream()
        with torch.cuda.stream(d2d_stream):
            for layer_id, (k_off, k_bytes, v_off, v_bytes) in enumerate(staged_entry.layer_data_offsets):
                # Reshape staging data to match KV buffer shape
                k_buf = kv_pool.k_buffer[layer_id]
                v_buf = kv_pool.v_buffer[layer_id]
                
                # k_buf[slot] shape: [num_heads, head_dim]
                k_slot_bytes = k_buf[0].numel() * k_buf[0].element_size()
                v_slot_bytes = v_buf[0].numel() * v_buf[0].element_size()
                
                # Copy each token's k data from staging to new slot
                staging_k = bump.buffer[k_off : k_off + k_bytes]
                staging_v = bump.buffer[v_off : v_off + v_bytes]
                
                # Reshape and scatter to new slots
                k_shaped = staging_k.view(n_tokens, -1)
                v_shaped = staging_v.view(n_tokens, -1)
                
                k_buf_flat = k_buf.view(k_buf.shape[0], -1)
                v_buf_flat = v_buf.view(v_buf.shape[0], -1)
                
                k_buf_flat[new_slots] = k_shaped
                v_buf_flat[new_slots] = v_shaped
        
        d2d_stream.synchronize()
        
        # Insert into radix tree
        token_ids = staged_entry.token_ids.tolist()
        inner = getattr(tree_cache, 'inner', tree_cache)
        if not inner.disable:
            from sglang.srt.mem_cache.radix_cache import RadixKey, InsertParams
            page_size = getattr(inner, 'page_size', 1)
            # page_align_keys: truncate to page boundary
            from sglang.srt.mem_cache.radix_cache import page_align_keys
            keys = page_align_keys(token_ids, page_size)
            values = new_slots[:len(keys)].to(dtype=torch.int64)
            radix_key = RadixKey(keys, None, is_bigram=False)
            try:
                inner.insert(InsertParams(key=radix_key, value=values, priority=0))
                # Free excess slots beyond page-aligned length
                if len(keys) < n_tokens:
                    allocator.free(new_slots[len(keys):])
            except Exception as e:
                logger.warning(f"  KV restore: radix insert failed: {e}")
                allocator.free(new_slots)
                continue
        
        total_restored += 1
        total_tokens += n_tokens
    
    # Clear staged KV data
    preload_mgr.kv_staged.clear()
    preload_mgr.kv_staging_bytes = 0
    
    if total_restored > 0:
        logger.info(f"  KV staging restored: {total_restored} entries, {total_tokens} tokens")


def do_model_switch(scheduler: "Scheduler", target_model_path: str, target_model_name: str = None) -> dict:
    """Switch to a different model by path.

    Steps:
    1. Release: flush cache, delete KV pools + model, gc
    2. Load: create new ModelConfig, load new model weights
    3. Reinit: create new KV pool + attention backend, propagate config

    Returns timing dict.
    """
    current_path = scheduler.server_args.model_path
    if target_model_path == current_path:
        return {"skipped": True, "message": "already active"}

    logger.info(f"Model switch: {current_path} -> {target_model_path}")
    timings = {}
    t_total = time.perf_counter()

    runner = scheduler.tp_worker.model_runner

    # ── 1. Release ──
    t0 = time.perf_counter()

    scheduler.flush_cache()

    # Delete KV cache pools from all holders
    for attr in ["req_to_token_pool", "token_to_kv_pool", "token_to_kv_pool_allocator"]:
        if hasattr(runner, attr):
            setattr(runner, attr, None)
    for attr in ["req_to_token_pool", "token_to_kv_pool_allocator"]:
        if hasattr(scheduler, attr):
            setattr(scheduler, attr, None)
    # Note: tree_cache is NOT nullified - it's reused after reinit
    if hasattr(runner, "attn_backend"):
        runner.attn_backend = None
    # Clear cuda graph runner if exists
    if hasattr(runner, "cuda_graph_runner"):
        runner.cuda_graph_runner = None
    # Clear any scheduler-level caches that hold pool refs
    if hasattr(scheduler, "tp_worker"):
        w = scheduler.tp_worker
        for attr in ["req_to_token_pool", "token_to_kv_pool_allocator"]:
            if hasattr(w, attr):
                setattr(w, attr, None)
    # Note: running_batch and waiting_queue are NOT cleared here

    # Delete model
    if hasattr(runner, "model"):
        del runner.model
        runner.model = None

    # Debug: count large CUDA tensors still alive
    import sys
    large_tensors = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.is_cuda and obj.numel() > 1_000_000:
            large_tensors.append((obj.shape, obj.dtype, obj.numel(), sys.getrefcount(obj)))
    logger.info(f"  Large CUDA tensors still alive: {len(large_tensors)}")
    for shape, dtype, numel, refcount in sorted(large_tensors, key=lambda x: -x[2])[:5]:
        logger.info(f"    {shape} {dtype} numel={numel} refs={refcount}")

    gc.collect()
    torch.cuda.empty_cache()
    logger.info(f"  After gc+empty_cache: GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")

    timings["release"] = time.perf_counter() - t0
    logger.info(f"  release: {timings["release"]:.3f}s  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── 2. Load new model ──
    t0 = time.perf_counter()

    runner.server_args.model_path = target_model_path

    new_config = ModelConfig.from_server_args(
        runner.server_args, model_path=target_model_path
    )
    runner.model_config = new_config

    from sglang.srt.model_loader import get_model_loader
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig

    load_config = LoadConfig(load_format=runner.server_args.load_format)
    loader = get_model_loader(load_config, new_config)
    runner.model = loader.load_model(
        model_config=new_config,
        device_config=DeviceConfig(runner.device, runner.gpu_id),
    )
    runner.load_config = load_config

    # Update derived attributes for new model
    runner.start_layer = 0
    runner.end_layer = new_config.num_hidden_layers
    runner.num_effective_layers = new_config.num_hidden_layers
    runner.dtype = new_config.dtype
    runner.kv_cache_dtype = new_config.dtype  # Phase 1: match model dtype

    timings["load"] = time.perf_counter() - t0
    logger.debug(f"  load: {timings["load"]:.3f}s  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # ── 3. Reinit KV pool + attention ──
    t0 = time.perf_counter()

    from sglang.srt.utils.common import get_available_gpu_memory
    pre_mem_gb = get_available_gpu_memory(runner.device, runner.gpu_id)
    logger.info(f"  Before init_memory_pool: pre_mem_gb={pre_mem_gb:.2f}GB")
    try:
        profiled = runner.profile_max_num_token(pre_mem_gb)
        logger.info(f"  profiled_tokens={profiled}, num_eff_layers={runner.num_effective_layers}, context_len={runner.model_config.context_len}")
        runner.init_memory_pool(pre_mem_gb)
    except Exception as e:
        logger.error(f"  init_memory_pool failed: {e}", exc_info=True)
        raise
    runner.init_attention_backend()

    # Propagate config to scheduler + worker
    worker = scheduler.tp_worker
    worker.model_config = new_config
    worker.max_total_num_tokens = runner.max_total_num_tokens
    worker.max_running_requests = runner.max_running_requests
    worker.max_req_len = min(
        new_config.context_len - 1,
        runner.max_token_pool_size - 1,
    )
    worker.max_req_input_len = worker.max_req_len - 5

    # Re-assign pool refs to scheduler
    scheduler.req_to_token_pool = runner.req_to_token_pool
    scheduler.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator

    # Update scheduler-level capacity/limits (must match new pool)
    scheduler.max_total_num_tokens = runner.max_total_num_tokens
    scheduler.max_running_requests = runner.max_running_requests
    scheduler.max_req_len = worker.max_req_len
    scheduler.max_req_input_len = worker.max_req_input_len

    # Re-init tree_cache with new pool refs
    # The tree_cache from old model may have incompatible structure
    # For now, flush and reinit
    scheduler.flush_cache()
    if hasattr(scheduler, 'tree_cache') and scheduler.tree_cache is not None:
        if hasattr(scheduler.tree_cache, 'reset'):
            scheduler.tree_cache.reset()
        # Update tree_cache's internal pool references
        if hasattr(scheduler.tree_cache, 'req_to_token_pool'):
            scheduler.tree_cache.req_to_token_pool = runner.req_to_token_pool
        if hasattr(scheduler.tree_cache, 'token_to_kv_pool_allocator'):
            scheduler.tree_cache.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator

    scheduler.model_config = new_config
    scheduler.server_args.model_path = target_model_path
    if hasattr(scheduler, "offload_tags"):
        scheduler.offload_tags.clear()

    # Notify detokenizer to update its tokenizer
    from sglang.srt.managers.io_struct import UpdateTokenizerNotification
    notif = UpdateTokenizerNotification(
        tokenizer_path=target_model_path,
        tokenizer_mode=scheduler.server_args.tokenizer_mode,
        trust_remote_code=scheduler.server_args.trust_remote_code,
    )
    if hasattr(scheduler, "send_to_detokenizer"):
        scheduler.send_to_detokenizer.socket.send_pyobj(notif)

    # Debug: check pool state after reinit
    alloc = scheduler.token_to_kv_pool_allocator
    if alloc:
        logger.info(f"  Pool after reinit: max={runner.max_total_num_tokens}, avail={alloc.available_size()}")

    timings["reinit"] = time.perf_counter() - t0
    logger.info(f"  reinit: {timings["reinit"]:.3f}s  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    timings["total"] = time.perf_counter() - t_total
    logger.info(f"  TOTAL: {timings["total"]:.3f}s")

    return timings


def do_model_switch_bump(scheduler, target_model_path, target_model_name=None):
    """Switch model without releasing bump regions. D2D staging + in-place overwrite."""
    runner = scheduler.tp_worker.model_runner
    bump = runner.bump_vram_manager
    timings = {}
    t_total = time.perf_counter()
    current_model_name = getattr(scheduler, 'active_model_name', None)
    logger.info(f'Bump switch: {scheduler.server_args.model_path} -> {target_model_path}')

    t_p1 = time.perf_counter()
    # === Phase 1: Save caches ===
    if current_model_name:
        if current_model_name not in _graph_cache:
            _save_graph_cache(runner, current_model_name)
        # Save KV pool cache for fast switch-back
        kv_r = bump.regions.get("kv_cache")
        if kv_r:
            _w = bump.regions.get("weights")
            _kv_pool_cache[current_model_name] = {
                "req_to_token_pool": runner.req_to_token_pool,
                "token_to_kv_pool": runner.token_to_kv_pool,
                "token_to_kv_pool_allocator": runner.token_to_kv_pool_allocator,
                "max_total_num_tokens": runner.max_total_num_tokens,
                "max_running_requests": runner.max_running_requests,
                "weight_bytes": _w.capacity if _w else 0,
                "kv_region_capacity": kv_r.capacity,
            }
            logger.debug(f"  KV pool cache saved: {current_model_name}")
        # Save model object cache (params point to bump, zero extra VRAM)
        if hasattr(runner, 'model') and runner.model is not None:
            _model_cache[current_model_name] = runner.model
            logger.debug(f"  Model cache saved: {current_model_name}")
        rt_r = bump.regions.get("runtime")
        if rt_r:
            _runtime_cache[current_model_name] = rt_r.capacity
        if hasattr(scheduler, 'memory_saver_adapter') and scheduler.memory_saver_adapter.enabled:
            cached = _graph_cache.get(current_model_name, {})
            old_tag = cached.get('vmm_graph_tag', runner.vmm_graph_tag)
            scheduler.memory_saver_adapter.pause(old_tag)
            logger.debug(f"  VMM: paused graph pool tag={old_tag}")

    timings['save_caches'] = time.perf_counter() - t_p1; logger.debug(f"  save_caches: {timings['save_caches']*1000:.1f}ms")
    # === Phase 2: Verify staging integrity (before clearing allocator) ===
    t0 = time.perf_counter()
    _preload_mgr = getattr(scheduler, "_preload_manager", None)
    _has_staging = (_preload_mgr is not None
                    and _preload_mgr.model_name == target_model_name
                    and _preload_mgr.num_staged > 0)
    _staging_info = None
    if _has_staging:
        _preload_mgr.wait_complete()
        cell_size = runner.get_cell_size_per_token(runner.num_effective_layers) if hasattr(runner, "get_cell_size_per_token") else 0
        _preload_mgr.verify_integrity(bump, scheduler.token_to_kv_pool_allocator, cell_size)
        _staging_info = _preload_mgr.staged
        logger.debug(f"  Staging: {_preload_mgr.num_staged} staged, {_preload_mgr.num_evicted} evicted")

    timings['staging_verify'] = time.perf_counter() - t0; logger.debug(f"  staging_verify: {timings['staging_verify']*1000:.1f}ms")
    # === Phase 3: Cleanup ===
    runner.graph_runner = None
    if 'runtime' in bump.regions:
        bump.release_region('runtime')
    if hasattr(runner, 'attn_backend'):
        runner.attn_backend = None
    from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
    _ROPE_DICT.clear()

    # === Phase 4: Load weights ===
    t0 = _t = time.perf_counter()
    runner.server_args.model_path = target_model_path
    new_config = ModelConfig.from_server_args(runner.server_args, model_path=target_model_path)
    _t1 = time.perf_counter()
    runner.model_config = new_config
    runner.start_layer = 0
    runner.end_layer = new_config.num_hidden_layers
    runner.num_effective_layers = new_config.num_hidden_layers
    runner.dtype = new_config.dtype
    runner.kv_cache_dtype = new_config.dtype

    _cpu_model = None
    if hasattr(scheduler, '_model_registry') and scheduler._model_registry is not None:
        _cpu_model, _ = scheduler._model_registry.get_cpu_model(target_model_name or '')
    _t2 = time.perf_counter()

    # Try model cache: restore model object + D2D only (skip Python param loop)
    _cached_model = _model_cache.get(target_model_name)
    if _cached_model is not None and _has_staging:
        # Model cache hit! Just D2D staging data to bump and restore model ref
        _preload_mgr.wait_complete()
        # Single memcpy: staging layout matches weights layout
        w_region = bump.regions.get("weights")
        if w_region:
            d2d_stream = torch.cuda.Stream()
            h2d_stream = torch.cuda.Stream()
            evict_boundary = _preload_mgr.get_eviction_boundary(w_region.start)
            valid_start = evict_boundary  # offset in bump buffer
            valid_end = w_region.start + w_region.capacity
            with torch.cuda.stream(d2d_stream):
                if valid_start < valid_end:
                    bump.buffer[valid_start : valid_end].copy_(
                        bump.buffer[valid_start - w_region.start + _preload_mgr.staging_right_ptr :
                                    valid_end - w_region.start + _preload_mgr.staging_right_ptr],
                        non_blocking=True)
            # H2D evicted portion from CPU
            if evict_boundary > w_region.start and hasattr(scheduler, '_model_registry'):
                cpu_model, _ = scheduler._model_registry.get_cpu_model(target_model_name or '')
                if cpu_model is not None:
                    evict_bytes = evict_boundary - w_region.start
                    with torch.cuda.stream(h2d_stream):
                        # Reconstruct left portion from CPU state dict
                        offset = 0
                        for name, p in cpu_model.named_parameters():
                            nbytes = p.numel() * p.element_size()
                            aligned = (nbytes + 255) & ~255
                            if offset + nbytes <= evict_bytes:
                                bump.buffer[w_region.start + offset : w_region.start + offset + nbytes].copy_(
                                    p.data.view(-1).view(torch.uint8), non_blocking=True)
                            elif offset < evict_bytes:
                                # Partial overlap - H2D the evicted part only
                                partial = evict_bytes - offset
                                bump.buffer[w_region.start + offset : w_region.start + offset + partial].copy_(
                                    p.data.view(-1).view(torch.uint8)[:partial], non_blocking=True)
                            else:
                                break
                            offset += aligned
                        for bname, buf in cpu_model.named_buffers():
                            if buf is None or buf.numel() == 0:
                                continue
                            nbytes = buf.numel() * buf.element_size()
                            aligned = (nbytes + 255) & ~255
                            if offset + nbytes <= evict_bytes:
                                bump.buffer[w_region.start + offset : w_region.start + offset + nbytes].copy_(
                                    buf.data.view(-1).view(torch.uint8), non_blocking=True)
                            elif offset < evict_bytes:
                                partial = evict_bytes - offset
                                bump.buffer[w_region.start + offset : w_region.start + offset + partial].copy_(
                                    buf.data.view(-1).view(torch.uint8)[:partial], non_blocking=True)
                            else:
                                break
                            offset += aligned
            d2d_stream.synchronize()
            h2d_stream.synchronize()
        runner.model = _cached_model
        runner.model.eval()
        if "weights" in bump.regions:
            bump.regions["weights"]._sub_offset = bump.regions["weights"].capacity
        bump._current_model = target_model_name
        total_weight_bytes = bump.regions["weights"].capacity if "weights" in bump.regions else 0
        _t3 = time.perf_counter()
        _t4 = time.perf_counter()
        logger.debug(f'  load detail: config {(_t1-_t)*1000:.1f}ms, '
                    f'MODEL CACHE HIT + D2D {(_t3-_t2)*1000:.1f}ms, '
                    f'total {(_t3-_t)*1000:.1f}ms')
    else:
        # No model cache: full load via _load_model_bump_from_cpu
        param_bytes = sum(p.numel() * p.element_size() for p in _cpu_model.parameters()) if _cpu_model else 0
        buf_bytes = sum(b.numel() * b.element_size() for b in _cpu_model.buffers() if b is not None and b.numel() > 0) if _cpu_model else 0
        total_weight_bytes = param_bytes + buf_bytes
        if "weights" in bump.regions:
            bump.reset_region("weights", total_weight_bytes)
        else:
            bump.allocate_region("weights", total_weight_bytes)
        _t3 = time.perf_counter()
        if _cpu_model is not None:
            runner._load_model_bump_from_cpu(_cpu_model, staging_info=_staging_info)
        else:
            runner._load_model_bump()
        _t4 = time.perf_counter()
        logger.debug(f'  load detail: config {(_t1-_t)*1000:.1f}ms, '
                    f'get_cpu_model {(_t2-_t1)*1000:.1f}ms, '
                    f'reset_region {(_t3-_t2)*1000:.1f}ms, '
                    f'load_bump {(_t4-_t3)*1000:.1f}ms, '
                    f'total {(_t4-_t)*1000:.1f}ms')
    timings['load'] = time.perf_counter() - t0
    logger.info(f'  load: {timings["load"]:.3f}s')

    # === Phase 5: KV cache + runtime rebuild ===
    t0 = time.perf_counter()
    # Try KV pool cache: skip init_memory_pool if same model switching back
    _kv_cached = _kv_pool_cache.get(target_model_name) if target_model_name else None
    if _kv_cached:
        # Same model, same weights size → same bump layout → KV pool objects still valid
        if "kv_cache" in bump.regions:
            bump.reset_region("kv_cache", bump.get_available_bytes())
        else:
            bump.allocate_region("kv_cache", bump.get_available_bytes())
        runner.req_to_token_pool = _kv_cached["req_to_token_pool"]
        runner.token_to_kv_pool = _kv_cached["token_to_kv_pool"]
        runner.token_to_kv_pool_allocator = _kv_cached["token_to_kv_pool_allocator"]
        runner.max_total_num_tokens = _kv_cached["max_total_num_tokens"]
        runner.max_running_requests = _kv_cached["max_running_requests"]
        runner.token_to_kv_pool_allocator.clear()
        scheduler.flush_cache()
        logger.debug(f"  KV pool cache hit: {target_model_name}")
    else:
        # KV pool cache miss: release kv_cache, clear refs, full init
        if 'kv_cache' in bump.regions:
            bump.release_region('kv_cache')
        scheduler.flush_cache()
        for attr in ['req_to_token_pool', 'token_to_kv_pool', 'token_to_kv_pool_allocator']:
            for obj in [runner, scheduler, scheduler.tp_worker]:
                if hasattr(obj, attr):
                    setattr(obj, attr, None)
        runner.init_memory_pool(0)
    _rt_cached = _runtime_cache.get(target_model_name)
    if _rt_cached and "runtime" not in bump.regions:
        bump.allocate_region("runtime", _rt_cached)
    else:
        runner._init_runtime_region()
    if not (target_model_name and target_model_name in _graph_cache):
        runner.init_attention_backend()

    from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool; _forward_input_buffer_pool.clear()
    # === Phase 5.5: KV staging restore ===
    if _has_staging and hasattr(_preload_mgr, 'kv_staged') and _preload_mgr.kv_staged:
        _restore_kv_from_staging(runner, _preload_mgr, bump, scheduler)
    t_p6 = time.perf_counter(); logger.debug(f"  kv_runtime: {(t_p6-t0)*1000:.1f}ms")
    # === Phase 6: CUDA graph restore ===
    if not scheduler.server_args.disable_cuda_graph:
        new_tag = f"cuda_graph:{target_model_name}" if target_model_name else "cuda_graph"
        runner.vmm_graph_tag = new_tag
        if hasattr(scheduler, 'memory_saver_adapter') and scheduler.memory_saver_adapter.enabled:
            cached = _graph_cache.get(target_model_name, {})
            resume_tag = cached.get('vmm_graph_tag')
            if resume_tag:
                scheduler.memory_saver_adapter.resume(resume_tag)
                logger.debug(f"  VMM: resumed graph pool tag={resume_tag}")
        # Wait for H2D weight transfer if using async stream
        if hasattr(runner, "_h2d_done_event"):
            torch.cuda.current_stream().wait_event(runner._h2d_done_event)
        if not (target_model_name and _restore_graph_cache(runner, target_model_name)):
            _set_graph_pool(None)
            runner.init_device_graphs()

    t_p7 = time.perf_counter(); logger.debug(f"  graph_restore: {(t_p7-t_p6)*1000:.1f}ms")
    # === Phase 7: Update scheduler/worker refs ===
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
    # tree_cache refs update (flush already done in Phase 5)
    if hasattr(scheduler, 'tree_cache') and scheduler.tree_cache is not None:
        scheduler.tree_cache.req_to_token_pool = runner.req_to_token_pool
        scheduler.tree_cache.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator

    scheduler.model_config = new_config
    scheduler.server_args.model_path = target_model_path
    if hasattr(scheduler, 'offload_tags'):
        scheduler.offload_tags.clear()
    from sglang.srt.managers.io_struct import UpdateTokenizerNotification
    notif = UpdateTokenizerNotification(
        tokenizer_path=target_model_path,
        tokenizer_mode=scheduler.server_args.tokenizer_mode,
        trust_remote_code=scheduler.server_args.trust_remote_code,
    )
    if hasattr(scheduler, 'send_to_detokenizer'):
        scheduler.send_to_detokenizer.socket.send_pyobj(notif)

    timings['update_refs'] = time.perf_counter() - t_p7; logger.debug(f"  update_refs: {timings['update_refs']*1000:.1f}ms")
    timings['reinit_total'] = time.perf_counter() - t0; logger.debug(f"  reinit_total: {timings['reinit_total']*1000:.1f}ms")
    timings['total'] = time.perf_counter() - t_total
    logger.info(f'  TOTAL: {timings["total"]:.3f}s')

    t_p8 = time.perf_counter()
    # === Phase 8: Preload previous model for fast switch-back ===
    if current_model_name and hasattr(scheduler, "_start_weight_preload"):
        scheduler._start_weight_preload(current_model_name)
    timings['preload_trigger'] = time.perf_counter() - t_p8; logger.debug(f"  preload_trigger: {timings['preload_trigger']*1000:.1f}ms")
    return timings
