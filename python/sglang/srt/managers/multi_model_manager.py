"""
Multi-model management: model switching logic.

Phase 1 MVP: switch by model_path.
Memory strategy: direct free (no VMM dependency).
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
        logger.info(f"  KV snapshot saved: {len(all_indices)} tokens for {model_name}")
    except Exception as e:
        logger.warning(f"  KV snapshot save failed: {e}")


def _restore_kv_snapshot(tree_cache, allocator, model_name):
    """Restore radix tree state + KV cache data from CPU after model switch."""
    if model_name not in _kv_snapshots:
        return False

    inner = getattr(tree_cache, 'inner', tree_cache)
    saved_state, saved_indices_cpu, kv_data_cpu = _kv_snapshots[model_name]

    try:
        saved_indices = saved_indices_cpu.to(allocator.device)
        pool = allocator.get_kvcache()
        pool.load_cpu_copy(kv_data_cpu, saved_indices)

        # Restore tree state
        inner.root_node = saved_state['root_node']
        inner.evictable_size_ = saved_state['evictable_size']
        inner.protected_size_ = saved_state['protected_size']
        inner.evictable_leaves = saved_state['evictable_leaves']

        # Remove restored indices from allocator free list
        used_set = set(saved_indices.tolist())
        free_list = allocator.free_pages.tolist()
        new_free = [p for p in free_list if p not in used_set]
        allocator.free_pages = torch.tensor(new_free, dtype=allocator.free_pages.dtype,
                                            device=allocator.free_pages.device)

        del _kv_snapshots[model_name]
        logger.info(f"  KV snapshot restored: {len(saved_indices)} tokens for {model_name}")
        return True
    except Exception as e:
        logger.warning(f"  KV snapshot restore failed: {e}")
        if model_name in _kv_snapshots:
            del _kv_snapshots[model_name]
        return False


def _get_graph_pool():
    from sglang.srt.model_executor.cuda_graph_runner import get_global_graph_memory_pool
    return get_global_graph_memory_pool()

def _set_graph_pool(pool):
    from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool
    set_global_graph_memory_pool(pool)

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
            'graph_runner': runner.graph_runner,
            'attn_backend': runner.attn_backend,
            'input_buffer_pool': dict(_forward_input_buffer_pool),
            'graph_pool_handle': _get_graph_pool(),  # pool id logged below
            'model_buffers': saved_buffers,
        }
        # Diagnostic: log key tensor addresses
        addrs = {}
        if hasattr(runner, 'attn_backend'):
            ab = runner.attn_backend
            addrs['workspace'] = ab.workspace_buffer.data_ptr() if hasattr(ab, 'workspace_buffer') else 'N/A'
            if hasattr(ab, 'cuda_graph_kv_indices') and ab.cuda_graph_kv_indices:
                addrs['kv_indices'] = ab.cuda_graph_kv_indices[0].data_ptr()
            if hasattr(ab, 'kv_indptr') and ab.kv_indptr:
                addrs['kv_indptr'] = ab.kv_indptr[0].data_ptr()
        if hasattr(runner, 'model'):
            for n, b in runner.model.named_buffers():
                if 'cos_sin' in n or 'inv_freq' in n:
                    addrs[n] = b.data_ptr()
                    break
        for n, p in list(runner.model.named_parameters())[:1]:
            addrs[f'param:{n}'] = p.data_ptr()
        addrs['pool_handle'] = id(_get_graph_pool()) if _get_graph_pool() else 'None'; logger.info(f"  Graph cache saved for {model_name}, addrs={addrs}")
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

        # DIAGNOSTIC: check buffer data before restore
        _saved_bufs = cached.get('model_buffers', {})
        for _bn, _sb in _saved_bufs.items():
            if 'cos_sin' in _bn:
                logger.info(f"  DIAG {_bn}: saved_ptr={_sb.data_ptr()}, data[:4]={_sb.flatten()[:4].tolist()}")
                _parts = _bn.split(".")
                _mod = runner.model
                try:
                    for _p in _parts[:-1]:
                        _mod = getattr(_mod, _p)
                    _nb = _mod._buffers.get(_parts[-1])
                    if _nb is not None:
                        logger.info(f"  DIAG {_bn}: new_ptr={_nb.data_ptr()}, data[:4]={_nb.flatten()[:4].tolist()}, same_ptr={_sb.data_ptr()==_nb.data_ptr()}")
                except Exception as _e:
                    logger.info(f"  DIAG {_bn}: new buf lookup failed: {_e}")

        # Restore model buffers to original addresses (cos_sin_cache etc.)
        # Key: the saved tensor's GPU address was overwritten during model switch.
        # We must copy the freshly computed buffer data to the saved address,
        # then replace the model's reference to point to the saved address.
        saved_buffers = cached.get('model_buffers', {})
        if saved_buffers and hasattr(runner, 'model') and runner.model is not None:
            for buf_name, saved_buf in saved_buffers.items():
                parts = buf_name.split(".")
                module = runner.model
                try:
                    for part in parts[:-1]:
                        module = getattr(module, part)
                    # Get the new model's freshly computed buffer (correct data, new address)
                    new_buf = module._buffers.get(parts[-1])
                    if new_buf is not None and new_buf.shape == saved_buf.shape and new_buf.dtype == saved_buf.dtype:
                        # Copy correct data to the saved address (where graph expects it)
                        saved_buf.copy_(new_buf)
                    # Replace model reference to use saved address
                    module._buffers[parts[-1]] = saved_buf
                except (AttributeError, KeyError):
                    pass

        del _graph_cache[model_name]
        # Diagnostic: log key tensor addresses after restore
        addrs = {}
        if hasattr(runner, 'attn_backend'):
            ab = runner.attn_backend
            addrs['workspace'] = ab.workspace_buffer.data_ptr() if hasattr(ab, 'workspace_buffer') else 'N/A'
            if hasattr(ab, 'cuda_graph_kv_indices') and ab.cuda_graph_kv_indices:
                addrs['kv_indices'] = ab.cuda_graph_kv_indices[0].data_ptr()
            if hasattr(ab, 'kv_indptr') and ab.kv_indptr:
                addrs['kv_indptr'] = ab.kv_indptr[0].data_ptr()
        if hasattr(runner, 'model'):
            for n, b in runner.model.named_buffers():
                if 'cos_sin' in n or 'inv_freq' in n:
                    addrs[n] = b.data_ptr()
                    break
        for n, p in list(runner.model.named_parameters())[:1]:
            addrs[f'param:{n}'] = p.data_ptr()
        addrs['pool_handle'] = id(_get_graph_pool()) if _get_graph_pool() else 'None'; logger.info(f"  Graph cache restored for {model_name}, addrs={addrs}")
        return True
    except Exception as e:
        logger.warning(f"  Graph cache restore failed: {e}")
        if model_name in _graph_cache:
            del _graph_cache[model_name]
        return False


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
    logger.info(f"  load: {timings["load"]:.3f}s  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")

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
    import time
    runner = scheduler.tp_worker.model_runner
    bump = runner.bump_vram_manager
    timings = {}
    t_total = time.perf_counter()
    logger.info(f'Bump switch: {scheduler.server_args.model_path} -> {target_model_path}')


    # Save KV cache + graph cache before release
    current_model_name = getattr(scheduler, 'active_model_name', None)
    if current_model_name and hasattr(scheduler, 'tree_cache') and scheduler.token_to_kv_pool_allocator is not None:
        _save_kv_snapshot(scheduler.tree_cache, scheduler.token_to_kv_pool_allocator, current_model_name)
    if current_model_name:
        _save_graph_cache(runner, current_model_name)

    t0 = time.perf_counter()
    runner.graph_runner = None
    import gc; gc.collect()
    # empty_cache skipped when graph cache is active (would free graph pool memory)
    if current_model_name not in _graph_cache:
        torch.cuda.empty_cache()
    scheduler.flush_cache()
    bump.release_region('kv_cache')
    bump.release_region('weights')
    for attr in ['req_to_token_pool', 'token_to_kv_pool', 'token_to_kv_pool_allocator']:
        for obj in [runner, scheduler, scheduler.tp_worker]:
            if hasattr(obj, attr):
                setattr(obj, attr, None)
    if hasattr(runner, 'attn_backend'):
        runner.attn_backend = None
    if hasattr(runner, 'model'):
        del runner.model
        runner.model = None
    timings['release'] = time.perf_counter() - t0
    logger.info(f'  release: {timings["release"]:.3f}s (bump, no gc)')
    t0 = time.perf_counter()
    # Clear rotary embedding cache (get_rope has a module-level _ROPE_DICT)
    # Without this, second load reuses the old RotaryEmbedding object whose
    # cos_sin_cache was migrated to bump and data was overwritten by other model
    from sglang.srt.layers.rotary_embedding.factory import _ROPE_DICT
    _ROPE_DICT.clear()

    runner.server_args.model_path = target_model_path
    new_config = ModelConfig.from_server_args(runner.server_args, model_path=target_model_path)
    runner.model_config = new_config
    runner.start_layer = 0
    runner.end_layer = new_config.num_hidden_layers
    runner.num_effective_layers = new_config.num_hidden_layers
    runner.dtype = new_config.dtype
    runner.kv_cache_dtype = new_config.dtype
    runner._load_model_bump()
    timings['load'] = time.perf_counter() - t0
    logger.info(f'  load: {timings["load"]:.3f}s')
    t0 = time.perf_counter()
    runner.init_memory_pool(0)
    if not (target_model_name and target_model_name in _graph_cache):
        runner.init_attention_backend()
    # else: attn_backend will be restored from cache

    # Clear shared input buffer pool (dtype may change across models)
    from sglang.srt.model_executor.input_buffers import _forward_input_buffer_pool
    _forward_input_buffer_pool.clear()

    # CUDA graph: try restore from cache, else recapture
    if not scheduler.server_args.disable_cuda_graph:
        if not (target_model_name and _restore_graph_cache(runner, target_model_name)):
            _set_graph_pool(None)  # Force new pool for this model
            runner.init_device_graphs()
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
    scheduler.flush_cache()
    if hasattr(scheduler, 'tree_cache') and scheduler.tree_cache is not None:
        if hasattr(scheduler.tree_cache, 'reset'):
            scheduler.tree_cache.reset()
        if hasattr(scheduler.tree_cache, 'req_to_token_pool'):
            scheduler.tree_cache.req_to_token_pool = runner.req_to_token_pool
        if hasattr(scheduler.tree_cache, 'token_to_kv_pool_allocator'):
            scheduler.tree_cache.token_to_kv_pool_allocator = runner.token_to_kv_pool_allocator

    # Try to restore KV cache from snapshot
    if target_model_name and hasattr(scheduler, 'tree_cache'):
        _restore_kv_snapshot(scheduler.tree_cache, scheduler.token_to_kv_pool_allocator, target_model_name)

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
    timings['reinit'] = time.perf_counter() - t0
    logger.info(f'  reinit: {timings["reinit"]:.3f}s')
    timings['total'] = time.perf_counter() - t_total
    logger.info(f'  TOTAL: {timings["total"]:.3f}s')
    return timings

