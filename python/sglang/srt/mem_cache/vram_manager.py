"""
Page-based bump allocator + per-model GPU resource cache for self-managed VRAM.

BumpVramAllocator: Layout [weights(left) | kv_cache(middle) | runtime(right)]
ModelResourceCache: CUDA graph / KV pool cache per model
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from math import prod
from typing import Dict

import torch

logger = logging.getLogger(__name__)

_ALIGNMENT = 256


def _align_up(n: int, alignment: int = _ALIGNMENT) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


@dataclass
class BumpRegion:
    tag: str  # region 名称 ("weights", "kv_cache", "runtime")
    start: int  # 在 bump buffer 中的起始字节偏移
    capacity: int  # region 总容量（字节，已对齐）
    _sub_offset: int = 0  # region 内部已分配的字节数，create_tensor 每次推进

    def reset(self):
        self._sub_offset = 0

    def mark_full(self) -> None:
        """Mark the whole region as consumed (for external copies that bypass create_tensor)."""
        self._sub_offset = self.capacity

    @property
    def used(self) -> int:
        return self._sub_offset


class BumpVramAllocator:
    def __init__(self, total_bytes: int, device: str = "cuda"):
        self.total_bytes = _align_up(total_bytes)
        self.device = device
        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=device)
        self.left_offset = 0
        self.right_offset = self.total_bytes
        self.regions: Dict[str, BumpRegion] = {}
        logger.info(
            f"BumpVramAllocator: allocated {self.total_bytes / 1024**3:.2f} GB "
            f"managed buffer on {device}"
        )

    def allocate_region(self, tag: str, size_bytes: int) -> BumpRegion:
        if tag in self.regions:
            raise ValueError(f"Region '{tag}' already exists. Use reset_region().")
        aligned_size = _align_up(size_bytes)
        avail = self.right_offset - self.left_offset
        if aligned_size > avail:
            raise RuntimeError(
                f"BumpVramAllocator OOM: need {aligned_size} for '{tag}', "
                f"avail {avail}"
            )
        if tag == "runtime":
            start = self.right_offset - aligned_size
            self.right_offset = start
        else:
            start = self.left_offset
            self.left_offset += aligned_size
        region = BumpRegion(tag=tag, start=start, capacity=aligned_size)
        self.regions[tag] = region
        logger.info(
            f"  Region '{tag}': {aligned_size / 1024**2:.1f} MB "
            f"@ offset {region.start} "
            f"(left={self.left_offset / 1024**3:.2f}GB, "
            f"right={self.right_offset / 1024**3:.2f}GB, "
            f"kv_avail={self.get_available_bytes() / 1024**3:.2f}GB)"
        )
        return region

    def release_region(self, tag: str):
        region = self.regions[tag]
        if tag == "runtime":
            self.right_offset = region.start + region.capacity
        else:
            self.left_offset = region.start
        del self.regions[tag]
        logger.info(f"  Region '{tag}' released")

    def reset_region(self, tag: str, new_capacity=None):
        if tag in self.regions:
            old_capacity = self.regions[tag].capacity
            self.release_region(tag)
        else:
            old_capacity = 0
        capacity = new_capacity if new_capacity is not None else old_capacity
        return self.allocate_region(tag, capacity)

    def create_tensor(
        self,
        tag: str,
        shape: tuple,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        region = self.regions[tag]
        elem_size = dtype.itemsize
        numel = prod(shape)
        nbytes = _align_up(numel * elem_size)
        if region._sub_offset + nbytes > region.capacity:
            raise RuntimeError(
                f"Region '{tag}' sub-alloc OOM: need {nbytes}, "
                f"free {region.capacity - region._sub_offset} (cap={region.capacity}, used={region._sub_offset})"
            )
        buf_start = region.start + region._sub_offset
        buf_end = buf_start + numel * elem_size
        tensor = self.buffer[buf_start:buf_end].view(dtype).reshape(shape)
        region._sub_offset += nbytes
        return tensor

    def get_available_bytes(self) -> int:
        return self.right_offset - self.left_offset

    def __repr__(self):
        region_strs = [
            f"  {tag}: {r.capacity / 1024**2:.1f}MB @ {r.start}, "
            f"used={r._sub_offset / 1024**2:.1f}MB"
            for tag, r in sorted(self.regions.items(), key=lambda x: x[1].start)
        ]
        return (
            f"BumpVramAllocator(total={self.total_bytes / 1024**3:.2f}GB, "
            f"left={self.left_offset / 1024**3:.2f}GB, "
            f"right={self.right_offset / 1024**3:.2f}GB, "
            f"kv_avail={self.get_available_bytes() / 1024**3:.2f}GB)\n"
            + "\n".join(region_strs)
        )


# ---------------------------------------------------------------------------
# Per-model GPU resource cache (CUDA graph, KV pool)
# ---------------------------------------------------------------------------


def _get_graph_pool():
    from sglang.srt.model_executor.cuda_graph_runner import get_global_graph_memory_pool

    return get_global_graph_memory_pool()


def _set_graph_pool(pool):
    from sglang.srt.model_executor.cuda_graph_runner import set_global_graph_memory_pool

    set_global_graph_memory_pool(pool)


def _set_graph_pool_id(pool):
    """Keep pynccl_allocator's graph_pool_id in sync with the global pool.

    capture path sets this inside cuda_graph_runner; the restore path skips
    recapture, so we must set it explicitly.
    """
    try:
        from sglang.srt.distributed.device_communicators.pynccl_allocator import (
            set_graph_pool_id,
        )

        set_graph_pool_id(pool)
    except Exception as e:
        logger.debug(f"  set_graph_pool_id skipped: {e}")


def _get_module_by_path(root, dotted_name):
    """Walk dotted path, return (parent_module, last_attr_name)."""
    parts = dotted_name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    return module, parts[-1]


class ModelResourceCache:
    """Per-model cache for GPU resources: CUDA graphs, KV pools."""

    def __init__(self):
        self.graph_cache: Dict[str, dict] = {}
        self.kv_pool_cache: Dict[str, dict] = {}

    def save_graph(self, runner, model_name: str, memory_saver_adapter=None):
        """Cache CUDA graph runner + attn_backend, and pause VMM graph pool.

        Pause runs BEFORE committing the cache entry: if pause fails, we must
        not claim we have a saved graph (restore would otherwise try to
        resume a tag that was never paused).
        """
        if runner.graph_runner is None:
            return
        vmm_tag = f"cuda_graph:{model_name}"
        try:
            from sglang.srt.model_executor.input_buffers import (
                _forward_input_buffer_pool,
            )

            saved_buffers = {}
            if runner.model is not None:
                for n, b in runner.model.named_buffers():
                    if b is not None and b.device.type == "cuda" and b.numel() > 0:
                        saved_buffers[n] = b

            self.graph_cache[model_name] = {
                "vmm_graph_tag": vmm_tag,
                "graph_runner": runner.graph_runner,
                "attn_backend": runner.attn_backend,
                "piecewise_cuda_graph_runner": getattr(
                    runner, "piecewise_cuda_graph_runner", None
                ),
                "attention_layers": getattr(runner, "attention_layers", None),
                "moe_layers": getattr(runner, "moe_layers", None),
                "moe_fusions": getattr(runner, "moe_fusions", None),
                "input_buffer_pool": dict(_forward_input_buffer_pool),
                "graph_pool_handle": _get_graph_pool(),
                "model_buffers": saved_buffers,
            }
            if memory_saver_adapter is not None and memory_saver_adapter.enabled:
                memory_saver_adapter.pause(vmm_tag)
                logger.debug(f"  VMM: paused graph pool tag={vmm_tag}")
            logger.info(f"  Graph cache saved for {model_name}")
        except Exception as e:
            logger.warning(f"  Graph cache save failed: {e}")
            # Leave no half state: pop whatever might have been written.
            self.graph_cache.pop(model_name, None)

    def restore_graph(self, runner, model_name: str, memory_saver_adapter=None) -> bool:
        """Restore cached CUDA graph runner + attn_backend, and resume VMM graph pool."""
        if model_name not in self.graph_cache:
            return False
        try:
            cached = self.graph_cache[model_name]
            vmm_tag = cached.get("vmm_graph_tag")
            if (
                memory_saver_adapter is not None
                and memory_saver_adapter.enabled
                and vmm_tag
            ):
                memory_saver_adapter.resume(vmm_tag)
                logger.debug(f"  VMM: resumed graph pool tag={vmm_tag}")
            runner.graph_runner = cached["graph_runner"]
            saved_pool = cached.get("graph_pool_handle")
            if saved_pool is not None:
                _set_graph_pool(saved_pool)
            runner.attn_backend = cached["attn_backend"]

            # keep module-level global_workspace_buffer in sync with
            # the restored backend. teardown cleared it to None; without this
            # sync any non-replay path (eager fallback / new wrapper / spec
            # draft) would read None and allocate a parallel workspace that
            # diverges from the cached backend.workspace_buffer.
            _ws = getattr(runner.attn_backend, "workspace_buffer", None)
            if _ws is not None:
                _mod = sys.modules.get(type(runner.attn_backend).__module__)
                if _mod is not None and hasattr(_mod, "global_workspace_buffer"):
                    _mod.global_workspace_buffer = _ws

            for attr in ("attention_layers", "moe_layers", "moe_fusions"):
                saved_val = cached.get(attr)
                if saved_val is not None:
                    setattr(runner, attr, saved_val)
            pcg_runner = cached.get("piecewise_cuda_graph_runner")
            runner.piecewise_cuda_graph_runner = pcg_runner
            if pcg_runner is not None:
                pcg_runner.attention_layers = runner.attention_layers
                pcg_runner.moe_layers = runner.moe_layers
                pcg_runner.moe_fusions = runner.moe_fusions
                from sglang.srt.compilation import piecewise_context_manager as _pcm

                if _pcm._pcg_capture_stream is None:
                    _pcm._pcg_capture_stream = torch.cuda.Stream()

            # Rebind all pointers that reference pool objects. Multiple
            # attention backends cache req_to_token / token_to_kv_pool[_allocator]
            # directly at __init__ (see flashmla / triton / trtllm / aiter /
            # flashattention / flashinfer); updating only the indices_updater_*
            # fields would leave the backend's own forward path using stale
            # KV / allocator references after a switch.
            ab = runner.attn_backend
            new_req_to_token = runner.req_to_token_pool.req_to_token
            new_allocator = runner.token_to_kv_pool_allocator
            new_kv_pool = runner.token_to_kv_pool
            for obj in (ab,
                        getattr(ab, "indices_updater_decode", None),
                        getattr(ab, "indices_updater_prefill", None)):
                if obj is None:
                    continue
                if hasattr(obj, "req_to_token"):
                    obj.req_to_token = new_req_to_token
                if hasattr(obj, "token_to_kv_pool_allocator"):
                    obj.token_to_kv_pool_allocator = new_allocator
                if hasattr(obj, "token_to_kv_pool"):
                    obj.token_to_kv_pool = new_kv_pool

            from sglang.srt.model_executor.input_buffers import (
                _forward_input_buffer_pool,
            )

            _forward_input_buffer_pool.clear()
            _forward_input_buffer_pool.update(cached["input_buffer_pool"])

            saved_buffers = cached.get("model_buffers", {})
            if saved_buffers and runner.model is not None:
                for buf_name, saved_buf in saved_buffers.items():
                    try:
                        module, attr = _get_module_by_path(runner.model, buf_name)
                        current_buf = module._buffers.get(attr)
                        if current_buf is not None:
                            # Require full structural match before copy — shape
                            # alone is not enough, mismatched dtype/stride
                            # would silently corrupt the model.
                            if (
                                current_buf.shape == saved_buf.shape
                                and current_buf.dtype == saved_buf.dtype
                                and current_buf.stride() == saved_buf.stride()
                                and current_buf.device == saved_buf.device
                            ):
                                saved_buf.copy_(current_buf)
                            else:
                                logger.warning(
                                    f"  Skip restore buffer {buf_name}: "
                                    f"shape/dtype/stride/device mismatch"
                                )
                                continue
                        module._buffers[attr] = saved_buf
                    except (AttributeError, KeyError) as e:
                        logger.warning(
                            f"  Skip restore buffer {buf_name}: path lookup failed ({e})"
                        )
            logger.info(f"  Graph cache restored for {model_name}")
            return True
        except Exception as e:
            logger.warning(f"  Graph cache restore failed: {e}")
            self.graph_cache.pop(model_name, None)
            return False

    def evict_graph(self, model_name: str, memory_saver_adapter=None):
        """Evict stale graph cache without resuming its VMM pages.

        The cached graph is about to be discarded and recaptured fresh,
        so we only drop the bookkeeping. Resuming the VMM tag would
        remap pages that pytorch's caching allocator already believes
        were freed, and the subsequent `empty_cache` in
        `init_device_graphs` hits them with invalid-argument errors.
        The paused pages remain reclaimable by the CUDA driver; the next
        capture will allocate fresh pages under a new tag.
        """
        stale = self.graph_cache.pop(model_name, None)
        if stale is None:
            return
        logger.info(f"  Graph cache evicted: {model_name} (VMM tag left paused)")

    def save_kv_pool(self, model_name: str, runner, bump):
        """Save KV pool state for fast switch-back."""
        self.kv_pool_cache[model_name] = {
            "req_to_token_pool": runner.req_to_token_pool,
            "token_to_kv_pool": runner.token_to_kv_pool,
            "token_to_kv_pool_allocator": runner.token_to_kv_pool_allocator,
            "max_total_num_tokens": runner.max_total_num_tokens,
            "max_running_requests": runner.max_running_requests,
            "weight_bytes": bump.regions["weights"].capacity,
            "kv_region_capacity": bump.regions["kv_cache"].capacity,
        }

    def get_kv_pool(self, model_name: str):
        return self.kv_pool_cache.get(model_name)

    def evict_kv_pool(self, model_name: str):
        self.kv_pool_cache.pop(model_name, None)
