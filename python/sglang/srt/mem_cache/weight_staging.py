"""
Scatter staging: background H2D of next model's weights into KV layer block free tails.

Instead of writing contiguous bytes at the KV region tail (which corrupts active KV),
staging distributes weight bytes across the free high-index rows of each KV layer block.
LIFO slot allocation keeps active slots at low indices, ensuring staging doesn't overlap.

Layout per block:
  [active rows 0..min_free-1 | gap | staging rows S..pool_size-1]

H2D proceeds top-down (highest rows first) across ALL blocks simultaneously.
Each round locks the current chunk's rows, launches H2D for all blocks (async),
records a CUDA event, then moves to next chunk without waiting. The preload thread
opportunistically queries earlier events and unlocks rows as H2D completes.
Zero synchronize() during the pipeline — PCIe stays fully saturated.

D2D gather at switch time: num_blocks copies → contiguous weights region (~21ms for 62GB).
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.bump_vram_manager import BumpVramAllocator

logger = logging.getLogger(__name__)

# Pre-load C++ batch H2D extension (compiled once, imported instantly after)
_batch_h2d_module = None
def _get_batch_h2d():
    global _batch_h2d_module
    if _batch_h2d_module is None:
        try:
            import importlib.util
            _so = "/home/mxc/.cache/torch_extensions/py312_cu128/batch_h2d/batch_h2d.so"
            spec = importlib.util.spec_from_file_location("batch_h2d", _so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _batch_h2d_module = mod
        except Exception:
            from torch.utils.cpp_extension import load as _cpp_load
            _batch_h2d_module = _cpp_load(
                name="batch_h2d",
                sources=["/home/mxc/sglang-feat-hot-switch/python/sglang/srt/mem_cache/csrc/batch_h2d.cpp"],
                extra_include_paths=["/usr/local/cuda-12.8/targets/x86_64-linux/include"],
                extra_ldflags=["-lcudart", "-L/usr/local/cuda-12.8/lib64"],
                verbose=False,
            )
    return _batch_h2d_module

H2D_CHUNK_SIZE = 32 * 1024 * 1024  # 32MB per block per round


@dataclass
class ScatterStagingInfo:
    """Metadata for scatter-staged weights across KV layer blocks."""
    staging_start: int           # first staging row in each block
    staging_rows: int            # number of staging rows per block
    staged_per_block: int        # weight bytes stored per block
    num_blocks: int              # total KV layer blocks (num_layers * 2)
    total_staged_bytes: int      # total weight bytes actually staged
    row_size: int                # bytes per row (uniform across blocks)
    pool_size: int               # total rows per block
    chunk_rows: int = 0          # rows per H2D/D2D chunk
    unsafe_blocks: int = 0       # blocks skipped (within expanded weights zone)
    # Per-chunk validity after verify_integrity. True = all rows free → D2D safe.
    # Length = ceil(staging_rows / chunk_rows). None before verify.
    chunk_valid: Optional[List[bool]] = None


class PreloadManager:
    def __init__(self):
        self.h2d_stream = torch.cuda.Stream()
        self.model_name: Optional[str] = None
        self.scatter_info: Optional[ScatterStagingInfo] = None
        self._block_tensors: List[torch.Tensor] = []
        self.layer_map = None  # List[LayerSlice] from bump allocator
        # Chunk pollution tracking (Issue M)
        self._chunk_dirty: Optional[List[bool]] = None
        self._staging_start: int = 0
        self._staging_end: int = 0
        self._chunk_rows: int = 0

    def mark_dirty_pages(self, allocated_pages: torch.Tensor):
        """Mark staging chunks as dirty when KV allocator assigns pages in staging range.

        Called from allocator.alloc() after each allocation. Uses GPU vectorized
        check to avoid per-page Python loop.
        """
        if self._chunk_dirty is None or self._chunk_rows == 0:
            return
        # Filter pages in staging range
        mask = (allocated_pages >= self._staging_start) & (allocated_pages < self._staging_end)
        if not mask.any():
            return
        staging_pages = allocated_pages[mask]
        chunk_indices = (staging_pages - self._staging_start) // self._chunk_rows
        unique_chunks = chunk_indices.unique()
        for ci in unique_chunks:
            idx = ci.item()
            if 0 <= idx < len(self._chunk_dirty):
                self._chunk_dirty[idx] = True

    def start_preload(
        self,
        model_name: str,
        cpu_state_dict: dict,
        bump: "BumpVramAllocator",
        kv_pool,          # MHATokenToKVPool
        min_free_slot: int,
        allocator=None,   # TokenToKVPoolAllocator — for per-chunk slot locking
        alloc_lock: threading.Lock = None,  # shared lock with allocator
        max_weights_bytes: int = 0,  # max(current, target) weights for layout safety
        max_runtime_bytes: int = 0,  # max(current, target) runtime for right boundary
        bump_total_bytes: int = 0,   # total bump buffer size
    ):
        """Scatter-stage weights into free tails of KV layer blocks."""
        if self.scatter_info is not None:
            self.cancel()

        self.model_name = model_name
        self.scatter_info = None

        # --- Collect blocks and compute layout ---
        blocks = list(kv_pool.k_buffer) + list(kv_pool.v_buffer)
        num_blocks = len(blocks)
        pool_size = blocks[0].shape[0]
        row_size = blocks[0][0].numel() * blocks[0][0].element_size()

        for i, blk in enumerate(blocks):
            rs = blk[0].numel() * blk[0].element_size()
            assert rs == row_size, (
                f"Block {i} row_size={rs} != block 0 row_size={row_size}. "
                f"Non-uniform KV head dims not supported yet."
            )

        import time as _time
        _t_layout = _time.perf_counter()
        staging_start, staging_rows, staged_per_block, staged_bytes, unsafe_blocks = (
            self._compute_layout(
                pool_size, row_size, num_blocks, min_free_slot, cpu_state_dict,
                max_weights_bytes=max_weights_bytes,
                kv_pool_physical_start=blocks[0].data_ptr() - bump.buffer.data_ptr(),
                max_runtime_bytes=max_runtime_bytes,
                bump_total_bytes=bump_total_bytes,
            )
        )
        logger.info(f"  TIMING: layout={(_time.perf_counter()-_t_layout)*1000:.0f}ms")
        if staged_bytes == 0:
            logger.info(f"Preload: no staging capacity for {model_name} (staged_bytes=0)")
            return

        # Reduce num_blocks by unsafe count (compute_layout only modifies local copy)
        num_blocks -= unsafe_blocks

        # Skip unsafe blocks (blocks within expanded weights zone)
        safe_blocks = blocks[unsafe_blocks:] if unsafe_blocks > 0 else blocks
        self._block_tensors = blocks  # Keep all for D2D gather offset calculation
        self._unsafe_blocks = unsafe_blocks

        # --- Get layer_map for offset mapping ---
        _layer_map = bump.get_layer_map(model_name) if bump else None
        _weights_start = bump.regions["weights"].start if bump and "weights" in bump.regions else 0
        self.layer_map = _layer_map

        # --- H2D scatter directly from pinned CPU params (no intermediate buffer) ---
        chunk_rows = max(1, H2D_CHUNK_SIZE // row_size)
        self._allocator_ref = allocator
        total_written = self._h2d_scatter(
            safe_blocks, cpu_state_dict, _layer_map, _weights_start,
            num_blocks, pool_size, row_size,
            staging_start, staged_per_block, chunk_rows,
        )
        self._allocator_ref = None
        _scatter_ms = getattr(self, "_last_scatter_ms", 0)
        logger.debug(f"  TIMING: h2d_scatter={_scatter_ms:.0f}ms written={total_written/1024**2:.0f}MB")

        self.scatter_info = ScatterStagingInfo(
            staging_start=staging_start,
            staging_rows=staging_rows,
            staged_per_block=staged_per_block,
            num_blocks=num_blocks,
            total_staged_bytes=total_written,
            row_size=row_size,
            pool_size=pool_size,
            chunk_rows=chunk_rows,
            unsafe_blocks=unsafe_blocks,
        )

        # Initialize chunk dirty tracking (Issue M)
        # Safety: _chunk_dirty starts as None during H2D scatter, so mark_dirty_pages()
        # is a no-op until here. This is safe because H2D locks each chunk's rows
        # (removes from free_pages) before writing, so the allocator cannot hand out
        # those rows to KV. After H2D completes and rows are unlocked, _chunk_dirty
        # is initialized to all-False, correctly reflecting that no corruption occurred
        # during the H2D phase itself.
        num_chunks = math.ceil(staging_rows / chunk_rows)
        self._chunk_dirty = [False] * num_chunks
        self._staging_start = staging_start
        self._staging_end = staging_start + staging_rows
        self._chunk_rows = chunk_rows

        logger.debug(
            f"Preload: {model_name}, "
            f"{total_written / 1024**2:.1f}MB staged across {num_blocks} blocks, "
            f"rows [{staging_start}, {pool_size}), "
            f"{staged_per_block / 1024**2:.1f}MB/block, "
            f"chunk_rows={chunk_rows}"
        )

    # ------------------------------------------------------------------
    # Layout computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_layout(pool_size, row_size, num_blocks, min_free_slot, cpu_state_dict,
                        max_weights_bytes=0, kv_pool_physical_start=0,
                        max_runtime_bytes=0, bump_total_bytes=0):
        """Compute staging layout with safety guarantee.

        Layout safety: staging rows must be at physical addresses beyond
        max(current_weights, target_weights) in the bump buffer. This prevents
        D2D gather source from overlapping with the expanded weights destination.

        Args:
            max_weights_bytes: max(current_model_weights, target_model_weights) in bytes.
kv_pool_physical_start: physical start of KV pool in bump buffer
                                    (= blocks[0].data_ptr() - bump.buffer.data_ptr()). Must use
                                    actual block address, not bump.left_offset, because KV pool
                                    may be restored from cache at a lower address.
        """
        # Layout safety floor: rows below this index could be overwritten
        # when weights region expands to max_weights_bytes.
        # Layout safety: when target weights > current, the first few KV blocks
        # fall within the expanded weights zone. Instead of per-row min_safe_row
        # (which can exceed pool_size), compute per-block: skip fully-unsafe blocks.
        unsafe_blocks = 0
        if max_weights_bytes > kv_pool_physical_start and kv_pool_physical_start > 0:
            unsafe_bytes = max_weights_bytes - kv_pool_physical_start
            block_bytes = pool_size * row_size
            if block_bytes > 0:
                unsafe_blocks = math.ceil(unsafe_bytes / block_bytes)
            num_blocks -= unsafe_blocks
            if num_blocks <= 0:
                logger.info(f"  _compute_layout: all {num_blocks + unsafe_blocks} blocks unsafe")
                return 0, 0, 0, 0, 0

        # Right boundary: staging rows must not extend into potential runtime region
        if bump_total_bytes > 0 and max_runtime_bytes > 0:
            safe_right_boundary = bump_total_bytes - max_runtime_bytes
            kv_pool_physical_end = kv_pool_physical_start + pool_size * row_size
            if kv_pool_physical_end > safe_right_boundary:
                unsafe_right_rows = math.ceil((kv_pool_physical_end - safe_right_boundary) / row_size)
                max_safe_row = pool_size - unsafe_right_rows
            else:
                max_safe_row = pool_size
        else:
            max_safe_row = pool_size

        effective_floor = min_free_slot
        effective_ceiling = min(pool_size, max_safe_row)
        free_rows = effective_ceiling - effective_floor
        if free_rows <= 0:
            logger.info(f"  _compute_layout: free_rows<=0, pool_size={pool_size} max_safe_row={max_safe_row} min_free_slot={min_free_slot} effective_floor={effective_floor} effective_ceiling={effective_ceiling}")
            return 0, 0, 0, 0, 0

        capacity_per_block = free_rows * row_size
        total_capacity = num_blocks * capacity_per_block

        _ALIGNMENT = 256
        def _align_up_local(n):
            return (n + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
        total_weight_bytes = sum(
            _align_up_local(t.numel() * t.element_size()) for t in cpu_state_dict.values()
        )
        staged_bytes = min(total_weight_bytes, total_capacity)

        if staged_bytes == 0:
            return 0, 0, 0, 0, 0

        staged_per_block = math.ceil(staged_bytes / num_blocks)
        # Align to row_size so D2D stride matches H2D write size
        staged_per_block = math.ceil(staged_per_block / row_size) * row_size
        staging_rows = math.ceil(staged_per_block / row_size)
        staging_start = effective_ceiling - staging_rows

        # Double-check: staging must be in safe zone
        assert staging_start >= effective_floor and staging_start + staging_rows <= max_safe_row, (
            f"staging_start={staging_start} effective_floor={effective_floor} "
            f"staging_end={staging_start + staging_rows} max_safe_row={max_safe_row}, "
            f"layout safety violated"
        )

        logger.info(
            f"  LAYOUT: pool_size={pool_size} row_size={row_size} num_blocks={num_blocks} "
            f"min_free_slot={min_free_slot} max_weights_bytes={max_weights_bytes} "
            f"kv_pool_physical_start={kv_pool_physical_start}")
        _unsafe = max_weights_bytes - kv_pool_physical_start if max_weights_bytes > kv_pool_physical_start and kv_pool_physical_start > 0 else 0
        logger.info(
            f"  LAYOUT SAFETY: unsafe_bytes={_unsafe} unsafe_blocks={unsafe_blocks} "
            f"effective_floor={effective_floor} free_rows={free_rows}")
        _staging_phys_offset = kv_pool_physical_start + staging_start * row_size
        _staging_phys_end = kv_pool_physical_start + (staging_start + staging_rows) * row_size
        logger.info(
            f"  LAYOUT RESULT: staging_start={staging_start} staging_rows={staging_rows} "
            f"staged_bytes={staged_bytes} staged_per_block={staged_per_block} "
            f"staging_phys_offset={_staging_phys_offset} staging_phys_end={_staging_phys_end} "
            f"weights_would_reach={max_weights_bytes} overlap={_staging_phys_offset < max_weights_bytes}")
        return staging_start, staging_rows, staged_per_block, staged_bytes, unsafe_blocks

    # ------------------------------------------------------------------
    # Param serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_params(cpu_state_dict, staged_bytes, num_blocks, staged_per_block,
                          layer_map=None, weights_start=0):
        """Serialize CPU params into a pinned flat byte stream matching bump layout.

        If layer_map is provided, each param is placed at its correct offset
        (matching the bump allocator's create_tensor layout). Otherwise falls
        back to sequential iteration order.
        """
        import time as _time_ser
        padded_len = num_blocks * staged_per_block
        _t_alloc = _time_ser.perf_counter()
        stream = torch.zeros(padded_len, dtype=torch.uint8, pin_memory=True)
        logger.info(f"    TIMING serialize_alloc: {(_time_ser.perf_counter()-_t_alloc)*1000:.0f}ms ({padded_len/1024**3:.1f}GB)")
        _t_copy = _time_ser.perf_counter()

        if layer_map:
            # Place each param at the exact offset used by bump allocator
            for ls in layer_map:
                rel_offset = ls.offset - weights_start
                if rel_offset < 0 or rel_offset >= staged_bytes:
                    continue
                tensor = cpu_state_dict.get(ls.name)
                if tensor is None:
                    continue
                raw_bytes = min(ls.nbytes, tensor.numel() * tensor.element_size())
                take = min(raw_bytes, staged_bytes - rel_offset)
                if take <= 0:
                    continue
                stream[rel_offset : rel_offset + take].copy_(
                    tensor.contiguous().view(-1).view(torch.uint8)[:take]
                )
        else:
            # Fallback: sequential iteration with 256B alignment
            _ALIGNMENT = 256
            def _align_up(n):
                return (n + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
            pos = 0
            for name, tensor in cpu_state_dict.items():
                raw_bytes = tensor.numel() * tensor.element_size()
                aligned_bytes = _align_up(raw_bytes)
                take = min(aligned_bytes, staged_bytes - pos)
                if take <= 0:
                    break
                raw_take = min(raw_bytes, take)
                stream[pos : pos + raw_take].copy_(
                    tensor.contiguous().view(-1).view(torch.uint8)[:raw_take]
                )
                pos += take
        logger.info(f"    TIMING serialize_copy: {(_time_ser.perf_counter()-_t_copy)*1000:.0f}ms")
        return stream

    # ------------------------------------------------------------------
    # H2D scatter loop (pipelined, zero synchronize)
    # ------------------------------------------------------------------

    def _h2d_scatter(
        self, blocks, cpu_state_dict, layer_map, weights_start,
        num_blocks, pool_size, row_size,
        staging_start, staged_per_block, chunk_rows,
    ):
        """Direct H2D via C++ batch dispatch. GIL-free per-chunk copies.

        All data_ptr() calls happen once in plan phase. Per-chunk loop
        only passes integer lists to C++ — zero Python tensor ops.
        """
        import time as _time_h2d
        import torch
        _batch_h2d = _get_batch_h2d()
        stream_ptr = self.h2d_stream.cuda_stream
        stream_ptr = self.h2d_stream.cuda_stream

        # === Plan phase: pre-compute ALL ops with data_ptr (one-time GIL cost) ===
        _t_plan = _time_h2d.perf_counter()
        staging_rows = math.ceil(staged_per_block / row_size)

        # Step 1: build per-block param list with raw byte views
        block_params = [[] for _ in range(num_blocks)]
        if layer_map:
            for ls in layer_map:
                t = cpu_state_dict.get(ls.name)
                if t is None:
                    continue
                rel = ls.offset - weights_start
                raw = min(ls.nbytes, t.numel() * t.element_size())
                if rel < 0 or raw <= 0:
                    continue
                cpu_view = t.contiguous().view(-1).view(torch.uint8)[:raw]
                first_bi = rel // staged_per_block
                last_bi = (rel + raw - 1) // staged_per_block
                for bi in range(max(0, first_bi), min(last_bi + 1, num_blocks)):
                    blk_start = bi * staged_per_block
                    cs = max(rel, blk_start)
                    ce = min(rel + raw, blk_start + staged_per_block)
                    if ce <= cs:
                        continue
                    block_params[bi].append((cs - blk_start, cpu_view[cs - rel : ce - rel]))
        else:
            _ALIGNMENT = 256
            pos = 0
            for name, tensor in cpu_state_dict.items():
                raw = tensor.numel() * tensor.element_size()
                aligned = (raw + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)
                if pos + aligned > num_blocks * staged_per_block:
                    break
                cpu_view = tensor.contiguous().view(-1).view(torch.uint8)[:raw]
                first_bi = pos // staged_per_block
                last_bi = (pos + raw - 1) // staged_per_block
                for bi in range(max(0, first_bi), min(last_bi + 1, num_blocks)):
                    blk_start = bi * staged_per_block
                    cs = max(pos, blk_start)
                    ce = min(pos + raw, blk_start + staged_per_block)
                    if ce <= cs:
                        continue
                    block_params[bi].append((cs - blk_start, cpu_view[cs - pos : ce - pos]))
                pos += aligned

        # Step 2: pre-compute data_ptr for ALL ops, grouped by chunk
        # chunk_ops[chunk_idx] = [(src_ptr, dst_ptr, nbytes), ...]
        block_flats = [blk.view(-1).view(torch.uint8) for blk in blocks]

        chunk_ops_list = []
        chunk_row_ranges = []  # (chunk_start_row, chunk_end_row) for reservation release
        written_per_block = 0
        chunk_end = staging_start + staging_rows - 1

        while chunk_end >= staging_start and written_per_block < staged_per_block:
            actual_rows = min(chunk_rows, chunk_end - staging_start + 1)
            chunk_start_row = chunk_end - actual_rows + 1
            chunk_bytes = min(actual_rows * row_size, staged_per_block - written_per_block)
            if chunk_bytes <= 0:
                break

            tail_off = staged_per_block - written_per_block - chunk_bytes
            ops = []
            chunk_written = 0

            for bi in range(num_blocks):
                dst_base = chunk_start_row * row_size
                for (off_in_blk, src_bytes) in block_params[bi]:
                    param_end = off_in_blk + len(src_bytes)
                    if param_end <= tail_off or off_in_blk >= tail_off + chunk_bytes:
                        continue
                    cs = max(off_in_blk, tail_off)
                    ce = min(param_end, tail_off + chunk_bytes)
                    n = ce - cs
                    src_s = cs - off_in_blk
                    dst_s = dst_base + (cs - tail_off)
                    ops.append((
                        src_bytes[src_s : src_s + n].data_ptr(),
                        block_flats[bi][dst_s : dst_s + n].data_ptr(),
                        n,
                    ))
                    chunk_written += n

            chunk_ops_list.append((ops, chunk_written))
            chunk_row_ranges.append(chunk_start_row)
            written_per_block += chunk_bytes
            chunk_end = chunk_start_row - 1

        _plan_ms = (_time_h2d.perf_counter() - _t_plan) * 1000
        _total_ops = sum(len(ops) for ops, _ in chunk_ops_list)
        logger.debug(f"    H2D plan: {_plan_ms:.0f}ms, {_total_ops} ops, {len(chunk_ops_list)} chunks")

        # === Dispatch phase: per-chunk C++ call + event sync + progressive release ===
        _t_dispatch = _time_h2d.perf_counter()
        total_written = 0
        staging_end = staging_start + staging_rows
        allocator_ref = getattr(self, '_allocator_ref', None)

        for ci, ((ops, chunk_written), chunk_start_row) in enumerate(
            zip(chunk_ops_list, chunk_row_ranges)
        ):
            # Per-chunk reservation: only reserve rows being H2D'd right now (~6%)
            if allocator_ref is not None:
                chunk_end_row = min(chunk_start_row + chunk_rows, staging_start + staging_rows)
                allocator_ref._reserved_range = (chunk_start_row, chunk_end_row)

            if ops:
                _batch_h2d.dispatch(ops, stream_ptr)

            event = torch.cuda.Event()
            event.record(self.h2d_stream)
            event.synchronize()

            # Chunk done: clear reservation (data written, LIFO protects it)
            if allocator_ref is not None:
                allocator_ref._reserved_range = None

            # Progressive release
            if allocator_ref is not None:
                new_end = chunk_start_row
                if new_end <= staging_start:
                    allocator_ref._reserved_range = None
                else:
                    allocator_ref._reserved_range = (staging_start, new_end)

            total_written += chunk_written

        _dispatch_ms = (_time_h2d.perf_counter() - _t_dispatch) * 1000
        logger.debug(f"    H2D dispatch+sync: {_dispatch_ms:.0f}ms")
        self._last_scatter_ms = (_time_h2d.perf_counter() - _t_plan) * 1000
        return total_written

    @staticmethod
    def _drain_completed_unlocks(pending_unlocks, allocator, alloc_lock):
        """Non-blocking: unlock chunks whose events have completed."""
        while pending_unlocks:
            event, locked = pending_unlocks[0]
            if not event.query():
                break  # This and all later events still pending
            pending_unlocks.pop(0)
            _lk = alloc_lock
            if _lk: _lk.acquire()
            try:
                allocator.free_pages = torch.cat([allocator.free_pages, locked])
            finally:
                if _lk: _lk.release()

    @staticmethod
    def _drain_all_unlocks(pending_unlocks, allocator, alloc_lock):
        """Blocking: wait for remaining events and unlock.

        CUDA events on the same stream complete in FIFO order, so only
        the last event needs synchronization — all earlier ones are
        guaranteed complete by then.
        """
        if not pending_unlocks:
            return
        # Only sync the last event (FIFO guarantees earlier ones are done)
        pending_unlocks[-1][0].synchronize()
        for _event, locked in pending_unlocks:
            _lk = alloc_lock
            if _lk: _lk.acquire()
            try:
                allocator.free_pages = torch.cat([allocator.free_pages, locked])
            finally:
                if _lk: _lk.release()
        pending_unlocks.clear()

    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------

    def verify_integrity(self, allocator):
        """Check which staging chunks are still valid.

        Uses dirty flags from mark_dirty_pages() if available (Issue M),
        falls back to free_pages check otherwise.
        """
        if self.scatter_info is None:
            return

        si = self.scatter_info
        chunk_rows = si.chunk_rows
        num_chunks = math.ceil(si.staging_rows / chunk_rows)

        if self._chunk_dirty is not None and len(self._chunk_dirty) == num_chunks:
            # Use dirty flags (Issue M: tracks actual KV allocations into staging)
            chunk_valid = [not d for d in self._chunk_dirty]
        else:
            # Fallback: check free_pages (original behavior)
            free_pages = allocator.free_pages
            if len(free_pages) == 0:
                si.chunk_valid = [False] * num_chunks
                logger.info(f"Staging integrity: fully corrupted (no free pages)")
                return

            staging_range = torch.arange(
                si.staging_start, si.pool_size, device=free_pages.device
            )
            is_free = torch.isin(staging_range, free_pages)
            chunk_valid = []
            for ci in range(num_chunks):
                row_off = ci * chunk_rows
                row_end = min(row_off + chunk_rows, si.staging_rows)
                valid = is_free[row_off:row_end].all().item()
                chunk_valid.append(valid)

        si.chunk_valid = chunk_valid
        n_corrupted = sum(1 for v in chunk_valid if not v)

        if n_corrupted == 0:
            logger.info(
                f"Staging integrity: all {num_chunks} chunks OK "
                f"(rows [{si.staging_start}, {si.pool_size}))"
            )
        else:
            corrupted_rows = sum(
                min(chunk_rows, si.staging_rows - ci * chunk_rows)
                for ci, v in enumerate(chunk_valid) if not v
            )
            corrupted_bytes = corrupted_rows * si.row_size * si.num_blocks
            logger.info(
                f"Staging integrity: {n_corrupted}/{num_chunks} chunks corrupted, "
                f"~{corrupted_bytes / 1024**2:.1f}MB need H2D fallback"
            )

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------

    def wait_complete(self):
        self.h2d_stream.synchronize()

    @property
    def is_valid(self) -> bool:
        return self.scatter_info is not None and self.scatter_info.total_staged_bytes > 0

    @property
    def block_tensors(self) -> List[torch.Tensor]:
        return self._block_tensors

    def cancel(self):
        self.h2d_stream.synchronize()
        self.scatter_info = None
        self._block_tensors = []
        self.layer_map = None
        self._chunk_dirty = None
        self._staging_start = 0
        self._staging_end = 0
        self._chunk_rows = 0
