"""
Scatter staging: background H2D of next model's weights into KV layer block free tails.

Instead of writing contiguous bytes at the KV region tail (which corrupts active KV),
staging distributes weight bytes across the free high-index rows of each KV layer block.
LIFO slot allocation keeps active slots at low indices, ensuring staging doesn't overlap.

Layout per block:
  [active rows 0..min_free-1 | gap | staging rows S..pool_size-1]

H2D proceeds top-down (highest rows first) across ALL blocks simultaneously.
Each round enters allocator.staging_chunk_guard(chunk), launches H2D for all
blocks (C++ batch dispatch), waits for completion (event.synchronize), then
exits the guard and moves to the next chunk.

D2D gather at switch time: num_blocks copies → contiguous weights region (~21ms for 62GB).
"""

from __future__ import annotations

import contextlib
import logging
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import torch
import torch.cuda

from sglang.srt.mem_cache.vram_manager import _align_up

if TYPE_CHECKING:
    from sglang.srt.mem_cache.vram_manager import BumpVramAllocator

logger = logging.getLogger(__name__)

# Pre-load C++ batch H2D extension (compiled once, imported instantly after)
_batch_h2d_module = None


def _get_batch_h2d():
    global _batch_h2d_module
    if _batch_h2d_module is not None:
        return _batch_h2d_module

    import importlib.util
    import os

    # Try fast import from torch extension cache (avoids 3.5s load() overhead)
    try:
        from torch.utils.cpp_extension import _get_build_directory

        _so = os.path.join(_get_build_directory("batch_h2d", False), "batch_h2d.so")
        if os.path.isfile(_so):
            spec = importlib.util.spec_from_file_location("batch_h2d", _so)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            _batch_h2d_module = mod
            return _batch_h2d_module
    except Exception:
        pass

    # Fallback: JIT compile from source
    from torch.utils.cpp_extension import load as _cpp_load, CUDA_HOME

    _src = os.path.join(os.path.dirname(__file__), "csrc", "batch_h2d.cpp")
    _cuda = CUDA_HOME or "/usr/local/cuda"
    _batch_h2d_module = _cpp_load(
        name="batch_h2d",
        sources=[_src],
        extra_include_paths=[
            os.path.join(_cuda, "targets", os.uname().machine + "-linux", "include"),
            os.path.join(_cuda, "include"),
        ],
        extra_ldflags=["-lcudart", f"-L{os.path.join(_cuda, 'lib64')}"],
        verbose=False,
    )
    return _batch_h2d_module


H2D_CHUNK_SIZE = 32 * 1024 * 1024  # 32MB per block per round


@dataclass
class ScatterStagingInfo:
    """Metadata for scatter-staged weights across KV layer blocks."""

    staging_start: int  # first staging row in each block
    staging_rows: int  # number of staging rows per block
    staged_per_block: int  # per-block byte span reserved for staging (row-aligned, may include tail padding beyond the real payload)
    num_blocks: int  # total KV layer blocks (num_layers * 2)
    total_staged_bytes: int  # payload bytes actually written across all blocks by _h2d_scatter
    row_size: int  # bytes per row (uniform across blocks)
    pool_size: int  # total rows per block
    chunk_rows: int = 0  # rows per H2D/D2D chunk
    unsafe_blocks: int = 0  # blocks skipped (within expanded weights zone)
    # Per-chunk validity after verify_integrity. True = all rows free → D2D safe.
    # Length = ceil(staging_rows / chunk_rows). None before verify.
    chunk_valid: Optional[List[bool]] = None


class PreloadManager:
    def __init__(self):
        self.h2d_stream = torch.cuda.Stream()
        self.model_name: Optional[str] = None
        self.scatter_info: Optional[ScatterStagingInfo] = None
        self._block_tensors: List[torch.Tensor] = []
        # Chunk pollution tracking
        self._chunk_dirty: Optional[List[bool]] = None
        self._staging_start: int = 0
        self._staging_end: int = 0
        self._chunk_rows: int = 0
        # Graceful-stop flag checked at each chunk boundary in _h2d_scatter.
        # Set by request_stop() when a switch wants to use the already-staged
        # chunks via partial D2D rather than wait for the full preload.
        self._stop_requested: bool = False

    def request_stop(self) -> None:
        """Ask the preload loop to stop at the next chunk boundary.

        Chunks already published in scatter_info.chunk_valid stay valid and
        can be consumed by gather_to_bump; unstaged chunks fall back to H2D.
        """
        self._stop_requested = True

    def mark_dirty_pages(self, allocated_pages: torch.Tensor):
        """Mark every staging chunk that overlaps the newly-allocated rows.

        Input is per-row (one row id per token slot allocated), but the
        dirty flag is per *chunk* (a chunk covers `chunk_rows` rows). Any
        row falling inside a chunk marks that whole chunk dirty, because
        staging writes the chunk as one unit.

        Called from allocator.alloc() after each allocation. Uses GPU
        vectorized filter to avoid a per-row Python loop.
        """
        if self._chunk_dirty is None or self._chunk_rows == 0:
            return
        mask = (allocated_pages >= self._staging_start) & (
            allocated_pages < self._staging_end
        )
        if not mask.any():
            return
        staging_pages = allocated_pages[mask]
        chunk_indices = (staging_pages - self._staging_start) // self._chunk_rows
        unique_chunks = chunk_indices.unique()
        for ci in unique_chunks:
            self._chunk_dirty[ci.item()] = True

    def start_preload(
        self,
        model_name: str,
        cpu_state_dict: dict,
        bump: "BumpVramAllocator",
        kv_pool,  # MHATokenToKVPool
        min_free_slot: int,
        allocator=None,  # TokenToKVPoolAllocator — for per-chunk slot locking
        max_weights_bytes: int = 0,  # max(current, target) weights for layout safety
    ):
        """Scatter-stage weights into free tails of KV layer blocks."""
        # Always reset prior staging state first. Any early return below
        # (e.g. staged_bytes == 0, _h2d_scatter raising) would otherwise leave
        # stale scatter_info / _block_tensors attached to the new model_name,
        # making `is_valid` falsely report readiness.
        self.h2d_stream.synchronize()
        self._reset_session()
        self.model_name = model_name

        # --- Collect blocks and compute layout ---
        blocks = list(kv_pool.k_buffer) + list(kv_pool.v_buffer)
        num_blocks = len(blocks)
        pool_size = blocks[0].shape[0]
        row_size = blocks[0][0].numel() * blocks[0][0].element_size()

        import time as _time

        _t_layout = _time.perf_counter()
        staging_start, staging_rows, staged_per_block, staged_bytes, unsafe_blocks = (
            self._compute_layout(
                pool_size,
                row_size,
                num_blocks,
                min_free_slot,
                cpu_state_dict,
                max_weights_bytes=max_weights_bytes,
                kv_pool_physical_start=blocks[0].data_ptr() - bump.buffer.data_ptr(),
            )
        )
        logger.debug(f"  TIMING: layout={(_time.perf_counter()-_t_layout)*1000:.0f}ms")
        if staged_bytes == 0:
            logger.info(
                f"Preload: no staging capacity for {model_name} (staged_bytes=0)"
            )
            return

        # Reduce num_blocks by unsafe count (compute_layout only modifies local copy)
        num_blocks -= unsafe_blocks

        # Skip unsafe blocks (blocks within expanded weights zone)
        safe_blocks = blocks[unsafe_blocks:] if unsafe_blocks > 0 else blocks
        self._block_tensors = blocks  # Keep all for D2D gather offset calculation

        # --- H2D scatter directly from pinned CPU params (no intermediate buffer) ---
        chunk_rows = max(1, H2D_CHUNK_SIZE // row_size)

        # Initialize dirty tracking BEFORE H2D so mark_dirty_pages() works during scatter
        num_chunks = math.ceil(staging_rows / chunk_rows)
        self._chunk_dirty = [False] * num_chunks
        self._staging_start = staging_start
        self._staging_end = staging_start + staging_rows
        self._chunk_rows = chunk_rows

        # Publish scatter_info *before* the H2D scatter runs so a switch that
        # cuts in mid-preload can read already-staged chunks via chunk_valid.
        # chunk_valid[i] flips to True inside _h2d_scatter after each chunk
        # has been event-synchronized on h2d_stream.
        self._stop_requested = False
        self.scatter_info = ScatterStagingInfo(
            staging_start=staging_start,
            staging_rows=staging_rows,
            staged_per_block=staged_per_block,
            num_blocks=num_blocks,
            total_staged_bytes=0,
            row_size=row_size,
            pool_size=pool_size,
            chunk_rows=chunk_rows,
            unsafe_blocks=unsafe_blocks,
            chunk_valid=[False] * num_chunks,
        )

        total_written = self._h2d_scatter(
            safe_blocks,
            cpu_state_dict,
            num_blocks,
            row_size,
            staging_start,
            staged_per_block,
            chunk_rows,
            allocator,
        )
        _scatter_ms = getattr(self, "_last_scatter_ms", 0)
        logger.debug(
            f"  TIMING: h2d_scatter={_scatter_ms:.0f}ms written={total_written/1024**2:.0f}MB"
        )

        # Update final total (chunk_valid was updated chunk-by-chunk inside scatter)
        if self.scatter_info is not None:
            self.scatter_info.total_staged_bytes = total_written

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
    def _compute_layout(
        pool_size,
        row_size,
        num_blocks,
        min_free_slot,
        cpu_state_dict,
        max_weights_bytes=0,
        kv_pool_physical_start=0,
    ):
        """Compute staging layout with safety guarantee.

        Staging rows must be at physical addresses beyond max(current_weights,
        target_weights) in the bump buffer, preventing D2D gather source from
        overlapping with the expanded weights destination.

        Args:
            max_weights_bytes: max(current_model_weights, target_model_weights) in bytes.
            kv_pool_physical_start: physical start of KV pool in bump buffer
                (= blocks[0].data_ptr() - bump.buffer.data_ptr()).
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
                logger.info(
                    f"  _compute_layout: all {num_blocks + unsafe_blocks} blocks unsafe"
                )
                return 0, 0, 0, 0, 0

        effective_floor = min_free_slot
        effective_ceiling = pool_size
        free_rows = effective_ceiling - effective_floor
        if free_rows <= 0:
            logger.info(
                f"  _compute_layout: free_rows<=0, pool_size={pool_size} min_free_slot={min_free_slot}"
            )
            return 0, 0, 0, 0, 0

        capacity_per_block = free_rows * row_size
        total_capacity = num_blocks * capacity_per_block

        total_weight_bytes = sum(
            _align_up(t.numel() * t.element_size()) for t in cpu_state_dict.values()
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
        assert (
            staging_start >= effective_floor
            and staging_start + staging_rows <= pool_size
        ), (
            f"staging_start={staging_start} effective_floor={effective_floor} "
            f"staging_end={staging_start + staging_rows} pool_size={pool_size}, "
            f"layout safety violated"
        )

        logger.debug(
            f"  LAYOUT: pool_size={pool_size} row_size={row_size} num_blocks={num_blocks} "
            f"min_free_slot={min_free_slot} max_weights_bytes={max_weights_bytes} "
            f"kv_pool_physical_start={kv_pool_physical_start}"
        )
        _unsafe = (
            max_weights_bytes - kv_pool_physical_start
            if max_weights_bytes > kv_pool_physical_start and kv_pool_physical_start > 0
            else 0
        )
        logger.debug(
            f"  LAYOUT SAFETY: unsafe_bytes={_unsafe} unsafe_blocks={unsafe_blocks} "
            f"effective_floor={effective_floor} free_rows={free_rows}"
        )
        # Staging is in safe blocks (block[unsafe_blocks:]), compute actual physical address
        _safe_block_phys_start = (
            kv_pool_physical_start + unsafe_blocks * pool_size * row_size
        )
        _staging_phys_offset = _safe_block_phys_start + staging_start * row_size
        _staging_phys_end = (
            _safe_block_phys_start + (staging_start + staging_rows) * row_size
        )
        # Keep this at info — it is the main actionable safety check.
        logger.info(
            f"  LAYOUT RESULT: staging_start={staging_start} staging_rows={staging_rows} "
            f"staged_bytes={staged_bytes} staged_per_block={staged_per_block} "
            f"staging_phys_offset={_staging_phys_offset} staging_phys_end={_staging_phys_end} "
            f"weights_would_reach={max_weights_bytes} overlap={_staging_phys_offset < max_weights_bytes}"
        )
        return (
            staging_start,
            staging_rows,
            staged_per_block,
            staged_bytes,
            unsafe_blocks,
        )

    # ------------------------------------------------------------------
    # H2D scatter loop (pipelined, zero synchronize)
    # ------------------------------------------------------------------

    def _h2d_scatter(
        self,
        blocks,
        cpu_state_dict,
        num_blocks,
        row_size,
        staging_start,
        staged_per_block,
        chunk_rows,
        allocator,
    ):
        """Scatter CPU params into staging rows via a C++ per-chunk dispatch.

        Hot path: all tensor work (data_ptr() calls, byte slicing) happens once
        in the plan phase; the per-chunk dispatch loop only hands integer
        (src_ptr, dst_ptr, nbytes) tuples to the C++ extension under the
        staging_chunk_guard, so the inner loop is GIL-friendly.

        Relies on `cpu_state_dict.items()` ordering — Python 3.7+ guarantees
        insertion order, and the bump layout assumes the same prefix-sum order.
        """
        import time as _time_h2d

        _batch_h2d = _get_batch_h2d()
        stream_ptr = self.h2d_stream.cuda_stream

        # === Plan phase: pre-compute ALL ops with data_ptr (one-time GIL cost) ===
        _t_plan = _time_h2d.perf_counter()
        staging_rows = math.ceil(staged_per_block / row_size)

        # Step 1: build per-block param list with raw byte views
        # Offsets computed as prefix sum over cpu_state_dict (same order as bump layout)
        block_params = [[] for _ in range(num_blocks)]
        rel = 0
        for name, t in cpu_state_dict.items():
            raw = t.numel() * t.element_size()
            aligned = _align_up(raw)
            if rel + aligned > num_blocks * staged_per_block:
                break
            cpu_view = t.contiguous().view(-1).view(torch.uint8)[:raw]
            first_bi = rel // staged_per_block
            last_bi = (rel + raw - 1) // staged_per_block
            for bi in range(max(0, first_bi), min(last_bi + 1, num_blocks)):
                blk_start = bi * staged_per_block
                cs = max(rel, blk_start)
                ce = min(rel + raw, blk_start + staged_per_block)
                block_params[bi].append((cs - blk_start, cpu_view[cs - rel : ce - rel]))
            rel += aligned

        # Step 2: pre-compute data_ptr for ALL ops, grouped by chunk
        # chunk_ops[chunk_idx] = [(src_ptr, dst_ptr, nbytes), ...]
        block_flats = [blk.view(-1).view(torch.uint8) for blk in blocks]

        chunk_ops_list = []
        chunk_row_ranges = []  # chunk_start_row for each chunk
        written_per_block = 0
        chunk_end = staging_start + staging_rows - 1

        # First iter writes the leftover rows at the right edge so each iter
        # aligns 1:1 with a gather-chunk (iter k ↔ gather-chunk num_chunks-1-k).
        # leftover = staging_rows mod chunk_rows, falls back to chunk_rows when
        # staging_rows is an exact multiple.
        leftover_rows = staging_rows % chunk_rows or chunk_rows

        while chunk_end >= staging_start and written_per_block < staged_per_block:
            iter_rows = leftover_rows if written_per_block == 0 else chunk_rows
            actual_rows = min(iter_rows, chunk_end - staging_start + 1)
            chunk_start_row = chunk_end - actual_rows + 1
            chunk_bytes = min(
                actual_rows * row_size, staged_per_block - written_per_block
            )
            if chunk_bytes <= 0:
                break

            tail_off = staged_per_block - written_per_block - chunk_bytes
            chunk_end_off = tail_off + chunk_bytes
            ops = []
            chunk_written = 0

            for bi in range(num_blocks):
                dst_base = chunk_start_row * row_size
                for off_in_blk, src_bytes in block_params[bi]:
                    param_end = off_in_blk + len(src_bytes)
                    if param_end <= tail_off or off_in_blk >= chunk_end_off:
                        continue
                    cs = max(off_in_blk, tail_off)
                    ce = min(param_end, chunk_end_off)
                    n = ce - cs
                    src_s = cs - off_in_blk
                    dst_s = dst_base + (cs - tail_off)
                    ops.append(
                        (
                            src_bytes[src_s : src_s + n].data_ptr(),
                            block_flats[bi][dst_s : dst_s + n].data_ptr(),
                            n,
                        )
                    )
                    chunk_written += n

            chunk_ops_list.append((ops, chunk_written))
            chunk_row_ranges.append(chunk_start_row)
            written_per_block += chunk_bytes
            chunk_end = chunk_start_row - 1

        _plan_ms = (_time_h2d.perf_counter() - _t_plan) * 1000
        _total_ops = sum(len(ops) for ops, _ in chunk_ops_list)
        logger.debug(
            f"    H2D plan: {_plan_ms:.0f}ms, {_total_ops} ops, {len(chunk_ops_list)} chunks"
        )

        # === Dispatch phase: per-chunk C++ call + event sync + progressive release ===
        _t_dispatch = _time_h2d.perf_counter()
        total_written = 0
        allocator_ref = allocator

        for ci, ((ops, chunk_written), chunk_start_row) in enumerate(
            zip(chunk_ops_list, chunk_row_ranges)
        ):
            # Check if KVC already allocated into this chunk — if so, stop.
            # KVC grows left-to-right, staging goes right-to-left, so hitting
            # a dirty chunk means all remaining (lower) chunks are unsafe too.
            dirty_idx = (chunk_start_row - staging_start) // chunk_rows
            if 0 <= dirty_idx < len(self._chunk_dirty) and self._chunk_dirty[dirty_idx]:
                logger.info(
                    f"    H2D stopped: chunk {ci} (rows {chunk_start_row}+) dirty, KVC boundary reached"
                )
                break

            # Reserve current chunk rows + hold staging lock around dispatch+sync.
            # Without the lock, main-thread alloc can read reserved_range=None,
            # pick rows in this chunk's range, then our H2D would overwrite KV data.
            if allocator_ref is not None:
                chunk_end_row = min(
                    chunk_start_row + chunk_rows, staging_start + staging_rows
                )
                guard = allocator_ref.staging_chunk_guard(chunk_start_row, chunk_end_row)
            else:
                guard = contextlib.nullcontext()

            if self._stop_requested:
                logger.info(
                    f"    H2D scatter stopped by caller at chunk {ci}/{len(chunk_ops_list)} "
                    f"(staged {total_written/1024**2:.0f}MB)"
                )
                break

            with guard:
                if ops:
                    _batch_h2d.dispatch(ops, stream_ptr)

                event = torch.cuda.Event()
                event.record(self.h2d_stream)
                event.synchronize()

            total_written += chunk_written

            # Publish this chunk completion. With the leftover-first layout
            # above, each iter writes exactly one gather-chunk: iter k covers
            # rows [ci*chunk_rows, (ci+1)*chunk_rows) for ci = num_chunks-1-k,
            # so floor division gives the right index directly.
            # event.synchronize() above guarantees the H2D bytes are committed
            # to GPU memory before the bit flips to True.
            si = self.scatter_info
            ci_arr = (chunk_start_row - staging_start) // chunk_rows
            si.chunk_valid[ci_arr] = True
            si.total_staged_bytes = total_written

        _dispatch_ms = (_time_h2d.perf_counter() - _t_dispatch) * 1000
        logger.debug(f"    H2D dispatch+sync: {_dispatch_ms:.0f}ms")
        self._last_scatter_ms = (_time_h2d.perf_counter() - _t_plan) * 1000
        return total_written

    # ------------------------------------------------------------------
    # D2D gather + H2D fallback
    # ------------------------------------------------------------------

    def gather_to_bump(
        self, bump, weights_start: int, cpu_sd: dict, total_weight_target: int
    ):
        """D2D gather valid chunks from staging, H2D fallback for dirty chunks.

        Consecutive valid/invalid chunks are merged into runs for efficiency.
        D2D and H2D use separate streams for hardware-level parallelism
        (GPU memory controller vs PCIe).
        Returns (d2d_bytes, h2d_bytes) tuple.
        """
        si = self.scatter_info
        d2d_stream = torch.cuda.Stream()
        h2d_stream = torch.cuda.Stream()

        chunk_valid = si.chunk_valid
        chunk_rows = si.chunk_rows
        num_chunks = len(chunk_valid)
        d2d_bytes = 0
        h2d_bytes = 0

        # Build runs of consecutive valid/invalid chunks
        # e.g., [T,T,F,T,T,T] → [(True,0,2), (False,2,3), (True,3,6)]
        runs = []
        i = 0
        while i < num_chunks:
            j = i + 1
            while j < num_chunks and chunk_valid[j] == chunk_valid[i]:
                j += 1
            runs.append((chunk_valid[i], i, j))
            i = j

        blocks = self._block_tensors
        _ub = si.unsafe_blocks
        for is_valid, c_start, c_end in runs:
            byte_off_start = c_start * chunk_rows * si.row_size
            run_row_start = si.staging_start + c_start * chunk_rows
            run_row_end = min(
                si.staging_start + c_end * chunk_rows,
                si.staging_start + si.staging_rows,
            )
            run_rows = run_row_end - run_row_start
            nbytes = min(run_rows * si.row_size, si.staged_per_block - byte_off_start)
            if nbytes <= 0:
                break

            if is_valid:
                src_off = run_row_start * si.row_size
                with torch.cuda.stream(d2d_stream):
                    for bi in range(si.num_blocks):
                        src = (
                            blocks[_ub + bi]
                            .view(-1)
                            .view(torch.uint8)[src_off : src_off + nbytes]
                        )
                        dst = weights_start + bi * si.staged_per_block + byte_off_start
                        bump.buffer[dst : dst + nbytes].copy_(src, non_blocking=True)
                d2d_bytes += nbytes * si.num_blocks
            else:
                # Dirty-chunk fallback: D2D would have written num_blocks
                # separate weights slices (one per block payload at offset
                # bi * staged_per_block + byte_off_start). H2D must cover all
                # of them — a single _h2d_params(byte_off_start, ...) only
                # refills bi=0's slice and leaves bi>=1 as stale bytes.
                with torch.cuda.stream(h2d_stream):
                    for bi in range(si.num_blocks):
                        block_off = bi * si.staged_per_block
                        self._h2d_params(
                            cpu_sd,
                            bump,
                            weights_start,
                            block_off + byte_off_start,
                            block_off + byte_off_start + nbytes,
                        )
                h2d_bytes += nbytes * si.num_blocks

        # H2D for unstaged params (didn't fit in staging)
        _total_covered = si.total_staged_bytes
        if _total_covered < total_weight_target:
            with torch.cuda.stream(h2d_stream):
                self._h2d_params(
                    cpu_sd, bump, weights_start, _total_covered, total_weight_target
                )
            h2d_bytes += total_weight_target - _total_covered

        d2d_stream.synchronize()
        h2d_stream.synchronize()

        # Staging has been consumed. Drop the whole session state so the
        # next switch re-runs preload cleanly. Only clearing _block_tensors
        # leaves scatter_info non-None, which makes is_valid still report
        # True and tricks _check_preload into skipping the next preload.
        self._reset_session()

        return d2d_bytes, h2d_bytes

    @staticmethod
    def _h2d_params(
        cpu_sd: dict, bump, weights_start: int, skip_bytes: int, total_bytes: int
    ):
        """H2D params from pinned cpu_sd to bump buffer.

        Offsets computed via prefix sum (same order/alignment as bump layout).
        Only params in [skip_bytes, total_bytes) are copied.
        """
        pos = 0
        for name, tensor in cpu_sd.items():
            raw_bytes = tensor.numel() * tensor.element_size()
            aligned = _align_up(raw_bytes)
            if pos + aligned <= skip_bytes:
                pos += aligned
                continue
            if pos >= total_bytes:
                break
            start_in_param = max(0, skip_bytes - pos)
            take = min(raw_bytes - start_in_param, total_bytes - max(pos, skip_bytes))
            cpu_bytes = tensor.view(-1).view(torch.uint8)[
                start_in_param : start_in_param + take
            ]
            write_off = weights_start + max(pos, skip_bytes)
            bump.buffer[write_off : write_off + take].copy_(
                cpu_bytes, non_blocking=True
            )
            pos += aligned

    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------

    def verify_integrity(self):
        """Finalize chunk_valid: chunk is valid only if H2D finished AND not dirty.

        _h2d_scatter sets chunk_valid[i]=True for chunks whose H2D completed.
        This AND-merges the dirty flag: if KV cache allocated into the chunk,
        flip it back to False so gather_to_bump falls back to H2D for it.
        """
        si = self.scatter_info
        if si is None or si.chunk_valid is None:
            return

        num_chunks = len(si.chunk_valid)
        for i in range(num_chunks):
            if self._chunk_dirty[i]:
                si.chunk_valid[i] = False
        n_corrupted = sum(1 for v in si.chunk_valid if not v)
        chunk_rows = si.chunk_rows
        chunk_valid = si.chunk_valid

        if n_corrupted == 0:
            logger.debug(
                f"Staging integrity: all {num_chunks} chunks OK "
                f"(rows [{si.staging_start}, {si.pool_size}))"
            )
        else:
            corrupted_rows = sum(
                min(chunk_rows, si.staging_rows - ci * chunk_rows)
                for ci, v in enumerate(chunk_valid)
                if not v
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
        return (
            self.scatter_info is not None and self.scatter_info.total_staged_bytes > 0
        )

    @property
    def block_tensors(self) -> List[torch.Tensor]:
        return self._block_tensors

    def _reset_session(self):
        """Drop all mutable state from the previous staging session."""
        self.scatter_info = None
        self._block_tensors = []
        self._chunk_dirty = None
        self._staging_start = 0
        self._staging_end = 0
        self._chunk_rows = 0

    def cancel(self):
        self.h2d_stream.synchronize()
        self._reset_session()
