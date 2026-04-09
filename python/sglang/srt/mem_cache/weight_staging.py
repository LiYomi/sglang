"""
Scatter staging: background H2D of next model's weights into KV layer block free tails.

Instead of writing contiguous bytes at the KV region tail (which corrupts active KV),
staging distributes weight bytes across the free high-index rows of each KV layer block.
LIFO slot allocation keeps active slots at low indices, ensuring staging doesn't overlap.

Layout per block:
  [active rows 0..min_free-1 | gap | staging rows S..pool_size-1]

H2D proceeds top-down (highest rows first) across ALL blocks simultaneously.
Each round locks only the current chunk of rows, syncs, then unlocks.
Previously staged rows may be overwritten by KV — verify_integrity detects this.

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
    # Per-chunk validity after verify_integrity. True = all rows free → D2D safe.
    # Length = ceil(staging_rows / chunk_rows). None before verify.
    chunk_valid: Optional[List[bool]] = None


class PreloadManager:
    def __init__(self):
        self.h2d_stream = torch.cuda.Stream()
        self.model_name: Optional[str] = None
        self.scatter_info: Optional[ScatterStagingInfo] = None
        self._block_tensors: List[torch.Tensor] = []

    def start_preload(
        self,
        model_name: str,
        cpu_state_dict: dict,
        bump: "BumpVramAllocator",
        kv_pool,          # MHATokenToKVPool
        min_free_slot: int,
        allocator=None,   # TokenToKVPoolAllocator — for per-chunk slot locking
        alloc_lock: threading.Lock = None,  # shared lock with allocator
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

        staging_start, staging_rows, staged_per_block, staged_bytes = (
            self._compute_layout(
                pool_size, row_size, num_blocks, min_free_slot, cpu_state_dict,
            )
        )
        if staged_bytes == 0:
            logger.info(f"Preload: no staging capacity")
            return

        self._block_tensors = blocks

        # --- Serialize CPU params → flat byte stream, padded to full block coverage ---
        cpu_stream = self._serialize_params(cpu_state_dict, staged_bytes,
                                            num_blocks, staged_per_block)

        # --- H2D scatter with per-chunk locking ---
        chunk_rows = max(1, H2D_CHUNK_SIZE // row_size)
        total_written = self._h2d_scatter(
            blocks, cpu_stream, num_blocks, pool_size, row_size,
            staging_start, staged_per_block, chunk_rows,
            allocator, alloc_lock,
        )

        self.scatter_info = ScatterStagingInfo(
            staging_start=staging_start,
            staging_rows=staging_rows,
            staged_per_block=staged_per_block,
            num_blocks=num_blocks,
            total_staged_bytes=total_written,
            row_size=row_size,
            pool_size=pool_size,
            chunk_rows=chunk_rows,
        )

        logger.info(
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
    def _compute_layout(pool_size, row_size, num_blocks, min_free_slot, cpu_state_dict):
        """Compute staging layout: how many rows/bytes per block, where to start."""
        free_rows = pool_size - min_free_slot
        capacity_per_block = free_rows * row_size
        total_capacity = num_blocks * capacity_per_block

        total_weight_bytes = sum(
            t.numel() * t.element_size() for t in cpu_state_dict.values()
        )
        staged_bytes = min(total_weight_bytes, total_capacity)

        if staged_bytes == 0:
            return 0, 0, 0, 0

        staged_per_block = math.ceil(staged_bytes / num_blocks)
        # Align to row_size so D2D stride matches H2D write size
        staged_per_block = math.ceil(staged_per_block / row_size) * row_size
        staging_rows = math.ceil(staged_per_block / row_size)
        staging_start = pool_size - staging_rows

        return staging_start, staging_rows, staged_per_block, staged_bytes

    # ------------------------------------------------------------------
    # Param serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_params(cpu_state_dict, staged_bytes, num_blocks, staged_per_block):
        """Serialize CPU params into a flat byte stream, padded to full block coverage."""
        chunks = []
        remaining = staged_bytes
        for name, tensor in cpu_state_dict.items():
            nbytes = tensor.numel() * tensor.element_size()
            take = min(nbytes, remaining)
            chunks.append(tensor.contiguous().view(-1).view(torch.uint8)[:take])
            remaining -= take
            if remaining <= 0:
                break
        stream = torch.cat(chunks)

        # Pad so every block has a full staged_per_block section.
        padded_len = num_blocks * staged_per_block
        if len(stream) < padded_len:
            stream = torch.cat([
                stream,
                torch.zeros(padded_len - len(stream), dtype=torch.uint8),
            ])
        return stream

    # ------------------------------------------------------------------
    # H2D scatter loop
    # ------------------------------------------------------------------

    def _h2d_scatter(
        self, blocks, cpu_stream, num_blocks, pool_size, row_size,
        staging_start, staged_per_block, chunk_rows,
        allocator, alloc_lock,
    ):
        """Top-down H2D with per-chunk locking. Returns total bytes written."""
        chunk_end = pool_size - 1
        written_per_block = 0
        total_written = 0

        while chunk_end >= staging_start and written_per_block < staged_per_block:
            actual_rows = min(chunk_rows, chunk_end - staging_start + 1)
            chunk_start = chunk_end - actual_rows + 1
            chunk_bytes = min(actual_rows * row_size, staged_per_block - written_per_block)

            if chunk_bytes <= 0:
                break

            # --- Lock: remove this chunk's rows from allocator ---
            locked = None
            if allocator is not None:
                _lk = alloc_lock
                if _lk: _lk.acquire()
                try:
                    fp = allocator.free_pages
                    mask = (fp >= chunk_start) & (fp <= chunk_end)
                    locked = fp[mask]
                    if locked.numel() < actual_rows:
                        # KV invaded this chunk — stop
                        logger.info(
                            f"Preload: early exit — only {locked.numel()}/{actual_rows} "
                            f"rows free in [{chunk_start}, {chunk_end}]"
                        )
                        locked = None
                        break
                    allocator.free_pages = fp[~mask]
                finally:
                    if _lk: _lk.release()

            # --- H2D: reversed byte order (last bytes → highest rows) ---
            try:
                with torch.cuda.stream(self.h2d_stream):
                    actual_total = 0
                    for bi in range(num_blocks):
                        # Reverse mapping: high rows get tail bytes, low rows get head bytes
                        src_off = bi * staged_per_block + (staged_per_block - written_per_block - chunk_bytes)
                        if src_off + chunk_bytes > len(cpu_stream):
                            actual = len(cpu_stream) - src_off
                            if actual <= 0:
                                break
                        else:
                            actual = chunk_bytes

                        dst_off = chunk_start * row_size
                        flat = blocks[bi].view(-1).view(torch.uint8)
                        flat[dst_off : dst_off + actual].copy_(
                            cpu_stream[src_off : src_off + actual],
                            non_blocking=True,
                        )
                        actual_total += actual

                self.h2d_stream.synchronize()
            finally:
                # Always unlock, even on error
                if allocator is not None and locked is not None and locked.numel() > 0:
                    _lk = alloc_lock
                    if _lk: _lk.acquire()
                    try:
                        allocator.free_pages = torch.cat([locked, allocator.free_pages])
                    finally:
                        if _lk: _lk.release()

            total_written += actual_total
            written_per_block += chunk_bytes
            chunk_end = chunk_start - 1

        return total_written

    # ------------------------------------------------------------------
    # Integrity check
    # ------------------------------------------------------------------

    def verify_integrity(self, allocator):
        """Check which staging chunks are still valid (all rows free).

        Produces a per-chunk validity list. model_switch D2D iterates chunks:
        valid chunks → D2D from staging, corrupted chunks → H2D from CPU.
        """
        if self.scatter_info is None:
            return

        si = self.scatter_info
        free_pages = allocator.free_pages
        chunk_rows = si.chunk_rows
        num_chunks = math.ceil(si.staging_rows / chunk_rows)

        if len(free_pages) == 0:
            si.chunk_valid = [False] * num_chunks
            logger.info(f"Staging integrity: fully corrupted (no free pages)")
            return

        # Per-row free mask (GPU-accelerated)
        staging_range = torch.arange(
            si.staging_start, si.pool_size, device=free_pages.device
        )
        is_free = torch.isin(staging_range, free_pages)

        # Group into chunks
        chunk_valid = []
        n_corrupted = 0
        for ci in range(num_chunks):
            row_off = ci * chunk_rows
            row_end = min(row_off + chunk_rows, si.staging_rows)
            valid = is_free[row_off:row_end].all().item()
            chunk_valid.append(valid)
            if not valid:
                n_corrupted += 1

        si.chunk_valid = chunk_valid

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
