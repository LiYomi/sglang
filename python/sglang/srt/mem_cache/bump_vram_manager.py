"""
Bump allocator for self-managed GPU VRAM.

Inspired by Aegaeon (SOSP'25): one large cudaMalloc at startup,
all weights and KV cache are views into this buffer.
Release = pointer move, O(1), zero GC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import prod
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# CUDA alignment requirement (256 bytes covers all common GPU architectures)
_ALIGNMENT = 256


def _align_up(n: int, alignment: int = _ALIGNMENT) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


@dataclass
class BumpRegion:
    """A named slice of the managed buffer."""
    tag: str
    start: int          # byte offset in the buffer
    capacity: int       # allocated capacity in bytes (aligned)
    state: str = "active"  # "active" or "free"
    _sub_offset: int = 0   # internal bump pointer for sub-allocations

    def reset(self, new_capacity: Optional[int] = None):
        """Reset region for reuse (e.g., after model switch)."""
        if new_capacity is not None:
            self.capacity = _align_up(new_capacity)
        self._sub_offset = 0
        self.state = "active"

    @property
    def used_bytes(self) -> int:
        return self._sub_offset

    @property
    def free_bytes(self) -> int:
        return self.capacity - self._sub_offset


class BumpVRAMManager:
    """Self-managed GPU VRAM via bump allocation.

    Usage:
        mgr = BumpVRAMManager(total_bytes=20 * 1024**3, device='cuda')
        w_region = mgr.allocate_region('weights', 2 * 1024**3)
        # Create tensors inside the region
        tensor = mgr.create_tensor('weights', (4096, 4096), torch.bfloat16)
        # Release region (O(1), no GC)
        mgr.release_region('weights')
        # Reuse for a different model
        mgr.reset_region('weights', new_capacity=1 * 1024**3)
        tensor2 = mgr.create_tensor('weights', (2048, 2048), torch.bfloat16)
    """

    def __init__(self, total_bytes: int, device: str = "cuda"):
        self.total_bytes = _align_up(total_bytes)
        self.device = device
        # One big allocation — the only cudaMalloc we ever do
        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=device)
        self.offset = 0  # global bump pointer
        self.regions: Dict[str, BumpRegion] = {}
        logger.info(
            f"BumpVRAMManager: allocated {self.total_bytes / 1024**3:.2f} GB "
            f"managed buffer on {device}"
        )

    def allocate_region(self, tag: str, size_bytes: int) -> BumpRegion:
        """Allocate a new region from the buffer."""
        if tag in self.regions:
            raise ValueError(f"Region '{tag}' already exists. Use reset_region().")
        aligned_size = _align_up(size_bytes)
        if self.offset + aligned_size > self.total_bytes:
            raise RuntimeError(
                f"BumpVRAMManager OOM: need {aligned_size} bytes for '{tag}', "
                f"but only {self.total_bytes - self.offset} available "
                f"(total={self.total_bytes}, used={self.offset})"
            )
        region = BumpRegion(tag=tag, start=self.offset, capacity=aligned_size)
        self.offset += aligned_size
        self.regions[tag] = region
        logger.info(
            f"  Region '{tag}': {aligned_size / 1024**2:.1f} MB "
            f"@ offset {region.start} (total used: {self.offset / 1024**3:.2f} GB)"
        )
        return region

    def release_region(self, tag: str):
        """Mark region as free. O(1), no CUDA API calls."""
        if tag not in self.regions:
            logger.warning(f"release_region: '{tag}' not found, skipping")
            return
        self.regions[tag].state = "free"
        self.regions[tag]._sub_offset = 0
        # Compact: if topmost region(s) are free, reclaim space
        self._compact_top()
        logger.info(f"  Region '{tag}' released (state=free)")

    def reset_region(self, tag: str, new_capacity: Optional[int] = None):
        """Reset a region for reuse. If capacity changes, must be topmost or re-allocate."""
        if tag not in self.regions:
            # First time — allocate
            return self.allocate_region(tag, new_capacity or 0)

        region = self.regions[tag]
        if new_capacity is not None:
            new_aligned = _align_up(new_capacity)
            if new_aligned != region.capacity:
                # Need to resize — only possible if this is topmost
                if region.start + region.capacity == self.offset:
                    # Topmost: adjust offset
                    self.offset = region.start + new_aligned
                    region.capacity = new_aligned
                else:
                    # Not topmost: release and re-allocate from top
                    region.state = "free"
                    self._compact_top()
                    del self.regions[tag]
                    return self.allocate_region(tag, new_capacity)

        region.reset(new_capacity)
        logger.info(
            f"  Region '{tag}' reset: capacity={region.capacity / 1024**2:.1f} MB"
        )
        return region

    def reset_all(self):
        """Reset everything. O(1) full cleanup."""
        self.offset = 0
        self.regions.clear()
        logger.info("BumpVRAMManager: all regions reset")

    def create_tensor(
        self, tag: str, shape: Tuple[int, ...], dtype: torch.dtype
    ) -> torch.Tensor:
        """Sub-allocate a tensor within a region."""
        region = self.regions[tag]
        assert region.state == "active", f"Region '{tag}' is {region.state}, not active"

        # Compute size
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = prod(shape)
        nbytes = _align_up(numel * elem_size)

        if region._sub_offset + nbytes > region.capacity:
            raise RuntimeError(
                f"Region '{tag}' sub-alloc OOM: need {nbytes} bytes, "
                f"but only {region.free_bytes} free "
                f"(capacity={region.capacity}, used={region._sub_offset})"
            )

        # Slice the buffer and view as target dtype
        buf_start = region.start + region._sub_offset
        buf_end = buf_start + numel * elem_size  # exact size, not aligned
        raw = self.buffer[buf_start:buf_end]
        tensor = raw.view(dtype).reshape(shape)
        region._sub_offset += nbytes

        return tensor

    def get_available_bytes(self) -> int:
        """Total unallocated space in the buffer."""
        return self.total_bytes - self.offset

    def get_region_available_bytes(self, tag: str) -> int:
        """Unallocated space within a specific region."""
        return self.regions[tag].free_bytes

    def _compact_top(self):
        """Reclaim space from topmost free regions."""
        # Sort regions by start offset, check from top
        if not self.regions:
            self.offset = 0
            return
        sorted_tags = sorted(self.regions.keys(), key=lambda t: self.regions[t].start, reverse=True)
        for tag in sorted_tags:
            r = self.regions[tag]
            if r.state == "free" and r.start + r.capacity == self.offset:
                self.offset = r.start
                del self.regions[tag]
            else:
                break

    def __repr__(self):
        region_strs = []
        for tag, r in sorted(self.regions.items(), key=lambda x: x[1].start):
            region_strs.append(
                f"  {tag}: {r.capacity/1024**2:.1f}MB, state={r.state}, "
                f"used={r._sub_offset/1024**2:.1f}MB"
            )
        return (
            f"BumpVRAMManager(total={self.total_bytes/1024**3:.2f}GB, "
            f"used={self.offset/1024**3:.2f}GB)\n"
            + "\n".join(region_strs)
        )
