"""
Page-based bump allocator for self-managed GPU VRAM.

Layout: [weights(left) | kv_cache(middle) | runtime(right)]
Release = pointer move, O(1), zero GC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import prod
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ALIGNMENT = 256

_DTYPE_SIZES = {
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.float32: 4,
    torch.float64: 8,
    torch.int8: 1,
    torch.uint8: 1,
    torch.int32: 4,
    torch.int64: 8,
}


def _align_up(n: int, alignment: int = _ALIGNMENT) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


@dataclass
class BumpRegion:
    tag: str
    start: int
    capacity: int
    _sub_offset: int = 0

    def reset(self, new_capacity=None):
        if new_capacity is not None:
            self.capacity = _align_up(new_capacity)
        self._sub_offset = 0

    @property
    def used_bytes(self):
        return self._sub_offset

    @property
    def free_bytes(self):
        return self.capacity - self._sub_offset


@dataclass
class LayerSlice:
    name: str
    offset: int
    nbytes: int


class BumpVramAllocator:
    def __init__(self, total_bytes: int, device: str = "cuda"):
        self.total_bytes = _align_up(total_bytes)
        self.device = device
        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=device)
        self.left_offset = 0
        self.right_offset = self.total_bytes
        self.regions: Dict[str, BumpRegion] = {}
        self.layer_map: Dict[str, List[LayerSlice]] = {}
        self._current_model: Optional[str] = None
        logger.info(
            f"BumpVramAllocator: allocated {self.total_bytes / 1024**3:.2f} GB "
            f"managed buffer on {device}"
        )

    def allocate_region(self, tag: str, size_bytes: int) -> BumpRegion:
        if tag in self.regions:
            raise ValueError(f"Region '{tag}' already exists. Use reset_region().")
        aligned_size = _align_up(size_bytes)
        if tag == "runtime":
            new_right = self.right_offset - aligned_size
            if new_right < self.left_offset:
                raise RuntimeError(
                    f"BumpVramAllocator OOM: need {aligned_size} for '{tag}', "
                    f"avail {self.right_offset - self.left_offset}"
                )
            region = BumpRegion(tag=tag, start=new_right, capacity=aligned_size)
            self.right_offset = new_right
        else:
            if self.left_offset + aligned_size > self.right_offset:
                raise RuntimeError(
                    f"BumpVramAllocator OOM: need {aligned_size} for '{tag}', "
                    f"avail {self.right_offset - self.left_offset}"
                )
            region = BumpRegion(tag=tag, start=self.left_offset, capacity=aligned_size)
            self.left_offset += aligned_size
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
        if tag not in self.regions:
            logger.warning(f"release_region: '{tag}' not found, skipping")
            return
        region = self.regions[tag]
        if tag == "runtime":
            self.right_offset = region.start + region.capacity
        else:
            self.left_offset = region.start
        del self.regions[tag]
        logger.info(f"  Region '{tag}' released")

    def reset_region(self, tag: str, new_capacity=None):
        if tag not in self.regions:
            return self.allocate_region(tag, new_capacity or 0)
        region = self.regions[tag]
        if tag == "runtime":
            self.right_offset = region.start + region.capacity
            del self.regions[tag]
            return self.allocate_region(tag, new_capacity or 0)
        if new_capacity is not None:
            new_aligned = _align_up(new_capacity)
            if new_aligned != region.capacity:
                # weights is always at stack bottom (offset 0), resize in-place
                assert region.start + region.capacity == self.left_offset, \
                    f"Can only resize topmost left region, but '{tag}' is not at top"
                self.left_offset = region.start + new_aligned
                region.capacity = new_aligned
        region.reset(new_capacity)
        logger.info(f"  Region '{tag}' reset: capacity={region.capacity / 1024**2:.1f} MB")
        return region

    def create_tensor(
        self, tag: str, shape: Tuple[int, ...], dtype: torch.dtype, name: Optional[str] = None
    ) -> torch.Tensor:
        region = self.regions[tag]
        elem_size = _DTYPE_SIZES.get(dtype, torch.tensor([], dtype=dtype).element_size())
        numel = prod(shape)
        nbytes = _align_up(numel * elem_size)
        if region._sub_offset + nbytes > region.capacity:
            raise RuntimeError(
                f"Region '{tag}' sub-alloc OOM: need {nbytes}, "
                f"free {region.free_bytes} (cap={region.capacity}, used={region._sub_offset})"
            )
        buf_start = region.start + region._sub_offset
        buf_end = buf_start + numel * elem_size
        raw = self.buffer[buf_start:buf_end]
        tensor = raw.view(dtype).reshape(shape)
        region._sub_offset += nbytes
        if tag == "weights" and name and self._current_model:
            self.layer_map.setdefault(self._current_model, []).append(
                LayerSlice(name=name, offset=buf_start, nbytes=nbytes)  # aligned size
            )
        return tensor

    def get_available_bytes(self) -> int:
        return self.right_offset - self.left_offset

    def get_layer_map(self, model_name: str) -> List[LayerSlice]:
        return self.layer_map.get(model_name, [])

    def __repr__(self):
        region_strs = []
        for tag, r in sorted(self.regions.items(), key=lambda x: x[1].start):
            region_strs.append(
                f"  {tag}: {r.capacity/1024**2:.1f}MB @ {r.start}, "
                f"used={r._sub_offset/1024**2:.1f}MB"
            )
        return (
            f"BumpVramAllocator(total={self.total_bytes/1024**3:.2f}GB, "
            f"left={self.left_offset/1024**3:.2f}GB, "
            f"right={self.right_offset/1024**3:.2f}GB, "
            f"kv_avail={self.get_available_bytes()/1024**3:.2f}GB)\n" + "\n".join(region_strs)
        )
