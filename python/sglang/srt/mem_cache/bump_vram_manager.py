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


def _align_up(n: int, alignment: int = _ALIGNMENT) -> int:
    return (n + alignment - 1) & ~(alignment - 1)


@dataclass
class BumpRegion:
    tag: str
    start: int
    capacity: int
    state: str = "active"
    _sub_offset: int = 0

    def reset(self, new_capacity=None):
        if new_capacity is not None:
            self.capacity = _align_up(new_capacity)
        self._sub_offset = 0
        self.state = "active"

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


class PageBumpManager:
    def __init__(self, total_bytes: int, device: str = "cuda"):
        self.total_bytes = _align_up(total_bytes)
        self.device = device
        self.buffer = torch.empty(self.total_bytes, dtype=torch.uint8, device=device)
        self.left_offset = 0
        self.right_offset = self.total_bytes
        self.offset = 0
        self.regions: Dict[str, BumpRegion] = {}
        self.layer_map: Dict[str, List[LayerSlice]] = {}
        self._current_model: Optional[str] = None
        logger.info(
            f"PageBumpManager: allocated {self.total_bytes / 1024**3:.2f} GB "
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
                    f"PageBumpManager OOM: need {aligned_size} for '{tag}', "
                    f"avail {self.right_offset - self.left_offset}")
            region = BumpRegion(tag=tag, start=new_right, capacity=aligned_size)
            self.right_offset = new_right
        else:
            if self.left_offset + aligned_size > self.right_offset:
                raise RuntimeError(
                    f"PageBumpManager OOM: need {aligned_size} for '{tag}', "
                    f"avail {self.right_offset - self.left_offset}")
            region = BumpRegion(tag=tag, start=self.left_offset, capacity=aligned_size)
            self.left_offset += aligned_size
            self.offset = self.left_offset
        self.regions[tag] = region
        logger.info(
            f"  Region '{tag}': {aligned_size / 1024**2:.1f} MB "
            f"@ offset {region.start} "
            f"(left={self.left_offset / 1024**3:.2f}GB, "
            f"right={self.right_offset / 1024**3:.2f}GB, "
            f"kv_avail={self.get_available_bytes() / 1024**3:.2f}GB)")
        return region

    def release_region(self, tag: str):
        if tag not in self.regions:
            logger.warning(f"release_region: '{tag}' not found, skipping")
            return
        region = self.regions[tag]
        region.state = "free"
        region._sub_offset = 0
        if tag == "runtime":
            self.right_offset = region.start + region.capacity
            del self.regions[tag]
        else:
            self._compact_top()
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
                if region.start + region.capacity == self.left_offset:
                    self.left_offset = region.start + new_aligned
                    self.offset = self.left_offset
                    region.capacity = new_aligned
                else:
                    region.state = "free"
                    self._compact_top()
                    del self.regions[tag]
                    return self.allocate_region(tag, new_capacity)
        region.reset(new_capacity)
        logger.info(f"  Region '{tag}' reset: capacity={region.capacity / 1024**2:.1f} MB")
        return region

    def reset_all(self):
        self.left_offset = 0
        self.right_offset = self.total_bytes
        self.offset = 0
        self.regions.clear()
        logger.info("PageBumpManager: all regions reset")

    def create_tensor(self, tag: str, shape: Tuple[int, ...], dtype: torch.dtype,
                      name: Optional[str] = None) -> torch.Tensor:
        region = self.regions[tag]
        assert region.state == "active", f"Region '{tag}' is {region.state}, not active"
        elem_size = torch.tensor([], dtype=dtype).element_size()
        numel = prod(shape)
        nbytes = _align_up(numel * elem_size)
        if region._sub_offset + nbytes > region.capacity:
            raise RuntimeError(
                f"Region '{tag}' sub-alloc OOM: need {nbytes}, "
                f"free {region.free_bytes} (cap={region.capacity}, used={region._sub_offset})")
        buf_start = region.start + region._sub_offset
        buf_end = buf_start + numel * elem_size
        raw = self.buffer[buf_start:buf_end]
        tensor = raw.view(dtype).reshape(shape)
        region._sub_offset += nbytes
        if tag == "weights" and name and self._current_model:
            self.layer_map.setdefault(self._current_model, []).append(
                LayerSlice(name=name, offset=buf_start, nbytes=numel * elem_size))
        return tensor

    def get_available_bytes(self) -> int:
        return self.right_offset - self.left_offset

    def get_region_available_bytes(self, tag: str) -> int:
        return self.regions[tag].free_bytes

    def _compact_top(self):
        if not self.regions:
            self.left_offset = 0
            self.offset = 0
            return
        left_regions = [(t, r) for t, r in self.regions.items() if t != "runtime"]
        if not left_regions:
            self.left_offset = 0
            self.offset = 0
            return
        left_regions.sort(key=lambda x: x[1].start, reverse=True)
        for tag, r in left_regions:
            if r.state == "free" and r.start + r.capacity == self.left_offset:
                self.left_offset = r.start
                self.offset = self.left_offset
                del self.regions[tag]
            else:
                break

    def get_layer_map(self, model_name: str) -> List[LayerSlice]:
        return self.layer_map.get(model_name, [])

    def clear_layer_map(self, model_name: str):
        self.layer_map.pop(model_name, None)

    def __repr__(self):
        region_strs = []
        for tag, r in sorted(self.regions.items(), key=lambda x: x[1].start):
            region_strs.append(
                f"  {tag}: {r.capacity/1024**2:.1f}MB @ {r.start}, "
                f"state={r.state}, used={r._sub_offset/1024**2:.1f}MB")
        return (
            f"PageBumpManager(total={self.total_bytes/1024**3:.2f}GB, "
            f"left={self.left_offset/1024**3:.2f}GB, "
            f"right={self.right_offset/1024**3:.2f}GB, "
            f"kv_avail={self.get_available_bytes()/1024**3:.2f}GB)\n"
            + "\n".join(region_strs))


