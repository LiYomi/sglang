"""
Preload manager: background H2D of next model's weights into staging area.

Staging grows from right to left (right_ptr moves leftward).
Stops when right_ptr hits left boundary (A's KV usage).
Soft occupation: A can reclaim staging pages by allocating KV slots.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

import torch
import torch.cuda

if TYPE_CHECKING:
    from sglang.srt.mem_cache.bump_vram_manager import PageBumpManager

logger = logging.getLogger(__name__)


@dataclass
class StagedLayer:
    name: str
    staging_offset: int
    nbytes: int
    status: str = "staged"  # "staged" | "evicted"


class PreloadManager:
    def __init__(self):
        self.h2d_stream = torch.cuda.Stream()
        self.staged: Dict[str, StagedLayer] = {}
        self._active = False
        self._cancel_event = None
        self.total_staged_bytes = 0
        self.model_name: Optional[str] = None
        self.staging_right_ptr = 0  # current right pointer (moves leftward)

    def start_preload(
        self,
        model_name: str,
        cpu_state_dict: Dict[str, torch.Tensor],
        bump: "PageBumpManager",
        staging_offset: int,  # ignored, we compute from right_ptr
        staging_capacity: int,  # ignored, we compute dynamically
    ):
        """Preload weights from right to left in bump buffer.

        Starts at bump.right_offset (runtime left boundary) and grows leftward.
        Stops when hitting left_offset (weights/KV right boundary).
        """
        if self._active:
            self.cancel()

        self.model_name = model_name
        self._active = True
        self.staged.clear()
        self.total_staged_bytes = 0

        # Right boundary = KV region right edge (staging writes into KV buffer tail)
        kv_region = bump.regions.get("kv_cache")
        if kv_region is None:
            logger.info("Preload: no kv_cache region")
            self._active = False
            return
        right_boundary = kv_region.start + kv_region.capacity
        # Left boundary = KV region start (dont overlap weights)
        left_boundary = kv_region.start

        self.staging_right_ptr = right_boundary

        # Reverse order: last layers first (they're at high addresses in weights)
        # This way layer_N is closest to runtime, layer_0 furthest
        # reversed(state_dict) order: right-to-left write produces weights-matching layout
        sorted_params = list(reversed(list(cpu_state_dict.items())))

        for name, cpu_tensor in sorted_params:
            nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
            # Align to 256 bytes
            aligned_nbytes = (nbytes + 255) & ~255

            new_ptr = self.staging_right_ptr - aligned_nbytes

            if new_ptr < left_boundary:
                logger.info(
                    f"Preload: staging full at {len(self.staged)} params, "
                    f"{self.total_staged_bytes / 1024**2:.1f}MB staged, "
                    f"ptr={new_ptr}, left_bound={left_boundary}"
                )
                break

            # H2D on separate stream
            cpu_bytes = cpu_tensor.contiguous().view(-1).view(torch.uint8)
            with torch.cuda.stream(self.h2d_stream):
                gpu_view = bump.buffer[new_ptr : new_ptr + nbytes]
                gpu_view.copy_(cpu_bytes, non_blocking=True)

            self.staged[name] = StagedLayer(
                name=name,
                staging_offset=new_ptr,
                nbytes=nbytes,
            )
            self.staging_right_ptr = new_ptr
            self.total_staged_bytes += nbytes

        logger.info(
            f"Preload: {model_name}, {len(self.staged)}/{len(cpu_state_dict)} params, "
            f"{self.total_staged_bytes / 1024**2:.1f}MB, "
            f"staging [{self.staging_right_ptr} .. {right_boundary}]"
        )

    def wait_complete(self):
        self.h2d_stream.synchronize()
        self._active = False

    def verify_integrity(self, bump, allocator=None, cell_size=0):
        """O(1) boundary check: find where KV data ends, mark everything below as evicted."""
        kv_region = bump.regions.get("kv_cache")
        if kv_region is None or allocator is None or cell_size == 0:
            logger.info("Staging integrity: skipped (no allocator/cell_size)")
            return
        try:
            if hasattr(allocator, "free_pages") and len(allocator.free_pages) > 0:
                min_free = allocator.free_pages.min().item()
                kv_data_end = kv_region.start + min_free * cell_size
            else:
                kv_data_end = kv_region.start
        except Exception:
            kv_data_end = kv_region.start + kv_region.capacity
        # O(1) check: if staging_right_ptr >= kv_data_end, all safe
        if self.staging_right_ptr >= kv_data_end:
            logger.info(f"Staging integrity: all {self.num_staged} params OK (KV end={kv_data_end})")
            return
        # Some evicted: mark from left until past kv_data_end
        evicted = 0
        for name, entry in self.staged.items():
            if entry.status == "staged" and entry.staging_offset < kv_data_end:
                entry.status = "evicted"
                evicted += 1
        logger.info(f"Staging integrity: {evicted} evicted (KV end={kv_data_end})")

    def get_eviction_boundary(self, weights_start: int) -> int:
        """O(1) eviction boundary in bump buffer coordinates.
        Returns the offset in bump buffer where valid staging begins.
        Everything in weights [weights_start, boundary) needs H2D from CPU.
        Everything in weights [boundary, weights_end) can be D2D from staging.
        """
        if not self.staged:
            return weights_start  # nothing staged, all needs H2D
        # staging_right_ptr is leftmost valid staging offset
        # Map to weights coordinate: staging offset - staging_right_ptr = weights offset
        # eviction_boundary = weights_start + (first_evicted_end_in_staging - staging_right_ptr)
        # But simpler: count evicted bytes from left
        evicted_bytes = sum(
            (e.nbytes + 255) & ~255
            for e in self.staged.values()
            if e.status == "evicted"
        )
        return weights_start + evicted_bytes

    @property
    def staging_total_bytes(self) -> int:
        return self.total_staged_bytes

    def is_staged(self, name: str) -> bool:
        entry = self.staged.get(name)
        return entry is not None and entry.status == "staged"

    def get_staged_layer(self, name: str) -> Optional[StagedLayer]:
        entry = self.staged.get(name)
        if entry and entry.status == "staged":
            return entry
        return None

    def evict(self, names: List[str]):
        for name in names:
            if name in self.staged:
                self.staged[name].status = "evicted"

    def cancel(self):
        self.h2d_stream.synchronize()
        self._active = False
        self.staged.clear()
        self.total_staged_bytes = 0

    @property
    def num_staged(self) -> int:
        return sum(1 for s in self.staged.values() if s.status == "staged")

    @property
    def num_evicted(self) -> int:
        return sum(1 for s in self.staged.values() if s.status == "evicted")


@dataclass
class StagedKVEntry:
    """A KV cache entry staged in bump buffer (between kv_region.start and weight staging)."""
    staging_offset: int           # start offset in bump buffer
    staging_bytes: int            # total bytes in staging for this entry
    num_tokens: int
    slot_indices: torch.Tensor    # original slot indices (CPU, for radix tree)
    token_ids: torch.Tensor       # token ids (CPU, for radix tree insert)
    layer_data_offsets: list      # [(k_offset, k_bytes, v_offset, v_bytes)] per layer
    status: str = "staged"        # "staged" | "evicted"


class KVStagingMixin:
    """Mixin for PreloadManager to add KV staging capability."""

    def _init_kv_staging(self):
        self.kv_staged: Dict[int, StagedKVEntry] = {}  # entry_index -> StagedKVEntry
        self.kv_staging_left_ptr: int = 0   # left boundary of KV staging region
        self.kv_staging_right_ptr: int = 0  # right boundary (= weight staging left edge)
        self.kv_staging_bytes: int = 0

    def start_kv_preload(
        self,
        model_name: str,
        hit_entries: list,  # List[KVEntry]
        bump: "PageBumpManager",
    ):
        """H2D KV data to staging area LEFT of weight staging.

        Layout in bump buffer:
          [weights_region | ... kv_staging | weight_staging | runtime_region]
                                ^                ^
                          kv_staging_left    staging_right_ptr (from weight preload)
        """
        if not hasattr(self, 'kv_staged'):
            self._init_kv_staging()

        self.kv_staged.clear()
        self.kv_staging_bytes = 0

        if not hit_entries:
            return

        # KV staging grows LEFT from weight staging's left edge
        right_boundary = self.staging_right_ptr  # left edge of weight staging
        kv_region = bump.regions.get("kv_cache")
        if kv_region is None:
            logger.info("KV staging: no kv_cache region")
            return

        left_boundary = kv_region.start  # can't go past weights region

        ptr = right_boundary  # write cursor, grows leftward

        for idx, entry in enumerate(hit_entries):
            # Calculate bytes for this entry: all layers' k+v data
            entry_bytes = 0
            layer_offsets = []

            for layer_id, (k_cpu, v_cpu) in enumerate(entry.kv_data):
                k_bytes = k_cpu.numel() * k_cpu.element_size()
                v_bytes = v_cpu.numel() * v_cpu.element_size()
                entry_bytes += k_bytes + v_bytes

            # Align to 256
            aligned_bytes = (entry_bytes + 255) & ~255

            new_ptr = ptr - aligned_bytes
            if new_ptr < left_boundary:
                logger.info(
                    f"KV staging: full at {idx}/{len(hit_entries)} entries, "
                    f"{self.kv_staging_bytes / 1024:.1f}KB staged"
                )
                break

            # H2D each layer's k,v into contiguous staging area
            write_pos = new_ptr
            layer_data_offsets = []

            with torch.cuda.stream(self.h2d_stream):
                for layer_id, (k_cpu, v_cpu) in enumerate(entry.kv_data):
                    k_bytes = k_cpu.numel() * k_cpu.element_size()
                    v_bytes = v_cpu.numel() * v_cpu.element_size()

                    # H2D k
                    k_flat = k_cpu.contiguous().view(-1).view(torch.uint8)
                    bump.buffer[write_pos : write_pos + k_bytes].copy_(
                        k_flat, non_blocking=True
                    )
                    k_offset = write_pos
                    write_pos += k_bytes

                    # H2D v
                    v_flat = v_cpu.contiguous().view(-1).view(torch.uint8)
                    bump.buffer[write_pos : write_pos + v_bytes].copy_(
                        v_flat, non_blocking=True
                    )
                    v_offset = write_pos
                    write_pos += v_bytes

                    layer_data_offsets.append((k_offset, k_bytes, v_offset, v_bytes))

            self.kv_staged[idx] = StagedKVEntry(
                staging_offset=new_ptr,
                staging_bytes=aligned_bytes,
                num_tokens=entry.num_tokens,
                slot_indices=entry.slot_indices,
                token_ids=entry.token_ids,
                layer_data_offsets=layer_data_offsets,
            )
            ptr = new_ptr
            self.kv_staging_bytes += aligned_bytes

        self.kv_staging_left_ptr = ptr
        self.kv_staging_right_ptr = right_boundary

        logger.info(
            f"KV staging: {model_name}, {len(self.kv_staged)} entries, "
            f"{self.kv_staging_bytes / 1024:.1f}KB, "
            f"region [{self.kv_staging_left_ptr} .. {self.kv_staging_right_ptr}]"
        )


# Monkey-patch PreloadManager to include KV staging methods
PreloadManager._init_kv_staging = KVStagingMixin._init_kv_staging
PreloadManager.start_kv_preload = KVStagingMixin.start_kv_preload
