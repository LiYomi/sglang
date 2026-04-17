"""
KV cache offload manager: continuous D2H backup of finished request KV data.

Placeholder — writes only. model_switch does not consume `_store` yet; the real
restore path depends on the radix-tree redesign (see REVIEW_NOTES §9 / plan).

Threading: all public methods are called from the single scheduler thread. The
D2H copies themselves run on `_d2h_stream`, a CUDA side stream. Events record
the last D2H per request so finalize/cancel can wait per-request instead of
blocking the shared stream for everyone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class KVEntry:
    token_ids: torch.Tensor  # [seq_len] CPU, int64
    kv_data: list  # [layer_num] of (k_cpu, v_cpu) tuples
    num_tokens: int
    cell_bytes: int  # bytes per token across all layers

    @property
    def total_bytes(self) -> int:
        return self.num_tokens * self.cell_bytes


@dataclass
class _IncState:
    """Per-request incremental offload state."""

    last_pos: int = 0
    kv: Dict[int, list] = field(default_factory=dict)  # {layer_id: [(k_cpu, v_cpu), ...]}
    event: Optional[torch.cuda.Event] = None  # records the last D2H for this req


class KVTransfer:
    """Manages continuous D2H offload of finished KV cache entries per model."""

    def __init__(self):
        self._store: Dict[str, Dict[tuple, KVEntry]] = {}
        self._d2h_stream = torch.cuda.Stream()
        self._cell_bytes_cache: Dict[str, int] = {}
        self._incremental: Dict[int, _IncState] = {}

    def _get_cell_bytes(self, model_name: str, kv_pool, num_layers: int) -> int:
        """Bytes per token across all KV layers (cached per model)."""
        if model_name not in self._cell_bytes_cache:
            cell_bytes = 0
            for lid in range(num_layers):
                cell_bytes += (
                    kv_pool.k_buffer[lid][0].numel()
                    * kv_pool.k_buffer[lid][0].element_size()
                )
                cell_bytes += (
                    kv_pool.v_buffer[lid][0].numel()
                    * kv_pool.v_buffer[lid][0].element_size()
                )
            self._cell_bytes_cache[model_name] = cell_bytes
        return self._cell_bytes_cache[model_name]

    def offload(
        self,
        model_name: str,
        token_ids: list,
        slot_indices: torch.Tensor,
        kv_pool,
        num_layers: int,
    ):
        """D2H offload slots to CPU. Blocks on d2h_stream until copies complete."""
        seq_len = len(slot_indices)
        if seq_len == 0:
            return

        token_ids_cpu = torch.tensor(token_ids, dtype=torch.int64)
        cell_bytes = self._get_cell_bytes(model_name, kv_pool, num_layers)

        kv_data = []
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for layer_id in range(num_layers):
                k_cpu = kv_pool.k_buffer[layer_id][slot_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = kv_pool.v_buffer[layer_id][slot_indices].to(
                    "cpu", non_blocking=True
                )
                kv_data.append((k_cpu, v_cpu))
        self._d2h_stream.synchronize()

        entry = KVEntry(
            token_ids=token_ids_cpu,
            kv_data=kv_data,
            num_tokens=seq_len,
            cell_bytes=cell_bytes,
        )

        if model_name not in self._store:
            self._store[model_name] = {}
        # NOTE: two concurrent reqs with the same prompt prefix overwrite each
        # other here. Acceptable for the current placeholder; revisit when the
        # real restore path lands (key needs a req-scoped component).
        key = tuple(token_ids_cpu.tolist())
        self._store[model_name][key] = entry

        logger.debug(
            f"KV offload: {model_name}, {seq_len} tokens, "
            f"{entry.total_bytes / 1024:.1f}KB, "
            f"total entries={len(self._store[model_name])}"
        )

    def offload_incremental(
        self,
        req_pool_idx,
        kv_committed_len,
        model_name,
        kv_pool,
        num_layers,
        req_to_token_pool,
    ):
        state = self._incremental.get(req_pool_idx)
        if state is None:
            state = _IncState()
            self._incremental[req_pool_idx] = state
        if kv_committed_len <= state.last_pos:
            return
        new_indices = req_to_token_pool.req_to_token[
            req_pool_idx, state.last_pos : kv_committed_len
        ]
        if new_indices.numel() == 0:
            return
        self._get_cell_bytes(model_name, kv_pool, num_layers)
        if not state.kv:
            state.kv = {lid: [] for lid in range(num_layers)}
        self._d2h_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._d2h_stream):
            for lid in range(num_layers):
                k_cpu = kv_pool.k_buffer[lid][new_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = kv_pool.v_buffer[lid][new_indices].to(
                    "cpu", non_blocking=True
                )
                state.kv[lid].append((k_cpu, v_cpu))
        # Record after each incremental chunk so finalize/cancel can wait on
        # just this req's pending D2H instead of the whole stream.
        event = torch.cuda.Event()
        event.record(self._d2h_stream)
        state.event = event
        state.last_pos = kv_committed_len

    def finalize_incremental(self, req_pool_idx, model_name, token_ids, num_layers):
        state = self._incremental.pop(req_pool_idx, None)
        if state is None or state.last_pos == 0:
            return
        if state.event is not None:
            state.event.synchronize()
        kv_data = []
        for lid in range(num_layers):
            chunks = state.kv.get(lid, [])
            if chunks:
                kv_data.append(
                    (
                        torch.cat([c[0] for c in chunks], dim=0),
                        torch.cat([c[1] for c in chunks], dim=0),
                    )
                )
            else:
                kv_data.append((torch.empty(0), torch.empty(0)))
        # offload_incremental populated this cache already; direct lookup so a
        # missing entry raises KeyError instead of silently producing 0.
        cell_bytes = self._cell_bytes_cache[model_name]
        entry = KVEntry(
            token_ids=torch.tensor(token_ids[: state.last_pos], dtype=torch.int64),
            kv_data=kv_data,
            num_tokens=state.last_pos,
            cell_bytes=cell_bytes,
        )
        if model_name not in self._store:
            self._store[model_name] = {}
        self._store[model_name][tuple(token_ids[: state.last_pos])] = entry
        logger.debug(f"KV offload finalized: {model_name}, {state.last_pos} tokens")

    def cancel_incremental(self, req_pool_idx):
        """Drop pending incremental offload for a req.

        Must wait for in-flight D2H to complete before returning — otherwise
        the backing slots can be recycled by the allocator and overwritten
        while the background stream is still reading them (see REVIEW_NOTES §9
        potential BLOCKER #1/#2).
        """
        state = self._incremental.pop(req_pool_idx, None)
        if state is None:
            return
        if state.event is not None:
            state.event.synchronize()

    def has_incremental(self, req_pool_idx) -> bool:
        return req_pool_idx in self._incremental

    def wait_offload(self):
        """Block until all pending D2H transfers complete."""
        self._d2h_stream.synchronize()

    def get_entries(self, model_name: str) -> List[KVEntry]:
        return list(self._store.get(model_name, {}).values())

    def get_hit_entries(
        self,
        model_name: str,
        pending_token_ids_list: Optional[List[list]] = None,
    ) -> List[KVEntry]:
        """Return entries whose token_ids prefix-match pending requests."""
        entries = self.get_entries(model_name)
        if not entries or pending_token_ids_list is None:
            return entries

        hits = []
        for entry in entries:
            entry_ids = entry.token_ids.tolist()
            entry_len = len(entry_ids)
            for pending_ids in pending_token_ids_list:
                if (
                    len(pending_ids) >= entry_len
                    and pending_ids[:entry_len] == entry_ids
                ):
                    hits.append(entry)
                    break
        return hits

    def get_total_bytes(self, entries: List[KVEntry]) -> int:
        return sum(e.total_bytes for e in entries)

    def has_data(self, model_name: str) -> bool:
        return bool(self._store.get(model_name, {}))

    def clear_model(self, model_name: str):
        if model_name in self._store:
            count = len(self._store[model_name])
            del self._store[model_name]
            logger.debug(f"KV offload: cleared {count} entries for {model_name}")

    def stats(self, model_name: str) -> dict:
        entries = self.get_entries(model_name)
        return {
            "entries": len(entries),
            "total_tokens": sum(e.num_tokens for e in entries),
            "total_bytes": sum(e.total_bytes for e in entries),
        }
