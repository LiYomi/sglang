"""
KV cache offload manager: continuous D2H backup of finished request KV data.

CPU backup is used when switching models destroys GPU KV cache.
On switch-back, entries are staged to bump buffer and D2D restored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class KVEntry:
    token_ids: torch.Tensor       # [seq_len] CPU, int64
    slot_indices: torch.Tensor    # [seq_len] CPU, int64 (original GPU slot indices)
    kv_data: list                 # [layer_num] of (k_cpu, v_cpu) tuples
    num_tokens: int
    cell_bytes: int               # bytes per slot across all layers (for staging budget)

    @property
    def total_bytes(self) -> int:
        return self.num_tokens * self.cell_bytes


class KVOffloadManager:
    """Manages continuous D2H offload of finished KV cache entries per model."""

    def __init__(self):
        self._store: Dict[str, List[KVEntry]] = {}
        self._d2h_stream = torch.cuda.Stream()

    def offload(
        self,
        model_name: str,
        token_ids: list,
        slot_indices: torch.Tensor,
        kv_pool,
        num_layers: int,
    ):
        """D2H offload slots to CPU. Runs on d2h_stream (non-blocking).

        Args:
            model_name: model this KV belongs to
            token_ids: list of int token ids for this sequence
            slot_indices: [seq_len] GPU tensor, KV pool slot indices
            kv_pool: MHATokenToKVPool with k_buffer/v_buffer
            num_layers: number of KV layers
        """
        seq_len = len(slot_indices)
        if seq_len == 0:
            return

        token_ids_cpu = torch.tensor(token_ids, dtype=torch.int64)
        slot_indices_cpu = slot_indices.cpu()

        # Compute cell_bytes: bytes per token across all layers
        # Each layer has k + v, each of shape [num_heads, head_dim] per token
        cell_bytes = 0
        for layer_id in range(num_layers):
            k_buf = kv_pool.k_buffer[layer_id]
            v_buf = kv_pool.v_buffer[layer_id]
            # k_buf shape: [pool_size, num_heads, head_dim]
            cell_bytes += k_buf[0].numel() * k_buf[0].element_size()
            cell_bytes += v_buf[0].numel() * v_buf[0].element_size()

        kv_data = []
        with torch.cuda.stream(self._d2h_stream):
            for layer_id in range(num_layers):
                k_cpu = kv_pool.k_buffer[layer_id][slot_indices].to(
                    "cpu", non_blocking=True
                )
                v_cpu = kv_pool.v_buffer[layer_id][slot_indices].to(
                    "cpu", non_blocking=True
                )
                kv_data.append((k_cpu, v_cpu))

        entry = KVEntry(
            token_ids=token_ids_cpu,
            slot_indices=slot_indices_cpu,
            kv_data=kv_data,
            num_tokens=seq_len,
            cell_bytes=cell_bytes,
        )

        if model_name not in self._store:
            self._store[model_name] = []
        self._store[model_name].append(entry)

        logger.debug(
            f"KV offload: {model_name}, {seq_len} tokens, "
            f"{entry.total_bytes / 1024:.1f}KB, "
            f"total entries={len(self._store[model_name])}"
        )

    def wait_offload(self):
        """Block until all pending D2H transfers complete."""
        self._d2h_stream.synchronize()

    def get_entries(self, model_name: str) -> List[KVEntry]:
        """Return all offloaded entries for a model."""
        return self._store.get(model_name, [])

    def get_hit_entries(
        self,
        model_name: str,
        pending_token_ids_list: Optional[List[list]] = None,
    ) -> List[KVEntry]:
        """Return entries whose token_ids prefix-match pending requests.

        If pending_token_ids_list is None, return all entries for the model.
        Otherwise, return entries where entry.token_ids is a prefix of any
        pending request's token_ids.
        """
        entries = self._store.get(model_name, [])
        if not entries:
            return []
        if pending_token_ids_list is None:
            return list(entries)

        hits = []
        for entry in entries:
            entry_ids = entry.token_ids.tolist()
            entry_len = len(entry_ids)
            for pending_ids in pending_token_ids_list:
                if len(pending_ids) >= entry_len and pending_ids[:entry_len] == entry_ids:
                    hits.append(entry)
                    break
        return hits

    def get_total_bytes(self, entries: List[KVEntry]) -> int:
        return sum(e.total_bytes for e in entries)

    def has_data(self, model_name: str) -> bool:
        entries = self._store.get(model_name, [])
        return len(entries) > 0

    def clear_model(self, model_name: str):
        if model_name in self._store:
            count = len(self._store[model_name])
            del self._store[model_name]
            logger.debug(f"KV offload: cleared {count} entries for {model_name}")

    def stats(self, model_name: str) -> dict:
        entries = self._store.get(model_name, [])
        total_tokens = sum(e.num_tokens for e in entries)
        total_bytes = sum(e.total_bytes for e in entries)
        return {
            "entries": len(entries),
            "total_tokens": total_tokens,
            "total_bytes": total_bytes,
        }
