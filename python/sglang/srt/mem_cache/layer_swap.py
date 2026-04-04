"""
Layer swap: overwrite model weights layer-by-layer during forward pass.

Uses PyTorch forward hooks on decoder layers. During A's last forward,
each layer's hook records a CUDA event after compute, then triggers
H2D overwrite of that layer with B's weights on a separate stream.

Usage:
    ctx = LayerSwapContext(cpu_state_dict, bump, src_model, dst_model)
    ctx.install_hooks(model)  # attach forward hooks to decoder layers
    # ... run A's last forward (must NOT use CUDA graph) ...
    ctx.finish()              # wait for all H2D, remove hooks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import torch.cuda


if TYPE_CHECKING:
    from sglang.srt.mem_cache.bump_vram_manager import PageBumpManager

logger = logging.getLogger(__name__)


@dataclass
class LayerSwapContext:
    """Manages layer-by-layer weight swap during a single forward pass."""

    # B's weights on CPU (from register_model preload)
    cpu_state_dict: Dict[str, torch.Tensor]

    # Bump manager with layer_map for current (A) model
    bump: "PageBumpManager"

    # Model names
    src_model: str  # A (being replaced)
    dst_model: str  # B (being loaded)

    # H2D stream (separate from compute)
    h2d_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)

    # Internal state
    _hooks: List[Any] = field(default_factory=list)
    _layer_prefix: str = ""  # e.g. "model.layers"
    layers_swapped: int = 0
    total_bytes_swapped: int = 0
    active: bool = True

    def install_hooks(self, model: torch.nn.Module):
        """Find decoder layers and attach forward hooks."""
        # Find the layers ModuleList (common patterns)
        layers_module = None
        prefix = ""
        for attr in ["model.layers", "model.decoder.layers", "transformer.h", "decoder.layers", "layers"]:
            parts = attr.split(".")
            obj = model
            try:
                for p in parts:
                    obj = getattr(obj, p)
                if isinstance(obj, torch.nn.ModuleList):
                    layers_module = obj
                    prefix = attr
                    break
            except AttributeError:
                continue

        if layers_module is None:
            logger.warning("LayerSwap: could not find decoder layers, skipping")
            self.active = False
            return

        self._layer_prefix = prefix
        num_layers = len(layers_module)

        for i, layer in enumerate(layers_module):
            hook = layer.register_forward_hook(
                self._make_hook(i, prefix)
            )
            self._hooks.append(hook)

        logger.info(
            f"LayerSwap: installed hooks on {num_layers} layers "
            f"(prefix={prefix})"
        )

    def _make_hook(self, layer_idx: int, prefix: str):
        """Create a forward hook for layer_idx."""
        def hook_fn(module, input, output):
            if not self.active:
                return
            self._swap_layer(layer_idx, prefix)
        return hook_fn

    def _swap_layer(self, layer_idx: int, prefix: str):
        """Overwrite layer_idx weights with B's weights via H2D stream."""
        # Record event on compute stream: layer_idx compute is done
        event = torch.cuda.Event(enable_timing=False)
        torch.cuda.current_stream().record_event(event)

        # Find A's layer slices in bump
        src_slices = self.bump.get_layer_map(self.src_model)
        if not src_slices:
            return

        layer_prefix = f"{prefix}.{layer_idx}."
        slices = [s for s in src_slices if s.name.startswith(layer_prefix)]
        if not slices:
            return

        # Wait for compute to finish on H2D stream
        self.h2d_stream.wait_event(event)

        bytes_this_layer = 0
        with torch.cuda.stream(self.h2d_stream):
            for s in slices:
                if s.name not in self.cpu_state_dict:
                    continue
                cpu_w = self.cpu_state_dict[s.name]
                cpu_bytes = cpu_w.contiguous().view(-1).view(torch.uint8)
                # Overwrite at same bump offset (same address for CUDA graph compat)
                gpu_view = self.bump.buffer[s.offset : s.offset + s.nbytes]
                actual = min(len(cpu_bytes), len(gpu_view))
                gpu_view[:actual].copy_(cpu_bytes[:actual], non_blocking=True)
                bytes_this_layer += actual

        self.layers_swapped += 1
        self.total_bytes_swapped += bytes_this_layer

    def finish(self):
        """Wait for all H2D to complete and remove hooks."""
        self.h2d_stream.synchronize()
        self.active = False
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        logger.info(
            f"LayerSwap done: {self.src_model} -> {self.dst_model}, "
            f"{self.layers_swapped} layers, "
            f"{self.total_bytes_swapped / 1024**2:.1f} MB swapped"
        )

    def remove_hooks(self):
        """Remove hooks without waiting (cleanup on error)."""
        self.active = False
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
