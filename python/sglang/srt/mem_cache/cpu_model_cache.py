"""CPU model cache for multi-model hot-switching.

Loads model weights from disk into pinned CPU memory in a background thread.
On model switch, the cached CPU model provides fast H2D transfer (~50 GB/s).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CpuModelEntry:
    """Cached CPU model entry."""
    name: str
    path: str
    cpu_model: Any = None           # nn.Module on CPU (pinned)
    cpu_model_config: Any = None    # ModelConfig
    cpu_state_dict: Any = None      # {name: pinned_tensor} zero-copy refs
    _load_thread: Any = None
    _load_error: Optional[str] = None


class CpuModelCache:
    """Cache of CPU models for hot-switching.

    register() stores an entry. Call load() to start background disk → pinned CPU loading.
    get_cpu_model() returns the cached model (blocks if loading still running).
    """

    def __init__(self):
        self._models: Dict[str, CpuModelEntry] = {}

    def register(self, name: str, path: str) -> None:
        """Register a model entry. Call load() separately to start background disk loading."""
        entry = CpuModelEntry(name=name, path=path)
        self._models[name] = entry
        logger.info(f"Model registered: {name} -> {path}")

    def load(self, name: str):
        """Start background thread to load model from disk into pinned CPU memory."""
        entry = self._models.get(name)
        if entry is None:
            return
        if entry._load_thread is not None and entry._load_thread.is_alive():
            return  # already loading

        def _do_load():
            try:
                import time as _time
                import torch
                from sglang.srt.model_loader.loader import (
                    DefaultModelLoader, get_model_loader,
                    _initialize_model, _get_quantization_config, set_default_torch_dtype,
                )
                from sglang.srt.configs.load_config import LoadConfig
                from sglang.srt.configs.model_config import ModelConfig
                from sglang.srt.server_args import get_global_server_args

                logger.info(f"CPU load started: {entry.name} ({entry.path})")
                server_args = get_global_server_args()
                model_config = ModelConfig.from_server_args(server_args, model_path=entry.path)
                load_config = LoadConfig(
                    load_format=server_args.load_format,
                    download_dir=server_args.download_dir,
                    model_loader_extra_config=server_args.model_loader_extra_config,
                )
                quant_config = _get_quantization_config(model_config, load_config)

                with set_default_torch_dtype(model_config.dtype):
                    with torch.device("cpu"):
                        model = _initialize_model(model_config, load_config, quant_config)

                    # Bulk pin: single cudaHostAlloc, params point to views
                    _t0 = _time.perf_counter()
                    _total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
                    _pin_buf = torch.empty(_total_bytes, dtype=torch.uint8, pin_memory=True)
                    _offset = 0
                    for p in model.parameters():
                        _nbytes = p.numel() * p.element_size()
                        p.data = _pin_buf[_offset:_offset + _nbytes].view(p.dtype).reshape(p.shape)
                        _offset += _nbytes
                    model._pinned_weight_buffer = _pin_buf

                    loader = get_model_loader(load_config=load_config, model_config=model_config)
                    weights = loader._get_all_weights(model_config, model)
                    DefaultModelLoader.load_weights_and_postprocess(model, weights, torch.device("cpu"))

                model.eval()

                entry.cpu_model = model
                entry.cpu_model_config = model_config
                entry.cpu_state_dict = dict(model.state_dict())

                _t1 = _time.perf_counter()
                logger.info(f"CPU load done: {entry.name}, {_total_bytes/1024**2:.0f}MB, {(_t1-_t0)*1000:.0f}ms")
            except Exception as e:
                logger.exception(f"CPU load failed: {entry.name}")
                entry._load_error = str(e)

        t = threading.Thread(target=_do_load, daemon=True)
        entry._load_thread = t
        t.start()

    def get_cpu_model(self, name: str):
        """Get cached CPU model. Blocks if loading still running."""
        entry = self._models.get(name)
        if entry is None:
            raise KeyError(f"Model '{name}' not registered")
        if entry._load_thread is not None and entry._load_thread.is_alive():
            logger.info(f"Waiting for CPU load: {name}")
            entry._load_thread.join()
        if entry._load_error:
            raise RuntimeError(f"CPU load failed for '{name}': {entry._load_error}")
        return entry.cpu_model, entry.cpu_model_config

    def get_entry(self, name: str) -> Optional[CpuModelEntry]:
        return self._models.get(name)


# Global singleton
_global_cache: Optional[CpuModelCache] = None


def get_cpu_model_cache() -> CpuModelCache:
    """Get or create the global CpuModelCache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = CpuModelCache()
    return _global_cache
