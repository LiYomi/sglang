"""Model registry for multi-model serving.

Maps model names to their metadata (path, tokenizer).
Supports startup registration + hot registration via API.
CPU preloading: models can be loaded into CPU memory at registration time.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    name: str
    path: str
    tokenizer: Any  # PreTrainedTokenizerBase
    cpu_model: Any = None  # nn.Module on CPU (preloaded)
    cpu_model_config: Any = None  # ModelConfig for preloaded model
    _preload_thread: Any = None  # background loading thread
    _preload_error: Optional[str] = None

    def to_dict(self) -> dict:
        d = {"name": self.name, "path": self.path}
        if self.cpu_model is not None:
            d["preloaded"] = True
        elif self._preload_thread is not None and self._preload_thread.is_alive():
            d["preloading"] = True
        return d


class ModelRegistry:
    """Registry of available models.

    Thread-safe for read; writes happen only at startup or via admin API
    (single-threaded in tokenizer_manager context).
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self.default_model: Optional[str] = None

    def register(self, name: str, path: str, tokenizer: Any, preload: bool = False) -> None:
        """Register a model. Overwrites if name already exists.
        If preload=True, starts background CPU loading."""
        info = ModelInfo(name=name, path=path, tokenizer=tokenizer)
        self._models[name] = info
        if self.default_model is None:
            self.default_model = name
        logger.info(f"Model registered: {name} -> {path}")
        if preload:
            self._start_preload(info)

    def _start_preload(self, info: ModelInfo):
        """Start background thread to load model weights into CPU memory."""
        def _load():
            try:
                logger.info(f"Preload started: {info.name} ({info.path})")
                import torch
                from sglang.srt.model_loader.loader import (
                    DefaultModelLoader, get_model_loader, _initialize_model,
                    _get_quantization_config, set_default_torch_dtype,
                )
                from sglang.srt.configs.load_config import LoadConfig
                from sglang.srt.configs.model_config import ModelConfig
                from sglang.srt.server_args import get_global_server_args
                from sglang.srt.utils.common import reserve_rope_cache_for_long_sequences

                server_args = get_global_server_args()
                model_config = ModelConfig.from_server_args(
                    server_args, model_path=info.path
                )
                load_config = LoadConfig(
                    load_format=server_args.load_format,
                    download_dir=server_args.download_dir,
                    model_loader_extra_config=server_args.model_loader_extra_config,
                )
                quant_config = _get_quantization_config(model_config, load_config)
                with set_default_torch_dtype(model_config.dtype):
                    with torch.device("cpu"):
                        model = _initialize_model(model_config, load_config, quant_config)
                    loader = get_model_loader(load_config=load_config, model_config=model_config)
                    weights = loader._get_all_weights(model_config, model)
                    DefaultModelLoader.load_weights_and_postprocess(
                        model, weights, torch.device("cpu"),
                    )
                model.eval()
                reserve_rope_cache_for_long_sequences(model, server_args, model_config)

                info.cpu_model = model
                info.cpu_model_config = model_config
                logger.info(f"Preload done: {info.name}, "
                           f"params={sum(p.numel()*p.element_size() for p in model.parameters())/1024**2:.0f}MB")
            except Exception as e:
                info._preload_error = str(e)
                logger.error(f"Preload failed for {info.name}: {e}")

        t = threading.Thread(target=_load, daemon=True)
        info._preload_thread = t
        t.start()

    def get(self, name: str) -> ModelInfo:
        """Get model info by name. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(
                f"Model \'{name}\' not registered. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def get_cpu_model(self, name: str):
        """Get preloaded CPU model if available. Waits for preload if in progress."""
        info = self._models.get(name)
        if info is None:
            return None, None
        # Wait for preload to finish if running
        if info._preload_thread is not None and info._preload_thread.is_alive():
            logger.info(f"Waiting for preload: {name}")
            info._preload_thread.join()
        if info.cpu_model is not None:
            return info.cpu_model, info.cpu_model_config
        return None, None

    def has(self, name: str) -> bool:
        return name in self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def get_tokenizer(self, name: str) -> Any:
        return self.get(name).tokenizer
