"""Model registry for multi-model serving.

Maps model names to their metadata (path, tokenizer).
Supports startup registration + hot registration via API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Metadata for a registered model."""
    name: str
    path: str
    tokenizer: Any  # PreTrainedTokenizerBase

    def to_dict(self) -> dict:
        return {"name": self.name, "path": self.path}


class ModelRegistry:
    """Registry of available models.

    Thread-safe for read; writes happen only at startup or via admin API
    (single-threaded in tokenizer_manager context).
    """

    def __init__(self):
        self._models: Dict[str, ModelInfo] = {}
        self.default_model: Optional[str] = None

    def register(self, name: str, path: str, tokenizer: Any) -> None:
        """Register a model. Overwrites if name already exists."""
        self._models[name] = ModelInfo(name=name, path=path, tokenizer=tokenizer)
        if self.default_model is None:
            self.default_model = name
        logger.info(f"Model registered: {name} -> {path}")

    def get(self, name: str) -> ModelInfo:
        """Get model info by name. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not registered. Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def resolve(self, name: Optional[str]) -> str:
        """Resolve model name: return name if registered, else default.

        Raises ValueError if no model is registered.
        """
        if name and name in self._models:
            return name
        if self.default_model:
            return self.default_model
        raise ValueError("No model registered")

    def has(self, name: str) -> bool:
        return name in self._models

    def list_models(self) -> List[str]:
        return list(self._models.keys())

    def get_path(self, name: str) -> str:
        return self.get(name).path

    def get_tokenizer(self, name: str) -> Any:
        return self.get(name).tokenizer
