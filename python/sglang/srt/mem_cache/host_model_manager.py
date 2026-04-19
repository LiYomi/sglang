"""CPU model cache for multi-model hot-switching.

Loads model weights from disk into pinned CPU memory in a background thread.
On model switch, the cached CPU model provides fast H2D transfer (~50 GB/s).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch

from sglang.srt.mem_cache.vram_manager import _align_up

logger = logging.getLogger(__name__)


@dataclass
class CpuModelEntry:
    """Cached CPU model entry."""

    name: str
    path: str
    cpu_model: Any = None  # nn.Module on CPU (pinned memory)
    cpu_state_dict: Optional[Dict[str, Any]] = None  # {param_name: pinned_tensor}
    _load_thread: Optional[threading.Thread] = None
    _load_error: Optional[BaseException] = None


class HostModelManager:
    """Cache of CPU models for hot-switching.

    register() stores an entry. Call load() separately to start background disk
    -> pinned CPU loading. get_cpu_model() returns the cached model (blocks if
    loading still running).
    """

    def __init__(self):
        self._models: Dict[str, CpuModelEntry] = {}

    def register(self, name: str, path: str) -> CpuModelEntry:
        """Register a model entry. Call load() separately to start loading.

        Same (name, path) is idempotent and returns the existing entry.
        Re-registering with a different path while the previous load is still
        running raises RuntimeError — the background thread's closure would
        otherwise write into an orphaned entry and the new one would stay None
        forever.
        """
        existing = self._models.get(name)
        if existing is not None:
            if existing.path == path:
                return existing
            if (
                existing._load_thread is not None
                and existing._load_thread.is_alive()
            ):
                raise RuntimeError(
                    f"Cannot re-register '{name}' with a different path while "
                    f"previous load is still running (old={existing.path}, "
                    f"new={path}). Wait for it to finish first."
                )
        entry = CpuModelEntry(name=name, path=path)
        self._models[name] = entry
        logger.info(f"Model registered: {name} -> {path}")
        return entry

    def load(
        self,
        name: str,
        on_ready: Optional[Callable[[str, Optional[BaseException]], None]] = None,
    ):
        """Start background thread to load model from disk into pinned CPU memory.

        `on_ready` (if provided) is invoked from the background thread when
        the load finishes — `on_ready(name, None)` on success,
        `on_ready(name, exc)` on failure. The callback must be cheap and
        must not raise; use it to flip a readiness flag / emit an IPC, not
        to do heavy work.
        """
        entry = self._models[name]
        if entry.cpu_model is not None:
            if on_ready is not None:
                try:
                    on_ready(name, None)
                except Exception:
                    logger.exception(f"on_ready callback failed for {name}")
            return
        if entry._load_thread is not None and entry._load_thread.is_alive():
            # Ongoing load already in progress — caller that wants the new
            # callback should have passed it the first time; we don't support
            # attaching multiple callbacks here.
            return

        entry._load_error = None

        def _do_load():
            # sglang internal imports stay lazy — keeps module-import order
            # flexible since HostModelManager is created early in scheduler init.
            try:
                from sglang.srt.configs.load_config import LoadConfig
                from sglang.srt.configs.model_config import ModelConfig
                from sglang.srt.model_loader.loader import (
                    DefaultModelLoader,
                    _get_quantization_config,
                    _initialize_model,
                    get_model_loader,
                    set_default_torch_dtype,
                )
                from sglang.srt.server_args import get_global_server_args

                t0 = time.perf_counter()
                logger.info(f"CPU load started: {entry.name} ({entry.path})")
                server_args = get_global_server_args()
                model_config = ModelConfig.from_server_args(
                    server_args, model_path=entry.path
                )
                load_config = LoadConfig(
                    load_format=server_args.load_format,
                    download_dir=server_args.download_dir,
                    model_loader_extra_config=server_args.model_loader_extra_config,
                )
                quant_config = _get_quantization_config(model_config, load_config)

                with set_default_torch_dtype(model_config.dtype):
                    with torch.device("meta"):
                        model = _initialize_model(
                            model_config, load_config, quant_config
                        )

                    # Bulk pin: single cudaHostAlloc, params point to views.
                    # 256B aligned to match bump allocator layout.
                    total_bytes = sum(
                        _align_up(p.numel() * p.element_size())
                        for p in model.parameters()
                    )
                    pin_buf = torch.empty(
                        total_bytes, dtype=torch.uint8, pin_memory=True
                    )
                    offset = 0
                    for pname, p in list(model.named_parameters()):
                        nbytes = p.numel() * p.element_size()
                        view = (
                            pin_buf[offset : offset + nbytes]
                            .view(p.dtype)
                            .reshape(p.shape)
                        )
                        # Meta Parameter → CPU view: `p.data = view` raises
                        # "incompatible tensor type" (meta vs cpu TensorImpl).
                        # Replace the whole Parameter instead. Safe here because
                        # _initialize_model has not yet attached quant/LoRA
                        # wrapper attrs (those arrive via load_weights_and_postprocess
                        # that runs below). If quant support is added, rehydrate
                        # wrapper attrs here before continuing.
                        parent_name, _, attr_name = pname.rpartition(".")
                        parent = model.get_submodule(parent_name) if parent_name else model
                        new_param = torch.nn.Parameter(view, requires_grad=False)
                        # Carry over attrs sglang models pin on Parameter
                        # (`weight_loader`, `output_dim`, shard metadata, etc.).
                        # These live in Tensor.__dict__; a fresh Parameter
                        # starts empty and load_weights would blow up.
                        try:
                            new_param.__dict__.update(p.__dict__)
                        except Exception:
                            pass
                        parent._parameters[attr_name] = new_param
                        offset += _align_up(nbytes)
                    model._pinned_weight_buffer = pin_buf

                    loader = get_model_loader(
                        load_config=load_config, model_config=model_config
                    )
                    weights = loader._get_all_weights(model_config, model)
                    DefaultModelLoader.load_weights_and_postprocess(
                        model, weights, torch.device("cpu")
                    )

                model.eval()

                entry.cpu_model = model
                entry.cpu_state_dict = dict(model.state_dict())

                logger.info(
                    f"CPU load done: {entry.name}, "
                    f"{total_bytes / 1024**2:.0f}MB, "
                    f"{(time.perf_counter() - t0) * 1000:.0f}ms"
                )
            except BaseException as e:
                logger.exception(f"CPU load failed: {entry.name}")
                entry._load_error = e
            finally:
                if on_ready is not None:
                    try:
                        on_ready(name, entry._load_error)
                    except Exception:
                        logger.exception(
                            f"on_ready callback raised for {name}"
                        )

        t = threading.Thread(target=_do_load, daemon=True)
        entry._load_thread = t
        t.start()

    def get_cpu_model(self, name: str):
        """Get cached CPU model. Blocks if loading still running."""
        entry = self._models[name]
        if entry._load_thread is None:
            raise RuntimeError(
                f"get_cpu_model('{name}') called before load() — "
                f"call register() then load() first."
            )
        if entry._load_thread.is_alive():
            logger.info(f"Waiting for CPU load: {name}")
            entry._load_thread.join()
        if entry._load_error is not None:
            raise RuntimeError(
                f"CPU load failed for '{name}'"
            ) from entry._load_error
        return entry.cpu_model

    def get_entry(self, name: str) -> Optional[CpuModelEntry]:
        return self._models.get(name)

    def get_status(self, name: str) -> str:
        """Return one of: "ready", "loading", "failed", "unknown".

        Surfaced to clients via GET /list_models so they can decide whether
        to poll before issuing requests (or just send and accept that the
        first switch to this model will block on get_cpu_model).
        """
        entry = self._models.get(name)
        if entry is None:
            return "unknown"
        if entry._load_error is not None:
            return "failed"
        if entry.cpu_model is not None:
            return "ready"
        if entry._load_thread is not None and entry._load_thread.is_alive():
            return "loading"
        return "unknown"

    def list_statuses(self) -> Dict[str, str]:
        return {name: self.get_status(name) for name in self._models}


_global_cache: Optional[HostModelManager] = None
_global_cache_lock = threading.Lock()


def get_host_model_manager() -> HostModelManager:
    """Get or create the global HostModelManager singleton."""
    global _global_cache
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = HostModelManager()
    return _global_cache
