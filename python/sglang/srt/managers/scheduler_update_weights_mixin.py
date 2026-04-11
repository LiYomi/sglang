from __future__ import annotations

import logging
import traceback
from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.constants import (
    GPU_MEMORY_ALL_TYPES,
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.io_struct import (
    CheckWeightsReqInput,
    CheckWeightsReqOutput,
    DestroyWeightsUpdateGroupReqInput,
    DestroyWeightsUpdateGroupReqOutput,
    GetWeightsByNameReqInput,
    GetWeightsByNameReqOutput,
    InitWeightsUpdateGroupReqInput,
    InitWeightsUpdateGroupReqOutput,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromDistributedReqOutput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromIPCReqOutput,
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromTensorReqOutput,
)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerUpdateWeightsMixin:

    def update_weights_from_disk(
        self: Scheduler, recv_req: UpdateWeightFromDiskReqInput
    ):
        """In-place update of the weights from disk."""
        success, message = self.tp_worker.update_weights_from_disk(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightFromDiskReqOutput(success, message, 0)

    def init_weights_update_group(
        self: Scheduler, recv_req: InitWeightsUpdateGroupReqInput
    ):
        """Initialize the online model parameter update group."""
        success, message = self.tp_worker.init_weights_update_group(recv_req)
        return InitWeightsUpdateGroupReqOutput(success, message)

    def destroy_weights_update_group(
        self: Scheduler, recv_req: DestroyWeightsUpdateGroupReqInput
    ):
        """Destroy the online model parameter update group."""
        success, message = self.tp_worker.destroy_weights_update_group(recv_req)
        return DestroyWeightsUpdateGroupReqOutput(success, message)

    def update_weights_from_distributed(
        self,
        recv_req: UpdateWeightsFromDistributedReqInput,
    ) -> Tuple[bool, str]:
        """Update the online model parameter."""
        success, message = self.tp_worker.update_weights_from_distributed(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        return UpdateWeightsFromDistributedReqOutput(success, message)

    def update_weights_from_tensor(
        self: Scheduler, recv_req: UpdateWeightsFromTensorReqInput
    ):
        """Update the online model parameter from tensors."""
        if recv_req.disable_draft_model:
            worker = self.tp_worker
        else:
            worker = self.draft_worker or self.tp_worker
        success, message = worker.update_weights_from_tensor(recv_req)
        # TODO extract common code b/t update_weights_from_distributed and update_weights_from_tensor later
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromTensorReqOutput(success, message)

    def update_weights_from_ipc(
        self: Scheduler, recv_req: UpdateWeightsFromIPCReqInput
    ):
        """Update the online model parameter from IPC for checkpoint-engine integration."""
        success, message = self.tp_worker.update_weights_from_ipc(recv_req)
        if success:
            if recv_req.flush_cache:
                flush_cache_success = self.flush_cache()
                assert flush_cache_success, "Cache flush failed after updating weights"
        else:
            logger.error(message)
        torch.distributed.barrier(group=self.tp_cpu_group)
        return UpdateWeightsFromIPCReqOutput(success, message)

    def get_weights_by_name(self: Scheduler, recv_req: GetWeightsByNameReqInput):
        parameter = self.tp_worker.get_weights_by_name(recv_req)
        return GetWeightsByNameReqOutput(parameter)

    def release_memory_occupation(
        self: Scheduler, recv_req: ReleaseMemoryOccupationReqInput
    ):
        assert (
            self.is_fully_idle()
        ), "release_memory_occupation should be called only when server is idle."

        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.add(tag)

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_KV_CACHE)
            self.flush_cache()

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.stashed_model_static_state = _export_static_state(
                self.tp_worker.model_runner.model
            )
            torch.distributed.barrier(self.tp_cpu_group)
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_WEIGHTS)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.pause(GPU_MEMORY_TYPE_CUDA_GRAPH)

        torch.get_device_module().synchronize()

        return ReleaseMemoryOccupationReqOutput()

    def resume_memory_occupation(
        self: Scheduler, recv_req: ResumeMemoryOccupationReqInput
    ):
        tags = recv_req.tags

        if tags is None or len(tags) == 0:
            tags = GPU_MEMORY_ALL_TYPES

        for tag in tags:
            self.offload_tags.remove(tag)

        if GPU_MEMORY_TYPE_CUDA_GRAPH in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_CUDA_GRAPH)

        if GPU_MEMORY_TYPE_WEIGHTS in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_WEIGHTS)
            torch.distributed.barrier(self.tp_cpu_group)
            _import_static_state(
                self.tp_worker.model_runner.model,
                self.stashed_model_static_state,
            )
            del self.stashed_model_static_state

        if GPU_MEMORY_TYPE_KV_CACHE in tags:
            self.memory_saver_adapter.resume(GPU_MEMORY_TYPE_KV_CACHE)

        return ResumeMemoryOccupationReqOutput()

    def check_weights(self: Scheduler, recv_req: CheckWeightsReqInput):
        try:
            self.tp_worker.model_runner.check_weights(action=recv_req.action)
            return CheckWeightsReqOutput(success=True, message="Success.")
        except Exception as e:
            logger.warning(f"check_weights see error: {e}")
            traceback.print_exc()
            return CheckWeightsReqOutput(success=False, message=f"{e}")

    def save_remote_model(self: Scheduler, params):
        url = params["url"]

        self.tp_worker.model_runner.save_remote_model(url)

        if self.draft_worker is not None:
            draft_url = params.get("draft_url", None)
            assert (
                draft_url is not None
            ), "draft_url must be provided when draft model is enabled"
            self.draft_worker.model_runner.save_remote_model(draft_url)

    def save_sharded_model(self: Scheduler, params):
        self.tp_worker.model_runner.save_sharded_model(
            path=params["path"],
            pattern=params["pattern"],
            max_size=params["max_size"],
        )


    def _prepare_model_switch(self: "Scheduler", target_model_name: str):
        import time as _time
        entry = self._cpu_model_cache.get_entry(target_model_name) if self._cpu_model_cache else None
        if entry is None:
            logger.warning(f"Model not registered: {target_model_name}")
            return
        self._switch_request_ts = _time.perf_counter()
        self._pending_switch = (target_model_name, entry.path)
        logger.info(f"Switch pending: {self.active_model_name} -> {target_model_name}")
        # Preload will be triggered by _check_preload() in the scheduler loop

    def _execute_pending_switch(self):
        if self._pending_switch is not None and self.running_batch.is_empty():
            # Also ensure no batch is currently being processed (last_batch cleared)
            if self.last_batch is not None and not self.last_batch.is_empty():
                return
            name, path = self._pending_switch
            self._execute_model_switch(name, path)

    def _execute_model_switch(self, target_model_name, target_path):
        import time as _time
        from sglang.srt.managers.model_switch import do_model_switch_bump
        _drain_done_ts = _time.perf_counter()
        prev_model_name = self.active_model_name
        logger.info(f"Auto-switching: {prev_model_name} -> {target_model_name}")
        try:
            timings = do_model_switch_bump(self, target_path, target_model_name)
            self.active_model_name = target_model_name
            self._prev_model_name = prev_model_name
            # E2E timing
            _req_ts = getattr(self, "_switch_request_ts", None)
            if _req_ts is not None:
                _e2e = _time.perf_counter() - _req_ts
                _drain_ms = (_drain_done_ts - _req_ts) * 1000
                _switch_ms = timings.get("total", 0) * 1000
                logger.info(f"  E2E: drain={_drain_ms:.1f}ms switch={_switch_ms:.1f}ms total={_e2e*1000:.1f}ms")
                self._switch_request_ts = None
        except Exception as e:
            logger.error(f"Auto-switch failed: {e}", exc_info=True)
        finally:
            self._pending_switch = None
            self._preload_attempted_target = None  # allow preload for next switch

    def _check_preload(self: "Scheduler"):
        """Check if a model needs preloading. Called from scheduler idle loop.

        Priority:
          1. Pending switch target (for upcoming switch)
          2. Previous model (for switch-back after completed switch)
        """
        # Determine preload target
        target = None
        if self._pending_switch is not None:
            target = self._pending_switch[0]
        elif getattr(self, "_prev_model_name", None):
            target = self._prev_model_name

        if target is None or target == self.active_model_name:
            return

        # Already preloading?
        if self._preload_thread is not None and self._preload_thread.is_alive():
            return

        # Already preloaded for this target?
        if (self._preload_manager is not None
                and self._preload_manager.model_name == target
                and self._preload_manager.is_valid):
            return

        # Already attempted preload for this target (success or failure)?
        # Reset only when target changes or switch completes.
        if self._preload_attempted_target == target:
            return

        self._start_weight_preload(target)

    def register_model(self: "Scheduler", recv_req):
        """Register a model for hot-switching.
        Preloads model weights to CPU in background via cpu_model_cache."""
        from sglang.srt.managers.io_struct import RegisterModelReqOutput

        logger.info(f"Scheduler registered model: {recv_req.model_name} -> {recv_req.model_path}")

        # Notify detokenizer to pre-load tokenizer for this model
        if hasattr(self, "send_to_detokenizer"):
            from sglang.srt.managers.io_struct import RegisterModelNotification
            self.send_to_detokenizer.socket.send_pyobj(
                RegisterModelNotification(model_name=recv_req.model_name, model_path=recv_req.model_path)
            )

        # Load model to CPU in background if bump allocator is enabled
        if self._cpu_model_cache is not None:
            self._cpu_model_cache.register(recv_req.model_name, recv_req.model_path)
            self._cpu_model_cache.load(recv_req.model_name)

        return RegisterModelReqOutput(success=True, message=f"Registered {recv_req.model_name}")


    def _start_weight_preload(self: "Scheduler", target_model_name: str):
        """Start background H2D of target model weights to GPU staging.

        Scatter staging: weights are distributed into free tails of KV layer blocks.
        Runs in a background thread so scheduler returns immediately.
        """
        import threading

        # Skip if a preload thread is already running
        _t = getattr(self, "_preload_thread", None)
        if _t is not None and _t.is_alive():
            logger.info(f"Preload already running, skip {target_model_name}")
            return

        bump = self.tp_worker.model_runner.bump_vram_manager
        entry = self._cpu_model_cache.get_entry(target_model_name)
        # TODO: non-blocking check + drain gate in get_next_batch_to_run
        # Currently blocks scheduler if CPU load not done yet.
        cpu_model, _ = self._cpu_model_cache.get_cpu_model(target_model_name)
        cpu_state_dict = entry.cpu_state_dict

        # Get KV pool and min_free_slot for scatter staging
        kv_pool = self.tp_worker.model_runner.token_to_kv_pool
        _allocator = self.token_to_kv_pool_allocator
        # min(free_pages) = lowest free slot; staging uses rows above this
        if len(_allocator.free_pages) > 0:
            _min_free = _allocator.free_pages.min().item()
        else:
            # All slots allocated — staging capacity will be 0
            _min_free = kv_pool.k_buffer[0].shape[0]

        from sglang.srt.mem_cache.weight_staging import PreloadManager
        from sglang.srt.managers.model_switch import _runtime_cache
        if self._preload_manager is None:
            self._preload_manager = PreloadManager()

        preload_mgr = self._preload_manager

        # Create shared lock for thread-safe free_pages access
        _alloc_lock = threading.Lock()
        _allocator._staging_lock = _alloc_lock
        # Set preload manager ref for chunk dirty tracking (Issue M)
        _allocator._preload_mgr = preload_mgr

        # Layout safety: compute max weights across current and target models
        _target_weight_bytes = sum(
            t.numel() * t.element_size() for t in cpu_state_dict.values()
        )
        _current_weight_bytes = bump.regions["weights"].capacity if "weights" in bump.regions else 0
        _max_weights_bytes = max(_current_weight_bytes, _target_weight_bytes)

        # Right boundary safety: max runtime across current and target models
        _current_runtime_bytes = bump.regions["runtime"].capacity if "runtime" in bump.regions else 0
        _target_runtime_bytes = _runtime_cache.get(target_model_name, _current_runtime_bytes)
        _max_runtime_bytes = max(_current_runtime_bytes, _target_runtime_bytes)
        _bump_total_bytes = bump.total_bytes

        def _do_preload():
            import traceback as _tb
            import time as _time_pl
            try:
                _t_contig = _time_pl.perf_counter()
                sd = {k: v.contiguous() for k, v in cpu_state_dict.items()}
                logger.info(f"  TIMING: contiguous_copy={(_time_pl.perf_counter()-_t_contig)*1000:.0f}ms ({len(cpu_state_dict)} tensors)")
                _t_preload = _time_pl.perf_counter()
                preload_mgr.start_preload(
                    model_name=target_model_name,
                    cpu_state_dict=sd,
                    bump=bump,
                    kv_pool=kv_pool,
                    min_free_slot=_min_free,
                    allocator=_allocator,
                    alloc_lock=_alloc_lock,
                    max_weights_bytes=_max_weights_bytes,
                    max_runtime_bytes=_max_runtime_bytes,
                    bump_total_bytes=_bump_total_bytes,
                )
                logger.info(f"  TIMING: start_preload_total={(_time_pl.perf_counter()-_t_preload)*1000:.0f}ms")
            except Exception as e:
                logger.error(f"Preload thread error: {e}", exc_info=True)
            finally:
                logger.info(f"Preload thread exiting for {target_model_name}, is_valid={preload_mgr.is_valid}")
                # Remove lock after preload completes — no contention outside preload
                _allocator._staging_lock = None
                # Keep _preload_mgr alive — dirty flags needed until switch consumes them

        logger.info(
            f"_start_weight_preload: target={target_model_name} "
            f"bump.left_offset={bump.left_offset} "
            f"current_weights_cap={_current_weight_bytes} "
            f"target_weight_bytes={_target_weight_bytes} "
            f"max_weights_bytes={_max_weights_bytes} "
            f"max_runtime_bytes={_max_runtime_bytes} "
            f"min_free_slot={_min_free} "
            f"free_pages_count={len(_allocator.free_pages)}")
        self._preload_thread = threading.Thread(target=_do_preload, daemon=True)
        self._preload_thread.start()
        logger.info(f"_start_weight_preload: background thread started for {target_model_name}")
        self._preload_attempted_target = target_model_name


def _export_static_state(model):
    return dict(
        buffers=[
            (name, buffer.detach().clone()) for name, buffer in model.named_buffers()
        ]
    )


def _import_static_state(model, static_params):
    self_named_buffers = dict(model.named_buffers())
    for name, tensor in static_params["buffers"]:
        self_named_buffers[name][...] = tensor
