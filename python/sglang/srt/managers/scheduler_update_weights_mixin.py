from __future__ import annotations

import logging
import threading
import time
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
    ModelCpuReadyNotification,
    RegisterModelNotification,
    RegisterModelReqOutput,
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
from sglang.srt.managers.model_switch import do_model_switch_bump
from sglang.srt.mem_cache.weight_staging import PreloadManager

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
        if self._host_model_mgr is None:
            logger.warning(
                f"Hot-switch requires enable_bump_allocator; ignoring switch to {target_model_name}"
            )
            return
        entry = self._host_model_mgr.get_entry(target_model_name)
        if entry is None:
            logger.warning(f"Model not registered: {target_model_name}")
            return
        # Dedup: if target already in queue anywhere, newly-parked reqs will be
        # flushed when that entry is executed — no need to enqueue again.
        # Also skip when target is already active and nothing is queued.
        if not self._switch_queue and target_model_name == self.active_model_name:
            return
        if any(entry_tuple[0] == target_model_name for entry_tuple in self._switch_queue):
            return
        self._switch_queue.append((target_model_name, entry.path, time.perf_counter()))
        logger.info(
            f"Switch queued: active={self.active_model_name} target={target_model_name} "
            f"(queue depth={len(self._switch_queue)})"
        )

    def _execute_pending_switch(self: "Scheduler"):
        if not self._switch_queue:
            return
        if not self.running_batch.is_empty():
            return
        if self.last_batch is not None and not self.last_batch.is_empty():
            return
        # Drain active-model requests before switching so they don't get
        # scheduled against the wrong model after the switch.
        if self.waiting_queue:
            return
        name, path, request_ts = self._switch_queue[0]
        self._execute_model_switch(name, path, request_ts)

    def _execute_model_switch(
        self: "Scheduler", target_model_name: str, target_path: str, request_ts: float
    ):
        drain_done_ts = time.perf_counter()
        prev_model_name = self.active_model_name
        logger.info(f"Auto-switching: {prev_model_name} -> {target_model_name}")
        try:
            timings = do_model_switch_bump(self, target_path, target_model_name)
            self.active_model_name = target_model_name
            drain_ms = (drain_done_ts - request_ts) * 1000
            switch_ms = timings["total"] * 1000
            e2e_ms = (time.perf_counter() - request_ts) * 1000
            logger.info(f"  E2E: drain={drain_ms:.1f}ms switch={switch_ms:.1f}ms total={e2e_ms:.1f}ms")
            # Flush parked reqs for the new active model back into waiting_queue.
            # They rerun the normal _add_request_to_queue path so priority/quota
            # checks happen now (against the new model's limits) rather than at park time.
            parked = self._parked_reqs_by_model.pop(target_model_name, None)
            if parked:
                logger.info(
                    f"Flushing {len(parked)} parked req(s) for {target_model_name} into waiting_queue"
                )
                for req in parked:
                    self._add_request_to_queue(req)
        except Exception:
            # Switch aborted mid-teardown: runtime region released,
            # attn_backend may be None, weights region possibly half-written.
            # No rollback path exists yet (F1 feature-level decision) — propagate
            # so the scheduler process dies loudly instead of silently serving
            # requests against a broken runtime.
            logger.exception(
                f"Auto-switch {prev_model_name} -> {target_model_name} failed; "
                f"server state is non-recoverable, re-raising"
            )
            raise
        finally:
            self._switch_queue.popleft()
            self._preload_attempted_target = None  # allow preload for next queued target

    def _check_preload(self: "Scheduler"):
        """Preload the next queued switch target in the background. Called from scheduler idle loop."""
        # Fire-and-forget drain for background CPU-load notifications. Cheap
        # O(1) when the queue is empty; each event emits one IPC.
        self._drain_cpu_ready_events()

        if not self._switch_queue:
            return
        target = self._switch_queue[0][0]
        if target == self.active_model_name:
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
        if self._host_model_mgr is None:
            msg = "Model registration requires enable_bump_allocator"
            logger.warning(msg)
            return RegisterModelReqOutput(success=False, message=msg)

        # Register first so conflicts (same name + different path while a prior
        # load is still in flight) are caught before we notify the detokenizer.
        try:
            self._host_model_mgr.register(recv_req.model_name, recv_req.model_path)
        except RuntimeError as e:
            logger.warning(f"register_model rejected: {e}")
            return RegisterModelReqOutput(success=False, message=str(e))

        self._host_model_mgr.load(recv_req.model_name, on_ready=self._on_cpu_load_done)

        logger.info(f"Scheduler registered model: {recv_req.model_name} -> {recv_req.model_path}")

        # Notify detokenizer to pre-load tokenizer for this model
        if self.send_to_detokenizer.socket is not None:
            self.send_to_detokenizer.socket.send_pyobj(
                RegisterModelNotification(model_name=recv_req.model_name, model_path=recv_req.model_path)
            )

        return RegisterModelReqOutput(success=True, message=f"Registered {recv_req.model_name}")

    # ------------------------------------------------------------------
    # Readiness notifications
    # ------------------------------------------------------------------

    def _on_cpu_load_done(self: "Scheduler", model_name: str, error):
        """Called from host_model_mgr's background load thread.

        Must NOT touch ZMQ (not thread-safe) — just enqueue, the event loop
        drains and emits the IPC.
        """
        self._cpu_ready_queue.put((model_name, error))

    def _drain_cpu_ready_events(self: "Scheduler"):
        """Forward any pending CPU-load completions to tokenizer_manager."""
        import queue as _queue
        if self.send_to_tokenizer.socket is None:
            # Nothing to forward to; drop events silently to avoid unbounded growth.
            try:
                while True:
                    self._cpu_ready_queue.get_nowait()
            except _queue.Empty:
                return
        while True:
            try:
                name, error = self._cpu_ready_queue.get_nowait()
            except _queue.Empty:
                return
            self.send_to_tokenizer.socket.send_pyobj(
                ModelCpuReadyNotification(
                    model_name=name,
                    success=error is None,
                    error=str(error) if error is not None else "",
                )
            )


    def _start_weight_preload(self: "Scheduler", target_model_name: str):
        """Start background H2D of target model weights to GPU staging.

        Scatter staging: weights are distributed into free tails of KV layer blocks.
        Runs in a background thread so scheduler returns immediately.
        """
        if self._preload_thread is not None and self._preload_thread.is_alive():
            logger.info(f"Preload already running, skip {target_model_name}")
            return

        bump = self.tp_worker.model_runner.vram_mgr
        entry = self._host_model_mgr.get_entry(target_model_name)
        kv_pool = self.tp_worker.model_runner.token_to_kv_pool
        kv_allocator = self.token_to_kv_pool_allocator
        current_weight_bytes = bump.regions["weights"].capacity

        if self._preload_manager is None:
            self._preload_manager = PreloadManager()
        preload_mgr = self._preload_manager
        alloc_lock = threading.Lock()

        def _do_preload():
            try:
                t_preload = time.perf_counter()
                # Wait for CPU load to finish before reading cpu_state_dict.
                # Runs in the background thread so the scheduler is not blocked.
                self._host_model_mgr.get_cpu_model(target_model_name)
                cpu_state_dict = entry.cpu_state_dict

                # Layout safety: compute max weights across current and target models
                target_weight_bytes = sum(
                    t.numel() * t.element_size() for t in cpu_state_dict.values()
                )
                max_weights_bytes = max(current_weight_bytes, target_weight_bytes)
                # Read staging floor just before staging starts so it reflects
                # the latest high-water-mark, not a stale snapshot from the
                # scheduler thread that queued this preload.
                min_free = kv_allocator.staging_floor

                logger.info(
                    f"_start_weight_preload: target={target_model_name} "
                    f"bump.left_offset={bump.left_offset} "
                    f"current_weights_cap={current_weight_bytes} "
                    f"target_weight_bytes={target_weight_bytes} "
                    f"max_weights_bytes={max_weights_bytes} "
                    f"min_free_slot={min_free} "
                    f"free_pages_count={len(kv_allocator.free_pages)}"
                )

                # Install staging hooks on the allocator only right before H2D
                # starts, otherwise alloc would pay lock + dirty-mark overhead
                # while we were still waiting for the CPU load to finish.
                kv_allocator.attach_staging(alloc_lock, preload_mgr)
                try:
                    preload_mgr.start_preload(
                        model_name=target_model_name,
                        cpu_state_dict=cpu_state_dict,
                        bump=bump,
                        kv_pool=kv_pool,
                        min_free_slot=min_free,
                        allocator=kv_allocator,
                        max_weights_bytes=max_weights_bytes,
                    )
                    logger.debug(
                        f"  TIMING: start_preload_total={(time.perf_counter()-t_preload)*1000:.0f}ms"
                    )
                finally:
                    kv_allocator.detach_staging()
            except Exception:
                logger.exception("Preload thread error")
            finally:
                logger.debug(
                    f"Preload thread exiting for {target_model_name}, is_valid={preload_mgr.is_valid}"
                )

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
