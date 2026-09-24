# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drained, prefix-safe capture control for a shared benchmark engine fleet.

Not a runtime routing fallback. Only used between complete experiment phases.
"""

import inspect
import multiprocessing
import os


class CaptureSwitch:
    def __init__(self, signature, transport, routers, environment):
        if (
            not signature.ENABLED
            or not transport.ENABLED
            or environment.get("ELDR") != "1"
            or not routers
            or any(router.capture_fn is None for router in routers)
        ):
            raise ValueError("Initial fully enabled native capture required")
        self.signature = signature
        self.transport = transport
        self.routers = tuple(routers)
        self.callbacks = tuple(router.capture_fn for router in routers)
        self.environment = environment
        self.enabled = True
        self.transitions = 0

    def set_enabled(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("Explicit boolean required")
        # Validate the entire old state before mutation, even an idempotent call.
        expected = self.callbacks if self.enabled else (None,) * len(self.routers)
        if (
            self.enabled != self.signature.ENABLED
            or self.enabled != self.transport.ENABLED
            or self.environment.get("ELDR") != str(int(self.enabled))
            or any(r.capture_fn is not cb for r, cb in zip(self.routers, expected))
        ):
            raise ValueError("Capture state changed outside phase controller")
        if enabled != self.enabled:
            old_refs, old_nt = self.signature._REFS, self.signature._NT
            try:
                for router, callback in zip(self.routers, self.callbacks):
                    router.set_capture_fn(callback if enabled else None)
                self.signature.ENABLED = enabled
                self.transport.ENABLED = enabled
                self.environment["ELDR"] = str(int(enabled))
                self.signature._REFS = [None] * self.signature._L
                self.signature._NT = 0
            except Exception:
                # Live routers are validated plain BaseRouters. Restore their
                # attributes directly rather than retrying a failed setter.
                for router, callback in zip(self.routers, expected):
                    router.__dict__["capture_fn"] = callback
                self.signature.ENABLED = self.transport.ENABLED = self.enabled
                self.environment["ELDR"] = str(int(self.enabled))
                self.signature._REFS, self.signature._NT = old_refs, old_nt
                raise
            self.enabled = enabled
            self.transitions += 1
        return dict(
            enabled=self.enabled,
            callbacks=sum(r.capture_fn is not None for r in self.routers),
            signature_enabled=self.signature.ENABLED,
            transport_enabled=self.transport.ENABLED,
            environment=self.environment["ELDR"],
            transitions=self.transitions,
            resident_signature_cache=self.signature._ESC is not None,
        )


def require_supported(config, process_name, core_pid, worker_pid, reset, mode):
    if mode not in ("on", "off", "status"):
        raise ValueError("Explicit phase mode required")
    if type(core_pid) is not int or core_pid != worker_pid:
        raise ValueError("Serialized core and worker must be the same process")
    if type(reset) is not bool or reset != (mode != "status"):
        raise ValueError("Core must confirm an unforced empty-prefix reset")
    parallel, cache = config.parallel_config, config.cache_config
    scheduler, model = config.scheduler_config, config.model_config
    transfer = config.kv_transfer_config
    if (
        scheduler.async_scheduling is not False
        or type(cache.enable_prefix_caching) is not bool
        or scheduler.enable_chunked_prefill is not False
        or scheduler.disable_hybrid_kv_cache_manager is not True
        or model.enforce_eager is not True
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.data_parallel_size != 1
        or parallel.enable_expert_parallel
        or parallel.distributed_executor_backend != "uni"
        or config.speculative_config is not None
        or transfer is None
        or transfer.kv_role != "kv_producer"
        or transfer.kv_connector != "NixlConnector"
        or not process_name.startswith("EngineCore")
    ):
        raise ValueError("Requires synchronous eager UniProc NIXL producer")
    if (
        type(cache.num_gpu_blocks) is not int
        or cache.num_gpu_blocks < 1
        or type(cache.block_size) is not int
        or not 0 < cache.block_size < 128
        or type(model.max_model_len) is not int
        or not 0 < model.max_model_len <= 32767
    ):
        raise ValueError("Initialized KV capacity and supported count bounds required")


def require_empty_core(core):
    if (
        core.async_scheduling is not False
        or core.batch_queue is not None
        or core.is_scheduler_paused()
    ):
        raise ValueError("Unpaused synchronous serialized core required")
    scheduler = core.scheduler
    if any(
        (
            scheduler.running,
            scheduler.waiting,
            scheduler.skipped_waiting,
            scheduler.requests,
            scheduler.delayed_free_req_ids,
        )
    ):
        raise ValueError("All generation and retained KV requests must be absent")
    pool = scheduler.kv_cache_manager.block_pool
    if (
        type(pool.num_gpu_blocks) is not int
        or pool.num_gpu_blocks < 1
        or len(pool.blocks) != pool.num_gpu_blocks
        or pool.num_gpu_blocks - pool.get_num_free_blocks() != 1
    ):
        raise ValueError("Only the reserved null KV block may remain used")
    return pool


def validate_worker(state, requested):
    enabled = state["enabled"]
    if (
        type(enabled) is not bool
        or state["pid"] != os.getpid()
        or state["prefix_cache_supported"] is not True
        or state["same_process_checked"] is not True
        or type(state["layers"]) is not int
        or state["layers"] < 1
        or type(state["callbacks"]) is not int
        or state["signature_enabled"] is not enabled
        or state["transport_enabled"] is not enabled
        or state["environment"] != str(int(enabled))
        or state["callbacks"] != (state["layers"] if enabled else 0)
        or (requested != "status" and enabled != (requested == "on"))
    ):
        raise ValueError("Consistent separately validated same-process worker required")


def capture_with_empty_prefix(self, mode):
    """Requires a separately validated worker; never expose directly to clients."""
    if mode not in ("on", "off", "status"):
        raise ValueError("Explicit capture mode required")
    pool = require_empty_core(self)
    (before,) = self.collective_rpc(
        "eldr_capture",
        timeout=20,
        args=("status", os.getpid(), False),
    )
    validate_worker(before, "status")
    if mode == "status":
        return dict(before, generation_empty=True, prefix_reset=False)
    # The block-pool reset checks free blocks and clears hashes with caching
    # either ON or OFF. Preserve the same drained-state checks in both ablations.
    # No running-request preemption, connector reset or forced KV release.
    if (
        self.reset_prefix_cache(reset_running_requests=False, reset_connector=False)
        is not True
    ):
        raise ValueError("Unforced prefix-cache reset did not succeed")
    require_empty_core(self)
    if any(block.block_hash is not None for block in pool.blocks):
        raise ValueError("Cached prefix hashes remain after reset")
    (after,) = self.collective_rpc(
        "eldr_capture",
        timeout=20,
        args=(mode, os.getpid(), True),
    )
    validate_worker(after, mode)
    return dict(after, generation_empty=True, prefix_reset=True)


def worker_capture(self, mode, core_pid, prefix_reset):
    from vllm.eldr import signature, transport
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter

    process_name = multiprocessing.current_process().name
    require_supported(
        self.vllm_config,
        process_name,
        core_pid,
        os.getpid(),
        prefix_reset,
        mode,
    )
    state = getattr(self, "_empty_prefix_capture_switch", None)
    if state is None:
        routers = [
            layer.router
            for layer in self.model_runner.model.modules()
            if isinstance(layer, FusedMoE)
        ]
        if len(routers) != signature._L or not routers:
            raise ValueError("Exactly one router per signature layer required")
        for router in routers:
            if (
                not isinstance(router, BaseRouter)
                or type(router).__setattr__ is not object.__setattr__
                or inspect.getattr_static(router, "set_capture_fn")
                is not BaseRouter.set_capture_fn
                or "capture_fn" not in router.__dict__
                or inspect.getattr_static(type(router), "capture_fn", None) is not None
            ):
                raise ValueError("Unmodified plain BaseRouter capture setter required")
        state = CaptureSwitch(signature, transport, routers, os.environ)
        self._empty_prefix_capture_switch = state
    result = state.set_enabled(state.enabled if mode == "status" else mode == "on")
    cache = self.vllm_config.cache_config
    return dict(
        result,
        pid=os.getpid(),
        process_name=process_name,
        layers=signature._L,
        experts=signature._E,
        prefix_cache_supported=True,
        prefix_cache_enabled=cache.enable_prefix_caching,
        same_process_checked=True,
        kv_capacity_tokens=cache.num_gpu_blocks * cache.block_size,
        kv_block_size=cache.block_size,
        max_model_len=self.vllm_config.model_config.max_model_len,
    )
