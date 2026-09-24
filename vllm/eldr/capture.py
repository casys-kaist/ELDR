# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe the monolithic gate's actual IDs without rerunning expert selection."""

from contextlib import contextmanager
from contextvars import ContextVar

_active: ContextVar[dict | None] = ContextVar("eldr_monolithic_capture", default=None)


@contextmanager
def observe(callback):
    if _active.get() is not None:
        raise RuntimeError("Nested monolithic capture is unsupported")
    state = {"callback": callback, "calls": 0}
    token = _active.set(state)
    try:
        yield
        if state["calls"] != 1:
            raise RuntimeError(
                "Monolithic backend did not expose exactly one gate result"
            )
    finally:
        _active.reset(token)


def record(ids):
    state = _active.get()
    if state is not None:
        if state["calls"]:
            raise RuntimeError("Multiple gate results in one monolithic layer")
        state["callback"](ids)
        state["calls"] += 1


def supports(method, layer):
    # Lazy import: ordinary decode and non-monolithic paths do not import this
    # optional backend or allocate a capture context.
    from vllm.model_executor.layers.fused_moe.experts import (
        gpt_oss_triton_kernels_moe as backend,
    )

    kernel = getattr(method, "moe_kernel", None)
    return (
        isinstance(
            getattr(kernel, "fused_experts", None),
            backend.OAITritonMxfp4ExpertsMonolithic,
        )
        and getattr(layer, "expert_map", None) is None
    )
