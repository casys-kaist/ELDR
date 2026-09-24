# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefill expert counts, indexed by the physical KV-cache block ID.

Cached prefixes retain their counts. Writing offset zero resets a reused block;
request signatures sum its blocks, including prefix hits (see transport.py).
No weights, gate decisions, or expert execution are changed.
"""

import os

import torch

ENABLED = os.environ.get("ELDR") == "1"
GATE_PROB = os.environ.get("ELDR_SIGNATURE", "count") == "gate_prob_all"
_L = _E = _NT = 0
_REFS: list = []
_GREFS: list = []  # Only populated by the explicit gate-probability ablation.
_ESC: torch.Tensor | None = None  # int8 [physical KV blocks, layers, experts]


def install(model) -> None:
    """Attach the upstream ID-only callback; disabled runs do nothing."""
    if not ENABLED:
        return
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    global _L, _E, _REFS, _GREFS
    layers = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if not layers or sorted(m.moe_layer_id for m in layers) != list(range(len(layers))):
        raise ValueError("ELDR requires contiguous MoE layer IDs")
    _L, _E = len(layers), layers[0].global_num_experts
    if any(m.global_num_experts != _E for m in layers):
        raise ValueError("ELDR requires the same expert count in each layer")
    _REFS = [None] * _L
    _GREFS = [None] * _L if GATE_PROB else []
    for layer in layers:
        layer.router.set_capture_fn(
            lambda ids, index=layer.moe_layer_id: record(index, ids)
        )
        if GATE_PROB:
            # Observe the actual gate projection, including Qwen's internal
            # router, without re-running or modifying model expert selection.
            if layer.runner.gate is not None:
                if layer.runner._fse_fuse_gate:
                    raise ValueError("Gate-probability ablation needs an unfused gate")

                def gate_hook(module, args, output, index=layer.moe_layer_id):
                    if ENABLED:
                        _GREFS[index] = (
                            output[0] if isinstance(output, tuple) else output
                        )

                layer.runner.gate.register_forward_hook(gate_hook)
            else:

                def expert_hook(module, args, kwargs, index=layer.moe_layer_id):
                    if ENABLED:
                        _GREFS[index] = (
                            kwargs["router_logits"]
                            if "router_logits" in kwargs
                            else args[1]
                        )

                layer.register_forward_pre_hook(expert_hook, with_kwargs=True)


def record(layer_id: int, ids: torch.Tensor) -> None:
    """Retain actual gate IDs without copying or launching another kernel."""
    global _NT
    _REFS[layer_id] = ids
    _NT = ids.shape[0]


def reduce(n, slot_mapping, block_size, num_gpu_blocks) -> None:
    """Accumulate this producer step; no per-request GPU work or host sync."""
    global _ESC
    if not ENABLED or n == 0:
        return
    if n != _NT or not _REFS or any(ids is None for ids in _REFS):
        raise RuntimeError("Incomplete expert capture for the scheduled prefill")
    if _ESC is None:
        _ESC = torch.zeros(
            num_gpu_blocks,
            _L,
            _E,
            dtype=torch.float32 if GATE_PROB else torch.int8,
            device=_REFS[0].device,
        )
    if GATE_PROB:
        if not _GREFS or any(row is None or row.shape != (n, _E) for row in _GREFS):
            raise RuntimeError("Incomplete actual gate-logit capture")
        from vllm.eldr.gate_probability import reduce_probabilities

        logits = torch.stack(_GREFS).contiguous()
        reduce_probabilities(logits, slot_mapping[:n], _ESC, block_size)
        for i in range(_L):
            _REFS[i] = _GREFS[i] = None
        return
    experts = torch.stack(_REFS)[:, :n, :].contiguous()
    for i in range(_L):
        _REFS[i] = None
    from vllm.eldr.counts import reduce_counts

    reduce_counts(experts, slot_mapping[:n], _ESC, block_size)
