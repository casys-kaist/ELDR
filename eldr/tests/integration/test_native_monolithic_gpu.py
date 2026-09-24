# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLm project
"""Actual Triton routing path; dense expert GEMMs are stubbed, not benchmarked."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.fused_moe.experts import (
    gpt_oss_triton_kernels_moe as backend,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner


@unittest.skipUnless(torch.cuda.is_available(), "Requires explicitly allocated GPU")
class ActualRoutingTests(unittest.TestCase):
    def test_hook_receives_actual_tensor_and_skips_duplicate_gate(self):
        for tied in (False, True):
            hidden = torch.zeros((129, 64), device="cuda", dtype=torch.bfloat16)
            generator = torch.Generator(device="cuda").manual_seed(58)
            logits = torch.randn((129, 128), device="cuda", generator=generator)
            if tied:
                logits.zero_()
            captures, executed, extra_gates = [], [], []

            def capture(ids, captures=captures):
                captures.append(ids)

            def extra_gate(
                hidden_states, router_logits, extra_gates=extra_gates, **kwargs
            ):
                extra_gates.append(True)
                weights, ids, _ = fused_topk(hidden_states, router_logits, 4, True)
                capture(ids)
                return weights, ids

            def monolithic(layer, x, router_logits, **kwargs):
                return backend.triton_kernel_moe_forward(
                    x, None, None, router_logits, 4, True
                )

            class Expert:
                pass

            class Runner:
                _apply_quant_method = MoERunner._apply_quant_method
                _shared_experts = None

                def _maybe_apply_shared_experts(self, *args):
                    pass

            runner = Runner()
            runner.router = SimpleNamespace(
                capture_fn=capture, select_experts=extra_gate
            )
            runner._quant_method = SimpleNamespace(
                is_monolithic=True,
                apply_monolithic=monolithic,
                moe_kernel=SimpleNamespace(fused_experts=Expert()),
            )
            original_make = backend.make_routing_data

            def make(ids, weights, n, executed=executed, original_make=original_make):
                executed.append((ids, weights))
                return original_make(ids, weights, n)

            with (
                patch.object(backend, "OAITritonMxfp4ExpertsMonolithic", Expert),
                patch.object(backend, "make_routing_data", make),
                patch.object(
                    backend,
                    "triton_kernel_fused_experts",
                    side_effect=lambda *args, **kwargs: args[1],
                ),
            ):
                _, result = runner._apply_quant_method(
                    SimpleNamespace(expert_map=None), hidden, logits, None
                )
                self.assertIs(result, hidden)
                self.assertEqual(len(extra_gates), 0)
                self.assertEqual(len(captures), 1)
                self.assertEqual(len(executed), 1)
                self.assertIs(captures[0], executed[0][0])
                self.assertIs(runner.router.capture_fn, capture)
                self.assertEqual(tuple(captures[0].shape), (129, 4))
                self.assertTrue(bool((captures[0] >= 0).all()))
                self.assertTrue(bool((captures[0] < 128).all()))


if __name__ == "__main__":
    unittest.main()
