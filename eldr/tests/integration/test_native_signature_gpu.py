# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run explicitly in the pinned serving image on one idle GPU."""

import unittest
from unittest.mock import patch

import torch

from vllm.eldr.batch_sum import sum_rows
from vllm.eldr.counts import _histogram_blocks, reduce_counts
from vllm.eldr.gate_probability import reduce_probabilities
from vllm.eldr.transport import PendingSignatureCopy


class NativeSignatureGPU(unittest.TestCase):
    def test_gate_probability_prefix_extension_reuse_and_float_transport(self):
        generator = torch.Generator(device="cuda").manual_seed(59)
        for layers, block_size in ((30, 16), (36, 64), (48, 16)):
            for length in (1, 15, 16, 63, 64, 65, 257, 4096):
                logits = torch.randn(
                    layers, length, 128, generator=generator, device="cuda"
                )
                cache = torch.zeros(300, layers, 128, device="cuda")
                slots = torch.arange(length, device="cuda", dtype=torch.int64)
                cut = min(length, block_size // 2)
                reduce_probabilities(
                    logits[:, :cut].contiguous(), slots[:cut], cache, block_size
                )
                if cut < length:
                    reduce_probabilities(
                        logits[:, cut:].contiguous(), slots[cut:], cache, block_size
                    )
                table = torch.arange(300, dtype=torch.int32, device="cuda").repeat(2, 1)
                table[1] = 2**30
                lengths = torch.tensor(
                    [length, length], dtype=torch.int32, device="cuda"
                )
                discard = torch.tensor([False, True], device="cuda")
                actual = sum_rows(cache, table, lengths, discard, 2, block_size)
                expected = logits.softmax(-1).sum(1)
                torch.testing.assert_close(actual[0], expected, rtol=2e-5, atol=2e-5)
                self.assertFalse(actual[1].any().item())
                with patch("vllm.eldr.signature.GATE_PROB", True):
                    copied = PendingSignatureCopy(
                        ["request", None], actual
                    ).finish_synchronously()
                self.assertEqual(len(copied["request"]), layers * 128 * 4)
                reduce_probabilities(logits, slots, cache, block_size)
                again = sum_rows(cache, table, lengths, discard, 2, block_size)
                torch.testing.assert_close(again, actual, rtol=2e-5, atol=2e-5)

    def test_geometries_lengths_and_prefix_reuse(self):
        generator = torch.Generator(device="cuda").manual_seed(58)
        for layers, experts, topk, block_size in (
            (30, 128, 8, 16),
            (36, 128, 4, 64),
            (48, 128, 8, 16),
        ):
            compiled = None
            for length in (1, 15, 16, 63, 64, 65, 127, 257, 1025, 4096):
                ids = (
                    torch.rand(
                        layers, length, experts, device="cuda", generator=generator
                    )
                    .argsort(-1)[..., :topk]
                    .contiguous()
                )
                cache = torch.zeros(
                    300, layers, experts, dtype=torch.int8, device="cuda"
                )
                slots = torch.arange(length, device="cuda", dtype=torch.int64)
                # A partial first block is extended; complete prefix blocks remain.
                cut = min(length, block_size // 2)
                reduce_counts(ids[:, :cut].contiguous(), slots[:cut], cache, block_size)
                if cut < length:
                    reduce_counts(
                        ids[:, cut:].contiguous(), slots[cut:], cache, block_size
                    )
                table = torch.arange(300, dtype=torch.int32, device="cuda").repeat(2, 1)
                table[1] = 2**30  # Discarded rows must not dereference these IDs.
                lengths = torch.tensor(
                    [length, length], dtype=torch.int32, device="cuda"
                )
                discard = torch.tensor([False, True], device="cuda")
                actual = sum_rows(cache, table, lengths, discard, 2, block_size)
                expected = torch.stack(
                    [torch.bincount(row.flatten(), minlength=experts) for row in ids]
                ).to(torch.int16)
                self.assertTrue(torch.equal(actual[0], expected))
                self.assertFalse(actual[1].any().item())
                copied = PendingSignatureCopy(
                    ["request", None], actual
                ).finish_synchronously()
                self.assertEqual(copied, {"request": expected.cpu().numpy().tobytes()})
                # Reallocated blocks overwrite old counts, never double count.
                reduce_counts(ids, slots, cache, block_size)
                again = sum_rows(cache, table, lengths, discard, 2, block_size)
                self.assertTrue(torch.equal(actual, again))
                variants = len(
                    _histogram_blocks.device_caches[torch.cuda.current_device()][0]
                )
                if compiled is None:
                    compiled = variants
                self.assertEqual(
                    variants, compiled, f"Length specialization at {length}"
                )

    def test_copy_single_consumption(self):
        value = torch.ones(1, 2, 3, dtype=torch.int16, device="cuda")
        pending = PendingSignatureCopy(["r"], value)
        with self.assertRaises(RuntimeError):
            pending.finish_after_sync()
        pending.finish_synchronously()
        with self.assertRaises(RuntimeError):
            pending.finish_synchronously()


if __name__ == "__main__":
    unittest.main()
