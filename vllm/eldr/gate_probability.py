# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in ablation: cache full-softmax gate sums alongside physical KV blocks."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _probability_blocks(
    Logits,
    Slots,
    Cache,
    N,
    L: tl.constexpr,
    E: tl.constexpr,
    BS: tl.constexpr,
    BINS: tl.constexpr,
):
    start, layer = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + start)
    block = slot // BS
    previous = tl.load(Slots + start - 1, mask=start > 0, other=-BS) // BS
    if (start == 0) | (block != previous):
        token = start + tl.arange(0, BS)
        expert = tl.arange(0, BINS)
        token_slot = tl.load(Slots + token, mask=token < N, other=-BS)
        valid = (token < N) & (token_slot // BS == block)
        logits = tl.load(
            Logits + (layer * N + token[:, None]) * E + expert[None, :],
            mask=(token[:, None] < N) & (expert[None, :] < E),
            other=-float("inf"),
        ).to(tl.float32)
        # Empty/padded rows must not introduce NaNs into the block sum.
        logits = tl.where(token[:, None] < N, logits, 0.0)
        numerator = tl.exp(logits - tl.max(logits, axis=1)[:, None])
        probabilities = numerator / tl.sum(numerator, axis=1)[:, None]
        total = tl.sum(tl.where(valid[:, None], probabilities, 0.0), axis=0)
        old = tl.load(
            Cache + (block * L + layer) * E + expert, mask=expert < E, other=0.0
        )
        total += tl.where(slot % BS == 0, 0.0, old)
        tl.store(Cache + (block * L + layer) * E + expert, total, mask=expert < E)


def reduce_probabilities(logits, slots, cache, block_size):
    if (
        logits.ndim != 3
        or cache.ndim != 3
        or logits.shape[0] != cache.shape[1]
        or logits.shape[2] != cache.shape[2]
        or cache.dtype != torch.float32
        or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or slots.ndim != 1
        or slots.numel() < logits.shape[1]
        or slots.dtype not in (torch.int32, torch.int64)
        or block_size not in (16, 32, 64)
        or not logits.is_cuda
        or any(t.device != logits.device for t in (slots, cache))
        or any(not t.is_contiguous() for t in (logits, slots, cache))
    ):
        raise ValueError("Unsupported gate-probability cache geometry")
    layers, tokens, experts = logits.shape
    if tokens:
        _probability_blocks[(tokens, layers)](
            logits,
            slots,
            cache,
            tokens,
            layers,
            experts,
            block_size,
            triton.next_power_of_2(experts),
            num_warps=4,
        )
