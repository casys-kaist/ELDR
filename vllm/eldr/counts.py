# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact count-cache reduction; prompt length is not a JIT specialization key."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N"])
def _histogram_blocks(
    Experts,
    Slots,
    Cache,
    N,
    L: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    BS: tl.constexpr,
    PICKS: tl.constexpr,
    BINS: tl.constexpr,
):
    start = tl.program_id(0)
    layer = tl.program_id(1)
    slot = tl.load(Slots + start)
    block = slot // BS
    previous = tl.load(Slots + start - 1, mask=start > 0, other=-BS) // BS
    if (start == 0) | (block != previous):
        p = tl.arange(0, PICKS)
        token = start + p // K
        token_slot = tl.load(Slots + token, mask=token < N, other=-BS)
        valid = (p < BS * K) & (token < N) & (token_slot // BS == block)
        expert = tl.load(
            Experts + (layer * N + token) * K + p % K,
            mask=valid,
            other=BINS,
        ).to(tl.int32)
        histogram = tl.histogram(expert, BINS, mask=valid)
        e = tl.arange(0, BINS)
        old = tl.load(Cache + (block * L + layer) * E + e, mask=e < E, other=0).to(
            tl.int32
        )
        value = histogram + tl.where(slot % BS == 0, 0, old)
        tl.store(Cache + (block * L + layer) * E + e, value, mask=e < E)


def reduce_counts(experts, slots, cache, block_size):
    """GPU-only reduction. Same contiguous-block invariant as baseline reduce."""
    layers, tokens, topk = experts.shape
    if not experts.is_cuda or not slots.is_cuda or not cache.is_cuda:
        raise ValueError("Fast signature reduction requires GPU tensors")
    if (
        not experts.is_contiguous()
        or not slots.is_contiguous()
        or not cache.is_contiguous()
    ):
        raise ValueError("Fast signature reduction requires contiguous tensors")
    if not 0 < block_size < 128 or cache.dtype != torch.int8:
        raise ValueError("Expected exact signed-int8 per-block counts")
    if not tokens:
        return
    _histogram_blocks[(tokens, layers)](
        experts,
        slots,
        cache,
        tokens,
        layers,
        cache.shape[2],
        topk,
        block_size,
        triton.next_power_of_2(block_size * topk),
        triton.next_power_of_2(cache.shape[2]),
        num_warps=4,
    )
