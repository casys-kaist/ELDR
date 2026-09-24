# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One native signature-sum launch over the already committed GPU metadata."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["TABLE_STRIDE", "ROWS"])
def _sum_rows(
    Cache,
    Table,
    SeqLens,
    Discard,
    Output,
    TABLE_STRIDE,
    ROWS,
    L: tl.constexpr,
    E: tl.constexpr,
    BS: tl.constexpr,
    BINS: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
    FLOAT: tl.constexpr,
):
    row, layer = tl.program_id(0), tl.program_id(1)
    final = ~tl.load(Discard + row, mask=row < ROWS, other=1).to(tl.int1)
    tokens = tl.load(SeqLens + row, mask=row < ROWS, other=0)
    count = tl.where(final, tl.cdiv(tokens, BS), 0)
    experts = tl.arange(0, BINS)
    offsets = tl.arange(0, BLOCK_TILE)
    total = tl.full((BINS,), 0, tl.float32 if FLOAT else tl.int32)
    for start in range(0, count, BLOCK_TILE):
        blocks = tl.load(
            Table + row * TABLE_STRIDE + start + offsets,
            mask=start + offsets < count,
            other=0,
        )
        values = tl.load(
            Cache + (blocks[:, None] * L + layer) * E + experts[None, :],
            mask=(start + offsets[:, None] < count) & (experts[None, :] < E),
            other=0,
        ).to(tl.float32 if FLOAT else tl.int32)
        total += tl.sum(values, axis=0)
    tl.store(Output + (row * L + layer) * E + experts, total, mask=experts < E)


def sum_rows(cache, table, seq_lens, discard, rows, block_size):
    """Caller owns metadata validity; this function launches without a host wait."""
    if (
        cache.ndim != 3
        or cache.dtype not in (torch.int8, torch.float32)
        or not cache.is_contiguous()
        or table.ndim != 2
        or table.stride(1) != 1
        or table.dtype not in (torch.int32, torch.int64)
        or not 0 < rows <= table.shape[0]
        or seq_lens.ndim != 1
        or discard.ndim != 1
        or seq_lens.numel() < rows
        or discard.numel() < rows
        or seq_lens.dtype not in (torch.int32, torch.int64)
        or discard.dtype != torch.bool
        or not seq_lens.is_contiguous()
        or not discard.is_contiguous()
        or not cache.is_cuda
        or any(t.device != cache.device for t in (table, seq_lens, discard))
    ):
        raise ValueError("Unsupported native batch-sum metadata")
    if not 0 < block_size < 128:
        raise ValueError("Signed block counts require block size below128")
    _, layers, experts = cache.shape
    output = torch.empty(
        (rows, layers, experts),
        dtype=torch.float32 if cache.dtype == torch.float32 else torch.int16,
        device=cache.device,
    )
    _sum_rows[(rows, layers)](
        cache,
        table,
        seq_lens,
        discard,
        output,
        table.stride(0),
        rows,
        layers,
        experts,
        block_size,
        triton.next_power_of_2(experts),
        8,
        cache.dtype == torch.float32,
    )
    return output
