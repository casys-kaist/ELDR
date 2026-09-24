# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in, step-owned signature transport through ModelRunnerOutput.

This transport replaces the per-request worker RPC, not the first-token
boundary. The signature becomes CPU bytes before the model result reaches the
scheduler. With async scheduling, the existing output-copy event covers both
tokens and signatures; preparation never waits for the GPU.
"""

import os

import torch

from vllm.eldr import signature

ENABLED = os.environ.get("ELDR") == "1"


def validate_config(config) -> None:
    """Fail closed outside the initially tested producer-only configuration."""
    if not ENABLED:
        return
    kv = config.kv_transfer_config
    parallel = config.parallel_config
    if not signature.ENABLED:
        raise ValueError("Signature and transport enablement disagree")
    if kv is None or kv.kv_role != "kv_producer" or kv.kv_connector != "NixlConnector":
        raise ValueError("Async signatures currently require a NIXL-only producer")
    if (
        parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or parallel.enable_expert_parallel
    ):
        raise ValueError("Async signatures currently require TP=PP=1 without EP")
    if (
        config.speculative_config is not None
        or config.model_config.runner_type != "generate"
    ):
        raise ValueError(
            "Async signatures currently require non-speculative generation"
        )
    if not config.model_config.enforce_eager:
        raise ValueError("Async signature transport requires eager model execution")
    if not 0 < config.cache_config.block_size < 128:
        raise ValueError("Signed-int8 block counts require block_size < 128")
    if config.model_config.max_model_len > 32767:
        raise ValueError("Signed-int16 signatures require max_model_len <= 32767")


def finishing_rows(batch, num_scheduled):
    """CPU-only selection; do not copy signatures for incomplete prefill chunks."""
    rows = []
    for rid, count in num_scheduled.items():
        if count <= 0:
            continue
        index = batch.req_id_to_index[rid]
        completed = int(batch.num_computed_tokens_cpu[index])
        prompt = int(batch.num_prompt_tokens[index])
        if completed + count > prompt:
            raise ValueError("Async signature producer must not execute decode tokens")
        if completed + count == prompt:
            rows.append((rid, index, prompt))
    return rows


class PendingSignatureCopy:
    """Own GPU and pinned host buffers until the enclosing output is resolved.

    No shared reusable buffers or request-ID cache: overlapping steps cannot
    overwrite one another, and an aborted request leaves no staged dictionary
    entry. GPU references are retained until the enclosing completion event.
    """

    def __init__(self, req_ids, counts):
        self.req_ids = tuple(req_ids)
        if counts.ndim != 3 or counts.shape[0] != len(self.req_ids):
            raise ValueError("Signature output shape disagrees with request IDs")
        expected = torch.float32 if signature.GATE_PROB else torch.int16
        if counts.dtype != expected:
            raise ValueError("Signature output dtype differs from the active capture")
        self.counts = counts
        self.host = None

    def enqueue_copy(self) -> None:
        if self.host is not None or self.counts is None:
            raise RuntimeError("Signature copy may only be enqueued once")
        self.host = torch.empty(
            self.counts.shape, dtype=self.counts.dtype, device="cpu", pin_memory=True
        )
        self.host.copy_(self.counts, non_blocking=True)

    def finish_after_sync(self) -> dict[str, bytes]:
        """Caller MUST first wait for the output event covering enqueue_copy."""
        if self.host is None or self.counts is None:
            raise RuntimeError("Signature copy is absent or already consumed")
        host = self.host.numpy()
        result = {
            rid: row.tobytes()
            for rid, row in zip(self.req_ids, host)
            if rid is not None
        }
        self.counts = None
        self.host = None
        return result

    def finish_synchronously(self) -> dict[str, bytes]:
        self.enqueue_copy()
        event = torch.cuda.Event()
        event.record()
        event.synchronize()
        return self.finish_after_sync()


def prepare(
    scheduler_output, batch, config, *, seq_lens=None, discard_mask=None
) -> PendingSignatureCopy | None:
    """Reduce on the compute stream; defer D2H until the normal output copy.

    Every scheduled token on this producer is a prefill token, even a one-token
    prompt or prefix-hit remainder. Never infer decode from batch shape here.
    The block cache still accumulates every chunk; only the final chunk is sent.
    """
    scheduled = scheduler_output.num_scheduled_tokens
    rows = finishing_rows(batch, scheduled)
    table = batch.block_table[0]
    block_size = config.cache_config.block_size
    if table.block_size != block_size or table.blocks_per_kv_block != 1:
        raise ValueError("Async signature transport does not support split KV blocks")
    signature.reduce(
        scheduler_output.total_num_scheduled_tokens,
        table.slot_mapping.gpu,
        block_size,
        config.cache_config.num_gpu_blocks,
    )
    if not rows:
        return None
    if signature._ESC is None:
        raise RuntimeError("Completed prefill has no expert-signature block cache")
    from vllm.eldr.batch_sum import sum_rows

    if seq_lens is None or discard_mask is None:
        raise ValueError("Batch sums require the committed GPU metadata")
    request_ids = [None] * len(batch.req_ids)
    for rid, index, prompt in rows:
        needed = (prompt + block_size - 1) // block_size
        if int(table.num_blocks_per_row[index]) < needed:
            raise RuntimeError("Completed prefill has an incomplete block table")
        request_ids[index] = rid
    counts = sum_rows(
        signature._ESC,
        table.block_table.gpu,
        seq_lens,
        discard_mask,
        len(request_ids),
        block_size,
    )
    return PendingSignatureCopy(request_ids, counts)
