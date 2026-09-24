# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-token NIXL handoff, independent of the decode routing policy.

The prompt is never extended: the prefiller's sampled token is generated
history, not input. Ordinary (non-handoff) vLLM requests are unchanged.
"""


def transfer_params(params):
    return (params.extra_args or {}).get("kv_transfer_params", {}) if params else {}


def first_token_ids(params):
    transfer = transfer_params(params)
    token = transfer.get("prefill_first_token_id")
    # NIXL clears do_remote_prefill after allocation; the generated history
    # must remain available when the worker is first scheduled afterward.
    return [token] if token is not None else []


def validate(config, params, prompt_token_ids, prompt_embeds):
    transfer = transfer_params(params)
    if not (transfer.get("prefill_first") or first_token_ids(params)):
        return
    from vllm import envs

    if (
        config.kv_transfer_config is None
        or config.kv_transfer_config.kv_connector != "NixlConnector"
        or config.speculative_config is not None
        or config.parallel_config.pipeline_parallel_size != 1
        or envs.VLLM_USE_V2_MODEL_RUNNER
        or config.model_config.is_encoder_decoder
        or config.model_config.is_hybrid
        or prompt_embeds is not None
        or not prompt_token_ids
    ):
        raise ValueError(
            "Prefill-first requires text-only V1 NIXL without PP/speculation"
        )
    if (
        params.n != 1
        or params.logprobs is not None
        or params.prompt_logprobs is not None
        or params.structured_outputs is not None
        or config.model_config.logits_processors
        or params.thinking_token_budget is not None
    ):
        raise ValueError(
            "Prefill-first supports one plain-text completion without logprobs"
        )
    if transfer.get("prefill_first") and not transfer.get("do_remote_decode"):
        raise ValueError("prefill_first is a producer-only parameter")
    if tokens := first_token_ids(params):
        token = tokens[0]
        if not transfer.get("do_remote_prefill") or transfer.get("prefill_first"):
            raise ValueError("A seeded first token is a consumer-only parameter")
        if (
            type(token) is not int
            or not 0 <= token < config.model_config.get_vocab_size()
        ):
            raise ValueError("Invalid prefill first token")
        if transfer.get("prefill_prompt_tokens") != len(prompt_token_ids):
            raise ValueError("Prefill/decode prompt length mismatch")
        if not transfer.get("prefill_continues") or params.max_tokens <= 1:
            raise ValueError("A completed prefill must not start decode generation")


def prime_detokenizer(detokenizer, params):
    """Keep stop-string/UTF-8 state while suppressing only already-sent text."""
    tokens = first_token_ids(params)
    if tokens:
        if detokenizer.update(tokens, False) is not None:
            raise ValueError("A stopped prefill must not start decode generation")
        detokenizer.get_next_output_text(finished=False, delta=True)
    return len(tokens)


def advance_seeded_generator(generator, vocab_size, device):
    """Consume exactly the native sampler's first draw, already used on P.

    Seeded requests use the native exponential sampler on our CUDA/ROCm V1
    path. Replaying its draw avoids guessing a platform-dependent RNG offset.
    """
    import torch

    torch.empty(vocab_size, dtype=torch.float32, device=device).exponential_(
        generator=generator
    )


def add_release_route(app):
    # Same /v1 authentication middleware as the generation endpoints. This
    # uses upstream's pre-aborted-request cleanup, never a dummy model forward.
    from fastapi import HTTPException, Request

    from vllm.utils import random_uuid

    async def release(request: Request):
        transfer = await request.json()
        if not isinstance(transfer, dict) or not transfer.get("do_remote_prefill"):
            raise HTTPException(400, "Expected a pending prefill KV transfer")
        await app.state.engine_client.notify_kv_transfer_request_rejected(
            "prefill-release-" + random_uuid(), transfer
        )
        return {"released": True}

    app.post("/v1/kv_transfer/release")(release)
