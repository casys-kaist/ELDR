# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HTTP helpers for single-completion requests with prefill-first delivery."""

import json

from aiohttp import web


def validate_request(path, payload):
    prompt = payload.get("prompt")
    if path != "/v1/completions" or not (
        isinstance(prompt, str)
        or (isinstance(prompt, list) and prompt and all(type(t) is int for t in prompt))
    ):
        raise web.HTTPBadRequest(reason="Prefill-first supports one text completion")
    unsupported = (
        "echo",
        "use_beam_search",
        "prompt_embeds",
        "suffix",
        "structured_outputs",
        "kv_transfer_params",
        "thinking_token_budget",
    )
    if (
        any(payload.get(key) for key in unsupported)
        or payload.get("n", 1) != 1
        or payload.get("best_of", 1) != 1
        or payload.get("logprobs") is not None
        or payload.get("prompt_logprobs") is not None
        or payload.get("response_format", {"type": "text"})
        not in (None, {"type": "text"})
        or (payload.get("max_tokens") is not None and payload["max_tokens"] < 1)
    ):
        raise web.HTTPBadRequest(reason="Unsupported prefill-first completion option")


def unpack_prefill(result):
    transfer = result.get("kv_transfer_params")
    choices = result.get("choices", [])
    if not isinstance(transfer, dict) or len(choices) != 1:
        raise web.HTTPBadGateway(reason="Missing prefill-first handoff")
    choice = choices[0]
    token = transfer.get("prefill_first_token_id")
    prompt = choice.get("prompt_token_ids")
    if (
        type(token) is not int
        or choice.get("token_ids") != [token]
        or not isinstance(prompt, list)
        or len(prompt) != transfer.get("prefill_prompt_tokens")
        or type(transfer.get("prefill_continues")) is not bool
    ):
        raise web.HTTPBadGateway(reason="Invalid prefill-first token history")
    return transfer, prompt


def public_prefill(result, payload, continues):
    # Only envelope fields change; do not copy a potentially long prompt-ID
    # list on the first-token critical path.
    result = {
        key: value for key, value in result.items() if key != "kv_transfer_params"
    }
    choice = dict(result["choices"][0])
    result["choices"] = [choice]
    if not payload.get("return_token_ids"):
        choice.pop("token_ids", None)
        choice.pop("prompt_token_ids", None)
    if continues:
        choice["finish_reason"] = None
        choice["stop_reason"] = None
    return result


def event(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


async def send_prefill(out, result, payload, continues):
    first = public_prefill(result, payload, continues)
    options = payload.get("stream_options") or {}
    usage = first.pop("usage", None)
    if options.get("continuous_usage_stats"):
        first["usage"] = usage
    await out.write(event(first))
    if not continues:
        if options.get("include_usage"):
            await out.write(event({**first, "choices": [], "usage": usage}))
        await out.write(b"data: [DONE]\n\n")


async def release_prefill(session, decoder, transfer):
    async with session.post(
        f"{decoder}/v1/kv_transfer/release", json=transfer
    ) as response:
        response.raise_for_status()
