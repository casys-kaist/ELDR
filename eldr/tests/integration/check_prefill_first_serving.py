# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Owned-worker correctness check, not a performance measurement.

Run from the repository root with a smoke config and a fresh output directory:
  .venv/bin/python -m eldr.tests.integration.check_prefill_first_serving \
      --config /path/config.json --output /path/new-check
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp

from eldr.experiments.runner.config import load_config
from eldr.experiments.runner.run import drain, endpoint, prime, reset, start_proxy
from eldr.experiments.runner.workers import WorkerGroup, prepare, write_new_json


async def complete(session, url, payload):
    started = time.perf_counter()
    async with session.post(url + "/v1/completions", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        if not payload.get("stream"):
            result = await response.json()
            if "error" in result:
                raise RuntimeError(result)
            return result
        tokens, texts, gaps = [], [], []
        first_time = previous = None
        usage = None
        done = False
        reason = None
        async for raw in response.content:
            if not raw.startswith(b"data: "):
                continue
            raw = raw[6:].strip()
            if raw == b"[DONE]":
                done = True
                continue
            item = json.loads(raw)
            if "error" in item:
                raise RuntimeError(item)
            if item.get("choices"):
                choice = item["choices"][0]
                now = time.perf_counter()
                if first_time is None:
                    first_time = now
                else:
                    gaps.append(now - previous)
                previous = now
                tokens.extend(choice.get("token_ids") or [])
                texts.append(choice["text"])
                reason = choice.get("finish_reason")
            if item.get("usage"):
                usage = item["usage"]
        if not done or reason is None or usage is None:
            raise AssertionError("Incomplete stream/usage/termination")
        assert usage["completion_tokens"] == len(tokens)
        return dict(
            choices=[dict(token_ids=tokens, text="".join(texts), finish_reason=reason)],
            usage=usage,
            ttft_ms=(first_time - started) * 1000,
            gaps_ms=[gap * 1000 for gap in gaps],
        )


async def check(session, prefill_url, decoder, proxy, policy):
    base = dict(
        model="eldr",
        prompt="The capital of France is",
        max_tokens=16,
        temperature=0,
        ignore_eos=True,
        return_token_ids=True,
    )
    rows = []
    # Compare actual generated IDs/text to a complete, unsplit D execution.
    cases = [
        ("greedy", {}),
        ("seeded", dict(seed=58, temperature=0.8, top_p=0.9)),
        (
            "penalties",
            dict(
                seed=73,
                temperature=0.8,
                repetition_penalty=1.1,
                frequency_penalty=0.2,
                presence_penalty=0.1,
            ),
        ),
        ("minimum", dict(min_tokens=8, ignore_eos=False)),
        ("one_token", dict(max_tokens=1)),
    ]
    for name, changes in cases:
        payload = {**base, **changes}
        reference = await complete(session, decoder, payload)
        streamed = await complete(
            session,
            proxy,
            {**payload, "stream": True, "stream_options": {"include_usage": True}},
        )
        actual = streamed["choices"][0]
        expected = reference["choices"][0]
        whole = (await complete(session, proxy, payload))["choices"][0]
        row = dict(
            policy=policy,
            case=name,
            expected=expected,
            actual=actual,
            ttft_ms=streamed["ttft_ms"],
            gaps_ms=streamed["gaps_ms"],
            usage=streamed["usage"],
            unsplit_match=actual["token_ids"] == expected["token_ids"],
            nonstream=whole,
        )
        rows.append(row)
        # Streaming must preserve the same P/D generation as a whole response.
        # Retain the unsplit comparison too: a seed alone does not guarantee
        # bitwise identity across eager and compiled execution layouts.
        assert actual["token_ids"] == whole["token_ids"], row
        assert actual["text"] == whole["text"], row
        assert actual["finish_reason"] == whole["finish_reason"], row
        if payload["temperature"] == 0:
            assert row["unsplit_match"], row
        assert streamed["usage"]["prompt_tokens"] == reference["usage"]["prompt_tokens"]
        print(
            json.dumps({k: row[k] for k in ("policy", "case", "ttft_ms")}), flush=True
        )

    # Exercise a real stop at T0 and a stop crossing the P/D boundary using
    # this model's own greedy text, without assuming its tokenizer vocabulary.
    reference = await complete(session, decoder, base)
    generated = reference["choices"][0]["text"]
    if generated:
        stop = generated[: min(8, len(generated))]
        expected = await complete(session, decoder, {**base, "stop": stop})
        actual = await complete(
            session,
            proxy,
            {
                **base,
                "stop": stop,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert actual["choices"][0] == {
            k: expected["choices"][0][k] for k in ("token_ids", "text", "finish_reason")
        }
        rows.append(dict(policy=policy, case="stop", result=actual))

    # First-token EOS and stop-token cases must release KV without decoding.
    first = reference["choices"][0]["token_ids"][0]
    for options in (dict(stop_token_ids=[first], ignore_eos=False),):
        actual = await complete(
            session,
            proxy,
            {
                **base,
                **options,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert actual["usage"]["completion_tokens"] == 1
        assert actual["choices"][0]["finish_reason"] == "stop"
        rows.append(dict(policy=policy, case="stop_token", result=actual))
    nonstream = await complete(session, proxy, base)
    assert nonstream["choices"][0]["token_ids"] == reference["choices"][0]["token_ids"]
    assert nonstream["choices"][0]["text"] == generated
    rows.append(dict(policy=policy, case="nonstream", result=nonstream))
    # Concurrent repeated prompts exercise async scheduling. These short
    # prompts do not guarantee full local prefix-cache block hits.
    concurrent = await asyncio.gather(
        *(
            complete(
                session,
                proxy,
                {**base, "stream": True, "stream_options": {"include_usage": True}},
            )
            for _ in range(4)
        )
    )
    for actual in concurrent:
        assert actual["choices"][0]["token_ids"] == reference["choices"][0]["token_ids"]
    rows.append(
        dict(
            policy=policy, case="concurrent_repeated_prompts", requests=len(concurrent)
        )
    )
    # Independent conditional oracle: sample T0 on P, then greedily continue
    # from its KV+generated history on D. Without penalties/stops, this suffix
    # must match D's normal forward on prompt IDs + T0. The extended prompt is
    # only a test oracle, never the actual proxy's implementation.
    sampled = await complete(
        session,
        prefill_url,
        {
            **base,
            "seed": 58,
            "temperature": 0.8,
            "top_p": 0.9,
            "kv_transfer_params": {"do_remote_decode": True, "prefill_first": True},
        },
    )
    transfer = sampled["kv_transfer_params"]
    transfer.pop("eldr_sig", None)
    token = transfer["prefill_first_token_id"]
    prompt = sampled["choices"][0]["prompt_token_ids"]
    continued = await complete(
        session,
        decoder,
        {
            **base,
            "prompt": prompt,
            "add_special_tokens": False,
            "kv_transfer_params": transfer,
        },
    )
    oracle = await complete(
        session,
        decoder,
        {
            **base,
            "prompt": prompt + [token],
            "max_tokens": base["max_tokens"] - 1,
            "add_special_tokens": False,
        },
    )
    continuation_ids = continued["choices"][0]["token_ids"]
    expected_ids = [token] + oracle["choices"][0]["token_ids"]
    assert continuation_ids == expected_ids, (continued, oracle, sampled)
    rows.append(
        dict(
            policy=policy,
            case="conditional_token_oracle",
            sampled_first=token,
            actual=continuation_ids,
            expected=expected_ids,
        )
    )
    return rows


async def diagnose_first_token(session, prefill, decoder):
    """No handoff flag or remote KV: isolate ordinary P/D execution differences."""
    rows = []
    for name, url in (("prefill", prefill), ("decode", decoder)):
        for budget in (1, 16):
            for seed in (58, 73):
                payload = dict(
                    model="eldr",
                    prompt="The capital of France is",
                    max_tokens=budget,
                    temperature=0.8,
                    top_p=0.9,
                    seed=seed,
                    ignore_eos=True,
                    return_token_ids=True,
                    logprobs=20,
                )
                result = await complete(session, url, payload)
                row = dict(worker=name, budget=budget, seed=seed, result=result)
                rows.append(row)
                print(
                    json.dumps(
                        dict(
                            worker=name,
                            budget=budget,
                            seed=seed,
                            first=result["choices"][0]["token_ids"][0],
                        )
                    ),
                    flush=True,
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnose-first-token", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config, profile="smoke")
    prepare(config, args.output, "smoke")
    worker_group = WorkerGroup(args.output)
    rows = []
    try:
        write_new_json(args.output / "preflight.json", worker_group.preflight())
        print("Starting pinned 1P/1D correctness workers", flush=True)
        worker_group.start_engines()
        prime(worker_group)
        if args.diagnose_first_token:
            reset(worker_group, "off")

            async def diagnose():
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as session:
                    return await diagnose_first_token(
                        session,
                        endpoint(config["prefills"][0]),
                        endpoint(config["decoders"][0]),
                    )

            rows = asyncio.run(diagnose())
            write_new_json(args.output / "diagnostic.json", rows)
            # This is evidence for classification, not a passed equivalence check.
            return
        for policy in ("rr", "jsq", "eldr"):
            reset(worker_group, "on" if policy == "eldr" else "off")
            entry, url = start_proxy(worker_group, policy, "correctness")

            async def run(url=url, policy=policy):
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as session:
                    return await check(
                        session,
                        endpoint(config["prefills"][0]),
                        endpoint(config["decoders"][0]),
                        url,
                        policy,
                    )

            result = asyncio.run(run())
            rows.extend(result)
            write_new_json(args.output / (policy + ".json"), result)
            drain(worker_group)
            worker_group.stop(entry)
        write_new_json(args.output / "results.json", rows)
    except BaseException as error:
        write_new_json(
            args.output / "failure.json", {"error": repr(error), "completed": rows}
        )
        raise
    finally:
        worker_group.cleanup()
    write_new_json(
        args.output / "complete.json",
        {
            "checks": len(rows),
            "cleanup": "complete",
            "unsplit_bitwise_mismatches": [
                {"policy": row["policy"], "case": row["case"]}
                for row in rows
                if row.get("unsplit_match") is False
            ],
        },
    )


if __name__ == "__main__":
    main()
