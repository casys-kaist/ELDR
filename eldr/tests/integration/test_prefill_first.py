# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real vLLM request/scheduler/detokenizer tests in the pinned serving image."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import torch
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from vllm.eldr.prefill_first import (
    add_release_route,
    advance_seeded_generator,
    first_token_ids,
    validate,
)
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.request import Request, RequestStatus
from vllm.v1.sample.ops.topk_topp_sampler import random_sample


def tokenizer():
    backend = Tokenizer(
        models.BPE(
            {"x": 0, "Ã": 1, "©": 2, "a": 3, "b": 4, "c": 5, "!": 6, "<eos>": 7}, []
        )
    )
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")


def engine_request(params):
    return EngineCoreRequest(
        request_id="internal",
        external_req_id="external",
        prompt_token_ids=[0],
        sampling_params=params,
        pooling_params=None,
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def handoff(token=3):
    return dict(
        do_remote_prefill=True,
        prefill_first_token_id=token,
        prefill_prompt_tokens=1,
        prefill_continues=True,
    )


class PrefillFirstTests(unittest.TestCase):
    def test_ordinary_request_is_not_modified(self):
        params = SamplingParams(max_tokens=3)
        validate(object(), params, [0], None)
        request = Request.from_engine_core_request(engine_request(params), None)
        self.assertEqual(list(request.output_token_ids), [])
        self.assertEqual(list(request.all_token_ids), [0])

    def test_release_uses_preaborted_request_not_model_generation(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        engine = app.state.engine_client = NS(
            notify_kv_transfer_request_rejected=AsyncMock()
        )
        add_release_route(app)
        with TestClient(app) as client:
            self.assertEqual(
                client.post("/v1/kv_transfer/release", json={}).status_code, 400
            )
            transfer = handoff()
            response = client.post("/v1/kv_transfer/release", json=transfer)
            self.assertEqual(response.status_code, 200)
            engine.notify_kv_transfer_request_rejected.assert_awaited_once()
            self.assertEqual(
                engine.notify_kv_transfer_request_rejected.await_args.args[1], transfer
            )

    def test_generated_history_is_not_prompt_and_counts_toward_budget(self):
        params = SamplingParams(
            max_tokens=3, extra_args={"kv_transfer_params": handoff()}
        )
        request = Request.from_engine_core_request(engine_request(params), None)
        self.assertEqual(request.prompt_token_ids, [0])
        self.assertEqual(list(request.output_token_ids), [3])
        self.assertEqual(list(request.all_token_ids), [0, 3])
        self.assertEqual(request.num_prompt_tokens, 1)
        scheduler = NS(max_model_len=100)
        _, stopped = Scheduler._update_request_with_output(scheduler, request, [4])
        self.assertFalse(stopped)
        _, stopped = Scheduler._update_request_with_output(scheduler, request, [5])
        self.assertTrue(stopped)
        self.assertEqual(request.num_output_tokens, 3)

    def test_full_kv_hit_computes_seed_not_last_prompt_token(self):
        params = SamplingParams(
            max_tokens=3, extra_args={"kv_transfer_params": handoff()}
        )
        request = Request.from_engine_core_request(engine_request(params), None)
        request.num_computed_tokens = 1
        scheduler = NS(
            connector=object(),
            failed_recving_kv_req_ids=set(),
            finished_recving_kv_req_ids={request.request_id},
            kv_cache_manager=MagicMock(),
        )
        Scheduler._update_waiting_for_remote_kv(scheduler, request)
        self.assertEqual(request.num_computed_tokens, 1)
        self.assertEqual(request.all_token_ids[request.num_computed_tokens], 3)
        params.extra_args["kv_transfer_params"]["do_remote_prefill"] = False
        self.assertEqual(first_token_ids(params), [3])

    def test_producer_handoff_preserves_min_tokens_and_actual_stop(self):
        for minimum, budget, token, expected_continues in (
            (5, 8, 3, True),
            (0, 1, 3, False),
            (0, 8, 7, False),
        ):
            with self.subTest(minimum=minimum, budget=budget, token=token):
                params = SamplingParams(
                    max_tokens=budget,
                    min_tokens=minimum,
                    stop_token_ids=[7],
                    extra_args={
                        "kv_transfer_params": {
                            "do_remote_decode": True,
                            "prefill_first": True,
                        }
                    },
                )
                request = Request.from_engine_core_request(engine_request(params), None)
                _, stopped = Scheduler._update_request_with_output(
                    NS(max_model_len=100), request, [token]
                )
                self.assertTrue(stopped)
                self.assertEqual(params.min_tokens, minimum)
                self.assertEqual(params.max_tokens, budget)
                self.assertEqual(
                    request.kv_transfer_params.get("prefill_continues", False),
                    expected_continues,
                )
                if token == 7:
                    self.assertEqual(request.status, RequestStatus.FINISHED_STOPPED)

    def outputs(self, tokens, *, split, stop=None, minimum=0, stream_interval=1):
        params = SamplingParams(
            max_tokens=len(tokens),
            stop=stop,
            min_tokens=minimum,
            output_kind=RequestOutputKind.DELTA,
        )
        if not split:
            processor = OutputProcessor(
                tokenizer(), log_stats=False, stream_interval=stream_interval
            )
            processor.add_request(engine_request(params), "x")
            pieces = []
        else:
            # Produce one token and its safe (non-final) detokenized prefix.
            producer_params = params.clone()
            producer_params.output_kind = RequestOutputKind.FINAL_ONLY
            producer = OutputProcessor(tokenizer(), log_stats=False)
            producer.add_request(engine_request(producer_params), "x")
            transfer = handoff(tokens[0])
            output = (
                producer.process_outputs(
                    [
                        EngineCoreOutput(
                            request_id="internal",
                            new_token_ids=[tokens[0]],
                            finish_reason=FinishReason.LENGTH,
                            kv_transfer_params=transfer,
                        )
                    ]
                )
                .request_outputs[0]
                .outputs[0]
            )
            pieces = [output.text]
            if not transfer["prefill_continues"]:
                return pieces, output.finish_reason
            params.extra_args = {"kv_transfer_params": transfer}
            processor = OutputProcessor(
                tokenizer(), log_stats=False, stream_interval=stream_interval
            )
            processor.add_request(engine_request(params), "x")
        reason = None
        for index in range(1 if split else 0, len(tokens)):
            result = processor.process_outputs(
                [
                    EngineCoreOutput(
                        request_id="internal",
                        new_token_ids=[tokens[index]],
                        finish_reason=FinishReason.LENGTH
                        if index == len(tokens) - 1
                        else None,
                    )
                ]
            )
            for output in result.request_outputs:
                pieces.append(output.outputs[0].text)
                reason = output.outputs[0].finish_reason
            if reason is not None:
                break
        return pieces, reason

    def test_detokenization_utf8_stop_boundary_and_stream_interval(self):
        for tokens, stop, minimum in (
            ([3, 4, 5], None, 0),
            ([1, 2, 3], None, 0),  # UTF-8 character split across P/D.
            ([3, 4, 5], "ab", 0),  # Stop spans the handoff.
            ([3, 4, 5], "a", 0),  # Stop within first token.
            ([3, 4, 5], "ab", 2),  # Suppressed until min_tokens is met.
        ):
            for interval in (1, 2):
                with self.subTest(tokens=tokens, stop=stop, interval=interval):
                    reference, reason = self.outputs(
                        tokens,
                        split=False,
                        stop=stop,
                        minimum=minimum,
                        stream_interval=interval,
                    )
                    actual, actual_reason = self.outputs(
                        tokens,
                        split=True,
                        stop=stop,
                        minimum=minimum,
                        stream_interval=interval,
                    )
                    self.assertEqual("".join(actual), "".join(reference))
                    self.assertEqual(actual_reason, reason)

    def test_seeded_rng_continues_at_second_draw(self):
        for device in ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",):
            for size in (8, 129, 151936):
                with self.subTest(device=device, vocab=size):
                    reference = torch.Generator(device=device).manual_seed(58)
                    decoder = torch.Generator(device=device).manual_seed(58)
                    probs = torch.ones((1, size), dtype=torch.float32, device=device)
                    random_sample(probs.clone(), {0: reference})
                    advance_seeded_generator(decoder, size, device)
                    self.assertTrue(
                        torch.equal(reference.get_state(), decoder.get_state())
                    )
                    self.assertTrue(
                        torch.equal(
                            random_sample(probs.clone(), {0: reference}),
                            random_sample(probs.clone(), {0: decoder}),
                        )
                    )

    def test_handoff_validation_rejects_incompatible_modes(self):
        config = NS(
            kv_transfer_config=NS(kv_connector="NixlConnector"),
            speculative_config=None,
            parallel_config=NS(pipeline_parallel_size=1),
            model_config=NS(
                is_encoder_decoder=False,
                is_hybrid=False,
                logits_processors=None,
                get_vocab_size=lambda: 8,
            ),
        )
        params = SamplingParams(
            max_tokens=3, extra_args={"kv_transfer_params": handoff()}
        )
        validate(config, params, [0], None)
        for key, value in (
            ("prefill_first_token_id", 8),
            ("prefill_first_token_id", True),
            ("prefill_prompt_tokens", 2),
            ("prefill_continues", False),
        ):
            copy = params.clone()
            copy.extra_args["kv_transfer_params"][key] = value
            with self.assertRaises(ValueError):
                validate(config, copy, [0], None)


class BenchmarkFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_ttft_and_itl_use_the_same_first_token_timestamp(self):
        from vllm.benchmarks.lib.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )

        async def chunks():
            yield b'data: {"choices":[{"text":"first"}]}\n\n'
            yield b'data: {"choices":[{"text":"second"}]}\n\n'
            yield b'data: {"usage":{"completion_tokens":2}}\n\n'

        response = NS(status=200, content=NS(iter_any=chunks))
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=response)
        manager.__aexit__ = AsyncMock(return_value=False)
        with patch(
            "vllm.benchmarks.lib.endpoint_request_func.time.perf_counter",
            side_effect=[10.0, 10.1, 10.2],
        ):
            result = await async_request_openai_completions(
                RequestFuncInput(
                    prompt="x",
                    api_url="http://test/v1/completions",
                    prompt_len=1,
                    output_len=2,
                    model="test",
                ),
                NS(post=MagicMock(return_value=manager)),
            )
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.output_tokens, 2)
        self.assertAlmostEqual(result.ttft, 0.1)
        self.assertAlmostEqual(result.latency - result.ttft, sum(result.itl))

    async def test_error_after_first_token_is_failed_not_successful(self):
        from vllm.benchmarks.lib.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )

        async def chunks():
            yield b'data: {"choices":[{"text":"first"}]}\n\n'
            yield b'data: {"error":{"message":"decode failed"}}\n\n'

        response = NS(status=200, content=NS(iter_any=chunks))
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=response)
        manager.__aexit__ = AsyncMock(return_value=False)
        session = NS(post=MagicMock(return_value=manager))
        result = await async_request_openai_completions(
            RequestFuncInput(
                prompt="x",
                api_url="http://test/v1/completions",
                prompt_len=1,
                output_len=3,
                model="test",
            ),
            session,
        )
        self.assertFalse(result.success)
        self.assertIn("decode failed", result.error)


if __name__ == "__main__":
    unittest.main()
