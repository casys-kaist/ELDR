# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P/D forwarding regressions; no server, GPU, or network required."""

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web

from eldr.serving.proxy import handle, make_decode_router


def context(value):
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=value)
    manager.__aexit__ = AsyncMock(return_value=False)
    return manager


class ProxyStreamTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.body = dict(
            prompt="hello",
            max_tokens=3,
            stream=True,
            stream_options={"include_usage": True},
        )
        self.app = dict(
            prefill=["p"],
            decode=["d"],
            prefill_inflight={"p": 0},
            decode_inflight={"d": 0},
            decode_router_name="rr",
            prefill_router=make_decode_router("rr", ["p"], 0),
            decode_router=make_decode_router("rr", ["d"], 0),
        )
        self.prefill = MagicMock()
        self.prefill.json = AsyncMock(
            return_value={
                "id": "cmpl-test",
                "created": 123,
                "object": "text_completion",
                "model": "test",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
                "choices": [
                    {
                        "index": 0,
                        "text": "A",
                        "token_ids": [7],
                        "prompt_token_ids": [3, 4],
                        "finish_reason": "length",
                    }
                ],
                "kv_transfer_params": {
                    "remote_engine_id": "p",
                    "do_remote_prefill": True,
                    "prefill_first_token_id": 7,
                    "prefill_prompt_tokens": 2,
                    "prefill_continues": True,
                },
            }
        )
        self.chunks = [b'data: {"text":"BC"}\n\n', b"data: [DONE]\n\n"]

        async def stream():
            for chunk in self.chunks:
                yield chunk

        self.decode = MagicMock()
        self.decode.content.iter_any = stream
        self.decode.headers = {"Content-Type": "text/event-stream"}
        self.session = self.app["http"] = MagicMock()
        self.session.post.side_effect = [context(self.prefill), context(self.decode)]
        self.request = NS(
            app=self.app,
            path="/v1/completions",
            headers={},
            json=AsyncMock(return_value=self.body),
        )
        self.output = NS(
            prepare=AsyncMock(),
            write=AsyncMock(),
            write_eof=AsyncMock(),
            force_close=MagicMock(),
        )

    async def test_prefill_precedes_routing_and_decode_without_dropping_chunks(self):
        original = copy.deepcopy(self.body)

        def route(*_args):
            self.assertEqual(self.output.write.await_count, 1)
            self.assertEqual(self.session.post.call_count, 1)
            return "d"

        with (
            patch("eldr.serving.proxy.web.StreamResponse", return_value=self.output),
            patch("eldr.serving.proxy.choose_decode", side_effect=route),
        ):
            self.assertIs(await handle(self.request), self.output)
        events = [c.args[0] for c in self.output.write.await_args_list]
        first = json.loads(events[0].decode().removeprefix("data: "))
        self.assertEqual(first["choices"][0]["text"], "A")
        self.assertIsNone(first["choices"][0]["finish_reason"])
        self.assertNotIn("kv_transfer_params", first)
        self.assertNotIn("token_ids", first["choices"][0])
        self.assertEqual(events[1:], self.chunks)
        prefill, decode = self.session.post.call_args_list
        self.assertEqual(prefill.kwargs["json"]["max_tokens"], 3)
        self.assertTrue(prefill.kwargs["json"]["kv_transfer_params"]["prefill_first"])
        self.assertFalse(prefill.kwargs["json"]["stream"])
        self.assertNotIn("stream_options", prefill.kwargs["json"])
        self.assertEqual(decode.kwargs["json"]["max_tokens"], original["max_tokens"])
        self.assertEqual(decode.kwargs["json"]["prompt"], [3, 4])
        self.assertEqual(
            decode.kwargs["json"]["kv_transfer_params"]["prefill_first_token_id"], 7
        )
        self.assertEqual(self.body, original)
        self.assertEqual(
            decode.kwargs["json"]["stream_options"], original["stream_options"]
        )
        self.assertEqual(self.app["prefill_inflight"], {"p": 0})
        self.assertEqual(self.app["decode_inflight"], {"d": 0})

    async def test_prefill_failure_releases_load_and_never_decodes(self):
        self.prefill.raise_for_status.side_effect = RuntimeError("prefill failed")
        with self.assertRaisesRegex(RuntimeError, "prefill failed"):
            await handle(self.request)
        self.assertEqual(self.session.post.call_count, 1)
        self.assertEqual(self.app["prefill_inflight"], {"p": 0})

    async def test_decode_failure_after_first_token_is_not_a_completed_stream(self):
        self.decode.raise_for_status.side_effect = RuntimeError("decode failed")
        with (
            patch("eldr.serving.proxy.web.StreamResponse", return_value=self.output),
            self.assertRaisesRegex(RuntimeError, "decode failed"),
        ):
            await handle(self.request)
        self.output.prepare.assert_awaited_once()
        self.output.write_eof.assert_not_awaited()
        self.output.force_close.assert_called_once()
        last = self.output.write.await_args_list[-1].args[0]
        self.assertIn(b'"error"', last)
        self.assertEqual(self.app["decode_inflight"], {"d": 0})

    async def test_cancellation_releases_decode_load(self):
        self.output.write.side_effect = [None, asyncio.CancelledError()]
        with (
            patch("eldr.serving.proxy.web.StreamResponse", return_value=self.output),
            self.assertRaises(asyncio.CancelledError),
        ):
            await handle(self.request)
        self.assertEqual(self.app["decode_inflight"], {"d": 0})

    async def test_early_cancellation_releases_prefill_without_decode(self):
        self.output.write.side_effect = asyncio.CancelledError()
        with (
            patch("eldr.serving.proxy.web.StreamResponse", return_value=self.output),
            self.assertRaises(asyncio.CancelledError),
        ):
            await handle(self.request)
        self.assertEqual(
            self.session.post.call_args_list[1].args[0], "d/v1/kv_transfer/release"
        )
        self.assertEqual(self.app["decode_inflight"], {"d": 0})

    async def test_terminal_prefill_emits_usage_and_releases_without_generation(self):
        self.prefill.json.return_value["kv_transfer_params"]["prefill_continues"] = (
            False
        )
        for reason in ("length", "stop"):
            with self.subTest(reason=reason):
                self.session.post.reset_mock(side_effect=True)
                self.session.post.side_effect = [
                    context(self.prefill),
                    context(self.decode),
                ]
                self.output.write.reset_mock()
                self.prefill.json.return_value["choices"][0]["finish_reason"] = reason
                with patch(
                    "eldr.serving.proxy.web.StreamResponse", return_value=self.output
                ):
                    await handle(self.request)
                events = [c.args[0] for c in self.output.write.await_args_list]
                self.assertEqual(len(events), 3)
                self.assertIn(b'"completion_tokens": 1', events[1])
                self.assertEqual(events[-1], b"data: [DONE]\n\n")
                self.assertEqual(
                    self.session.post.call_args_list[1].args[0],
                    "d/v1/kv_transfer/release",
                )

    async def test_nonstream_decoder_returns_complete_history_once(self):
        self.body["stream"] = False
        self.decode.json = AsyncMock(return_value={"choices": [{"text": "ABC"}]})
        result = await handle(self.request)
        self.assertEqual(json.loads(result.body), {"choices": [{"text": "ABC"}]})

    async def test_unsupported_options_rejected_before_prefill(self):
        for option in (
            {"n": 2},
            {"echo": True},
            {"logprobs": 0},
            {"prompt": ["a", "b"]},
            {"max_tokens": 0},
        ):
            with self.subTest(option=option):
                self.request.json.return_value = {**self.body, **option}
                with self.assertRaises(web.HTTPBadRequest):
                    await handle(self.request)
        self.session.post.assert_not_called()

    async def test_missing_handoff_fails_instead_of_local_prefill(self):
        self.prefill.json.return_value = {}
        with self.assertRaises(web.HTTPBadGateway):
            await handle(self.request)
        self.assertEqual(self.session.post.call_count, 1)
