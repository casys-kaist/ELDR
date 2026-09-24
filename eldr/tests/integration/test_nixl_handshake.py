# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only real-ZMQ regression tests for the source-bound NIXL listener.

Load only the actual static method by AST to avoid importing GPU engines.
Run directly with a venv containing pyzmq/msgspec, or through unittest.
"""

import ast
import contextlib
import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import msgspec
import zmq

ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py"
GET_META_MSG = b"get_meta_msg"


class Listener:
    """One listener-owned socket, bounded waits, no reconnects or GPU access."""

    def __init__(self, source=SCHEDULER):
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error = None
        self.endpoint = None
        self.logger = Mock()
        self.metadata = {0: b"exact-rank-zero-metadata", 2: b"exact-rank-two-metadata"}
        cls = next(
            n
            for n in ast.parse(source.read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "NixlConnectorScheduler"
        )
        method = next(
            n for n in cls.body if getattr(n, "name", "") == "_nixl_handshake_listener"
        )
        method.decorator_list = []
        scope = dict(
            Any=Any,
            threading=threading,
            msgspec=msgspec,
            zmq=zmq,
            envs=SimpleNamespace(VLLM_NIXL_SIDE_CHANNEL_HOST="127.0.0.1"),
            make_zmq_path=lambda *_args: "tcp://127.0.0.1:*",
            zmq_ctx=self.socket_context,
            GET_META_MSG=GET_META_MSG,
            logger=self.logger,
        )
        exec(
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
            scope,
        )

        def target():
            try:
                scope[method.name](self.metadata, self.ready, self.stop, 0)
            except BaseException as exc:
                self.error = exc
                self.ready.set()

        self.thread = threading.Thread(target=target, daemon=True)

    @contextlib.contextmanager
    def socket_context(self, socket_type, path):
        with zmq.Context() as context, context.socket(socket_type) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.bind(path)
            self.endpoint = socket.getsockopt_string(zmq.LAST_ENDPOINT)
            yield socket

    def __enter__(self):
        self.thread.start()
        if not self.ready.wait(5) or self.error is not None:
            raise RuntimeError("Listener did not start") from self.error
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(3)
        if self.thread.is_alive():
            raise RuntimeError("Listener did not stop within its receive timeout")

    @contextlib.contextmanager
    def client(self, kind=zmq.REQ, *, correlate=False, probe=False):
        with zmq.Context() as context, context.socket(kind) as socket:
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, 2000)
            socket.setsockopt(zmq.SNDTIMEO, 2000)
            socket.setsockopt(zmq.RECONNECT_IVL, -1)
            if correlate:
                socket.setsockopt(zmq.REQ_CORRELATE, 1)
            if probe:
                socket.setsockopt(zmq.PROBE_ROUTER, 1)
            socket.connect(self.endpoint)
            yield socket

    def query(self, rank=0, **kwargs):
        with self.client(**kwargs) as socket:
            socket.send(msgspec.msgpack.encode((GET_META_MSG, rank)))
            return socket.recv()


class NixlHandshakeTests(unittest.TestCase):
    def test_standard_and_correlated_req_preserve_exact_metadata(self):
        with Listener() as listener:
            for correlate in (False, True):
                for rank in (0, 2):
                    for _ in range(3):
                        self.assertEqual(
                            listener.query(rank, correlate=correlate),
                            listener.metadata[rank],
                        )
            self.assertIsNone(listener.error)
            listener.logger.warning.assert_not_called()

    def test_empty_connection_probe_does_not_kill_listener(self):
        with Listener() as listener:
            self.assertEqual(listener.query(probe=True), listener.metadata[0])
            self.assertEqual(listener.query(), listener.metadata[0])
            self.assertIsNone(listener.error)
            listener.logger.warning.assert_not_called()

    def test_bad_envelopes_payloads_commands_and_ranks_then_valid_request(self):
        valid = msgspec.msgpack.encode((GET_META_MSG, 0))
        malformed = [
            [b"not-a-request"],  # ROUTER receives two frames: previous crash.
            [b"", valid, b"unexpected-extra-frame"],  # Previous >3 crash.
            [b"bad-delimiter", valid],
            [b"", b"\xc1"],  # Reserved/invalid MessagePack byte.
            [b"", msgspec.msgpack.encode(None)],
            [b"", msgspec.msgpack.encode([GET_META_MSG])],
            [b"", msgspec.msgpack.encode([b"other-command", 0])],
            [b"", msgspec.msgpack.encode([GET_META_MSG, 1])],
            [b"", msgspec.msgpack.encode([GET_META_MSG, -1])],
            [b"", msgspec.msgpack.encode([GET_META_MSG, True])],
            [b"", msgspec.msgpack.encode([GET_META_MSG, 0.0])],
            [b"", msgspec.msgpack.encode([GET_META_MSG, []])],
        ]
        with Listener() as listener, listener.client(zmq.DEALER) as socket:
            for frames in malformed:
                with self.subTest(frame_sizes=list(map(len, frames))):
                    # Same peer/socket preserves ordering. A valid query sent
                    # after the bad message must still get its exact response;
                    # no reply is invented for the rejected request itself.
                    socket.send_multipart(frames)
                    socket.send_multipart([b"", valid])
                    self.assertEqual(
                        socket.recv_multipart(), [b"", listener.metadata[0]]
                    )
                    self.assertIsNone(listener.error)
            warnings = listener.logger.warning.call_args_list
            self.assertEqual([call.args[1] for call in warnings], [1, 2, 4, 8])
            for call in warnings:
                self.assertIn("127.0.0.1", call.args[2])
                self.assertIsInstance(call.args[3], list)
                self.assertNotIn("not-a-request", str(call.args))

    def test_valid_requests_from_multiple_independent_peers(self):
        failures = []
        with Listener() as listener:

            def client(rank, correlate):
                try:
                    for _ in range(8):
                        if (
                            listener.query(rank, correlate=correlate)
                            != listener.metadata[rank]
                        ):
                            raise AssertionError("Metadata was sent to the wrong peer")
                except BaseException as exc:
                    failures.append(exc)

            threads = [
                threading.Thread(target=client, args=(rank, correlate), daemon=True)
                for rank in (0, 2)
                for correlate in (False, True)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
                self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertIsNone(listener.error)

    def test_idle_shutdown_is_bounded(self):
        with Listener() as listener:
            self.assertTrue(listener.thread.is_alive())
        self.assertIsNone(listener.error)
        self.assertFalse(listener.thread.is_alive())

    @unittest.skipUnless(
        os.environ.get("ELDR_NIXL_ORIGINAL_SCHEDULER"),
        "Optional original-image counterfactual",
    )
    def test_original_image_reproduces_both_observed_frame_count_crashes(self):
        original = Path(os.environ["ELDR_NIXL_ORIGINAL_SCHEDULER"])
        valid = msgspec.msgpack.encode((GET_META_MSG, 0))
        for frames, expected in (
            ([b"invalid"], "not enough values to unpack"),
            ([b"", valid, b"extra"], "too many values to unpack"),
        ):
            with Listener(original) as listener, listener.client(zmq.DEALER) as socket:
                socket.send_multipart(frames)
                listener.thread.join(3)
                self.assertFalse(listener.thread.is_alive())
                self.assertIsInstance(listener.error, ValueError)
                self.assertIn(expected, str(listener.error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
