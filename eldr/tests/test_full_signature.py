"""Full signature and online JSQ regressions; no GPU or private inputs."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pybase64
from aiohttp import web

from eldr.experiments.runner.fit import fit_centroids
from eldr.serving.centroid_update import OnlineJSQRouter, _run_refit
from eldr.serving.clustering import build_signature, load_centroid_file
from eldr.serving.proxy import (
    choose_decode,
    create_app,
    handle_admin_decoders,
    parse_args,
    routing_status,
)
from eldr.serving.routing import locality_candidates, route_jsq


def full_fit():
    return dict(
        L=2,
        e=3,
        k=2,
        variant="count_idf",
        sig_dtype="<i2",
        layer_mask=[1],
        sig_idf=[2.0, 0.0, 0.5],
        centroids=[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
    )


class FullSignatureTests(unittest.TestCase):
    def test_singleton_expansion_preserves_zero_tau_and_existing_bands(self):
        for scores, tau, expected in (
            ([0.9, 0.7, 0.2], 0, [0]),
            ([0.9, 0.7, 0.2], 0.1, [0, 1]),
            ([0.7, 0.9, 0.2], 0.1, [0, 1]),
            ([0.9, 0.85, 0.83, 0.2], 0.1, [0, 1, 2]),
            ([0.9, 0.7, 0.7], 0.1, [0, 1]),
            ([0.9, 0.9, 0.7], 0, [0, 1]),
            ([0, 0, 0], 0.1, [0, 1, 2]),
            ([0.9], 0.1, [0]),
        ):
            with self.subTest(scores=scores, tau=tau):
                np.testing.assert_array_equal(
                    locality_candidates(np.array(scores, dtype=np.float32), tau),
                    expected,
                )

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "fit.json"
        self.path.write_text(json.dumps(full_fit()))
        self.data = load_centroid_file(self.path)

    def test_packed_transform_is_the_measured_count_idf_mask_l2_formula(self):
        counts = np.array(
            [[[3, 5, 1], [1, 2, 8]], [[1, 1, 1], [0, 8, 0]]], dtype=np.int16
        )
        original = counts.copy()
        expected = counts[:, [1], :].astype(np.float32).reshape(2, -1)
        expected *= np.array([2, 0, 0.5], dtype=np.float32)
        expected /= np.clip(np.linalg.norm(expected, axis=1, keepdims=True), 1e-9, None)
        actual = build_signature(self.data, counts)
        np.testing.assert_array_equal(actual, expected)
        for row, vector in zip(counts, actual):
            np.testing.assert_array_equal(build_signature(self.data, row), vector)
        np.testing.assert_array_equal(counts, original)
        self.assertFalse(np.shares_memory(actual, counts))
        with self.assertRaisesRegex(ValueError, "geometry"):
            build_signature(self.data, counts[:, :1])

    def test_invalid_full_and_counts_metadata_are_rejected(self):
        mutations = (
            dict(layer_mask=[]),
            dict(layer_mask=[1, 1]),
            dict(layer_mask=[2]),
            dict(layer_mask=[True]),
            dict(sig_idf=[1, 2]),
            dict(sig_idf=[1, -1, 1]),
            dict(sig_idf=[0, 0, 0]),
            dict(sig_idf=[1, float("nan"), 1]),
            dict(sig_per_layer_norm=True),
            dict(sig_mean=[0, 0, 0]),
            dict(variant="count"),
            dict(sig_dtype="<f4"),
            dict(k=3),
            dict(centroids=[[1, 0], [0, 1]]),
            dict(centroids=[[2, 0, 0], [0, 0, 1]]),
        )
        for change in mutations:
            with self.subTest(change=change):
                self.path.write_text(json.dumps(full_fit() | change))
                with self.assertRaises(ValueError):
                    load_centroid_file(self.path)

    def test_online_jsq_matches_static_until_first_refit_and_uses_packed_vectors(self):
        router = OnlineJSQRouter(["one", "two"], self.data["centroid_matrix"])
        self.addCleanup(router.close)
        counts = np.random.default_rng(0).integers(0, 20, (32, 2, 3), dtype=np.int16)
        with patch("eldr.serving.centroid_update.time.monotonic", return_value=0):
            for row in counts:
                self.assertEqual(
                    router.route_transformed(build_signature(self.data, row), [2, 1]),
                    route_jsq(self.data, row, [2, 1]),
                )
        vectors = build_signature(self.data, counts)
        fitted = _run_refit(vectors, self.data["centroid_matrix"])
        self.assertEqual(fitted.shape, (2, 3))
        self.assertFalse(fitted.flags.writeable)
        np.testing.assert_allclose(np.linalg.norm(fitted, axis=1), 1, atol=1e-6)

    def test_formal_eldr_proxy_selects_jsq_without_starting_itl(self):
        argv = [
            "proxy",
            "--prefiller-hosts",
            "prefill",
            "--prefiller-ports",
            "8000",
            "--decoder-hosts",
            "one",
            "two",
            "--decoder-ports",
            "8001",
            "8002",
            "--decode-router",
            "eldr",
            "--eldr-centroids",
            str(self.path),
        ]
        with patch("sys.argv", argv):
            args = parse_args()
        with patch.dict("sys.modules", {"eldr.serving.itl_observer": None}):
            app = create_app(args)
        router = app["online_jsq_router"]
        self.addCleanup(router.close)
        self.assertNotIn("itl_observer", app)
        self.assertEqual((router.window_seconds, router.refit_seconds), (5.0, 2.5))
        app["decode_inflight"].update({"http://one:8001": 2, "http://two:8002": 1})
        counts = np.array([[7, 1, 0], [1, 3, 4]], dtype="<i2")
        payload = pybase64.b64encode(counts).decode()
        self.assertEqual(choose_decode(app, payload, {}), "http://two:8002")
        frozen = copy.deepcopy(self.data["layer_mask"])
        np.testing.assert_array_equal(app["centroid_data"]["sig_idf"], [2, 0, 0.5])
        self.assertEqual(app["centroid_data"]["layer_mask"], frozen)

        args.decode_router = "eldr-static"
        with patch("eldr.serving.proxy.OnlineJSQRouter") as updater:
            static_app = create_app(args)
        updater.assert_not_called()
        self.assertNotIn("online_jsq_router", static_app)
        static_app["decode_inflight"].update(
            {"http://one:8001": 2, "http://two:8002": 1}
        )
        before = static_app["centroid_data"]["centroid_matrix"].copy()
        self.assertEqual(choose_decode(static_app, payload, {}), "http://two:8002")
        np.testing.assert_array_equal(
            static_app["centroid_data"]["centroid_matrix"], before
        )

    def test_removed_policy_names_are_rejected_not_reinterpreted(self):
        for name in (
            "global-itl",
            "eldr-itl",
            "eldr-itl-static",
            "eldr-jsq",
            "eldr-jsq-online",
        ):
            argv = [
                "proxy",
                "--prefiller-hosts",
                "one",
                "--prefiller-ports",
                "8000",
                "--decoder-hosts",
                "two",
                "--decoder-ports",
                "8001",
                "--decode-router",
                name,
            ]
            with (
                self.subTest(name=name),
                patch("sys.argv", argv),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                with self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_refit_retains_frozen_transform_and_rejects_zero_signatures(self):
        source = self.path.parent / "counts.npy"
        counts = np.random.default_rng(0).integers(1, 20, (32, 2, 3), dtype=np.int16)
        np.save(source, counts)
        before = self.path.read_bytes()
        for balanced in (True, False):
            output = self.path.parent / f"fit-{balanced}.json"
            result = fit_centroids(
                source,
                output,
                "test",
                2,
                0,
                balanced=balanced,
                transform_from=self.path,
            )
            self.assertEqual(result["variant"], "count_idf")
            self.assertEqual(result["layer_mask"], [1])
            self.assertEqual(result["sig_idf"], [2, 0, 0.5])
            self.assertEqual(
                load_centroid_file(output)["centroid_matrix"].shape, (2, 3)
            )
        self.assertEqual(self.path.read_bytes(), before)
        counts[:, 1, [0, 2]] = 0
        np.save(source, counts)
        with self.assertRaisesRegex(ValueError, "zero under the frozen transform"):
            fit_centroids(
                source,
                self.path.parent / "invalid.json",
                "test",
                2,
                0,
                transform_from=self.path,
            )

    def test_jsq_ties_and_homogeneous_scores_remain_load_only(self):
        row = np.array([[1, 1, 1], [1, 3, 4]], dtype=np.int16)
        self.assertEqual(route_jsq(self.data, row, [3, 1]), 1)
        self.assertEqual(route_jsq(self.data, row, [1, 1]), 0)
        self.assertEqual(route_jsq(self.data, np.zeros_like(row), [3, 1]), 1)

    def test_pending_full_refit_never_blocks_or_starts_another_fit(self):
        router = OnlineJSQRouter(["one", "two"], self.data["centroid_matrix"])
        self.addCleanup(router.close)
        pending = Future()
        router._fit_future = pending
        counts = np.array([[1, 1, 1], [1, 3, 4]], dtype=np.int16)
        with patch.object(pending, "result", side_effect=AssertionError("fit waited")):
            for now in range(8):
                with patch(
                    "eldr.serving.centroid_update.time.monotonic", return_value=now
                ):
                    router.route_transformed(build_signature(self.data, counts), [3, 1])
        self.assertIs(router._fit_future, pending)
        self.assertEqual(router.refits_started, 0)


class JSQProxySafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_routing_status_reports_zero_tau_control_and_minimum_two(self):
        for tau, workers, minimum in ((0, 3, 1), (0.1, 3, 2), (0.1, 1, 1)):
            app = dict(
                decode=list(range(workers)),
                decode_inflight=dict.fromkeys(range(workers), 0),
                decode_router_name="eldr-static",
                centroid_data={"variant": "count_idf"},
                tau=tau,
            )
            status = json.loads((await routing_status(MagicMock(app=app))).text)
            self.assertEqual(status["minimum_candidates"], minimum)
            self.assertEqual(status["rule"], "locality_band_jsq_min2_v1")

    async def test_worker_mutation_is_rejected_before_state_change(self):
        app = dict(decode=["one", "two"], decode_inflight={"one": 0, "two": 0})
        before = copy.deepcopy(app)
        with self.assertRaises(web.HTTPBadRequest):
            await handle_admin_decoders(MagicMock(app=app, method="POST"))
        self.assertEqual(app, before)


if __name__ == "__main__":
    unittest.main()
