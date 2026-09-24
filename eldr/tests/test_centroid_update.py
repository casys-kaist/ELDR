# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import unittest
from unittest.mock import patch

import numpy as np

from eldr.serving.centroid_update import (
    OnlineJSQRouter,
    _align_centroids,
)
from eldr.serving.clustering import build_signature
from eldr.serving.routing import route_jsq as route


def _unit(values) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class CentroidUpdateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0.0
        self._clock_patch = patch(
            "eldr.serving.centroid_update.time.monotonic",
            side_effect=lambda: self.now,
        )
        self._clock_patch.start()
        self.addCleanup(self._clock_patch.stop)

    def _advance(self, seconds: float) -> None:
        self.now += seconds

    def _router(self, workers, centroids, **kwargs) -> OnlineJSQRouter:
        router = OnlineJSQRouter(workers, centroids, **kwargs)
        self.addCleanup(router.close)
        return router

    def _finish_refit(self, router: OnlineJSQRouter) -> None:
        future = router._fit_future
        self.assertIsNotNone(future)
        future.result(timeout=5)
        router._poll_refit()
        self.assertIsNone(router._fit_future)

    def test_locality_band_then_jsq(self) -> None:
        centroids = np.asarray(
            [
                [1.0, 0.0],
                [0.995, 0.1],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        router = self._router(
            ["d0", "d1", "d2"],
            centroids,
            tau=0.1,
        )
        before = router._centroids.copy()

        # d0 and d1 are in the locality band. d1 wins by queue length; d2 is
        # outside the band even though its queue is equally short.
        self.assertEqual(router.route([1.0, 0.0], [3, 0, 0]), 1)
        np.testing.assert_array_equal(router._centroids, before)

    def test_alignment_preserves_decoder_identity(self) -> None:
        current = np.eye(2, dtype=np.float32)
        fitted = current[::-1].copy()

        aligned = _align_centroids(
            current,
            fitted,
        )

        np.testing.assert_array_equal(aligned, current)

    def test_rolling_balanced_refit_tracks_drift(self) -> None:
        router = self._router(
            ["d0", "d1"],
            np.eye(2, dtype=np.float32),
            window_seconds=2,
            refit_seconds=1,
        )
        upper_right = _unit([1.0, 1.0])
        upper_left = _unit([-1.0, 1.0])

        signatures = (
            upper_right,
            upper_left,
            upper_right,
            upper_left,
        )
        for index, signature in enumerate(signatures):
            if index:
                self._advance(2 / 3)
            router.route(signature, [0, 0])

        self.assertEqual(router.window_seconds, 2)
        self.assertEqual(router.refit_seconds, 1)
        self.assertEqual(router.refits_started, 1)
        self._finish_refit(router)

        self.assertEqual(router.refits_completed, 1)
        self.assertGreater(router._centroids[0, 1], 0.6)
        self.assertLess(router._centroids[1, 0], -0.6)
        np.testing.assert_allclose(
            np.linalg.norm(router._centroids, axis=1),
            np.ones(2),
            atol=1e-6,
        )

    def test_no_forced_balancing_before_refit(self) -> None:
        router = self._router(
            ["d0", "d1"],
            np.eye(2, dtype=np.float32),
            tau=0.1,
        )

        picks = [router.route([1.0, 0.0], [0, 0]) for _ in range(20)]
        self.assertEqual(picks, [0] * 20)
        self.assertEqual(router.refits_started, 0)

    def test_refit_cadence_uses_elapsed_time(self) -> None:
        router = self._router(
            ["d0", "d1"],
            np.eye(2, dtype=np.float32),
            window_seconds=2,
            refit_seconds=1,
        )

        for _ in range(20):
            router.route([1.0, 0.0], [0, 0])
        self._advance(1.99)
        for _ in range(20):
            router.route([0.0, 1.0], [0, 0])
        self.assertEqual(router.refits_started, 0)

        self._advance(0.01)
        router.route([0.0, 1.0], [0, 0])
        self.assertEqual(router.refits_started, 1)
        self._finish_refit(router)

        for _ in range(20):
            router.route([1.0, 0.0], [0, 0])
        self.assertEqual(router.refits_started, 1)
        self._advance(1)
        router.route([1.0, 0.0], [0, 0])
        self.assertEqual(router.refits_started, 2)

    def test_refit_excludes_expired_signatures(self) -> None:
        router = self._router(
            ["d0"],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            window_seconds=2,
            refit_seconds=1,
        )
        router.route([1.0, 0.0], [0])

        self._advance(2.01)
        router.route([0.0, 1.0], [0])
        self._finish_refit(router)

        np.testing.assert_allclose(
            router._centroids[0],
            np.asarray([0.0, 1.0], dtype=np.float32),
        )

    def test_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "window_seconds"):
            OnlineJSQRouter(
                ["d0"],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                window_seconds=0,
            )
        with self.assertRaisesRegex(ValueError, "refit_seconds"):
            OnlineJSQRouter(
                ["d0"],
                np.asarray([[1.0, 0.0]], dtype=np.float32),
                refit_seconds=0,
            )
        router = self._router(
            ["d0"],
            np.asarray([[1.0, 0.0]], dtype=np.float32),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            router.route([np.nan, 0.0], [0])

    def test_transformed_route_matches_static_policy(self) -> None:
        rng = np.random.default_rng(42)
        seeds = rng.random((16, 12), dtype=np.float32)
        seeds /= np.linalg.norm(seeds, axis=1, keepdims=True)
        router = self._router([f"d{i}" for i in range(16)], seeds)
        centroid_data = {"centroid_matrix": router._centroids, "L": 3, "e": 4}
        for _ in range(1000):
            raw = rng.integers(0, 100, (3, 4), dtype=np.int16)
            loads = rng.integers(0, 100, 16).tolist()
            signature = build_signature(centroid_data, raw)
            self.assertEqual(
                router.route_transformed(signature, loads),
                route(centroid_data, raw, loads, tau=0.1),
            )

    def test_transformed_route_transfers_ownership_without_normalizing(self) -> None:
        router = self._router(["d0", "d1"], np.eye(2, dtype=np.float32))
        signature = _unit([1, 2])
        with patch("numpy.linalg.norm", side_effect=AssertionError("redundant norm")):
            router.route_transformed(signature, [0, 0])
        self.assertIs(router._signatures[-1][1], signature)
        self.assertFalse(signature.flags.writeable)
        zero = np.zeros(2, dtype=np.float32)
        self.assertEqual(router.route_transformed(zero, [1, 0]), 1)
        self.assertEqual(len(router._signatures), 1)

    def test_public_route_preserves_caller_buffer_safety(self) -> None:
        router = self._router(["d0", "d1"], np.eye(2, dtype=np.float32))
        signature = np.asarray([3, 4], dtype=np.float32)
        router.route(signature, [0, 0])
        self.assertTrue(signature.flags.writeable)
        signature[:] = 0
        np.testing.assert_allclose(router._signatures[-1][1], [0.6, 0.8])

    def test_alignment_and_validation_run_on_background_thread(self) -> None:
        router = self._router(
            ["d0", "d1"], np.eye(2, dtype=np.float32), window_seconds=1
        )
        threads = []

        def align(*args):
            threads.append(threading.get_ident())
            return _align_centroids(*args)

        with patch("eldr.serving.centroid_update._align_centroids", side_effect=align):
            router.route([1, 0], [0, 0])
            self._advance(1)
            router.route([0, 1], [0, 0])
            future = router._fit_future
            assert future is not None
            fitted = future.result(timeout=5)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertFalse(fitted.flags.writeable)
        with patch(
            "eldr.serving.centroid_update._align_centroids",
            side_effect=AssertionError("request-path alignment"),
        ):
            router._poll_refit()
        self.assertIs(router._centroids, fitted)

    def test_failed_fit_keeps_current_centroids(self) -> None:
        router = self._router(
            ["d0"], np.asarray([[1, 0]], dtype=np.float32), window_seconds=1
        )
        original = router._centroids
        router.route([1, 0], [0])
        self._advance(1)
        with patch(
            "eldr.serving.centroid_update.balanced_spherical_kmeans",
            return_value=np.full((1, 2), np.nan, dtype=np.float32),
        ):
            router.route([0, 1], [0])
            future = router._fit_future
            assert future is not None
            with self.assertRaisesRegex(ValueError, "invalid centroid"):
                future.result(timeout=5)
        with self.assertLogs("eldr.serving.centroid_update", level="ERROR"):
            router._poll_refit()
        self.assertIs(router._centroids, original)
        self.assertEqual(router.refits_completed, 0)
        self.assertIn("invalid centroid", router.last_refit_error or "")
        self._advance(3)
        router.route([1, 0], [0])
        self._finish_refit(router)
        self.assertIsNone(router.last_refit_error)

    def test_refitter_is_warmed_before_first_request(self) -> None:
        threads = []
        with patch(
            "eldr.serving.centroid_update._warm_refitter",
            side_effect=lambda: threads.append(threading.get_ident()),
        ):
            router = self._router(["d0"], np.asarray([[1, 0]], dtype=np.float32))
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], threading.get_ident())
        self.assertEqual(router.refits_started, 0)

    def test_fast_path_rejects_invalid_inputs(self) -> None:
        router = self._router(["d0"], np.asarray([[1, 0]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "float32"):
            router.route_transformed(np.ones(2, dtype=np.float64), [0])
        with self.assertRaisesRegex(ValueError, "shape"):
            router.route_transformed(np.ones(3, dtype=np.float32), [0])
        with self.assertRaisesRegex(ValueError, "finite"):
            router.route_transformed(np.asarray([np.inf, 0], dtype=np.float32), [0])
        with self.assertRaisesRegex(ValueError, "loads length"):
            router.route_transformed(_unit([1, 0]), [])
        for tau in (np.nan, np.inf, -1):
            with self.assertRaisesRegex(ValueError, "tau"):
                self._router(["d0"], np.asarray([[1, 0]], dtype=np.float32), tau=tau)

    def test_fixed_fleet_rejects_missing_seeds(self):
        with self.assertRaisesRegex(ValueError, "one nonzero seed"):
            self._router(["d0", "d1", "d2"], np.eye(2, dtype=np.float32))
