# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Background balanced centroid updates for locality-band JSQ.

Recent normalized signature vectors form a rolling window. One fit at a time runs off
the request thread; completed, decoder-aligned centroids are installed there.
The decoder group is fixed. This is workload adaptation, not elasticity.
"""

import logging
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from eldr.serving.clustering import balanced_spherical_kmeans
from eldr.serving.routing import locality_candidates

WINDOW_SECONDS = 5.0
REFIT_SECONDS = 2.5
_EPS = 1e-9
_LOG = logging.getLogger(__name__)


def _run_refit(
    signature_rows: Sequence[np.ndarray],
    centroids: np.ndarray,
) -> np.ndarray:
    signatures = np.stack(signature_rows).astype(
        np.float32,
        copy=False,
    )
    fitted = balanced_spherical_kmeans(
        signatures,
        centroids,
    ).astype(
        np.float32,
        copy=False,
    )
    if fitted.shape != centroids.shape or not np.isfinite(fitted).all():
        raise ValueError(
            f"invalid centroid shape/value: {fitted.shape}, expected {centroids.shape}"
        )
    aligned = _align_centroids(centroids, fitted)
    aligned.flags.writeable = False
    return aligned


def _align_centroids(
    current: np.ndarray,
    fitted: np.ndarray,
) -> np.ndarray:
    """Align fitted centroids to existing decoder identities."""
    from scipy.optimize import linear_sum_assignment

    cost = -(current @ fitted.T)
    rows, columns = linear_sum_assignment(cost)
    aligned = np.empty_like(fitted)
    aligned[rows] = fitted[columns]
    return aligned


def _warm_refitter() -> None:
    """Pay thread startup and the lazy SciPy import before accepting requests."""
    from scipy.optimize import linear_sum_assignment

    linear_sum_assignment(np.zeros((1, 1), dtype=np.float32))


class CentroidUpdater:
    def __init__(
        self,
        seed_centroids,
        *,
        window_seconds: float = WINDOW_SECONDS,
        refit_seconds: float = REFIT_SECONDS,
    ):
        if not np.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        if not np.isfinite(refit_seconds) or refit_seconds <= 0:
            raise ValueError("refit_seconds must be positive and finite")

        seeds = np.asarray(seed_centroids, dtype=np.float32)
        if seeds.ndim != 2 or not len(seeds) or seeds.shape[1] < 1:
            raise ValueError("seed_centroids must have shape [K, dim]")
        if not np.isfinite(seeds).all():
            raise ValueError("seed_centroids must be finite")

        self.dim = int(seeds.shape[1])
        self.window_seconds = float(window_seconds)
        self.refit_seconds = float(refit_seconds)

        norms = np.linalg.norm(seeds, axis=1, keepdims=True)
        if np.any(norms <= _EPS):
            raise ValueError("Require one nonzero seed per decoder in a fixed group")
        self._centroids = seeds / norms
        self._centroids.flags.writeable = False

        self._signatures: deque[tuple[float, np.ndarray]] = deque()
        self._history_started_at: float | None = None
        self._last_refit_started_at: float | None = None
        self._fit_future: Future[np.ndarray] | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="eldr-balanced-refit",
        )
        self._closed = False
        self.refits_started = 0
        self.refits_completed = 0
        self.last_refit_error: str | None = None
        try:
            self._executor.submit(_warm_refitter).result()
        except Exception:
            self._executor.shutdown(wait=True)
            raise

    def update(self, signature, now):
        """Take ownership of a fresh normalized vector; never wait for a fit.

        Called on the routing thread. Centroid installation is one reference
        replacement, and fitting/validation/alignment run on the worker thread.
        """
        self._poll_refit()
        self._expire_signatures(now)
        if not self._closed and signature.any():
            if self._history_started_at is None:
                self._history_started_at = now
            signature.flags.writeable = False
            self._signatures.append((now, signature))
            self._maybe_start_refit(now)
        return self._centroids

    def status(self):
        return dict(
            window_seconds=self.window_seconds,
            refit_seconds=self.refit_seconds,
            refits_started=self.refits_started,
            refits_completed=self.refits_completed,
            pending=self._fit_future is not None,
            last_error=self.last_refit_error,
        )

    def _expire_signatures(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._signatures and self._signatures[0][0] < cutoff:
            self._signatures.popleft()

    def _maybe_start_refit(self, now: float) -> None:
        first_window_ready = (
            self._history_started_at is not None
            and now - self._history_started_at >= self.window_seconds
        )
        interval_ready = (
            self._last_refit_started_at is None
            or now - self._last_refit_started_at >= self.refit_seconds
        )
        if (
            self._closed
            or self._fit_future is not None
            or len(self._signatures) < len(self._centroids)
            or not first_window_ready
            or not interval_ready
        ):
            return

        # Each stored row is an immutable per-request copy. Capturing their
        # references is cheap and keeps the stack operation off the request
        # path; evicted deque rows remain alive through this tuple.
        signature_rows = tuple(row for _, row in self._signatures)
        self._fit_future = self._executor.submit(
            _run_refit,
            signature_rows,
            self._centroids,
        )
        self._last_refit_started_at = now
        self.refits_started += 1

    def _poll_refit(self) -> None:
        future = self._fit_future
        if future is None or not future.done():
            return

        self._fit_future = None
        try:
            fitted = future.result()
        except Exception as error:
            self.last_refit_error = repr(error)
            _LOG.exception("ELDR background balanced refit failed")
            return

        if self._closed:
            return
        self._centroids = fitted
        self.refits_completed += 1
        self.last_refit_error = None

    def close(self) -> None:
        """Stop accepting refits and release the background worker."""
        if self._closed:
            return
        self._closed = True
        if self._fit_future is not None:
            self._fit_future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class OnlineJSQRouter(CentroidUpdater):
    """ELDR's locality-band JSQ sharing the background centroid updater."""

    def __init__(self, worker_urls, seed_centroids, tau=0.1, **kwargs):
        if not worker_urls:
            raise ValueError("worker_urls must be non-empty")
        if not np.isfinite(tau) or tau < 0:
            raise ValueError("tau must be finite and non-negative")
        if len(worker_urls) != len(seed_centroids):
            raise ValueError("Require one nonzero seed per decoder in a fixed group")
        super().__init__(seed_centroids, **kwargs)
        self.worker_urls, self.tau = list(worker_urls), float(tau)
        self._similarity_scores = np.empty(len(worker_urls), dtype=np.float32)
        self._loads = np.empty(len(worker_urls), dtype=np.float64)

    def route(
        self,
        signature_vector,
        loads: Sequence[int],
    ) -> int:
        """Route one transformed signature and advance background refitting."""
        signature = np.asarray(signature_vector, dtype=np.float32)
        if signature.shape != (self.dim,):
            raise ValueError(f"sig_vec must have shape [{self.dim}]")
        if not np.isfinite(signature).all():
            raise ValueError("sig_vec must be finite")
        norm = float(np.linalg.norm(signature))
        if norm > _EPS:
            signature = signature / norm
        # Normalization already allocated an owned array. Do not copy it again.
        return self._route_owned(signature, loads)

    def route_transformed(
        self,
        signature: np.ndarray,
        loads: Sequence[int],
    ) -> int:
        """Consume a fresh unit-or-zero float32 vector from the shared transform.

        Ownership transfers to the router: callers must not reuse or mutate
        the array. The proxy's transform allocates a new vector per request.
        Use route() for arbitrary vectors or caller-owned reusable buffers.
        """
        if signature.dtype != np.float32 or signature.shape != (self.dim,):
            raise ValueError(f"signature must be float32 with shape [{self.dim}]")
        if not np.isfinite(signature).all():
            raise ValueError("signature must be finite")
        return self._route_owned(signature, loads)

    def _route_owned(self, signature, loads):
        if len(loads) != len(self.worker_urls):
            raise ValueError("loads length must equal number of workers")
        centroids = self.update(signature, time.monotonic())
        np.matmul(centroids, signature, out=self._similarity_scores)
        self._loads[:] = loads
        candidates = locality_candidates(self._similarity_scores, self.tau)
        return int(candidates[np.argmin(self._loads[candidates])])
