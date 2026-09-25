# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosine locality band followed by minimum in-flight request count."""

import numpy as np

from eldr.serving.clustering import build_signature


def locality_candidates(scores: np.ndarray, tau: float) -> np.ndarray:
    """Workers within tau of the best cosine score; singleton bands stay intact."""
    return np.flatnonzero(scores >= scores.max() - tau)


def route_jsq(centroid_data, counts, loads, tau=0.1):
    """Locality-band JSQ with fixed centroids (the static control)."""
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("A finite nonnegative locality threshold is required")
    vector = build_signature(centroid_data, counts)
    k = len(loads)
    if not k or len(centroid_data["centroid_matrix"]) != k:
        raise ValueError("static ELDR requires exactly one centroid per decoder")
    if "_similarity_buffer" not in centroid_data:
        centroid_data["_similarity_buffer"] = np.empty(k, dtype=np.float32)
        centroid_data["_load_buffer"] = np.empty(k, dtype=np.float64)
    scores, work = centroid_data["_similarity_buffer"], centroid_data["_load_buffer"]
    np.matmul(centroid_data["centroid_matrix"], vector, out=scores)
    work[:] = loads
    eligible = locality_candidates(scores, tau)
    return int(eligible[np.argmin(work[eligible])])
