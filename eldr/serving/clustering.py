# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared count/IDF signature geometry, clustering and centroid validation."""

import json

import numpy as np


def normalize_counts(counts):
    """[L,E] or [N,L,E] -> globally L2-normalized float32 vectors."""
    counts = np.asarray(counts, dtype=np.float32)
    if counts.ndim not in (2, 3):
        raise ValueError("Counts must have shape [L,E] or [N,L,E]")
    single = counts.ndim == 2
    vectors = counts.reshape(1 if single else len(counts), -1)
    vectors = vectors / np.clip(
        np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None
    )
    return vectors[0] if single else vectors


def balanced_spherical_kmeans(X, C, iters=50):
    """Balanced spherical (cosine) k-means via Hungarian assignment
    (Malinen & Fränti 2014, "Balanced K-Means for Clustering"). Each
    iteration computes the *optimal* cap-constrained assignment of points
    to clusters by solving the linear sum assignment problem on a
    [N x K*cap] cost matrix, where each cluster is replicated
    `cap=ceil(N/K)` times. The assignment is optimal (not a greedy
    heuristic), giving tight cluster-size balance. O(N^3) per iter via
    scipy.optimize.linear_sum_assignment.

    Distance metric: cosine distance (1 - X·C.T). X is unit-normed by
    the upstream transform; C is unit-normed at every iteration so the
    cost is true cosine throughout — matching the §design claim
    'assignment minimizing total cosine distance'."""
    from scipy.optimize import linear_sum_assignment

    n, k = len(X), len(C)
    cap = -(-n // k)  # ceil(N/K)
    cluster_of_slot = np.repeat(np.arange(k), cap)  # [slots]
    # Centroids must be unit-norm so X·C.T is cosine similarity.
    C = C / np.clip(np.linalg.norm(C, axis=1, keepdims=True), 1e-9, None)
    for _ in range(iters):
        # Cosine distance = 1 - cosine similarity. Both X and C are unit.
        D = 1.0 - X @ C.T  # [n, k]
        cost = D[:, cluster_of_slot]  # [n, slots]
        # linear_sum_assignment on a rectangular cost (n < slots) matches
        # each of the n rows to a distinct column; columns beyond n are
        # left unassigned -- the empty cluster slots, exactly the slack
        # we need for the cap to not bind.
        _, col = linear_sum_assignment(cost)
        assign = cluster_of_slot[col]  # [n] cluster ids
        newC = np.stack(
            [X[assign == c].mean(0) if (assign == c).any() else C[c] for c in range(k)]
        )
        # Renormalize centroids: the mean of unit vectors is not unit,
        # but spherical k-means requires unit centroids so cos(x,c)=x·c.
        newC = newC / np.clip(np.linalg.norm(newC, axis=1, keepdims=True), 1e-9, None)
        if np.allclose(newC, C):
            C = newC
            break
        C = newC
    return C


def initialize_centroids(X, K, rng=None):
    """Seeded k-means++; identical signatures may share a centroid."""
    if not 1 <= K <= len(X):
        raise ValueError("Require 1 <= clusters <= training requests")
    rng = rng or np.random.default_rng(0)
    indices = [int(rng.integers(len(X)))]
    for _ in range(1, K):
        distances = ((X[:, None, :] - X[indices][None, :, :]) ** 2).sum(-1).min(1)
        total = distances.sum()
        indices.append(
            int(rng.choice(len(X), p=distances / total))
            if total > 0
            else int(rng.integers(len(X)))
        )
    return X[indices].copy()


def validate_signature_metadata(centroid_data):
    """Return the packed dimension; reject unsupported or ambiguous transforms."""
    layers = centroid_data["L"]
    if (
        type(layers) is not int
        or layers < 1
        or type(centroid_data["e"]) is not int
        or centroid_data["e"] < 1
    ):
        raise ValueError("Positive integer model geometry required")
    if (
        any(centroid_data.get(key) is not None for key in ("sig_mean", "_mean"))
        or centroid_data.get("sig_per_layer_norm", False)
        or centroid_data.get("_per_layer_norm", False)
        or centroid_data.get("variant", "count")
        not in ("count", "count_idf", "gate_prob_all")
        or centroid_data.get("sig_dtype", "<i2")
        != ("<f4" if centroid_data.get("variant") == "gate_prob_all" else "<i2")
    ):
        raise ValueError("Unsupported signature transform or wire dtype")
    mask = centroid_data.get("layer_mask", list(range(layers)))
    if (
        not isinstance(mask, list)
        or not mask
        or any(type(i) is not int or not 0 <= i < layers for i in mask)
        or mask != sorted(set(mask))
        or ("_mask" in centroid_data and list(centroid_data["_mask"]) != mask)
    ):
        raise ValueError("Layer mask must contain sorted, distinct model layers")
    dim = len(mask) * centroid_data["e"]
    if centroid_data.get("variant", "count") == "count":
        if mask != list(range(layers)) or any(
            centroid_data.get(key) is not None for key in ("sig_idf", "_idf")
        ):
            raise ValueError("Counts requires all layers and no IDF weights")
    elif centroid_data.get("variant") == "gate_prob_all":
        if "layer_mask" not in centroid_data or any(
            centroid_data.get(key) is not None for key in ("sig_idf", "_idf")
        ):
            raise ValueError("Gate-probability signatures require a mask and no IDF")
    else:
        if "layer_mask" not in centroid_data:
            raise ValueError("Full signatures require an explicit layer mask")
        idf = np.asarray(centroid_data.get("sig_idf"), dtype=np.float32)
        if (
            idf.shape != (dim,)
            or not np.isfinite(idf).all()
            or np.any(idf < 0)
            or not np.any(idf)
            or (
                "_idf" in centroid_data
                and not np.array_equal(centroid_data["_idf"], idf)
            )
        ):
            raise ValueError(
                "Full signatures require finite nonnegative packed IDF weights"
            )
    return dim


def load_centroid_file(path):
    with open(path) as stream:
        centroid_data = json.load(stream)
    dim = validate_signature_metadata(centroid_data)
    # Preserve the geometry and transform of immutable calibration artifacts.
    centroids = np.asarray(
        centroid_data.get("centroids", centroid_data.get("micro_centroids")),
        dtype=np.float32,
    )
    if (
        centroids.ndim != 2
        or not len(centroids)
        or centroids.shape[1] != dim
        or not np.isfinite(centroids).all()
    ):
        raise ValueError("Centroids must have finite shape [K, signature dimension]")
    if centroid_data.get("k", len(centroids)) != len(centroids):
        raise ValueError(f"metadata k={centroid_data['k']} differs from centroid rows")
    if not np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-3):
        raise ValueError("Centroids must be unit-normed")
    centroid_data["centroid_matrix"] = centroids
    if centroid_data.get("variant", "count") != "count":
        centroid_data["_mask"] = np.asarray(centroid_data["layer_mask"], dtype=np.int64)
    if centroid_data.get("variant", "count") == "count_idf":
        centroid_data["_idf"] = np.asarray(centroid_data["sig_idf"], dtype=np.float32)
    return centroid_data


def build_signature(centroid_data, counts):
    """Transform one request or a calibration batch with the same fixed metadata."""
    array = np.asarray(counts, dtype=np.float32)
    if array.ndim not in (2, 3) or array.shape[-2:] != (
        centroid_data["L"],
        centroid_data["e"],
    ):
        raise ValueError("Counts do not match fitted model geometry")
    if centroid_data.get("variant", "count") == "count":
        return normalize_counts(array)
    single = array.ndim == 2
    array = array[None] if single else array
    vectors = array[:, centroid_data["_mask"], :].reshape(len(array), -1)
    if centroid_data.get("variant") == "count_idf":
        vectors = vectors * centroid_data["_idf"]
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9, None)
    return vectors[0] if single else vectors
