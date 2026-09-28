# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centroid fitting from training counts with an explicit, frozen transform."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from eldr.experiments.runner.config import FIT_SEED
from eldr.serving.clustering import (
    balanced_spherical_kmeans,
    build_signature,
    initialize_centroids,
    load_centroid_file,
    normalize_counts,
)


def training_dataset(source: Path, evaluation: Path):
    """Convert builder-produced [prompt, label] pairs; reject train/eval overlap."""
    raw = source.read_bytes()
    pairs = json.loads(raw)
    if (
        not isinstance(pairs, list)
        or not pairs
        or any(
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not row[0].strip()
            for row in pairs
        )
    ):
        raise ValueError("Training input must be nonempty [prompt, label] pairs")
    prompts = [row[0] for row in pairs]
    held_out = {
        row["conversations"][0]["value"] for row in json.loads(evaluation.read_bytes())
    }
    if len(set(prompts)) != len(prompts) or set(prompts) & held_out:
        raise ValueError("Training prompts must be unique and disjoint from evaluation")
    dataset = [
        {
            "conversations": [
                {"from": "human", "value": p},
                {"from": "gpt", "value": "ok"},
            ]
        }
        for p in prompts
    ]
    return dataset, hashlib.sha256(raw).hexdigest()


def read_capture(raw, model):
    """Read the proxy's length-prefixed int16 training signatures."""
    from eldr.experiments.runner.config import MODEL_GEOMETRY

    layers, experts, _ = MODEL_GEOMETRY[model]
    size, offset, rows = layers * experts * 2, 0, []
    while offset < len(raw):
        if offset + 4 > len(raw) or struct.unpack_from("<I", raw, offset)[0] != size:
            raise ValueError("Invalid training signature frame/geometry")
        offset += 4
        if offset + size > len(raw):
            raise ValueError("Truncated training signature")
        rows.append(
            np.frombuffer(raw, dtype="<i2", count=layers * experts, offset=offset)
        )
        offset += size
    if not rows:
        raise ValueError("Empty training capture")
    return np.stack(rows).reshape(-1, layers, experts)


def read_activations(path, model):
    """Validate a fresh request-aligned capture, including token conservation."""
    from eldr.experiments.runner.config import MODEL_GEOMETRY

    with np.load(path, allow_pickle=False) as archive:
        data = dict(archive)
    expected = {
        "request_ids",
        "prefill_counts",
        "gate_probabilities",
        "decode_counts",
        "prompt_tokens",
        "decode_tokens",
        "metadata",
    }
    if set(data) != expected:
        raise ValueError("Unexpected calibration fields")
    metadata = json.loads(str(data.pop("metadata")))
    if metadata.get("format") != "eldr-calibration-v1" or metadata["model"] != model:
        raise ValueError("Calibration format/model mismatch")
    ids = data["request_ids"]
    if (
        ids.ndim != 1
        or not len(ids)
        or ids.dtype.kind != "U"
        or len(set(ids)) != len(ids)
        or any(
            len(value) != 64 or set(value) - set("0123456789abcdef") for value in ids
        )
    ):
        raise ValueError("Unique calibration prompt SHA-256 IDs required")
    layers, experts, _ = MODEL_GEOMETRY[model]
    shape = (len(ids), layers, experts)
    topk = metadata["top_k"]
    output_tokens = metadata["output_tokens"]
    if (
        type(topk) is not int
        or not 1 <= topk <= experts
        or type(output_tokens) is not int
        or output_tokens < 2
    ):
        raise ValueError("Invalid capture top-k/output length")
    for key in ("prompt_tokens", "decode_tokens"):
        values = data[key]
        if (
            values.shape != (len(ids),)
            or values.dtype.kind not in "iu"
            or np.any(values <= 0)
        ):
            raise ValueError("Invalid calibration token counts")
    if np.any(data["decode_tokens"] != output_tokens - 1):
        raise ValueError("Incomplete calibration generation")
    for key, length in (
        ("prefill_counts", "prompt_tokens"),
        ("decode_counts", "decode_tokens"),
    ):
        values = data[key]
        if (
            values.shape != shape
            or values.dtype.kind not in "iu"
            or np.any(values < 0)
            or not np.all(values.sum(-1) == data[length][:, None] * topk)
        ):
            raise ValueError("Calibration expert counts do not conserve tokens")
    probabilities = data["gate_probabilities"]
    if (
        probabilities.shape != shape
        or probabilities.dtype.kind != "f"
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(-1), data["prompt_tokens"][:, None], rtol=1e-4, atol=1e-3
        )
    ):
        raise ValueError("Invalid calibration probabilities")
    data["metadata"] = metadata
    return data


def fit_centroids(
    source: Path,
    output: Path,
    model: str,
    k: int,
    seed: int,
    *,
    balanced=True,
    transform_from: Path | None = None,
):
    if output.exists():
        raise ValueError("Refusing to replace an existing fit")
    raw = source.read_bytes()
    if source.suffix == ".npz":
        counts = read_activations(source, model)["prefill_counts"]
    elif source.suffix == ".npy":
        counts = np.load(source, allow_pickle=False)
    else:
        counts = read_capture(raw, model)
    if (
        counts.ndim != 3
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts < 0)
        or np.any(counts > 32767)
        or k < 1
        or len(counts) < k
        or min(counts.shape) < 1
        or np.any(counts.sum(axis=(1, 2)) == 0)
    ):
        raise ValueError(
            "Training capture must be nonzero integer counts [N,L,E], N >= K"
        )
    layers, experts = counts.shape[1:]
    transform = dict(
        variant="count", layer_select="all", layer_mask=list(range(layers))
    )
    if transform_from is not None:
        template = load_centroid_file(transform_from)
        vectors = build_signature(template, counts)
        transform = dict(
            variant=template.get("variant", "count"),
            layer_mask=template.get("layer_mask", list(range(layers))),
        )
        if transform["variant"] == "count_idf":
            transform["sig_idf"] = template["sig_idf"]
        transform["transform_sha256"] = hashlib.sha256(
            transform_from.read_bytes()
        ).hexdigest()
    else:
        vectors = normalize_counts(counts)
    if np.any(np.linalg.norm(vectors, axis=1) < 1e-9):
        raise ValueError("Training signature is zero under the frozen transform")
    initial = initialize_centroids(vectors, k, np.random.default_rng(seed))
    if balanced:
        centroids = balanced_spherical_kmeans(vectors, initial)
    else:
        # Same cosine geometry and initialization; only the capacity constraint
        # is removed for the clustering ablation. Empty clusters keep their seed.
        centroids = initial.copy()
        for _ in range(50):
            labels = np.argmax(vectors @ centroids.T, axis=1)
            updated = np.stack(
                [
                    vectors[labels == i].mean(0) if np.any(labels == i) else center
                    for i, center in enumerate(centroids)
                ]
            )
            updated /= np.clip(
                np.linalg.norm(updated, axis=1, keepdims=True), 1e-9, None
            )
            converged = np.allclose(updated, centroids)
            centroids = updated
            if converged:
                break
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    result = dict(
        model=model,
        algo="kmbal" if balanced else "km",
        k=k,
        m=k,
        L=layers,
        e=experts,
        n_fit=len(counts),
        **transform,
        sig_dtype="<i2",
        seed=seed,
        centroids=centroids.tolist(),
        training_sha256=hashlib.sha256(raw).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def greedy_mask(signal, decode, *, pairs=30000, seed=FIT_SEED):
    """Paper's deterministic incremental forward-greedy Spearman selection."""
    if signal.shape != decode.shape or signal.ndim != 3 or len(signal) < 2:
        raise ValueError("Matched calibration profiles [N,L,E], N >= 2 required")
    n, layers, _ = signal.shape
    rng = np.random.default_rng(seed)
    left = rng.integers(0, n, pairs)
    right = rng.integers(0, n, pairs)
    left, right = left[left != right], right[left != right]
    self_sq = np.einsum("nle,nle->nl", signal, signal)
    pair_dot = np.einsum("kle,kle->kl", signal[left], signal[right])
    target = decode.astype(np.float32).reshape(n, -1)
    target /= np.clip(np.linalg.norm(target, axis=1, keepdims=True), 1e-9, None)
    target_rank = rankdata(1.0 - (target[left] * target[right]).sum(1)).astype(
        np.float64
    )
    target_rank -= target_rank.mean()
    if not np.any(target_rank):
        raise ValueError("Decode profiles have no rank variation")
    numerator = np.zeros(len(left), dtype=np.float32)
    norm_left, norm_right = numerator.copy(), numerator.copy()
    remaining, order, correlations = list(range(layers)), [], []
    for _ in range(layers):
        best_layer, best_rho = None, -np.inf
        for layer in remaining:
            cosine = (numerator + pair_dot[:, layer]) / np.sqrt(
                (norm_left + self_sq[left, layer])
                * (norm_right + self_sq[right, layer])
                + 1e-9
            )
            rank = rankdata(1.0 - cosine).astype(np.float64)
            rank -= rank.mean()
            rho = float(
                (rank * target_rank).sum()
                / np.sqrt(
                    (rank * rank).sum() * (target_rank * target_rank).sum() + 1e-9
                )
            )
            if rho > best_rho:
                best_layer, best_rho = layer, rho
        numerator += pair_dot[:, best_layer]
        norm_left += self_sq[left, best_layer]
        norm_right += self_sq[right, best_layer]
        order.append(best_layer)
        correlations.append(best_rho)
        remaining.remove(best_layer)
    peak = int(np.argmax(correlations))
    return sorted(order[: peak + 1]), correlations[peak]


def fit_signature(source, output, model, variant, k=16):
    if variant not in ("count_idf", "gate_prob_all") or Path(output).exists():
        raise ValueError("Known signature variant and a new fit output required")
    from eldr.experiments.runner.config import MODEL_GEOMETRY, file_sha256

    layers, experts, _ = MODEL_GEOMETRY[model]
    before = file_sha256(source)
    capture = read_activations(source, model)
    counts = capture["prefill_counts"]
    probability = capture["gate_probabilities"]
    decode = capture["decode_counts"]
    if len(counts) < k or k < 1:
        raise ValueError("Calibration must contain at least K requests")
    idf = np.log((len(counts) + 1.0) / ((counts > 0).sum(0) + 1.0)).astype(np.float32)
    signal = counts.astype(np.float32) * idf if variant == "count_idf" else probability
    mask, rho = greedy_mask(signal, decode, seed=FIT_SEED)
    # Preserve the original fitter's float32 IDF arithmetic for the runtime
    # transform (its greedy-search helper used float64 document frequencies).
    if variant == "count_idf":
        frequencies = (counts > 0).sum(0).astype(np.float32)
        idf = np.log((len(counts) + 1) / (frequencies + 1)).astype(np.float32)
        signal = counts.astype(np.float32) * idf
    vectors = signal[:, mask].reshape(len(signal), -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-9):
        raise ValueError("Calibration signature vanished under the fitted transform")
    vectors /= norms
    initial = initialize_centroids(vectors, k, np.random.default_rng(FIT_SEED))
    centroids = balanced_spherical_kmeans(vectors, initial)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    result = dict(
        model=model,
        algo="kmbal",
        k=k,
        m=k,
        L=layers,
        e=experts,
        n_fit=len(counts),
        variant=variant,
        layer_select="forward_greedy",
        layer_mask=mask,
        fit_peak_rho=rho,
        seed=FIT_SEED,
        sig_dtype="<f4" if variant == "gate_prob_all" else "<i2",
        centroids=centroids.tolist(),
        training_sha256=before,
        calibration_format="eldr-calibration-v1",
    )
    if variant == "count_idf":
        result["sig_idf"] = idf[mask].reshape(-1).tolist()
    if file_sha256(source) != before:
        raise ValueError("Calibration capture changed during fitting")
    with Path(output).open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result
