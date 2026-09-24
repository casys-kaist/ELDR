# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Centroid fitting from training counts with an explicit, frozen transform."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

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
    counts = (
        np.load(source, allow_pickle=False)
        if source.suffix == ".npy"
        else read_capture(raw, model)
    )
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
