"""Count·IDF versus full-softmax gate signatures, with matched RR controls.

Both transforms use the same explicit paired calibration capture, seed,
forward-greedy layer-selection algorithm and balanced K-means. Separate worker
groups isolate the capture mode; each includes RR with capture disabled.
Both fitted centroid tables stay fixed throughout warmup and measurement.
"""

import struct
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from eldr.experiments.runner.config import FIT_SEED, MODEL_GEOMETRY, file_sha256
from eldr.experiments.runner.experiment import main, plan_worker_group
from eldr.experiments.runner.workers import write_new_json
from eldr.serving.clustering import balanced_spherical_kmeans, initialize_centroids

RATES = (60,)
POLICIES = ("rr", "eldr-static")
VARIANTS = ("count_idf", "gate_prob_all")


def read_paired_capture(path, layers, experts):
    """Original four-field format: prefill counts/logits/probs, decode counts.

    All records are retained. This legacy format has no request IDs; callers
    must supply a known calibration artifact, not an arbitrary evaluation dump.
    """
    raw = Path(path).read_bytes()
    size = layers * experts
    lengths = (2 * size, 4 * size, 4 * size, 2 * size)
    stride = 16 + sum(lengths)
    if not raw or len(raw) % stride:
        raise ValueError("Incomplete four-field paired calibration capture")
    prefill, probability, decode = [], [], []
    for offset in range(0, len(raw), stride):
        if struct.unpack_from("<4I", raw, offset) != lengths:
            raise ValueError("Paired capture frame/model geometry mismatch")
        for rows, dtype, start in (
            (prefill, "<i2", offset + 16),
            (probability, "<f4", offset + 16 + 6 * size),
            (decode, "<i2", offset + 16 + 10 * size),
        ):
            array = np.frombuffer(raw, dtype=dtype, count=size, offset=start)
            if not np.isfinite(array).all() or np.any(array < 0) or not np.any(array):
                raise ValueError("Invalid/empty paired expert profile")
            rows.append(array.reshape(layers, experts))
    return tuple(np.stack(rows) for rows in (prefill, probability, decode))


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
    if variant not in VARIANTS or Path(output).exists():
        raise ValueError("Known signature variant and a new fit output required")
    layers, experts, _ = MODEL_GEOMETRY[model]
    before = file_sha256(source)
    counts, probability, decode = read_paired_capture(source, layers, experts)
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
        calibration_format="legacy_paired_four_field_no_request_ids",
    )
    if variant == "count_idf":
        result["sig_idf"] = idf[mask].reshape(-1).tolist()
    if file_sha256(source) != before:
        raise ValueError("Calibration capture changed during fitting")
    write_new_json(output, result)
    return result


def configure(config, output, policies):
    if "paired_signatures" not in config:
        raise ValueError("Signature ablation requires explicit paired_signatures")
    groups = []
    for variant in VARIANTS:
        folder = output / f"{config['model']}-{variant}"
        group = plan_worker_group(
            config,
            folder,
            [("rr", "rr", {}), (variant, "eldr-static", {})],
            inputs=("paired_signatures",),
        )
        group["site"].update(
            centroids=str(folder / "centroids.json"), signature_variant=variant
        )
        group["site"].pop("centroids_sha256", None)
        groups.append(group)
    return groups


def prepare(worker_group):
    config = worker_group["site"]
    fit_signature(
        worker_group["source_site"]["paired_signatures"],
        config["centroids"],
        config["model"],
        config["signature_variant"],
        len(config["decoders"]),
    )


if __name__ == "__main__":
    main("signature_ablation")
