# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate saved measurements, compute metrics and draw experiment plots."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from eldr.experiments.runner.workers import write_new_json

QUANTILES = (50, 95, 99)


def checked_path(root: Path, row: dict) -> Path:
    relative = Path(row["path"])
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
    ):
        raise ValueError("Recipe paths must stay inside the data root")
    return path


def verify(recipe: dict, root: Path) -> list[dict]:
    root = root.resolve()
    verified = []
    for row in recipe["measurements"]:
        path = checked_path(root, row)
        with path.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if checksum != row["sha256"]:
            raise ValueError("SHA-256 mismatch: " + row["path"])
        verified.append(
            dict(path=row["path"], sha256=checksum, bytes=path.stat().st_size)
        )
    return verified


def request_metrics(raw: dict, expected_count: int, output_tokens: int):
    """Token-normalized TPOT, including bundled SSE events without imputation.

    Sum each request's saved inter-event intervals and divide by generated
    tokens minus one, not by SSE event count. Check benchmark-reported
    aggregates within 0.0001 ms.
    """
    for key in ("output_lens", "input_lens", "itls", "ttfts", "errors"):
        if len(raw[key]) != expected_count:
            raise ValueError("Incomplete " + key)
    if (
        raw["completed"] != expected_count
        or raw.get("failed", 0) != 0
        or any(raw["errors"])
        or raw["output_lens"] != [output_tokens] * expected_count
    ):
        raise ValueError("Failed/incomplete requests or wrong token counts")
    spans = []
    for gaps in raw["itls"]:
        values = np.asarray(gaps, dtype=np.float64)
        if (
            values.ndim != 1
            or not 1 <= len(values) <= output_tokens - 1
            or not np.isfinite(values).all()
            or np.any(values < 0)
        ):
            raise ValueError("Invalid streaming intervals")
        spans.append(float(values.sum()))
    tpot = np.asarray(spans) * (1000.0 / (output_tokens - 1))
    ttft = np.asarray(raw["ttfts"], dtype=np.float64) * 1000
    if (
        ttft.shape != (expected_count,)
        or not np.isfinite(tpot).all()
        or np.any(tpot <= 0)
        or not np.isfinite(ttft).all()
        or np.any(ttft <= 0)
    ):
        raise ValueError("Invalid TTFT/TPOT values")
    for prefix, values in (("tpot", tpot), ("ttft", ttft)):
        for statistic, computed in (
            ("mean", values.mean()),
            ("median", np.median(values)),
            ("std", values.std()),
            ("p99", np.percentile(values, 99)),
        ):
            reported = raw[f"{statistic}_{prefix}_ms"]
            if not np.isfinite(reported) or abs(reported - computed) > 1e-4:
                raise ValueError(f"Benchmark metric disagreement: {statistic}_{prefix}")
    return tpot, ttft


def collect(plan, root):
    """Recompute all saved cells; a missing or failed cell invalidates the figure."""
    rows = []
    for worker_group in plan["fleets"]:
        if Path(worker_group["key"]).name != worker_group["key"] or worker_group[
            "key"
        ] in (".", ".."):
            raise ValueError("Invalid worker-group directory key")
        run = root / worker_group["key"] / "run"
        if not (run / "complete.json").is_file() or (run / "failed.json").exists():
            raise ValueError("Incomplete worker-group run: " + str(run))
        if json.loads((run / "cleanup.json").read_text())["remaining"]:
            raise ValueError("Worker-group cleanup is incomplete")
        actual = json.loads((run / "trials.json").read_text())
        if actual != json.loads(json.dumps(worker_group["trials"])):
            raise ValueError("Executed trials differ from experiment plan")
        for label, policy, protocol in actual:
            path = run / label / "measure" / "raw.json"
            content = path.read_bytes()
            raw = json.loads(content)
            if raw["request_rate"] != protocol["request_rate"]:
                raise ValueError("Measured rate differs from experiment plan")
            tpot, ttft = request_metrics(
                raw, protocol["requests"], protocol["output_tokens"]
            )
            rows.append(
                dict(
                    setting=worker_group["key"],
                    policy=policy,
                    variant=protocol["variant"],
                    rate=protocol["request_rate"],
                    requests=len(tpot),
                    tpot50_ms=float(np.percentile(tpot, 50)),
                    tpot95_ms=float(np.percentile(tpot, 95)),
                    tpot99_ms=float(np.percentile(tpot, 99)),
                    ttft50_ms=float(np.median(ttft)),
                    raw=str(path.relative_to(root)),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
    return rows


def write_report(rows, output, experiment):
    import matplotlib.pyplot as plt

    from eldr.experiments import plot_style

    write_new_json(output / "summary.json", rows)
    with (output / "summary.csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_style.apply()
    settings = list(dict.fromkeys(r["setting"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, axes = plt.subplots(
        3, len(settings), squeeze=False, figsize=(plot_style.WIDTH, 6.4)
    )
    for column, setting in enumerate(settings):
        for row, metric in enumerate(("tpot50_ms", "tpot99_ms", "ttft50_ms")):
            ax = axes[row, column]
            for index, variant in enumerate(variants):
                values = sorted(
                    [
                        r
                        for r in rows
                        if r["setting"] == setting and r["variant"] == variant
                    ],
                    key=lambda r: r["rate"],
                )
                if not values:
                    continue
                ax.plot(
                    [r["rate"] for r in values],
                    [r[metric] for r in values],
                    label=variant,
                    linewidth=1.6,
                    marker="o" if len(values) == 1 else None,
                    color=(plot_style.BLUES + ["#595959", "#d62728"])[index % 7],
                    linestyle="--" if variant == "rr" else "-",
                )
            ax.grid(alpha=0.2)
            if column == 0:
                ax.set_ylabel(
                    {
                        "tpot50_ms": "TPOT P50 (ms)",
                        "tpot99_ms": "TPOT P99 (ms)",
                        "ttft50_ms": "TTFT P50 (ms)",
                    }[metric]
                )
            if row == 0:
                ax.set_title(setting)
            if row == 2:
                ax.set_xlabel("Requests/s")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    plot_style.legend_top(fig, min(4, len(labels)), handles, labels)
    plot_style.save(fig, output / experiment)
    plt.close(fig)
