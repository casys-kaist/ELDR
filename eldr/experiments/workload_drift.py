# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen Task drift with RR, static ELDR and online ELDR on matched traffic."""

import json
from pathlib import Path

import numpy as np

from eldr.experiments.drift_plot import plot
from eldr.experiments.drift_workload import BASELINE_MIX, DOMAINS, DRIFT_MIX, build
from eldr.experiments.runner.config import file_sha256
from eldr.experiments.runner.experiment import (
    load_labeled_inputs,
    main,
    plan_worker_group,
)
from eldr.experiments.runner.results import audit_generalization_run, request_metrics
from eldr.experiments.runner.workers import write_new_json

RATES = (100,)
POLICIES = ("rr", "eldr-static", "eldr")
METHODS = (("rr", "rr"), ("static", "eldr-static"), ("online", "eldr"))
DURATION, GUARD, WIDTH = 120, 15, 5


def configure(config, output, policies):
    _, pairs, _ = load_labeled_inputs(config)
    if config["model"] != "qwen" or {p[1] for p in pairs} != set(DOMAINS):
        raise ValueError("The drift experiment requires Qwen and the Task workload")
    folder = output / "qwen"
    group = plan_worker_group(
        config,
        folder,
        [
            (
                label,
                policy,
                dict(
                    traffic_seconds=DURATION + 2 * GUARD,
                    shuffle=False,
                    no_oversample=True,
                ),
            )
            for label, policy in METHODS
        ],
        inputs=("training_prompts", "labels"),
    )
    group["site"]["dataset"] = str(folder / "traffic/task_domain_gradual.json")
    group["site"].pop("dataset_sha256", None)
    return [group]


def validate_plan(plan):
    if len(plan["fleets"]) != 1:
        raise ValueError("Drift requires one Qwen worker group")
    trials = plan["fleets"][0]["trials"]
    if len(trials) != 3 or [p for _, p, _ in trials] != list(POLICIES):
        raise ValueError("Drift requires exactly one RR/static/online trial")
    rates = {p["request_rate"] for _, _, p in trials}
    if len(rates) != 1 or (plan["profile"] == "paper" and rates != {100}):
        raise ValueError("Paper drift requires 100 req/s; smoke uses one lower rate")
    for _, _, p in trials:
        if (
            p["requests"] != (DURATION + 2 * GUARD) * p["request_rate"]
            or p["shuffle"]
            or not p["no_oversample"]
            or (p["window_seconds"], p["refit_seconds"]) != (5, 1)
        ):
            raise ValueError("Drift protocol changed")


def prepare(group):
    source, site = group["source_site"], group["site"]
    build(
        Path(source["dataset"]),
        Path(source["labels"]),
        Path(group["directory"]) / "traffic",
        group["trials"][0][2]["request_rate"],
        DURATION,
        42,
        profile="moderate",
        guard_s=GUARD,
    )
    site["dataset_sha256"] = file_sha256(site["dataset"])


def windowed(
    raw, domains, *, guard=GUARD, duration=DURATION, width=WIDTH, output_tokens=512
):
    """Whole-request TPOT in actual-arrival windows, not index/rate estimates."""
    if guard < 0 or duration <= 0 or width <= 0 or duration % width:
        raise ValueError("Invalid analysis interval")
    tpots, _ = request_metrics(raw, len(domains), output_tokens)
    elapsed = np.asarray(raw["start_times"], dtype=float)
    if elapsed.shape != tpots.shape or not np.isfinite(elapsed).all():
        raise ValueError("Complete measured request start_times required")
    if np.any(np.diff(elapsed) < 0):
        raise ValueError("Request timestamps are not in input order")
    elapsed = elapsed - elapsed[0] - guard
    labels = np.asarray(domains)
    if any(d not in DOMAINS for d in labels):
        raise ValueError("Missing source domain labels")
    if elapsed[-1] < duration:
        raise ValueError("Traffic does not cover the complete analysis interval")
    windows = []
    for start in range(0, duration, width):
        selected = (elapsed >= start) & (elapsed < start + width)
        values = tpots[selected]
        if not len(values):
            raise ValueError("Empty arrival window")
        windows.append(
            dict(
                requests=len(values),
                **{
                    f"tpot_p{q}_ms": float(np.percentile(values, q))
                    for q in (50, 95, 99)
                },
                domain_percent={
                    d: float(100 * np.mean(labels[selected] == d)) for d in DOMAINS
                },
            )
        )
    return windows


def validate(plan, root, rows):
    validate_plan(plan)
    for path, expected in plan["inputs"].items():
        if file_sha256(path) != expected:
            raise ValueError("Drift source input changed: " + path)
    group = plan["fleets"][0]
    folder = root / group["key"]
    dataset = folder / "traffic/task_domain_gradual.json"
    manifest = json.loads((folder / "traffic/manifest.json").read_text())
    if (
        manifest["dataset"]["sha256"] != file_sha256(dataset)
        or manifest["guard_s"] != GUARD
        or manifest["duration_s"] != DURATION
        or manifest["rate"] != group["trials"][0][2]["request_rate"]
        or manifest["records"] != group["trials"][0][2]["requests"]
        or manifest["baseline_mix"] != BASELINE_MIX
        or manifest["drift_mix"] != DRIFT_MIX
        or manifest["profile"] != "moderate"
        or manifest["seed"] != 42
        or manifest["ordered"] is not True
    ):
        raise ValueError("Drift traffic or schedule changed")
    for key, original in (("task_eval", "dataset"), ("domain_map", "labels")):
        if (
            manifest["source"][key]["sha256"]
            != plan["inputs"][group["source_site"][original]]
        ):
            raise ValueError("Drift traffic source does not match the frozen inputs")
    sources = json.loads((root / "sources.json").read_text())
    audit_generalization_run(
        folder / "run", dataset, Path(group["source_site"]["centroids"]), sources
    )
    domains = [r["_eldr_domain"] for r in json.loads(dataset.read_text())]
    if len(domains) != manifest["records"] or len(rows) != len(group["trials"]):
        raise ValueError("Drift request count changed")
    for row, (label, policy, protocol) in zip(rows, group["trials"]):
        if row["policy"] != policy or row["variant"] != protocol["variant"]:
            raise ValueError("Drift policy order changed")
        raw = json.loads((folder / "run" / label / "measure/raw.json").read_text())
        row.update(
            arrival_windows=windowed(
                raw, domains, output_tokens=protocol["output_tokens"]
            ),
            protocol=protocol,
            profile=plan["profile"],
            provenance={
                str(p.relative_to(root)): file_sha256(p)
                for p in (
                    dataset,
                    folder / "traffic/manifest.json",
                    folder / "run/prepared.json",
                    root / "sources.json",
                )
            },
        )


def compare(rows):
    lookup = {r["variant"]: r for r in rows}
    if len(rows) != 3 or set(lookup) != {label for label, _ in METHODS}:
        raise ValueError("Drift comparison requires exactly RR/static/online")
    if any(len(r["arrival_windows"]) != DURATION // WIDTH for r in rows):
        raise ValueError("Drift requires all 24 arrival windows")
    windows = [
        dict(
            start_s=i * WIDTH,
            end_s=(i + 1) * WIDTH,
            domain_percent=lookup["rr"]["arrival_windows"][i]["domain_percent"],
            **{label: lookup[label]["arrival_windows"][i] for label, _ in METHODS},
        )
        for i in range(DURATION // WIDTH)
    ]
    steady = {
        label: float(np.mean([w[label]["tpot_p50_ms"] for w in windows[10:16]]))
        for label, _ in METHODS
    }
    return dict(
        duration_s=DURATION,
        window_s=WIDTH,
        windows=windows,
        phase_boundaries_s=[0, 40, 80, 120],
        transition_intervals_s=[[40, 50], [80, 90]],
        domain_mix_reference="rr",
        steady_drift=dict(
            interval_s=[50, 80],
            mean_bin_median_tpot_ms=steady,
            online_reduction_percent={
                p: 100 * (1 - steady["online"] / steady[p]) for p in ("rr", "static")
            },
        ),
        provenance=dict(
            runs_per_method={label: 1 for label, _ in METHODS},
            raw_sha256={r["raw"]: r["sha256"] for r in rows},
            protocols={r["variant"]: r["protocol"] for r in rows},
            profile=rows[0]["profile"],
        ),
    )


def report(panels, stem):
    comparison = compare([r for rows in panels.values() for r in rows])
    write_new_json(stem.with_suffix(".json"), comparison)
    plot(
        comparison,
        stem,
        larger_mix=True,
        shade_transitions=False,
        static_color="#595959",
    )


if __name__ == "__main__":
    main("workload_drift")
