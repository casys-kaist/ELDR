# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fresh category-only calibration and RR/ELDR measurements for Table 2."""

import csv
import json
from pathlib import Path

from eldr.experiments.runner.config import FIT_SEED, file_sha256
from eldr.experiments.runner.experiment import (
    load_labeled_inputs,
    main,
    plan_worker_group,
    prompt_hash,
)
from eldr.experiments.runner.fit import fit_centroids, read_capture
from eldr.experiments.runner.results import audit_generalization_run, request_metrics
from eldr.experiments.runner.run import execute
from eldr.experiments.runner.workers import prepare as prepare_run
from eldr.experiments.runner.workers import write_new_json
from eldr.serving.clustering import load_centroid_file

RATES = (20, 40, 60, 80, 100)
POLICIES = ("rr", "eldr")
CATEGORIES = {"math": "Math-only", "English": "English-only"}
METRICS = ("tpot50_ms", "tpot95_ms", "tpot99_ms", "ttft50_ms")


def select(config):
    category = config.get("category")
    if config["model"] != "qwen" or category not in CATEGORIES:
        raise ValueError("Single-category requires Qwen with category math or English")
    rows, pairs, labels = load_labeled_inputs(config)
    selected = [
        r
        for r in rows
        if labels[prompt_hash(r["conversations"][0]["value"])] == category
    ]
    training = [p for p in pairs if p[1] == category]
    if not selected or len(training) < len(config["decoders"]):
        raise ValueError("Insufficient category-only calibration/evaluation data")
    return selected, training


def configure(config, output, policies):
    select(config)  # Validate disjoint category data without capturing anything.
    folder = output / ("qwen-" + config["category"].lower())
    group = plan_worker_group(
        config,
        folder,
        [(p, p, {}) for p in POLICIES],
        inputs=("training_prompts", "labels"),
    )
    for key, filename in (
        ("dataset", "evaluation.json"),
        ("training_prompts", "training.json"),
        ("centroids", "centroids.json"),
    ):
        group["site"][key] = str(folder / filename)
        group["site"].pop(key + "_sha256", None)
    return [group]


def validate_plan(plan):
    for group in plan["fleets"]:
        trials = group["trials"]
        cells = {(p["request_rate"], policy) for _, policy, p in trials}
        rates = {rate for rate, _ in cells}
        if (
            len(cells) != len(trials)
            or cells != {(rate, policy) for rate in rates for policy in POLICIES}
            or (plan["profile"] == "paper" and rates != set(RATES))
        ):
            raise ValueError("Single-category requires matched RR/ELDR at every rate")


def prepare(group):
    source, site, folder = group["source_site"], group["site"], Path(group["directory"])
    rows, training = select(source)
    write_new_json(folder / "evaluation.json", rows)
    write_new_json(folder / "training.json", training)
    capture_site = dict(
        source, dataset=site["dataset"], dataset_sha256=file_sha256(site["dataset"])
    )
    print(f"[phase] {source['category']}/calibration", flush=True)
    capture = prepare_run(
        capture_site, folder / "capture", "paper", training=folder / "training.json"
    )
    execute(capture)
    fit_centroids(
        capture / "signatures.bin",
        folder / "centroids.json",
        "qwen",
        len(site["decoders"]),
        FIT_SEED,
        transform_from=Path(source["centroids"]),
    )
    for key in ("dataset", "training_prompts", "centroids"):
        site[key + "_sha256"] = file_sha256(site[key])
    print(f"[phase] {source['category']}/evaluation", flush=True)


def validate(plan, root, rows):
    validate_plan(plan)
    for path, expected in plan["inputs"].items():
        if file_sha256(path) != expected:
            raise ValueError("Single-category source input changed: " + path)
    sources = json.loads((root / "sources.json").read_text())
    for group in plan["fleets"]:
        folder = root / group["key"]
        source = group["source_site"]
        selected, training = select(source)
        if (
            json.loads((folder / "evaluation.json").read_text()) != selected
            or json.loads((folder / "training.json").read_text()) != training
        ):
            raise ValueError("Category-only inputs differ from the frozen selection")
        capture = folder / "capture"
        prepared = audit_generalization_run(
            capture,
            folder / "evaluation.json",
            Path(source["centroids"]),
            sources,
            capture=True,
        )
        metadata = prepared["capture"]
        if (
            metadata["source_sha256"] != file_sha256(folder / "training.json")
            or metadata["dataset_sha256"] != file_sha256(capture / "training.json")
            or metadata["requests"] != len(training)
        ):
            raise ValueError("Calibration input changed")
        raw = json.loads((capture / "rr/measure/raw.json").read_text())
        request_metrics(raw, len(training), 2)
        counts = read_capture((capture / "signatures.bin").read_bytes(), "qwen")
        fit = load_centroid_file(folder / "centroids.json")
        template = load_centroid_file(source["centroids"])
        if (
            len(counts) != len(training)
            or fit["n_fit"] != len(training)
            or fit["training_sha256"] != file_sha256(capture / "signatures.bin")
            or fit["transform_sha256"] != file_sha256(source["centroids"])
            or fit["layer_mask"] != template["layer_mask"]
            or fit["sig_idf"] != template["sig_idf"]
            or fit["seed"] != FIT_SEED
            or fit["algo"] != "kmbal"
        ):
            raise ValueError("Fresh category fit or frozen Full transform changed")
        audit_generalization_run(
            folder / "run",
            folder / "evaluation.json",
            folder / "centroids.json",
            sources,
        )
        provenance = {
            str(p.relative_to(root)): file_sha256(p)
            for p in (
                folder / "centroids.json",
                folder / "evaluation.json",
                folder / "training.json",
                capture / "signatures.bin",
                capture / "rr/measure/raw.json",
                folder / "run/prepared.json",
                root / "sources.json",
            )
        }
        for row in rows:
            if row["setting"] == group["key"]:
                row.update(category=source["category"], provenance=provenance)


def compare(rows):
    """Keep every rate, including regressions; average relative changes by rate."""
    lookup = {(r["category"], r["rate"], r["policy"]): r for r in rows}
    if (
        not rows
        or len(lookup) != len(rows)
        or {r["category"] for r in rows} - set(CATEGORIES)
    ):
        raise ValueError("Empty or duplicate category measurements")
    cases = {}
    for category in CATEGORIES:
        rates = sorted({r["rate"] for r in rows if r["category"] == category})
        if not rates:
            continue
        expected = {(category, rate, policy) for rate in rates for policy in POLICIES}
        if {k for k in lookup if k[0] == category} != expected:
            raise ValueError("Each category/rate requires matched RR and ELDR")
        per_rate = {
            str(rate): {
                metric: 100
                * (
                    lookup[category, rate, "eldr"][metric]
                    / lookup[category, rate, "rr"][metric]
                    - 1
                )
                for metric in METRICS
            }
            for rate in rates
        }
        cases[category] = dict(
            rates=rates,
            per_rate_change_pct=per_rate,
            mean_change_pct={
                m: sum(v[m] for v in per_rate.values()) / len(rates) for m in METRICS
            },
        )
    return dict(
        aggregation="Mean per-rate RR-relative change (%); negative is faster",
        cases=cases,
        source_runs=[
            {k: r[k] for k in ("setting", "rate", "policy", "raw", "sha256")}
            for r in rows
        ],
    )


def report(panels, stem):
    rows = [row for values in panels.values() for row in values]
    comparison = compare(rows)
    write_new_json(stem.with_suffix(".json"), comparison)
    table = [
        {
            "Workload": CATEGORIES[c],
            "TPOT P50 reduction (%)": -v["mean_change_pct"]["tpot50_ms"],
            "TPOT P99 reduction (%)": -v["mean_change_pct"]["tpot99_ms"],
        }
        for c, v in comparison["cases"].items()
    ]
    with stem.with_suffix(".csv").open("x") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Workload & TPOT P50 & TPOT P99 \\",
        r"\midrule",
    ]
    for row in table:
        lines.append(
            f"{row['Workload']} & {row['TPOT P50 reduction (%)']:.1f}\\% & "
            f"{row['TPOT P99 reduction (%)']:.1f}\\% " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    with stem.with_suffix(".tex").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main("single_category")
