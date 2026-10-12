# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only generalization plans, synthetic run/replay and tamper checks."""

import copy
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eldr.experiments import single_category, workload_drift
from eldr.experiments.drift_workload import DOMAINS, build
from eldr.experiments.runner.config import IMAGE, file_sha256
from eldr.experiments.runner.experiment import (
    check_resume,
    plan_experiment,
    prompt_hash,
    run_experiment,
)
from eldr.experiments.runner.fit import training_dataset
from eldr.experiments.runner.results import read_panels, request_metrics, write_report
from eldr.experiments.runner.workers import source_files, write_new_json


def raw_result(count, rate, tokens, tpot=20):
    values = np.arange(count, dtype=float) / max(count, 1) + tpot
    raw = dict(
        completed=count,
        failed=0,
        request_rate=rate,
        input_lens=[10] * count,
        output_lens=[tokens] * count,
        errors=[""] * count,
        ttfts=[0.06] * count,
        itls=[[v * (tokens - 1) / 1000] for v in values],
        start_times=(1000 + np.arange(count) / rate).tolist(),
    )
    for prefix, data in (("tpot", values), ("ttft", np.full(count, 60.0))):
        for key, value in (
            ("mean", data.mean()),
            ("median", np.median(data)),
            ("std", data.std()),
            ("p99", np.percentile(data, 99)),
        ):
            raw[f"{key}_{prefix}_ms"] = float(value)
    return raw


def fixture_config(root, name, domains, category=None):
    folder = root / name
    folder.mkdir()
    pairs = [[f"{name}-train-{d}-{i}", d] for d in domains for i in range(16)]
    evaluation = [(f"{name}-eval-{d}-{i}", d) for d in domains for i in range(20)]
    write_new_json(folder / "training.json", pairs)
    write_new_json(
        folder / "evaluation.json",
        [
            {
                "conversations": [
                    {"from": "human", "value": text},
                    {"from": "gpt", "value": "ok"},
                ]
            }
            for text, _ in evaluation
        ],
    )
    write_new_json(
        folder / "labels.json", {prompt_hash(text): d for text, d in pairs + evaluation}
    )
    write_new_json(
        folder / "centroids.json",
        dict(
            L=2,
            e=4,
            k=16,
            variant="count_idf",
            layer_mask=[0],
            sig_idf=[1, 2, 3, 4],
            centroids=[[0.5] * 4] * 16,
        ),
    )
    config = dict(
        model="qwen",
        repo=str(root),
        category=category,
        prefills=[{}] * 8,
        decoders=[{}] * 16,
        dataset=str(folder / "evaluation.json"),
        training_prompts=str(folder / "training.json"),
        labels=str(folder / "labels.json"),
        centroids=str(folder / "centroids.json"),
        signature_variant="count_idf",
        prefix_cache=True,
    )
    for key in ("dataset", "centroids"):
        config[key + "_sha256"] = file_sha256(config[key])
    path = folder / "config.json"
    write_new_json(path, config)
    return path


def fake_prepare(site, output, profile, *, training=None):
    """Write the real runner's evidence schema without cluster/compiled extensions."""
    output.mkdir()
    capture = None
    if training is not None:
        rows, digest = training_dataset(training, Path(site["dataset"]))
        write_new_json(output / "training.json", rows)
        capture = dict(
            dataset=str(output / "training.json"),
            dataset_sha256=file_sha256(output / "training.json"),
            source_sha256=digest,
            requests=len(rows),
        )
    root = Path(site["repo"])
    write_new_json(
        output / "prepared.json",
        dict(
            site=site,
            capture=capture,
            profile=profile,
            image=IMAGE,
            sources={
                str(p.relative_to(root)): file_sha256(p) for p in source_files(root)
            },
        ),
    )
    return output


def fake_execute(output, *, trials=None):
    prepared = json.loads((output / "prepared.json").read_text())
    capture = prepared["capture"]
    if capture:
        trials = [
            (
                "rr",
                "rr",
                dict(
                    requests=capture["requests"],
                    request_rate=10,
                    output_tokens=2,
                ),
            )
        ]
        counts = np.random.default_rng(1).integers(
            1, 20, (capture["requests"], 2, 4), dtype=np.int16
        )
        (output / "signatures.bin").write_bytes(
            b"".join(
                struct.pack("<I", row.nbytes) + row.astype("<i2").tobytes()
                for row in counts
            )
        )
    write_new_json(output / "trials.json", trials)
    for label, policy, protocol in trials:
        for stage in ("measure",) if capture else ("warmup", "measure"):
            folder = output / label / stage
            folder.mkdir(parents=True)
            measured = stage == "measure"
            count = protocol["requests" if measured else "warmup_requests"]
            rate = protocol["request_rate" if measured else "warmup_rate"]
            raw = raw_result(count, rate, protocol["output_tokens"])
            tpot, ttft = request_metrics(raw, count, protocol["output_tokens"])
            write_new_json(folder / "raw.json", raw)
            write_new_json(
                folder / "metrics.json",
                dict(
                    policy=policy,
                    protocol=protocol,
                    requests=count,
                    tpot_ms={
                        str(q): float(np.percentile(tpot, q)) for q in (50, 95, 99)
                    },
                    ttft50_ms=float(np.median(ttft)),
                ),
            )
            enabled = bool(capture) or policy.startswith("eldr")
            state = dict.fromkeys(
                (
                    "generation_empty",
                    "prefix_reset",
                    "prefix_cache_enabled",
                    "same_process_checked",
                    "resident_signature_cache",
                ),
                True,
            ) | dict.fromkeys(
                (
                    "enabled",
                    "signature_enabled",
                    "transport_enabled",
                ),
                enabled,
            )
            write_new_json(folder / "capture.json", [state] * 8)
            if policy.startswith("eldr"):
                online = policy == "eldr"
                write_new_json(
                    folder / "routing.json",
                    dict(
                        policy=policy,
                        centroid_mode="online" if online else "static",
                        rule="locality_band_jsq_v1",
                        selector="jsq",
                        signature="count_idf",
                        tau=0.1,
                        inflight_by_worker=[0] * 16,
                        refit=dict(
                            last_error=None,
                            balanced=True,
                            refits_completed=3,
                            window_seconds=5,
                            refit_seconds=1,
                        )
                        if online
                        else None,
                    ),
                )
    write_new_json(output / "cleanup.json", dict(remaining=[]))
    write_new_json(output / "complete.json", {})


class GeneralizationTests(unittest.TestCase):
    def test_plans_are_read_only_and_freeze_camera_ready_protocols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            math = fixture_config(root, "task", DOMAINS, "math")
            english = fixture_config(
                root, "language", ("English", "Chinese"), "English"
            )
            before = sorted(root.rglob("*"))
            with (
                patch(
                    "eldr.experiments.runner.experiment.load_config",
                    side_effect=lambda p, **_: json.loads(p.read_text()),
                ),
                patch(
                    "eldr.experiments.runner.remote.Remote.run",
                    side_effect=AssertionError("No cluster access"),
                ),
                patch("eldr.experiments.single_category.execute") as gpu,
            ):
                category = plan_experiment(
                    "single_category",
                    [math, english],
                    root / "out",
                    "paper",
                    list(single_category.RATES),
                    ["rr", "eldr"],
                )
                self.assertEqual(sum(len(g["trials"]) for g in category["fleets"]), 20)
                drift = plan_experiment(
                    "workload_drift",
                    [math],
                    root / "drift",
                    "paper",
                    [100],
                    list(workload_drift.POLICIES),
                )
                trials = drift["fleets"][0]["trials"]
                self.assertEqual(
                    [p for _, p, _ in trials], list(workload_drift.POLICIES)
                )
                self.assertEqual(
                    [p["warmup_requests"] for _, _, p in trials], [14400, 3840, 3840]
                )
                for _, _, p in trials:
                    self.assertEqual(
                        (p["requests"], p["output_tokens"], p["seed"]),
                        (15000, 512, 1234),
                    )
                    self.assertFalse(p["shuffle"])
                    self.assertTrue(p["no_oversample"])
                with self.assertRaisesRegex(ValueError, "100 req/s"):
                    plan_experiment(
                        "workload_drift",
                        [math],
                        root / "bad",
                        "paper",
                        [80],
                        list(workload_drift.POLICIES),
                    )
                with self.assertRaisesRegex(ValueError, "every rate"):
                    plan_experiment(
                        "single_category",
                        [math],
                        root / "bad",
                        "paper",
                        [80],
                        ["rr", "eldr"],
                    )
                gpu.assert_not_called()
            self.assertEqual(sorted(root.rglob("*")), before)

    def test_ordered_drift_is_deterministic_and_uses_actual_arrival_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(fixture_config(root, "task", DOMAINS).read_text())
            args = (Path(config["dataset"]), Path(config["labels"]))
            manifests = [
                build(*args, root / name, 100, 120, 42, guard_s=15)
                for name in ("a", "b")
            ]
            self.assertEqual(manifests[0], manifests[1])
            data = root / "a/task_domain_gradual.json"
            self.assertEqual(file_sha256(data), manifests[0]["dataset"]["sha256"])
            rows = json.loads(data.read_text())
            self.assertEqual(len(rows), 15000)
            self.assertEqual(
                (rows[0]["_eldr_second"], rows[-1]["_eldr_second"]), (-15, 134)
            )
            with self.assertRaises(FileExistsError):
                build(*args, root / "a", 100, 120, 42, guard_s=15)
            raw = raw_result(15000, 100, 512)
            # Actual arrivals are 10% slower than nominal index/rate estimates.
            raw["start_times"] = (1000 + np.arange(15000) * 0.011).tolist()
            windows = workload_drift.windowed(raw, [r["_eldr_domain"] for r in rows])
            self.assertEqual(len(windows), 24)
            self.assertEqual(windows[0]["requests"], 455)
            self.assertNotEqual(windows[0]["requests"], 500)
            raw["start_times"][-1] = raw["start_times"][-2] - 1
            with self.assertRaisesRegex(ValueError, "input order"):
                workload_drift.windowed(raw, [r["_eldr_domain"] for r in rows])

    def test_category_table_keeps_regressions_and_averages_relative_changes(self):
        rows = []
        for category in single_category.CATEGORIES:
            for rate, rr, eldr in ((20, 10, 9), (100, 100, 110)):
                for policy, value in (("rr", rr), ("eldr", eldr)):
                    rows.append(
                        dict(
                            category=category,
                            rate=rate,
                            policy=policy,
                            setting=category,
                            raw=f"{category}/{rate}-{policy}",
                            sha256="test",
                            **dict.fromkeys(single_category.METRICS, value),
                        )
                    )
        comparison = single_category.compare(rows)
        for case in comparison["cases"].values():
            self.assertAlmostEqual(case["mean_change_pct"]["tpot50_ms"], 0)
            self.assertGreater(case["per_rate_change_pct"]["100"]["tpot50_ms"], 0)
        with self.assertRaisesRegex(ValueError, "matched RR"):
            single_category.compare(rows[:-1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report({"Workload": rows}, root, "single_category", stem="tab2")
            self.assertTrue((root / "tab2-summary.json").is_file())
            self.assertEqual(json.loads((root / "tab2.json").read_text()), comparison)
            self.assertIn(r"\begin{tabular}", (root / "tab2.tex").read_text())

    def test_both_studies_execute_replay_and_reject_changed_evidence_without_gpus(self):
        for study in ("single_category", "workload_drift"):
            with self.subTest(study=study), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                code = root / "eldr/serving/source.py"
                code.parent.mkdir(parents=True)
                code.write_text("# frozen source fixture\n")
                configs = [fixture_config(root, "task", DOMAINS, "math")]
                if study == "single_category":
                    configs.append(
                        fixture_config(root, "language", ("English",), "English")
                    )
                output = root / "run"
                with (
                    patch(
                        "eldr.experiments.runner.experiment.load_config",
                        side_effect=lambda p, **_: json.loads(p.read_text()),
                    ),
                    patch(
                        "eldr.experiments.runner.config.MODEL_GEOMETRY",
                        {"qwen": (2, 4, 2)},
                    ),
                    patch(
                        "eldr.experiments.runner.experiment.prepare",
                        side_effect=fake_prepare,
                    ),
                    patch(
                        "eldr.experiments.single_category.prepare_run",
                        side_effect=fake_prepare,
                    ),
                    patch(
                        "eldr.experiments.runner.experiment.execute",
                        side_effect=fake_execute,
                    ),
                    patch(
                        "eldr.experiments.single_category.execute",
                        side_effect=fake_execute,
                    ) as capture,
                    patch(
                        "eldr.experiments.runner.remote.Remote.run",
                        side_effect=AssertionError("No cluster access"),
                    ),
                ):
                    plan = plan_experiment(
                        study, configs, output, "smoke", [2], ["rr", "eldr"]
                    )
                    saved_plan = copy.deepcopy(plan)
                    run_experiment(plan, output)
                    self.assertEqual(
                        capture.call_count, 2 if study == "single_category" else 0
                    )
                    panels = read_panels(output, study)
                    self.assertEqual(
                        sum(map(len, panels.values())), 4 if capture.call_count else 3
                    )
                    self.assertTrue(check_resume(saved_plan, output))
                    replay = root / "replay"
                    replay.mkdir()
                    write_report(panels, replay, study, stem="paper")
                    self.assertTrue(
                        (
                            replay
                            / ("paper.tex" if capture.call_count else "paper.pdf")
                        ).is_file()
                    )
                    # Completion cannot bypass cleanup, raw or routing checks.
                    group = saved_plan["fleets"][0]
                    trial = group["trials"][-1][0]
                    stage = output / group["key"] / "run" / trial / "measure"
                    for path, mutate in (
                        (
                            output / group["key"] / "run/cleanup.json",
                            lambda d: d.update(remaining=["owned"]),
                        ),
                        (
                            stage / "raw.json",
                            lambda d: d["errors"].__setitem__(0, "failed"),
                        ),
                        (
                            stage / "routing.json",
                            lambda d: d["refit"].update(refit_seconds=10),
                        ),
                        (output / f"{study}.json", lambda d: d.update(unverified=True)),
                    ):
                        old = path.read_bytes()
                        value = json.loads(old)
                        mutate(value)
                        path.write_text(json.dumps(value))
                        with self.assertRaises(ValueError):
                            read_panels(output, study)
                        path.write_bytes(old)
                    if capture.call_count:
                        path = output / group["key"] / "capture/cleanup.json"
                        path.write_text('{"remaining": ["calibration-worker"]}')
                        with self.assertRaisesRegex(ValueError, "Cleanup"):
                            check_resume(saved_plan, output)


if __name__ == "__main__":
    unittest.main()
