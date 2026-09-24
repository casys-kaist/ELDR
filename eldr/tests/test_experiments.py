"""Experiment plan/replay checks; none of these tests launch GPU containers."""

import copy
import io
import json
import random
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eldr.experiments import cluster_balance, prefix_cache
from eldr.experiments.runner.config import FIT_SEED, RUN_PROFILES, file_sha256
from eldr.experiments.runner.experiment import (
    BASELINES,
    EXPERIMENTS,
    build_domain_mapping,
    collect,
    plan_experiment,
    prompt_hash,
    run_experiment,
)
from eldr.experiments.runner.experiment import (
    main as experiment_main,
)
from eldr.experiments.runner.fit import fit_centroids
from eldr.experiments.runner.workers import write_new_json
from eldr.tests.test_native_capture import module
from eldr.tests.test_reproducibility import raw_result


class ExperimentTests(unittest.TestCase):
    def test_public_experiments_are_exactly_the_six_ae_experiments(self):
        self.assertEqual(
            set(EXPERIMENTS),
            {
                "main_task",
                "main_language",
                "signature_ablation",
                "cluster_balance",
                "locality_band",
                "prefix_cache",
            },
        )
        for removed in ("homogeneous", "topology", "drift"):
            with self.subTest(experiment=removed), self.assertRaises(ValueError):
                plan_experiment(removed, [], Path("unused"), "paper", [60], ["rr"])

    def test_ae_refits_use_seed_one_without_changing_traffic(self):
        self.assertEqual(FIT_SEED, 1)
        self.assertTrue(
            all(profile["seed"] == 1234 for profile in RUN_PROFILES.values())
        )
        with patch("eldr.experiments.cluster_balance.fit_centroids") as fit:
            cluster_balance.prepare(
                dict(
                    directory="unused-output",
                    source_site=dict(
                        training_signatures="unused-counts",
                        centroids="unused-transform",
                        model="qwen",
                        decoders=[None] * 16,
                    ),
                )
            )
        self.assertEqual([call.args[4] for call in fit.call_args_list], [1, 1])

    def test_cli_config_preserves_planning_defaults_and_policy_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = [root / "qwen.json", root / "gemma.json"]
            for experiment in EXPERIMENTS:
                argv = [experiment, "--output", str(root / "out")]
                for config in configs:
                    argv += ["--config", str(config)]
                with (
                    self.subTest(experiment=experiment),
                    patch("sys.argv", argv),
                    patch("builtins.print"),
                    patch(
                        "eldr.experiments.runner.experiment.plan_experiment",
                        return_value={},
                    ) as plan,
                    patch(
                        "eldr.experiments.runner.experiment.run_experiment"
                    ) as execute,
                ):
                    experiment_main(experiment)
                    self.assertEqual(
                        plan.call_args.args[:4],
                        (experiment, configs, root / "out", "paper"),
                    )
                    if experiment in ("main_task", "main_language"):
                        self.assertEqual(plan.call_args.args[5], list(BASELINES))
                    execute.assert_not_called()
            with (
                patch(
                    "sys.argv",
                    [
                        "main_task",
                        "--config",
                        str(configs[0]),
                        "--output",
                        str(root / "out"),
                        "--profile",
                        "smoke",
                        "--policies",
                        "rr",
                        "eldr",
                        "--execute",
                    ],
                ),
                patch("builtins.print"),
                patch(
                    "eldr.experiments.runner.experiment.plan_experiment",
                    return_value={},
                ) as plan,
                patch("eldr.experiments.runner.experiment.run_experiment") as execute,
            ):
                experiment_main("main_task")
                self.assertEqual(
                    plan.call_args.args[3:6], ("smoke", [2], ["rr", "eldr"])
                )
                execute.assert_called_once_with({}, root / "out")

    def test_cli_rejects_unused_or_ignored_experiment_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            for experiment, flags, message in (
                ("main_task", ["--site", "old.json"], "unrecognized arguments"),
                ("main_task", ["--domain", "math"], "unrecognized arguments"),
                ("locality_band", ["--policies", "rr"], "unrecognized arguments"),
                *(
                    (
                        "main_task",
                        ["--replay", "saved", flag, value],
                        "uses saved settings",
                    )
                    for flag, value in (
                        ("--config", "config.json"),
                        ("--profile", "paper"),
                        ("--rates", "60"),
                        ("--policies", "rr"),
                    )
                ),
            ):
                with (
                    self.subTest(experiment=experiment, flags=flags),
                    patch(
                        "sys.argv",
                        [experiment, "--output", str(Path(directory) / "out"), *flags],
                    ),
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                    patch("eldr.experiments.runner.experiment.plan_experiment") as plan,
                    patch(
                        "eldr.experiments.runner.experiment.run_experiment"
                    ) as execute,
                    self.assertRaises(SystemExit) as error,
                ):
                    experiment_main(experiment)
                self.assertEqual(error.exception.code, 2)
                self.assertIn(message, stderr.getvalue())
                plan.assert_not_called()
                execute.assert_not_called()

    def config(self, root):
        rows = [
            {"conversations": [{"value": f"eval-{i}"}, {"value": "ok"}]}
            for i in range(2000)
        ]
        pairs = [[f"train-{i}", "math" if i % 2 else "code"] for i in range(8)]
        labels = {
            prompt_hash(f"eval-{i}"): "math" if i % 2 else "code" for i in range(2000)
        }
        labels.update({prompt_hash(text): domain for text, domain in pairs})
        for name, value in (
            ("eval", rows),
            ("train", pairs),
            ("labels", labels),
            ("fit", {}),
        ):
            write_new_json(root / f"{name}.json", value)
        np.save(root / "counts.npy", np.ones((8, 2, 4), dtype=np.int16))
        config = dict(
            model="qwen",
            repo=str(root),
            prefills=[{}] * 8,
            decoders=[{}] * 16,
            dataset=str(root / "eval.json"),
            centroids=str(root / "fit.json"),
            training_prompts=str(root / "train.json"),
            labels=str(root / "labels.json"),
            training_signatures=str(root / "counts.npy"),
            paired_signatures=str(root / "counts.npy"),  # Plan-only fixture.
            first_token_mode="prefill",
        )
        write_new_json(root / "config.json", config)
        return config, root / "config.json"

    def test_plans_freeze_rates_warmups_and_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = self.config(root)
            for experiment, expected in (
                ("main_task", 6),
                ("main_language", 6),
                ("locality_band", 5),
                ("cluster_balance", 3),
                ("prefix_cache", 2),
                ("signature_ablation", 2),
            ):
                with (
                    self.subTest(experiment=experiment),
                    patch(
                        "eldr.experiments.runner.experiment.load_config",
                        return_value=config,
                    ),
                ):
                    plan = plan_experiment(
                        experiment,
                        [path],
                        root / experiment,
                        "paper",
                        [20, 60],
                        BASELINES,
                    )
                for worker_group in plan["fleets"]:
                    trials = worker_group["trials"]
                    self.assertEqual(len(trials), 2 * expected)
                    self.assertEqual(trials[0][2]["warmup_requests"], 14400)
                    self.assertTrue(
                        all(p["warmup_requests"] == 3840 for _, _, p in trials[1:])
                    )
                    self.assertTrue(
                        all(
                            p["requests"] == 120 * p["request_rate"]
                            for _, _, p in trials
                        )
                    )
                    self.assertEqual(
                        {p for _, p, _ in trials if p.startswith("eldr")},
                        {"eldr-static"},
                    )
                    # Each variant reuses its fit across request rates.
                    by_variant = {}
                    for _, policy, protocol in trials:
                        if policy == "eldr-static":
                            by_variant.setdefault(protocol["variant"], set()).add(
                                protocol.get(
                                    "centroids", worker_group["site"]["centroids"]
                                )
                            )
                    self.assertTrue(
                        all(len(paths) == 1 for paths in by_variant.values())
                    )
                    self.assertIn(config["centroids"], plan["inputs"])
                self.assertFalse((root / experiment).exists())
                if experiment == "prefix_cache":
                    self.assertEqual(
                        {f["site"]["centroids"] for f in plan["fleets"]},
                        {config["centroids"]},
                    )
                    self.assertEqual(
                        [f["site"]["prefix_cache"] for f in plan["fleets"]],
                        [False, True],
                    )
                    for worker_group in plan["fleets"]:
                        self.assertEqual(
                            {policy for _, policy, _ in worker_group["trials"]},
                            {"rr", "eldr-static"},
                        )
                        self.assertTrue(
                            all(
                                p["shuffle"] and not p["no_oversample"]
                                for _, _, p in worker_group["trials"]
                            )
                        )
            self.assertEqual(RUN_PROFILES["paper"]["warmup_requests"], 3840)

    def test_prefix_pool_is_seeded_identical_across_cache_modes_and_not_expanded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self.config(root)
            rows = [
                {"conversations": [{"value": f"eval-{i}"}, {"value": "ok"}]}
                for i in range(10668)
            ]
            Path(config["dataset"]).write_text(json.dumps(rows))
            original_hash = file_sha256(config["dataset"])
            groups = prefix_cache.configure(config, root, [])
            expected = random.Random(42).sample(rows, 2000)
            self.assertEqual(
                [r["conversations"][0]["value"] for r in expected[:5]],
                [f"eval-{i}" for i in (10476, 1824, 409, 4506, 4012)],
            )
            for group in groups:
                Path(group["directory"]).mkdir()
                prefix_cache.prepare(group)
                actual = json.loads(Path(group["site"]["dataset"]).read_text())
                self.assertEqual(actual, expected)
                self.assertEqual(len(actual), 2000)
                with self.assertRaises(FileExistsError):
                    prefix_cache.prepare(group)
            self.assertEqual(file_sha256(config["dataset"]), original_hash)
            for invalid in (rows[:1999], [rows[0]] * 2000):
                Path(config["dataset"]).write_text(json.dumps(invalid))
                with self.assertRaisesRegex(ValueError, "2,000"):
                    prefix_cache.prepare(groups[0])

    def test_new_experiment_runs_recomputes_and_renders_every_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = self.config(root)
            output = root / "out"
            with patch(
                "eldr.experiments.runner.experiment.load_config", return_value=config
            ):
                plan = plan_experiment(
                    "main_task",
                    [path],
                    output,
                    "smoke",
                    [2],
                    ["rr", "jsq", "eldr"],
                )
            for _, _, protocol in plan["fleets"][0]["trials"]:
                protocol.update(requests=2, output_tokens=4)

            def prepare_run(config, run, profile):
                run.mkdir()

            def run_fake(run, *, trials):
                write_new_json(run / "trials.json", trials)
                for label, _, protocol in trials:
                    folder = run / label / "measure"
                    folder.mkdir(parents=True)
                    raw = raw_result()
                    raw["request_rate"] = protocol["request_rate"]
                    write_new_json(folder / "raw.json", raw)
                write_new_json(run / "complete.json", {})
                write_new_json(run / "cleanup.json", {"remaining": []})

            with (
                patch(
                    "eldr.experiments.runner.experiment.load_config",
                    return_value=config,
                ),
                patch(
                    "eldr.experiments.runner.experiment.prepare",
                    side_effect=prepare_run,
                ),
                patch(
                    "eldr.experiments.runner.experiment.execute", side_effect=run_fake
                ),
            ):
                run_experiment(plan, output)
            rows = collect(plan, output)
            self.assertEqual(len(rows), 3)
            self.assertAlmostEqual(rows[0]["tpot95_ms"], 39.0)
            self.assertTrue((output / "main_task.pdf").stat().st_size > 1000)
            self.assertTrue((output / "summary.csv").is_file())
            self.assertEqual(rows, json.loads((output / "summary.json").read_text()))
            relocated = root / "moved"
            output.rename(relocated)
            self.assertEqual(collect(plan, relocated), rows)
            rendered = root / "redrawn with spaces"
            command = [
                "bash",
                str(
                    Path(__file__).resolve().parents[1]
                    / "scripts/plot_fig10_main_task.sh"
                ),
                str(relocated),
                "--output",
                str(rendered),
            ]
            replay = subprocess.run(
                command, cwd=root, capture_output=True, text=True, timeout=60
            )
            self.assertEqual(replay.returncode, 0, replay.stderr)
            self.assertTrue((rendered / "main_task.pdf").stat().st_size > 1000)
            self.assertEqual(rows, json.loads((rendered / "summary.json").read_text()))
            (relocated / "qwen/run/complete.json").unlink()
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                collect(plan, relocated)

    def test_domain_control_uses_calibration_not_evaluation_frequencies(self):
        pairs = [[str(i), "a" if i < 8 else "b"] for i in range(10)]
        result = build_domain_mapping(pairs, {"arbitrary": "b"}, 4)
        self.assertEqual(result["dom2dec"], {"a": [0, 1, 2]})
        self.assertEqual(result["misc_dec"], [3])

    def test_changed_input_is_rejected_before_fleet_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input"
            source.write_text("original")
            plan = dict(inputs={str(source): file_sha256(source)})
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "input changed"):
                run_experiment(plan, root / "out")
            self.assertFalse((root / "out").exists())

    def test_vanilla_fit_retains_counts_geometry_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "counts.npy", np.ones((8, 2, 4), dtype=np.int16))
            for balanced in (False, True):
                result = fit_centroids(
                    root / "counts.npy",
                    root / f"{balanced}.json",
                    "toy",
                    2,
                    58,
                    balanced=balanced,
                )
                self.assertEqual(result["algo"], "kmbal" if balanced else "km")
                self.assertEqual(result["layer_mask"], [0, 1])
                np.testing.assert_allclose(
                    np.linalg.norm(result["centroids"], axis=1), 1
                )

    def test_capture_switch_supports_cache_off_without_relaxing_empty_guard(self):
        from types import SimpleNamespace as NS

        control = module("control")
        config = NS(
            parallel_config=NS(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                data_parallel_size=1,
                enable_expert_parallel=False,
                distributed_executor_backend="uni",
            ),
            cache_config=NS(
                enable_prefix_caching=True, num_gpu_blocks=16, block_size=64
            ),
            scheduler_config=NS(
                async_scheduling=False,
                enable_chunked_prefill=False,
                disable_hybrid_kv_cache_manager=True,
            ),
            model_config=NS(enforce_eager=True, max_model_len=4096),
            speculative_config=None,
            kv_transfer_config=NS(kv_role="kv_producer", kv_connector="NixlConnector"),
        )
        for enabled in (False, True):
            config.cache_config.enable_prefix_caching = enabled
            control.require_supported(config, "EngineCore", 123, 123, True, "on")
            with self.assertRaises(ValueError):
                control.require_supported(config, "EngineCore", 123, 456, True, "on")
            unsafe = copy.deepcopy(config)
            unsafe.scheduler_config.async_scheduling = True
            with self.assertRaises(ValueError):
                control.require_supported(unsafe, "EngineCore", 123, 123, True, "on")


if __name__ == "__main__":
    unittest.main()
