# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eldr.experiments.runner.inputs import configure_inputs
from eldr.experiments.runner.results import (
    checked_path,
    request_metrics,
)


def raw_result():
    raw = dict(
        completed=2,
        failed=0,
        request_rate=60,
        errors=["", ""],
        input_lens=[10, 20],
        output_lens=[4, 4],
        ttfts=[0.04, 0.06],
        itls=[[0.01, 0.02, 0.03], [0.03, 0.09]],
    )
    # Second request bundles three tokens into two events: divide by 3, not 2.
    for prefix, values in (
        ("tpot", np.array([20.0, 40.0])),
        ("ttft", np.array([40.0, 60.0])),
    ):
        for key, value in (
            ("mean", values.mean()),
            ("median", np.median(values)),
            ("std", values.std()),
            ("p99", np.percentile(values, 99)),
        ):
            raw[f"{key}_{prefix}_ms"] = float(value)
    return raw


class ReproducibilityTests(unittest.TestCase):
    def test_input_bundle_relocation_validation_and_no_cluster_access(self):
        fields = (
            "dataset",
            "centroids",
            "training_prompts",
            "labels",
            "training_signatures",
            "paired_signatures",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original"
            source.mkdir()
            payload = b'{"seed": 1}'  # Other content parsers are tested separately.
            (source / "data.bin").write_bytes(payload)
            manifest = dict(
                schema_version=1,
                files=[
                    dict(path="data.bin", sha256=hashlib.sha256(payload).hexdigest())
                ],
                settings={
                    f"{model}-{workload}": dict.fromkeys(fields, "data.bin")
                    for model in ("qwen", "gptoss", "gemma")
                    for workload in ("task", "language")
                },
            )
            (source / "manifest.json").write_text(json.dumps(manifest))
            inputs = root / "relocated inputs"
            shutil.copytree(source, inputs)
            cluster = root / "cluster.json"
            cluster.write_text(
                json.dumps(
                    dict(
                        model_paths={
                            m: str(root / m) for m in ("qwen", "gptoss", "gemma")
                        },
                        prefills=[dict(gpu=i) for i in range(8)],
                        decoders=[dict(gpu=i) for i in range(16)],
                    )
                )
            )
            with (
                patch(
                    "eldr.experiments.runner.config.load_config",
                    side_effect=lambda path, **kw: json.loads(path.read_text()),
                ),
                patch(
                    "eldr.experiments.runner.experiment.load_labeled_inputs",
                    return_value=([], [None] * 16, {}),
                ),
                patch(
                    "eldr.experiments.runner.fit.read_capture",
                    return_value=np.ones((16, 2, 2)),
                ) as counts,
                patch(
                    "eldr.experiments.signature_ablation.read_paired_capture",
                    return_value=(np.ones((16, 2, 2)),) * 3,
                ),
                patch(
                    "eldr.experiments.runner.remote.Remote.run",
                    side_effect=AssertionError("No SSH allowed"),
                ),
            ):
                output = root / "configs"
                self.assertEqual(
                    configure_inputs(inputs, cluster, output)["configs"], 7
                )
                for path in output.glob("*.json"):
                    config = json.loads(path.read_text())
                    self.assertEqual(config["dataset"], str(inputs / "data.bin"))
                    self.assertEqual(
                        config["dataset_sha256"], manifest["files"][0]["sha256"]
                    )
                    self.assertEqual(
                        len(config["prefills"]),
                        1 if path.stem.endswith("-prefix") else 8,
                    )
                    self.assertEqual(len(config["decoders"]), 16)
                with self.assertRaisesRegex(ValueError, "already exists"):
                    configure_inputs(inputs, cluster, output)
                counts.return_value = np.ones((15, 2, 2))
                with self.assertRaisesRegex(ValueError, "row-count mismatch"):
                    configure_inputs(inputs, cluster, root / "invalid-count")
                self.assertFalse((root / "invalid-count").exists())
                counts.return_value = np.ones((16, 2, 2))
                (inputs / "data.bin").write_bytes(b'{"seed": 0}')
                wrong_seed = copy.deepcopy(manifest)
                wrong_seed["files"][0]["sha256"] = hashlib.sha256(
                    (inputs / "data.bin").read_bytes()
                ).hexdigest()
                (inputs / "manifest.json").write_text(json.dumps(wrong_seed))
                with self.assertRaisesRegex(ValueError, "fitting seed 1"):
                    configure_inputs(inputs, cluster, root / "invalid-seed")
                self.assertFalse((root / "invalid-seed").exists())
                (inputs / "data.bin").write_bytes(payload)
                for bad in (
                    dict(manifest, files=manifest["files"] * 2),
                    dict(manifest, settings={}),
                    dict(
                        manifest,
                        files=[dict(path="../outside", sha256="bad")],
                        settings={
                            k: dict.fromkeys(fields, "../outside")
                            for k in manifest["settings"]
                        },
                    ),
                ):
                    (inputs / "manifest.json").write_text(json.dumps(bad))
                    with self.assertRaises(ValueError):
                        configure_inputs(inputs, cluster, root / "invalid")
                    self.assertFalse((root / "invalid").exists())
                (inputs / "manifest.json").write_text(json.dumps(manifest))
                (inputs / "data.bin").write_bytes(payload + b"changed")
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    configure_inputs(inputs, cluster, root / "invalid")

    def test_figure_scripts_forward_quoted_arguments_and_exit_status(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        figures = {
            "fig10_main_task": "main_task",
            "fig11_main_language": "main_language",
            "fig13_signature": "signature_ablation",
            "fig14_cluster_balance": "cluster_balance",
            "fig15_locality_band": "locality_band",
            "fig16_prefix_cache": "prefix_cache",
        }
        self.assertEqual(
            {p.name for p in scripts.glob("*.sh")},
            {f"{prefix}{stem}.sh" for stem in figures for prefix in ("", "plot_")}
            | {"run_all.sh", "plot_all.sh"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout with spaces"
            target = root / "eldr/scripts"
            target.mkdir(parents=True)
            interpreter = root / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text('#!/bin/sh\nprintf "%s\\n" "$PWD" "$@"\nexit 19\n')
            interpreter.chmod(0o755)
            (target / "run_all.sh").write_text(
                '#!/bin/sh\nprintf "%s\\n" "$@"\nexit 19\n'
            )
            for stem, module in figures.items():
                for prefix in ("", "plot_"):
                    name = f"{prefix}{stem}.sh"
                    shutil.copyfile(scripts / name, target / name)
                    arguments = ["--output", "output with spaces"]
                    if prefix:
                        arguments = ["saved run with spaces", *arguments]
                    else:
                        arguments = ["--config", "config with spaces.json", *arguments]
                    with self.subTest(script=name):
                        subprocess.run(["bash", "-n", str(target / name)], check=True)
                        result = subprocess.run(
                            ["bash", str(target / name), *arguments],
                            cwd=directory,
                            capture_output=True,
                            text=True,
                            timeout=15,
                        )
                        self.assertEqual(result.returncode, 19, result.stderr)
                        self.assertEqual(
                            result.stdout.splitlines(),
                            [
                                str(root),
                                "-m",
                                f"eldr.experiments.{module}",
                                "--replay",
                                *arguments,
                            ]
                            if prefix
                            else ["--figure", stem, *arguments],
                        )

    def test_figure_scripts_help_missing_arguments_and_overwrite_guards(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory() as directory:
            for script in scripts.glob("*.sh"):
                plotting = script.name.startswith("plot_")
                inputs = ["saved"] if plotting else ["--config", "config.json"]
                for arguments, code in (
                    (["--help"], 0),
                    ([] if plotting else ["--config", "/nonexistent/eldr.json"], 2),
                    ([*inputs, "--output", directory], 2),
                    *(
                        [(["saved", "--execute", "--output", directory], 2)]
                        if plotting
                        else []
                    ),
                ):
                    with self.subTest(script=script.name, arguments=arguments):
                        result = subprocess.run(
                            ["bash", str(script), *arguments],
                            cwd=directory,
                            capture_output=True,
                            text=True,
                            timeout=30,
                        )
                        self.assertEqual(result.returncode, code, result.stderr)
                        self.assertIn("usage:", result.stdout + result.stderr)

    def test_run_all_orders_scripts_and_stops_without_retry(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout with spaces"
            shutil.copytree(scripts, root / "eldr/scripts")
            (root / "eldr/inputs").mkdir()
            (root / "eldr/inputs/cluster.json").write_text("{}")
            (root / "cluster with spaces.json").write_text("{}")
            binaries = root / "bin"
            binaries.mkdir()
            (binaries / "uv").write_text("#!/bin/sh\nexit 0\n")
            (binaries / "uv").chmod(0o755)
            # Do not contend with a real AE run on the shared controller.
            (binaries / "flock").write_text('#!/bin/sh\nexit "${ELDR_TEST_BUSY:-0}"\n')
            (binaries / "flock").chmod(0o755)
            interpreter = root / ".venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_text(
                f"#!{sys.executable}\n"
                + textwrap.dedent("""\
                    import json, os, sys
                    from pathlib import Path
                    args = sys.argv[1:]
                    with Path('calls.jsonl').open('a') as stream:
                        stream.write(json.dumps(args) + '\\n')
                    if os.environ.get('ELDR_TEST_FAIL') in args:
                        sys.exit(19)
                    if 'configure' in args:
                        Path(args[args.index('--output') + 1]).mkdir(parents=True)
                    print('fixture: no GPU or SSH commands')
                    """)
            )
            interpreter.chmod(0o755)
            expected = [
                ("main_task", "task"),
                ("main_language", "language"),
                ("signature_ablation", "task"),
                ("signature_ablation", "language"),
                ("cluster_balance", "task"),
                ("cluster_balance", "language"),
                ("locality_band", "task"),
                ("locality_band", "language"),
                ("prefix_cache", "task-prefix"),
            ]
            trace = root / "calls.jsonl"
            for mode in (
                "plan",
                "execute",
                "failure",
                "defaults",
                "single",
                "bootstrap-failure",
                "busy",
            ):
                output = root / ("output " + mode)
                trace.write_text("")
                cmd = [
                    "bash",
                    str(root / "eldr/scripts/run_all.sh"),
                    "--config",
                    "cluster with spaces.json",
                    "--output",
                    str(output),
                ]
                if mode == "plan":
                    cmd += ["--plan"]
                else:
                    cmd += ["--profile", "smoke"]
                if mode == "defaults":
                    del cmd[2:4]  # Default bundled config; no --execute needed.
                if mode == "single":
                    cmd += ["--figure", "fig13_signature"]
                result = subprocess.run(
                    cmd,
                    cwd=directory,
                    text=True,
                    capture_output=True,
                    timeout=20,
                    env=dict(
                        os.environ,
                        PATH=str(binaries) + os.pathsep + os.environ["PATH"],
                        ELDR_TEST_FAIL=(
                            "eldr.experiments.main_language"
                            if mode == "failure"
                            else "bootstrap"
                            if mode == "bootstrap-failure"
                            else ""
                        ),
                        ELDR_TEST_BUSY="1" if mode == "busy" else "0",
                    ),
                )
                self.assertEqual(
                    result.returncode,
                    1 if mode == "busy" else 19 if "failure" in mode else 0,
                    result.stderr,
                )
                calls = [json.loads(row) for row in trace.read_text().splitlines()]
                if mode == "busy":
                    self.assertEqual(calls, [])
                    self.assertFalse(output.exists())
                    continue
                self.assertEqual(
                    calls[0][1:3], ["eldr.experiments.runner", "configure"]
                )
                start = 1 if mode == "plan" else 2
                if start == 2:
                    self.assertEqual(
                        calls[1][1:3], ["eldr.experiments.runner", "bootstrap"]
                    )
                selected = expected[2:4] if mode == "single" else expected
                if mode == "bootstrap-failure":
                    selected = []
                self.assertEqual(
                    len(calls), 4 if mode == "failure" else start + len(selected)
                )
                for args, (module, workload) in zip(calls[start:], selected):
                    self.assertEqual(args[1], "eldr.experiments." + module)
                    self.assertEqual("--execute" in args, mode != "plan")
                    self.assertEqual(
                        args[args.index("--profile") + 1],
                        "paper" if mode == "plan" else "smoke",
                    )
                    models = (
                        ("gptoss",)
                        if module == "prefix_cache"
                        else ("qwen", "gptoss", "gemma")
                    )
                    self.assertEqual(
                        [
                            args[i + 1]
                            for i, value in enumerate(args)
                            if value == "--config"
                        ],
                        [
                            str(output / "configs" / f"{model}-{workload}.json")
                            for model in models
                        ],
                    )
                    self.assertTrue(
                        Path(args[args.index("--output") + 1]).is_relative_to(output)
                    )
                self.assertEqual(len(list(output.glob("*.log"))), len(calls) - start)
                if mode == "failure":
                    self.assertIn("Stopped; see", result.stderr)
                # Repeating an output never relaunches jobs or overwrites results.
                again = subprocess.run(
                    cmd, cwd=directory, capture_output=True, timeout=20
                )
                self.assertEqual(again.returncode, 2)
                self.assertEqual(len(trace.read_text().splitlines()), len(calls))

    def test_plot_all_validates_inputs_and_stops_without_overwriting(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        entries = (
            "fig10_main_task",
            "fig11_main_language",
            "fig13_signature/task",
            "fig13_signature/language",
            "fig14_cluster_balance/task",
            "fig14_cluster_balance/language",
            "fig15_locality_band/task",
            "fig15_locality_band/language",
            "fig16_prefix_cache",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "checkout with spaces"
            target = root / "eldr/scripts"
            target.mkdir(parents=True)
            shutil.copyfile(scripts / "plot_all.sh", target / "plot_all.sh")
            for entry in entries:
                path = root / "saved run" / entry
                path.mkdir(parents=True)
                (path / "complete.json").write_text("{}")
                (target / ("plot_" + entry.split("/")[0] + ".sh")).write_text(
                    '#!/bin/bash\nset -eu\nprintf "%s\\n" "$1" "$3" >> calls.txt\n'
                    'if [[ $1 == *"${ELDR_TEST_FAIL:-never}"* ]]; then exit 19; fi\n'
                    'mkdir -p -- "$3"\nprintf "test plot\\n" > "$3/plot.pdf"\n'
                )
            trace = root / "calls.txt"
            marker = root / "saved run/fig16_prefix_cache/complete.json"
            for mode in ("missing", "failure", "success"):
                output = root / ("plots " + mode)
                trace.write_text("")
                if mode == "missing":
                    marker.unlink()
                else:
                    marker.write_text("{}")
                command = [
                    "bash",
                    str(target / "plot_all.sh"),
                    "saved run",
                    "--output",
                    str(output),
                ]
                result = subprocess.run(
                    command,
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    env=dict(
                        os.environ,
                        ELDR_TEST_FAIL=(
                            "fig11_main_language" if mode == "failure" else "never"
                        ),
                    ),
                    timeout=20,
                )
                self.assertEqual(
                    result.returncode,
                    {"missing": 1, "failure": 19, "success": 0}[mode],
                    result.stderr,
                )
                calls = trace.read_text().splitlines()
                if mode == "missing":
                    self.assertEqual(calls, [])
                    self.assertFalse(output.exists())
                    continue
                selected = entries[:2] if mode == "failure" else entries
                self.assertEqual(
                    calls,
                    [
                        item
                        for entry in selected
                        for item in (f"saved run/{entry}", str(output / entry))
                    ],
                )
                again = subprocess.run(command, capture_output=True, timeout=20)
                self.assertEqual(again.returncode, 2)
                self.assertEqual(trace.read_text().splitlines(), calls)
                if mode == "success":
                    self.assertEqual(len(list(output.rglob("*.pdf"))), len(entries))

    def test_token_not_event_denominator(self):
        tpot, ttft = request_metrics(raw_result(), 2, 4)
        np.testing.assert_allclose(tpot, [20.0, 40.0])
        np.testing.assert_allclose(ttft, [40.0, 60.0])

    def test_invalid_requests_never_filtered(self):
        cases = [
            dict(completed=1),
            dict(errors=["bad", ""]),
            dict(failed=1),
            dict(output_lens=[4, 3]),
            dict(itls=[[], [0.01]]),
            dict(itls=[[float("nan")], [0.01]]),
            dict(ttfts=[0.1]),
            dict(ttfts=[float("inf"), 0.1]),
            dict(median_tpot_ms=100),
        ]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                request_metrics(raw_result() | change, 2, 4)

    def test_paths_confined_to_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for path in ("../out.json", "/tmp/out.json"):
                with self.assertRaises(ValueError):
                    checked_path(root, dict(path=path))


if __name__ == "__main__":
    unittest.main()
