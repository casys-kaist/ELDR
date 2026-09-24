# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import io
import json
import shlex
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from eldr.experiments.runner.__main__ import main as runner_main
from eldr.experiments.runner.bootstrap import deploy, restore
from eldr.experiments.runner.config import RUN_PROFILES, load_config
from eldr.experiments.runner.fit import fit_centroids, read_capture, training_dataset
from eldr.experiments.runner.remote import Remote
from eldr.experiments.runner.run import benchmark, execute, start_proxy
from eldr.experiments.runner.workers import (
    IMAGE,
    WorkerGroup,
    check_ports,
    engine_command,
    prepare,
    source_files,
)
from eldr.serving.clustering import load_centroid_file

ROOT = Path(__file__).resolve().parents[2]


class RunnerTests(unittest.TestCase):
    def test_worker_sources_include_runtime_assets_but_not_private_data(self):
        included = (
            "vllm/__init__.py",
            "vllm/native.so",
            "vllm/model_executor/layers/fused_moe/configs/kernel.json",
            "vllm/transformers_utils/chat_templates/template.jinja",
            "vllm/utils/numa_wrapper.sh",
            "vllm/entrypoints/serve/instrumentator/static/swagger-ui.css",
            "vllm/entrypoints/serve/instrumentator/static/swagger-ui-bundle.js",
            "eldr/serving/proxy.py",
        )
        excluded = (
            "vllm/__pycache__/cached.pyc",
            "vllm/.cache/profile.json",
            "eldr/experiments/private-cluster.json",
            "eldr/artifacts/data.json",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in included + excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertEqual(
                {str(p.relative_to(root)) for p in source_files(root)}, set(included)
            )

    def test_ports_reject_ephemeral_overlap_and_existing_tcp_users(self):
        for ports, sockets, error in (
            ({28000, 28127}, "", None),
            ({42256}, "", "ephemeral range"),
            ({32768}, "", "ephemeral range"),
            ({60999}, "", "ephemeral range"),
            ({32767, 61000}, "", None),
            ({28000}, "LISTEN 0 128 0.0.0.0:28000 0.0.0.0:*\n", "already in use"),
            ({28000}, "ESTAB 0 0 10.0.0.1:28000 10.0.0.2:80\n", "already in use"),
            ({28000}, "LISTEN 0 128 [::]:29000 [::]:*\n", None),
        ):
            with self.subTest(ports=ports, sockets=sockets):
                remote = MagicMock()
                remote.run.side_effect = [
                    subprocess.CompletedProcess([], 0, "32768\t60999\n", ""),
                    subprocess.CompletedProcess([], 0, sockets, ""),
                ]
                if error:
                    with self.assertRaisesRegex(ValueError, error):
                        check_ports(remote, "worker", ports)
                else:
                    self.assertEqual(
                        check_ports(remote, "worker", ports),
                        dict(ephemeral_range=[32768, 60999], checked_ports=len(ports)),
                    )
                if error == "ephemeral range":
                    self.assertEqual(remote.run.call_count, 1)

    def test_ports_wait_for_closed_connections_but_not_live_owners(self):
        time_wait = "TIME-WAIT 0 0 [::1]:28000 [::1]:80\n"
        for final, error in (
            ("", None),
            (time_wait, TimeoutError),
            ("LISTEN 0 128 0.0.0.0:28000 0.0.0.0:*\n", ValueError),
        ):
            with self.subTest(final=final):
                remote = MagicMock()
                remote.run.side_effect = [
                    subprocess.CompletedProcess([], 0, text, "")
                    for text in ("32768 60999\n", time_wait, final)
                ]
                with (
                    patch(
                        "eldr.experiments.runner.workers.time.monotonic",
                        side_effect=[0, 1, 91],
                    ),
                    patch("eldr.experiments.runner.workers.time.sleep") as sleep,
                ):
                    if error:
                        with self.assertRaises(error):
                            check_ports(remote, "worker", {28000})
                    else:
                        self.assertEqual(
                            check_ports(remote, "worker", {28000})["checked_ports"], 1
                        )
                    sleep.assert_called_once_with(2)
                    self.assertEqual(remote.run.call_count, 3)

    def test_preflight_finds_rocm_smi_outside_path_and_keeps_gpu_guards(self):
        for usage, rdma, error in (
            (0, "4: ACTIVE", None),
            (1, "4: ACTIVE", "GPU already in use"),
            (0, "2: INIT", "RDMA rail is not active"),
        ):
            with self.subTest(usage=usage, rdma=rdma):
                group = WorkerGroup.__new__(WorkerGroup)
                group.prepared = {"sources": {}}
                group.config = dict(
                    repo="/repo",
                    model_path="/model",
                    dataset="/data",
                    dataset_sha256="hash",
                    centroids="/fit",
                    centroids_sha256="hash",
                    proxy_port=10001,
                    prefills=[],
                    decoders=[
                        dict(
                            host="node2",
                            gpu=0,
                            cpus="0-47",
                            nic="mlx5_0:1",
                            port=22001,
                            side_channel_port=5600,
                            internal_port=28000,
                        )
                    ],
                )

                def remote(host, argv, *, usage=usage, rdma=rdma, **kwargs):
                    output = ""
                    if argv[:2] == ["sh", "-c"]:
                        self.assertIn("command -v /opt/rocm/bin/rocm-smi", argv[2])
                        output = "/opt/rocm/bin/rocm-smi\n"
                    elif argv[0] == "/opt/rocm/bin/rocm-smi":
                        output = json.dumps(
                            {
                                "card0": {
                                    "PCI Bus": "0001:01:00.0",
                                    "GPU use (%)": usage,
                                    "GPU Memory Allocated (VRAM%)": 0,
                                }
                            }
                        )
                    elif argv[0] == "cat":
                        output = {
                            "numa_node": "0",
                            "cpulist": "0-47",
                            "state": rdma,
                            "ip_local_port_range": "32768 60999",
                        }[Path(argv[1]).name]
                    return subprocess.CompletedProcess(argv, 0, output, "")

                group.remote = MagicMock()
                group.remote.run.side_effect = remote
                with patch(
                    "eldr.experiments.runner.workers.file_sha256", return_value="hash"
                ):
                    if error:
                        with self.assertRaisesRegex(ValueError, error):
                            group.preflight()
                    else:
                        self.assertEqual(group.preflight()["hosts"], ["local", "node2"])
                group.remote.run.assert_any_call(
                    "node2",
                    [
                        "/opt/rocm/bin/rocm-smi",
                        "--showbus",
                        "--showuse",
                        "--showmemuse",
                        "--json",
                    ],
                )

    def test_cli_config_and_prepared_run_keep_their_profiles(self):
        for command, flags, profile in (
            ("prepare", [], "smoke"),
            ("capture", ["--prompts", "train.json"], "smoke"),
            ("run", [], "smoke"),
            ("run", ["--profile", "paper"], "paper"),
        ):
            with (
                self.subTest(command=command, flags=flags),
                patch(
                    "sys.argv",
                    [
                        "runner",
                        command,
                        "--config",
                        "config.json",
                        "--output",
                        "out",
                        *flags,
                    ],
                ),
                patch("builtins.print"),
                patch(
                    "eldr.experiments.runner.config.load_config", return_value={}
                ) as load,
                patch("eldr.experiments.runner.workers.prepare") as prepare_run,
                patch(
                    "eldr.experiments.runner.run.execute", return_value={}
                ) as execute_run,
            ):
                runner_main()
                self.assertEqual(load.call_args.args, (Path("config.json"),))
                self.assertEqual(load.call_args.kwargs["profile"], profile)
                self.assertEqual(prepare_run.call_args.args, ({}, Path("out"), profile))
                self.assertEqual(execute_run.call_count, int(command != "prepare"))
        with (
            patch("sys.argv", ["runner", "run", "--run", "prepared"]),
            patch("builtins.print"),
            patch("eldr.experiments.runner.workers.prepare") as prepare_run,
            patch(
                "eldr.experiments.runner.run.execute", return_value={}
            ) as execute_run,
        ):
            runner_main()
            prepare_run.assert_not_called()
            execute_run.assert_called_once_with(Path("prepared"))
        with (
            patch(
                "sys.argv", ["runner", "run", "--run", "prepared", "--profile", "paper"]
            ),
            patch("sys.stderr", new_callable=io.StringIO) as stderr,
            patch("eldr.experiments.runner.run.execute") as execute_run,
            self.assertRaises(SystemExit) as error,
        ):
            runner_main()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("uses the saved profile", stderr.getvalue())
        execute_run.assert_not_called()

    def test_initial_warmup_is_longer_only_for_default_paper_run(self):
        for profile, custom, expected in (
            ("paper", None, [14400, 3840, 3840]),
            ("smoke", None, [8, 8, 8]),
            (
                "paper",
                [("eldr-static", "eldr-static", dict(RUN_PROFILES["paper"]))],
                [3840],
            ),
            (
                "paper",
                [("tau-0", "eldr-static", dict(RUN_PROFILES["paper"], tau=0))],
                [3840],
            ),
        ):
            with (
                self.subTest(profile=profile, custom=bool(custom)),
                tempfile.TemporaryDirectory() as directory,
            ):
                worker_group = MagicMock()
                worker_group.output = Path(directory)
                worker_group.prepared = {"profile": profile}
                worker_group.config = {
                    "first_token_mode": "prefill",
                    "signature_variant": "count_idf",
                    "decoders": [None] * 16,
                }
                worker_group.preflight.return_value = {}
                status = {}

                def measured(
                    worker_group, policy, stage, protocol, *, label, status=status
                ):
                    (worker_group.output / label / stage).mkdir(parents=True)
                    status["centroid_mode"] = {
                        "eldr": "online",
                        "eldr-static": "static",
                    }.get(policy, "none")
                    status.update(
                        selector="jsq",
                        signature="count_idf",
                        tau=protocol.get("tau", 0.1),
                        rule="locality_band_jsq_min2_v1",
                        minimum_candidates=2 if protocol.get("tau", 0.1) > 0 else 1,
                        refit=dict(
                            last_error=None,
                            refits_completed=int(policy == "eldr"),
                            window_seconds=5.0,
                            refit_seconds=2.5,
                        ),
                    )
                    return {"policy": policy}

                with (
                    patch(
                        "eldr.experiments.runner.run.WorkerGroup",
                        return_value=worker_group,
                    ),
                    patch("eldr.experiments.runner.run.prime"),
                    patch("eldr.experiments.runner.run.reset", return_value=[]),
                    patch("eldr.experiments.runner.run.drain"),
                    patch(
                        "eldr.experiments.runner.run.http",
                        side_effect=lambda *_, status=status: json.dumps(
                            status
                        ).encode(),
                    ),
                    patch(
                        "eldr.experiments.runner.run.start_proxy",
                        return_value=(None, "url"),
                    ),
                    patch(
                        "eldr.experiments.runner.run.benchmark", side_effect=measured
                    ) as bench,
                    patch("builtins.print"),
                ):
                    execute(worker_group.output, trials=custom)
                worker_group.start_engines.assert_called_once()
                worker_group.cleanup.assert_called_once()
                plan = json.loads((worker_group.output / "trials.json").read_text())
                self.assertEqual([p["warmup_requests"] for _, _, p in plan], expected)
                self.assertEqual(
                    [
                        c.args[3]["warmup_requests"]
                        for c in bench.call_args_list
                        if c.args[2] == "warmup"
                    ],
                    expected,
                )
                for label, _, protocol in plan:
                    for stage in ("warmup", "measure"):
                        saved = json.loads(
                            (
                                worker_group.output / label / stage / "metrics.json"
                            ).read_text()
                        )
                        self.assertEqual(saved["protocol"], protocol)
                    self.assertEqual(
                        protocol["requests"], RUN_PROFILES[profile]["requests"]
                    )
        self.assertEqual(RUN_PROFILES["paper"]["warmup_requests"], 3840)

    def test_all_policies_and_capture_use_prefix_hash_prefill(self):
        worker_group = MagicMock()
        worker_group.output = Path("/srv/run")
        worker_group.run_id = "run"
        worker_group.config = load_config(
            ROOT / "eldr/experiments/config.example.json",
            profile="smoke",
            check_files=False,
        )
        worker_group.inspect.return_value = {"State": {"Running": True}}
        for capture, policies, stages in (
            (None, ("rr", "jsq", "eldr", "eldr-static"), ("warmup", "measure")),
            ({"requests": 1000}, ("rr",), ("measure",)),
        ):
            worker_group.prepared = {"capture": capture}
            for policy in policies:
                for stage in stages:
                    with (
                        self.subTest(capture=capture, policy=policy, stage=stage),
                        patch("eldr.experiments.runner.run.http", return_value=b"{}"),
                    ):
                        start_proxy(worker_group, policy, stage)
                    args = shlex.split(worker_group.launch.call_args.args[2][-1])
                    for flag, value in (
                        ("--decode-router", policy),
                        ("--prefill-router", "prefix-hash"),
                        ("--prefill-prefix-bytes", "1024"),
                        ("--prefill-load-factor", "1.25"),
                    ):
                        self.assertEqual(args[args.index(flag) + 1], value)

    def test_benchmark_uses_seeded_shuffle_but_capture_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            worker_group = MagicMock()
            worker_group.output = Path(directory)
            worker_group.run_id = "run"
            worker_group.prepared = {}
            worker_group.config = load_config(
                ROOT / "eldr/experiments/config.example.json",
                profile="smoke",
                check_files=False,
            )
            raw = worker_group.output / "rr/measure/raw.json"

            def completed(_):
                raw.write_text("{}")

            worker_group.wait_client.side_effect = completed
            protocol = dict(
                requests=32, request_rate=2, output_tokens=32, no_oversample=False
            )
            with patch(
                "eldr.experiments.runner.run.request_metrics",
                return_value=(np.array([1.0]), np.array([2.0])),
            ):
                benchmark(worker_group, "rr", "measure", protocol)
            command = worker_group.launch.call_args.args[2][-1]
            self.assertNotIn("--disable-shuffle", command)
            self.assertNotIn("--no-oversample", command)
            self.assertIn("--seed 1234", command)
            self.assertIn("--ready-check-timeout-sec 0", command)
            self.assertNotIn("--temperature", command)
            self.assertIn("--num-prompts 32", command)

            worker_group.prepared = {"capture": {"dataset": "/srv/train.json"}}
            raw = worker_group.output / "rr/capture/raw.json"
            with patch(
                "eldr.experiments.runner.run.request_metrics",
                return_value=(np.array([1.0]), np.array([2.0])),
            ):
                benchmark(
                    worker_group,
                    "rr",
                    "capture",
                    dict(output_tokens=2, warmup_requests=32, warmup_rate=10),
                )
            command = worker_group.launch.call_args.args[2][-1]
            self.assertIn("--ready-check-timeout-sec 0", command)
            self.assertIn("--no-oversample", command)
            self.assertIn("--dataset-path /srv/train.json", command)

    def test_training_rejects_overlap_duplicates_and_bad_format(self):
        with tempfile.TemporaryDirectory() as directory:
            train, evaluation = (
                Path(directory) / "train.json",
                Path(directory) / "eval.json",
            )
            evaluation.write_text(
                json.dumps([{"conversations": [{"value": "held out"}]}])
            )
            train.write_text(json.dumps([["training prompt", "math"]]))
            rows, sha = training_dataset(train, evaluation)
            self.assertEqual(rows[0]["conversations"][0]["value"], "training prompt")
            self.assertEqual(sha, hashlib.sha256(train.read_bytes()).hexdigest())
            for bad in (
                [],
                [["held out", "math"]],
                [["same", "x"]] * 2,
                ["not a pair"],
            ):
                train.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    training_dataset(train, evaluation)

    def test_capture_preparation_freezes_inputs_without_centroids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, evaluation = root / "train.json", root / "eval.json"
            train.write_text(json.dumps([["training prompt", "math"]]))
            evaluation.write_text(
                json.dumps([{"conversations": [{"value": "held out"}]}])
            )
            config = load_config(
                ROOT / "eldr/experiments/config.example.json",
                profile="smoke",
                check_files=False,
            )
            config.update(repo=str(ROOT), model_path=str(root), dataset=str(evaluation))
            config.pop("centroids")
            path = root / "config.json"
            path.write_text(json.dumps(config))
            config = load_config(path, profile="smoke", require_centroids=False)
            self.assertEqual(
                config["dataset_sha256"],
                hashlib.sha256(evaluation.read_bytes()).hexdigest(),
            )
            with (
                patch("eldr.experiments.runner.workers.EXTENSIONS", {}),
                patch("eldr.experiments.runner.workers.source_files", return_value=[]),
            ):
                output = prepare(config, root / "run", "smoke", training=train)
            worker_group = WorkerGroup(output)
            capture = worker_group.prepared["capture"]
            self.assertEqual(capture["requests"], 1)
            Path(capture["dataset"]).write_text("changed")
            with (
                self.assertRaisesRegex(ValueError, "training input changed"),
                patch.object(worker_group.remote, "run") as run,
            ):
                worker_group.preflight()
            run.assert_not_called()
            config["dataset_sha256"] = "0" * 64
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_config(path, profile="smoke", require_centroids=False)

    def test_capture_failure_cleans_up_without_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            worker_group = MagicMock()
            worker_group.output = Path(directory)
            worker_group.prepared = {"capture": {"requests": 2}, "profile": "smoke"}
            worker_group.config = {"model": "qwen"}
            worker_group.preflight.return_value = {}
            (worker_group.output / "signatures.bin").write_bytes(b"invalid")
            (worker_group.output / "rr/measure").mkdir(parents=True)
            with (
                patch(
                    "eldr.experiments.runner.run.WorkerGroup", return_value=worker_group
                ),
                patch("eldr.experiments.runner.run.prime"),
                patch("eldr.experiments.runner.run.reset", return_value=[]),
                patch(
                    "eldr.experiments.runner.run.start_proxy",
                    return_value=(None, "url"),
                ),
                patch("eldr.experiments.runner.run.benchmark", return_value={}),
                patch("eldr.experiments.runner.run.drain"),
                self.assertRaises(ValueError),
            ):
                execute(worker_group.output)
            worker_group.cleanup.assert_called_once()
            self.assertTrue((worker_group.output / "failed.json").exists())
            self.assertFalse((worker_group.output / "complete.json").exists())

    def test_training_wire_format_and_truncation(self):
        row = np.ones((48, 128), dtype="<i2")
        framed = struct.pack("<I", row.nbytes) + row.tobytes()
        np.testing.assert_array_equal(read_capture(framed * 2, "qwen"), [row, row])
        for broken in (b"", framed[:-1], framed + b"x", struct.pack("<I", 2)):
            with self.assertRaises(ValueError):
                read_capture(broken, "qwen")

    def test_homogeneous_counts_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.npy"
            np.save(source, np.ones((16, 2, 4), dtype=np.int16))
            result = fit_centroids(source, Path(directory) / "fit.json", "test", 4, 0)
            centroids = np.asarray(result["centroids"])
            np.testing.assert_allclose(centroids, np.full((4, 8), 1 / np.sqrt(8)))

    def test_bootstrap_checks_existing_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm").mkdir()
            (root / "vllm/__init__.py").touch()
            extension = root / "vllm/test.so"
            extension.write_bytes(b"known")
            expected = {"test.so": hashlib.sha256(b"known").hexdigest()}
            with patch("eldr.experiments.runner.bootstrap.EXTENSIONS", expected):
                self.assertEqual(restore(root)["restored"], [])
                extension.write_bytes(b"user's different build")
                with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                    restore(root)
                self.assertEqual(extension.read_bytes(), b"user's different build")

    def test_deploy_reuses_matching_sources_and_never_overwrites_or_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("known source")
            config = dict(
                repo=str(root),
                controller_address="10.0.0.1",
                prefills=[dict(host="worker")],
                decoders=[],
            )
            for exists in (0, 1):
                remote = MagicMock()

                def run(host, argv, *, exists=exists, **kwargs):
                    if argv[0] == "ip":
                        return subprocess.CompletedProcess(
                            [], 0, '[{"addr_info":[{"local":"10.0.0.1"}]}]', ""
                        )
                    return subprocess.CompletedProcess(
                        [], exists if argv[0] == "test" else 0, "", ""
                    )

                remote.run.side_effect = run
                with (
                    patch(
                        "eldr.experiments.runner.bootstrap.Remote", return_value=remote
                    ),
                    patch("eldr.experiments.runner.bootstrap.restore", return_value={}),
                    patch(
                        "eldr.experiments.runner.workers.source_files",
                        return_value=[source],
                    ),
                    patch(
                        "eldr.experiments.runner.bootstrap.subprocess.run"
                    ) as transfer,
                ):
                    self.assertEqual(deploy(config)["hosts"], ["local", "worker"])
                    self.assertEqual(transfer.call_count, exists)
                    if exists:
                        command = transfer.call_args.args[0]
                        self.assertIn("--ignore-existing", command)
                        self.assertIn(
                            "ProxyCommand=false", command[command.index("-e") + 1]
                        )
                        transfer.side_effect = subprocess.CalledProcessError(
                            12, "rsync"
                        )
                        with self.assertRaises(subprocess.CalledProcessError):
                            deploy(config)
                        self.assertEqual(
                            transfer.call_count, 2
                        )  # One attempt per invocation.
                    else:
                        remote.run.side_effect = RuntimeError("source mismatch")
                        with self.assertRaisesRegex(RuntimeError, "source mismatch"):
                            deploy(config)
                        transfer.assert_not_called()

    def test_cleanup_does_not_reconnect_to_failed_host(self):
        with tempfile.TemporaryDirectory() as directory:
            worker_group = WorkerGroup.__new__(WorkerGroup)
            worker_group.output = Path(directory)
            worker_group.remote = Remote()
            worker_group.remote.failed.add("failed")
            failed = dict(host="failed", name="owned-a", id="a" * 64)
            healthy = dict(host="local", name="owned-b", id="b" * 64)
            worker_group.owned = [failed, healthy]
            with patch.object(worker_group, "stop") as stop:
                with self.assertRaisesRegex(RuntimeError, "Cleanup incomplete"):
                    worker_group.cleanup()
                stop.assert_called_once_with(healthy)
            report = json.loads((Path(directory) / "cleanup.json").read_text())
            self.assertEqual(report["remaining"][0]["name"], "owned-a")

    def test_unowned_container_is_never_stopped(self):
        worker_group = WorkerGroup.__new__(WorkerGroup)
        worker_group.run_id = "our-run"
        worker_group.remote = MagicMock()
        worker_group.remote.run.return_value.stdout = json.dumps(
            [{"Config": {"Labels": {"eldr.artifact.run": "someone-else"}}}]
        )
        with self.assertRaisesRegex(ValueError, "Ownership mismatch"):
            worker_group.stop(dict(host="local", id="a" * 64, name="not-ours"))
        self.assertEqual(worker_group.remote.run.call_count, 1)
        self.assertEqual(worker_group.remote.run.call_args.args[1][1], "inspect")

    def test_ssh_reuse_and_no_retry_after_failure(self):
        remote = Remote()
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 255, "", "failed"),
        ) as run:
            with self.assertRaises(RuntimeError):
                remote.run("worker", ["true"])
            with self.assertRaises(RuntimeError):
                remote.run("worker", ["docker", "ps"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("ProxyCommand=false", run.call_args.args[0])

    def test_timeout_poisoned_host_and_local_no_ssh(self):
        remote = Remote()
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired("test", 1)),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            remote.run("worker", ["true"])
        self.assertIn("worker", remote.failed)
        with patch(
            "subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")
        ) as run:
            remote.run("local", ["true"])
        self.assertEqual(run.call_args.args[0], ["true"])

    def test_counts_fit_is_uniform_reproducible_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "train.npy", Path(directory) / "fit.json"
            np.save(
                source,
                np.random.default_rng(1).integers(1, 10, (32, 3, 8), dtype=np.int16),
            )
            result = fit_centroids(source, output, "test", 4, 0)
            self.assertEqual(result["layer_mask"], [0, 1, 2])
            self.assertNotIn("sig_idf", result)
            self.assertEqual(
                load_centroid_file(output)["centroid_matrix"].shape, (4, 24)
            )
            again = fit_centroids(source, Path(directory) / "other.json", "test", 4, 0)
            self.assertEqual(result, again)
            with self.assertRaises(ValueError):
                fit_centroids(source, output, "test", 4, 0)

    def test_one_prefill_geometry_is_explicit_and_keeps_other_checks(self):
        config = load_config(
            ROOT / "eldr/experiments/config.example.json",
            profile="smoke",
            check_files=False,
        )
        decoder = config["decoders"][0]
        config["prefills"] = config["prefills"][:1]
        config["decoders"] = [
            dict(
                decoder,
                gpu=i,
                port=25001 + i,
                side_channel_port=5700 + i,
                internal_port=28000 + 128 * i,
            )
            for i in range(16)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "8P16D"):
                load_config(path, check_files=False)
            actual = load_config(path, check_files=False, prefix_cache_experiment=True)
            self.assertEqual(len(actual["prefills"]), 1)
            self.assertEqual(len(actual["decoders"]), 16)
            config["decoders"][1]["port"] = config["decoders"][0]["port"]
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_config(path, check_files=False, prefix_cache_experiment=True)

    def test_config_and_docker_commands(self):
        config = load_config(
            ROOT / "eldr/experiments/config.example.json",
            profile="smoke",
            check_files=False,
        )
        with self.assertRaises(ValueError):
            load_config(
                ROOT / "eldr/experiments/config.example.json", check_files=False
            )
        worker = config["prefills"][0]
        command = engine_command(config, worker, True, "name", "run", "0" * 64)
        self.assertIn(IMAGE, command)
        self.assertIn("ELDR=1", command)
        self.assertIn("--no-async-scheduling", command[-1])
        decoder = engine_command(
            config, config["decoders"][0], False, "d", "run", "0" * 64
        )
        self.assertIn("--async-scheduling", decoder[-1])
        self.assertNotIn("--no-async-scheduling", decoder[-1])
        self.assertIn("--cpuset-cpus", command)
        self.assertFalse(any("artifacts/diagnostics" in arg for arg in command))
        self.assertNotIn("--privileged", command)
        config["tiktoken_cache"] = "/srv/harmony"
        cached = engine_command(config, worker, True, "name", "run", "0" * 64)
        self.assertIn("/srv/harmony:/srv/harmony:ro", cached)
        self.assertIn("TIKTOKEN_ENCODINGS_BASE=/srv/harmony", cached)
        with tempfile.TemporaryDirectory() as directory:
            config["decoders"][0]["port"] = config["decoders"][1]["port"]
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_config(path, profile="smoke", check_files=False)


if __name__ == "__main__":
    unittest.main()
