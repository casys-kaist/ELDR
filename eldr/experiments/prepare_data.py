# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate prompts, capture fresh activations, and fit all AE inputs."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from eldr.experiments.runner.bootstrap import deploy
from eldr.experiments.runner.config import (
    IMAGE,
    MODEL_GEOMETRY,
    file_sha256,
    load_config,
)
from eldr.experiments.runner.fit import (
    fit_signature,
    read_activations,
    training_dataset,
)
from eldr.experiments.runner.inputs import configure_inputs
from eldr.experiments.runner.remote import SSH
from eldr.experiments.runner.results import verify
from eldr.experiments.runner.workers import (
    VLLM_PACKAGE_PATH,
    WorkerGroup,
    base_container,
    in_venv,
    source_files,
    write_new_json,
)
from eldr.experiments.runner.workers import (
    prepare as prepare_workers,
)

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = tuple(f"{m}-{w}" for m in MODEL_GEOMETRY for w in ("task", "language"))


def build_prompts(output: Path, models: dict):
    from transformers import AutoTokenizer

    for workload, stem, sidecar in (
        ("task", "task", "task_unbal_full_hash2dom.json"),
        ("language", "lang", "wildchat_full_hash2lang.json"),
    ):
        parsed = output / workload
        subprocess.run(
            [
                sys.executable,
                "-m",
                f"eldr.experiments.datasets.build_{workload}",
                "--tokenizer",
                models["qwen"],
                "--output",
                str(parsed),
            ],
            check=True,
        )
        target = output / "data" / workload
        target.mkdir(parents=True)
        shutil.copyfile(parsed / f"{stem}.fit.json", target / "training.json")
        shutil.copyfile(parsed / f"{stem}.eval.json", target / "evaluation.json")
        labels = json.loads((parsed / "captures" / sidecar).read_text())
        if workload == "task":
            shutil.copyfile(parsed / "captures" / sidecar, target / "labels.json")
            continue
        training = json.loads((target / "training.json").read_text())
        evaluation = json.loads((target / "evaluation.json").read_text())
        prompts = [row[0] for row in training] + [
            row["conversations"][0]["value"] for row in evaluation
        ]
        used = {hashlib.md5(p.encode()).hexdigest()[:16] for p in prompts}
        write_new_json(
            target / "labels.json", {k: v for k, v in labels.items() if k in used}
        )
        for model in ("gptoss", "gemma"):
            tokenizer = AutoTokenizer.from_pretrained(
                models[model], local_files_only=True
            )
            rows = [
                row
                for row in training
                if 4
                <= len(tokenizer(row[0], add_special_tokens=False)["input_ids"])
                <= 1024
            ]
            write_new_json(target / f"training-{model}.json", rows)


def generation_spec(root, models):
    # The complete file-by-file source inventory is also saved with each GPU run.
    sources = {
        str(p.relative_to(root)): file_sha256(p)
        for p in source_files(root)
        if p.suffix in (".py", ".sh")
    }
    for name in ("eldr/requirements.txt", "eldr/experiments/datasets/requirements.txt"):
        sources[name] = file_sha256(root / name)
    metadata = {}
    for model, folder in models.items():
        directory = Path(folder)
        if not directory.is_dir():
            raise ValueError("Missing local model: " + str(directory))
        metadata[model] = {
            p.name: file_sha256(p)
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in (".json", ".model", ".tiktoken")
        }
        if "config.json" not in metadata[model]:
            raise ValueError("Missing model config: " + model)
    return dict(
        recipe="fresh-paired-full-v1",
        fitting_seed=1,
        output_tokens=128,
        temperature=0,
        image=IMAGE,
        model_metadata=metadata,
        source_sha256=hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()
        ).hexdigest(),
    )


def file_record(root, path):
    return dict(
        path=str(path.relative_to(root)),
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
    )


def save_manifest(inputs, manifest):
    # Replacing this generated index is atomic; the referenced data is immutable.
    with tempfile.NamedTemporaryFile(mode="w", dir=inputs, delete=False) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, inputs / "manifest.json")


def capture(config, training, output):
    """Run one offline GPU process using the existing preflight/ownership checks."""
    prepare_workers(config, output, "smoke", training=training)
    group = WorkerGroup(output)
    worker = config["prefills"][0]
    host = worker["host"]
    prompts = output / "prompts.json"
    shutil.copyfile(training, prompts)
    target = output / "activations.npz"
    try:
        write_new_json(output / "preflight.json", group.preflight())
        if host != "local":
            group.remote.run(host, ["mkdir", "-p", "--", str(output)])
            try:
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--protect-args",
                        "--ignore-existing",
                        "-e",
                        shlex.join(SSH),
                        str(prompts),
                        host + ":" + str(prompts),
                    ],
                    check=True,
                    timeout=120,
                )
            except subprocess.SubprocessError:
                group.remote.failed.add(host)
                raise
            group.remote.run(
                host,
                ["sha256sum", "--check", "--status", "-"],
                stdin=f"{file_sha256(prompts)}  {prompts}\n",
            )
        name = group.run_id + "-calibration"
        command = base_container(config, name, group.run_id, worker["cpus"])
        command += [
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "--group-add",
            "video",
            "--group-add",
            "render",
            "--cap-add",
            "IPC_LOCK",
            "--security-opt",
            "seccomp=unconfined",
            "--shm-size",
            "32g",
            "-v",
            f"{config['repo']}/vllm:{VLLM_PACKAGE_PATH}:ro",
            "-v",
            f"{output}:{output}",
            "-e",
            f"HIP_VISIBLE_DEVICES={worker['gpu']}",
            "-e",
            f"VLLM_PORT={worker['internal_port']}",
            "-e",
            "ELDR=0",
            "-e",
            "VLLM_USE_LAYERNAME=0",
            "-e",
            "VLLM_SSM_CONV_STATE_LAYOUT=DS",
        ]
        if config.get("tiktoken_cache"):
            path = config["tiktoken_cache"]
            command += [
                "-v",
                f"{path}:{path}:ro",
                "-e",
                f"TIKTOKEN_ENCODINGS_BASE={path}",
            ]
        command += [IMAGE] + in_venv(
            [
                "-m",
                "vllm.eldr.calibration",
                "--model",
                config["model"],
                "--model-path",
                config["model_path"],
                "--prompts",
                str(prompts),
                "--output",
                str(target),
            ]
        )
        entry = group.launch(host, name, command)
        print(
            f"Calibration started: {config['model']} on {host}/GPU{worker['gpu']}; "
            f"logs: {output}",
            flush=True,
        )
        group.wait_client(entry)
        if host != "local":
            try:
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--protect-args",
                        "--ignore-existing",
                        "-e",
                        shlex.join(SSH),
                        host + ":" + str(target),
                        str(target),
                    ],
                    check=True,
                    timeout=120,
                )
            except subprocess.SubprocessError:
                group.remote.failed.add(host)
                raise
            group.remote.run(
                host,
                ["sha256sum", "--check", "--status", "-"],
                stdin=f"{file_sha256(target)}  {target}\n",
            )
        data = read_activations(target, config["model"])
        pairs = json.loads(training.read_text())
        if list(data["request_ids"]) != [
            hashlib.sha256(p.encode()).hexdigest() for p, _ in pairs
        ] or data["metadata"]["training_sha256"] != file_sha256(training):
            raise ValueError("Fresh capture does not match the calibration prompts")
        return target
    finally:
        group.cleanup()


def prepare(inputs, cluster, settings=SETTINGS):
    inputs, cluster = inputs.resolve(), cluster.resolve()
    shared = json.loads(cluster.read_text())
    shared["repo"] = str(ROOT)
    models = shared.pop("model_paths")
    specification = generation_spec(ROOT, models)
    manifest_path = inputs / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version") != 2
            or manifest.get("generation") != specification
        ):
            raise ValueError(
                "Inputs use an old/different recipe; choose a new --output directory"
            )
        verify(dict(measurements=manifest["files"]), inputs)
    else:
        if inputs.exists() and any(inputs.iterdir()):
            raise ValueError(
                "Unrecognized nonempty input directory; refusing to adopt old files"
            )
        inputs.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".parse-", dir=inputs) as temporary:
            stage = Path(temporary)
            build_prompts(stage, models)
            (stage / "data").rename(inputs / "data")
        manifest = dict(
            schema_version=2,
            generation=specification,
            settings={},
            files=[
                file_record(inputs, p)
                for p in sorted((inputs / "data").rglob("*.json"))
            ],
        )
        save_manifest(inputs, manifest)
    deployed = False
    for setting in settings:
        if setting in manifest["settings"]:
            print("Verified; reusing fresh calibration: " + setting, flush=True)
            continue
        model, workload = setting.split("-")
        data = inputs / "data" / workload
        training = data / (
            f"training-{model}.json"
            if workload == "language" and model != "qwen"
            else "training.json"
        )
        target = inputs / "calibration" / setting
        if target.exists():
            raise ValueError(
                "Unindexed calibration; preserve it and choose a new --output: "
                + str(target)
            )
        dataset = data / "evaluation.json"
        training_dataset(training, dataset)
        config = dict(
            shared, model=model, model_path=models[model], dataset=str(dataset)
        )
        with tempfile.TemporaryDirectory(prefix="eldr-capture-config-") as temporary:
            path = Path(temporary) / "config.json"
            write_new_json(path, config)
            config = load_config(path, profile="paper", require_centroids=False)
        if not deployed:
            deploy(config)
            deployed = True
        logs = ROOT / "eldr/artifacts/preparation"
        logs.mkdir(parents=True, exist_ok=True)
        run = Path(tempfile.mkdtemp(prefix=setting + "-", dir=logs))
        print(f"Preparing {setting}; run directory: {run}", flush=True)
        activations = capture(config, training, run / "capture")
        fit = run / "centroids.json"
        fit_signature(activations, fit, model, "count_idf", k=len(config["decoders"]))
        # Publish only a successfully captured and fitted setting.
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".publish-", dir=target.parent
        ) as temporary:
            stage = Path(temporary) / setting
            stage.mkdir()
            shutil.copyfile(activations, stage / "activations.npz")
            shutil.copyfile(fit, stage / "centroids.json")
            stage.rename(target)
        manifest["settings"][setting] = dict(
            dataset=str(dataset.relative_to(inputs)),
            training_prompts=str(training.relative_to(inputs)),
            labels=str((data / "labels.json").relative_to(inputs)),
            activations=str((target / "activations.npz").relative_to(inputs)),
            centroids=str((target / "centroids.json").relative_to(inputs)),
        )
        manifest["files"].extend(
            file_record(inputs, p) for p in sorted(target.iterdir())
        )
        save_manifest(inputs, manifest)
        print("Prepared fresh calibration: " + setting, flush=True)
    if set(manifest["settings"]) == set(SETTINGS):
        with tempfile.TemporaryDirectory(prefix="eldr-input-check-") as temporary:
            configure_inputs(inputs, cluster, Path(temporary) / "configs")
        print(f"All six settings verified: {inputs}", flush=True)
    else:
        print(
            f"Prepared {len(manifest['settings'])}/6 settings. "
            "Rerun without --setting to finish.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "eldr/experiments/cluster.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "eldr/inputs")
    parser.add_argument(
        "--setting", choices=SETTINGS, help="Prepare one setting; default: all six"
    )
    args = parser.parse_args()
    previous = signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    try:
        prepare(args.output, args.config, (args.setting,) if args.setting else SETTINGS)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
