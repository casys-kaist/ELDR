# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate supplied inputs and generate the six AE experiment configs."""

import json
import tempfile
from pathlib import Path


def configure_inputs(inputs: Path, cluster: Path, output: Path):
    """Verify a relocated input bundle; write seven configs without cluster access."""
    from eldr.experiments.runner.config import FIT_SEED, MODEL_GEOMETRY, load_config
    from eldr.experiments.runner.experiment import load_labeled_inputs
    from eldr.experiments.runner.fit import read_capture
    from eldr.experiments.runner.results import checked_path, verify
    from eldr.experiments.runner.workers import write_new_json
    from eldr.experiments.signature_ablation import read_paired_capture

    inputs, output = inputs.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Configuration output already exists")
    manifest = json.loads((inputs / "manifest.json").read_text())
    settings = manifest["settings"]
    expected = {f"{m}-{w}" for m in MODEL_GEOMETRY for w in ("task", "language")}
    if manifest["schema_version"] != 1 or set(settings) != expected:
        raise ValueError("Input bundle must contain all six model/workload settings")
    files = manifest["files"]
    paths = [row["path"] for row in files]
    fields = {
        "dataset",
        "centroids",
        "training_prompts",
        "labels",
        "training_signatures",
        "paired_signatures",
    }
    if any(set(setting) != fields for setting in settings.values()):
        raise ValueError("Each setting requires exactly six input fields")
    referenced = {p for setting in settings.values() for p in setting.values()}
    if len(paths) != len(set(paths)) or set(paths) != referenced:
        raise ValueError("Bundle file list must match its inputs without duplicates")
    verify(dict(measurements=files), inputs)
    hashes = {row["path"]: row["sha256"] for row in files}
    shared = json.loads(cluster.read_text())
    shared.setdefault("repo", str(Path(__file__).resolve().parents[3]))
    models = shared.pop("model_paths")
    configs = {}
    # Validate every generated config before creating the destination directory.
    with tempfile.TemporaryDirectory(prefix="eldr-config-check-") as temporary:
        for name, setting in settings.items():
            model = name.rsplit("-", 1)[0]
            config = dict(shared, model=model, model_path=models[model])
            for field, relative in setting.items():
                config[field] = str(checked_path(inputs, dict(path=relative)))
                config[field + "_sha256"] = hashes[relative]
            path = Path(temporary) / (name + ".json")
            write_new_json(path, config)
            config = load_config(path, profile="paper")
            fit_seed = json.loads(Path(config["centroids"]).read_text()).get("seed")
            if type(fit_seed) is not int or fit_seed != FIT_SEED:
                raise ValueError(
                    f"AE calibration requires fitting seed {FIT_SEED}: {name}"
                )
            _, training, _ = load_labeled_inputs(config)
            counts = read_capture(
                Path(config["training_signatures"]).read_bytes(), model
            )
            if len(counts) != len(training):
                raise ValueError("Training prompt/capture row-count mismatch: " + name)
            layers, experts, _ = MODEL_GEOMETRY[model]
            paired = read_paired_capture(config["paired_signatures"], layers, experts)
            if len(paired[0]) < len(config["decoders"]):
                raise ValueError("Paired calibration has fewer than K records: " + name)
            configs[name] = config
        prefix = dict(configs["gptoss-task"])
        prefix["prefills"] = prefix["prefills"][:1]
        path = Path(temporary) / "gptoss-task-prefix.json"
        write_new_json(path, prefix)
        configs[path.stem] = load_config(path, prefix_cache_experiment=True)
    output.mkdir(parents=True)
    for name, config in configs.items():
        write_new_json(output / (name + ".json"), config)
    return dict(files=len(files), configs=len(configs), output=str(output))
