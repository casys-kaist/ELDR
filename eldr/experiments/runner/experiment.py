# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared planning and execution; variants live in each experiment module."""

import argparse
import hashlib
import importlib
import json
from collections import Counter
from pathlib import Path

from eldr.experiments.runner.config import RUN_PROFILES, file_sha256, load_config
from eldr.experiments.runner.fit import training_dataset
from eldr.experiments.runner.results import collect, write_report
from eldr.experiments.runner.run import execute
from eldr.experiments.runner.workers import prepare, source_files, write_new_json

BASELINES = ("rr", "random", "jsq", "p2c", "domain", "eldr-static")
EXPERIMENTS = (
    "main_task",
    "main_language",
    "locality_band",
    "cluster_balance",
    "prefix_cache",
    "signature_ablation",
)


def load_experiment(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown experiment")
    return importlib.import_module(f"eldr.experiments.{experiment}")


def plan_worker_group(config, folder, variants, *, inputs=()):
    """One worker group; experiment files declare variants and derived inputs."""
    # Keep persisted field names so completed experiments still replay unchanged.
    return dict(
        key=folder.name,
        directory=str(folder),
        source_site=config,
        site=dict(config, prefix_cache=True),
        variants=variants,
        extra_inputs=inputs,
    )


def main_worker_group(config, output, policies):
    folder = output / config["model"]
    inputs = ()
    if "domain" in policies:
        load_labeled_inputs(config)
        inputs = ("training_prompts", "labels")
    variants = [
        (
            p,
            p,
            {"domain_artifact": str(folder / "domain.json")} if p == "domain" else {},
        )
        for p in policies
    ]
    return [plan_worker_group(config, folder, variants, inputs=inputs)]


def prepare_domain_baseline(worker_group):
    if any(policy == "domain" for _, policy, _ in worker_group["trials"]):
        config = worker_group["source_site"]
        _, pairs, labels = load_labeled_inputs(config)
        write_new_json(
            Path(worker_group["directory"]) / "domain.json",
            build_domain_mapping(pairs, labels, len(config["decoders"])),
        )


def prompt_hash(text):
    return hashlib.md5(text.encode()).hexdigest()[:16]


def load_labeled_inputs(config):
    """Read explicit builder outputs; never infer an oracle label from text."""
    labels = json.loads(Path(config["labels"]).read_text())
    dataset = json.loads(Path(config["dataset"]).read_text())
    pairs = json.loads(Path(config["training_prompts"]).read_text())
    training_dataset(Path(config["training_prompts"]), Path(config["dataset"]))
    for text, label in pairs:
        if labels.get(prompt_hash(text)) != label:
            raise ValueError("Training labels disagree with the label sidecar")
    for row in dataset:
        key = prompt_hash(row["conversations"][0]["value"])
        if not isinstance(labels.get(key), str) or not labels[key]:
            raise ValueError("Missing evaluation label")
    return dataset, pairs, labels


def build_domain_mapping(pairs, labels, k):
    """Original Domain control: calibration-proportional pools and a misc tail."""
    mix = Counter(label for _, label in pairs)
    head = {d: n for d, n in mix.items() if n >= len(pairs) / k}
    tail = sum(n for d, n in mix.items() if d not in head)
    # None cannot collide with a real string label.
    groups = head | ({None: tail} if tail else {})
    quota = {d: k * n / len(pairs) for d, n in groups.items()}
    allocation = {d: int(n) for d, n in quota.items()}
    for d in sorted(groups, key=lambda d: quota[d] - allocation[d], reverse=True)[
        : k - sum(allocation.values())
    ]:
        allocation[d] += 1
    pools, index = {}, 0
    for d, size in allocation.items():
        pools[d] = list(range(index, index + size))
        index += size
    return dict(
        K=k,
        dom2dec={d: v for d, v in pools.items() if d is not None},
        misc_dec=pools.get(None, []),
        hash2dom=labels,
        calibration_mix=dict(mix),
    )


def plan_experiment(experiment, configs, output, profile, rates, policies):
    """Plan all cells before any cluster access. No file writes or GPU actions."""
    if experiment not in EXPERIMENTS or not rates or len(set(rates)) != len(rates):
        raise ValueError("Known experiment and distinct request rates required")
    if any(type(r) is not int or r <= 0 for r in rates):
        raise ValueError("Rates must be positive integers")
    if (
        not policies
        or len(set(policies)) != len(policies)
        or set(policies) - {*BASELINES, "eldr"}
    ):
        raise ValueError("Known, distinct policies required")
    plan = dict(study=experiment, profile=profile, fleets=[], inputs={})
    experiment_module = load_experiment(experiment)
    for path in configs:
        config = load_config(
            path,
            profile=profile,
            prefix_cache_experiment=experiment == "prefix_cache",
        )
        plan["inputs"][str(path.resolve())] = file_sha256(path)
        for worker_group in experiment_module.configure(config, output, policies):
            for name in ("dataset", "centroids", *worker_group.pop("extra_inputs")):
                source = Path(config[name]).resolve()
                plan["inputs"][str(source)] = file_sha256(source)
            trials = []
            for rate in rates:
                for variant, policy, options in worker_group["variants"]:
                    protocol = dict(RUN_PROFILES[profile], **options)
                    protocol.update(
                        request_rate=rate,
                        requests=120 * rate
                        if profile == "paper"
                        else RUN_PROFILES[profile]["requests"],
                        variant=variant,
                    )
                    if not trials:
                        protocol["warmup_requests"] = protocol.get(
                            "initial_warmup_requests", protocol["warmup_requests"]
                        )
                    trials.append((f"r{rate}-{variant}", policy, protocol))
            worker_group.pop("variants")
            worker_group["trials"] = trials
            plan["fleets"].append(worker_group)
    keys = [f["key"] for f in plan["fleets"]]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError("Supply distinct model configs")
    return plan


def prepare_experiment_inputs(worker_group, experiment):
    """Create derived inputs inside the new experiment's output directory."""
    folder = Path(worker_group["directory"])
    folder.mkdir()
    prepare_inputs = getattr(load_experiment(experiment), "prepare", None)
    if prepare_inputs is not None:
        prepare_inputs(worker_group)
    config_path = folder / "config.json"
    write_new_json(config_path, worker_group["site"])
    return load_config(config_path, profile="smoke")


def run_experiment(plan, output):
    for source, expected in plan["inputs"].items():
        if file_sha256(source) != expected:
            raise ValueError("Experiment input changed: " + source)
    output.mkdir(parents=True, exist_ok=False)
    write_new_json(output / "plan.json", plan)
    sources = {
        str(path): file_sha256(path)
        for root in {f["site"]["repo"] for f in plan["fleets"]}
        for path in source_files(Path(root))
    }
    write_new_json(output / "sources.json", sources)
    for worker_group in plan["fleets"]:
        for path, expected in (sources | plan["inputs"]).items():
            if file_sha256(path) != expected:
                raise ValueError("Source/input changed between worker groups: " + path)
        config = prepare_experiment_inputs(worker_group, plan["study"])
        run = Path(worker_group["directory"]) / "run"
        prepare(config, run, plan["profile"])
        execute(run, trials=worker_group["trials"])
    rows = collect(plan, output)
    write_report(rows, output, plan["study"])
    write_new_json(output / "complete.json", dict(measurements=len(rows)))


def main(experiment):
    parser = argparse.ArgumentParser(
        description=f"{experiment}: plan, run, validate and plot."
    )
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        default=[],
        help="Experiment environment JSON; repeat for multiple models",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=RUN_PROFILES, help="Run size (default: paper)"
    )
    parser.add_argument("--rates", nargs="+", type=int)
    parser.set_defaults(policies=None)
    if experiment in ("main_task", "main_language"):
        parser.add_argument(
            "--policies",
            nargs="+",
            choices=(*BASELINES, "eldr"),
            help="Policy subset; default uses static ELDR ('eldr' opts into online)",
        )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--execute",
        action="store_true",
        help="Launch GPU experiments; default only prints plan",
    )
    action.add_argument(
        "--replay", type=Path, help="Completed experiment directory; no cluster access"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("--output must be a new directory")
    if args.replay:
        if args.config or any(
            value is not None for value in (args.profile, args.rates, args.policies)
        ):
            parser.error("--replay uses saved settings; accepts only --output")
        source = args.replay.resolve()
        if not (source / "complete.json").is_file():
            parser.error("Replay requires a complete experiment")
        plan = json.loads((source / "plan.json").read_text())
        if plan["study"] != experiment:
            parser.error("Experiment type mismatch")
        rows = collect(plan, source)
        previous = json.loads((source / "summary.json").read_text())
        if rows != previous:
            parser.error("Raw data or metrics changed since the completed experiment")
        output.mkdir(parents=True)
        write_report(rows, output, experiment)
        return
    if not args.config:
        parser.error("At least one --config is required")
    profile = args.profile or "paper"
    policies = args.policies or list(
        getattr(load_experiment(experiment), "POLICIES", BASELINES)
    )
    rates = args.rates or (
        [2] if profile == "smoke" else list(load_experiment(experiment).RATES)
    )
    plan = plan_experiment(experiment, args.config, output, profile, rates, policies)
    print(json.dumps(plan, indent=2))
    if args.execute:
        run_experiment(plan, output)
