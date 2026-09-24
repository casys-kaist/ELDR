# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configure inputs, fit centroids and run ELDR experiments."""

import argparse
import json
from pathlib import Path

from eldr.experiments.runner.config import FIT_SEED

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("configure")
    command.add_argument("--inputs", type=Path, required=True)
    command.add_argument(
        "--config", type=Path, required=True, help="Shared cluster/model paths"
    )
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("bootstrap")
    command.add_argument(
        "--config", type=Path, help="Also deploy source to idle workers"
    )
    command = sub.add_parser("fit")
    command.add_argument("--signatures", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--model", required=True)
    command.add_argument("--k", type=int, default=16)
    command.add_argument("--seed", type=int, default=FIT_SEED)
    transform = command.add_mutually_exclusive_group(required=True)
    transform.add_argument(
        "--transform-from",
        type=Path,
        help="Existing calibration artifact; retain its IDF and layer mask",
    )
    transform.add_argument(
        "--counts-only",
        action="store_true",
        help="Explicit Counts/L2 diagnostic fit",
    )
    command = sub.add_parser("prepare")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    command = sub.add_parser("preflight")
    command.add_argument("--run", type=Path, required=True)
    command = sub.add_parser("capture")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--prompts", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run")
    source = command.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path, help="Previously prepared run directory")
    source.add_argument("--config", type=Path, help="Environment JSON; prepare and run")
    command.add_argument("--output", type=Path)
    command.add_argument(
        "--profile",
        choices=("smoke", "paper"),
        help="With --config only (default: smoke)",
    )
    args = parser.parse_args()
    if args.command == "run":
        if bool(args.config) != bool(args.output):
            parser.error(
                "run --config requires --output; run --run accepts no --output"
            )
        if args.run and args.profile is not None:
            parser.error("run --run uses the saved profile; omit --profile")
    try:
        if args.command == "configure":
            from eldr.experiments.runner.inputs import configure_inputs

            print(
                json.dumps(
                    configure_inputs(args.inputs, args.config, args.output), indent=2
                )
            )
            return
        if args.command == "bootstrap":
            from eldr.experiments.runner.bootstrap import deploy, restore
            from eldr.experiments.runner.config import load_config

            result = deploy(load_config(args.config)) if args.config else restore(ROOT)
            print(json.dumps(result, indent=2))
            return
        if args.command == "fit":
            from eldr.experiments.runner.fit import fit_centroids

            fit_centroids(
                args.signatures,
                args.output,
                args.model,
                args.k,
                args.seed,
                transform_from=args.transform_from,
            )
            print(f"Centroid fit -> {args.output}")
            return
        if args.command == "prepare":
            from eldr.experiments.runner.config import load_config
            from eldr.experiments.runner.workers import prepare

            config = load_config(args.config, profile=args.profile)
            print(prepare(config, args.output, args.profile))
            return
        if args.command in ("preflight", "run", "capture"):
            if args.command == "preflight":
                from eldr.experiments.runner.workers import WorkerGroup

                print(json.dumps(WorkerGroup(args.run).preflight(), indent=2))
            else:
                from eldr.experiments.runner.config import load_config
                from eldr.experiments.runner.run import execute
                from eldr.experiments.runner.workers import prepare

                capturing = args.command == "capture"
                output = args.output if args.config else args.run
                if args.config:
                    profile = "smoke" if capturing else (args.profile or "smoke")
                    config = load_config(
                        args.config, profile=profile, require_centroids=not capturing
                    )
                    prepare(
                        config,
                        output,
                        profile,
                        training=args.prompts if capturing else None,
                    )
                print(json.dumps(execute(output), indent=2))
            return
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"runner: {error}\n")


if __name__ == "__main__":
    main()
