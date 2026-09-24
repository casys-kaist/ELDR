"""Explicit, non-overwriting output paths for dataset preparation."""

import argparse
import json
from pathlib import Path


def write_json(value, path):
    with Path(path).open("x") as stream:
        json.dump(value, stream)


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tokenizer", required=True, help="Pinned local tokenizer directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="New dataset directory"
    )
    args = parser.parse_args()
    if not Path(args.tokenizer).is_dir():
        parser.error("--tokenizer must be a local directory")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "captures").mkdir()
    return args
