"""Prefix cache: RR/static ELDR on/off over a seeded 2,000-prompt pool."""

import json
import random
from pathlib import Path

from eldr.experiments.runner.experiment import main, plan_worker_group
from eldr.experiments.runner.workers import write_new_json

RATES = (100,)
POLICIES = ("rr", "eldr-static")
PREFIX_PROMPTS = 2000


def configure(config, output, policies):
    worker_groups = []
    for cache in (False, True):
        folder = output / (config["model"] + f"-cache-{int(cache)}")
        worker_group = plan_worker_group(
            config, folder, [(p, p, dict(no_oversample=False)) for p in POLICIES]
        )
        worker_group["site"].update(
            prefix_cache=cache, dataset=str(folder / "evaluation.json")
        )
        worker_group["site"].pop("dataset_sha256", None)
        worker_groups.append(worker_group)
    return worker_groups


def prepare(worker_group):
    rows = json.loads(Path(worker_group["source_site"]["dataset"]).read_text())
    if len(rows) < PREFIX_PROMPTS:
        raise ValueError("Prefix composition requires at least 2,000 held-out prompts")
    selected = random.Random(42).sample(rows, PREFIX_PROMPTS)
    if len({r["conversations"][0]["value"] for r in selected}) != PREFIX_PROMPTS:
        raise ValueError("Prefix composition requires 2,000 distinct prompts")
    # vLLM shuffles this pool once with the trial seed, visits each prompt,
    # then samples with replacement to reach the requested measurement count.
    write_new_json(
        Path(worker_group["directory"]) / "evaluation.json",
        selected,
    )


if __name__ == "__main__":
    main("prefix_cache")
