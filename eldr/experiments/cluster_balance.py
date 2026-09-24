"""Balanced versus vanilla spherical K-means, same frozen signature and seed."""

from pathlib import Path

from eldr.experiments.runner.config import FIT_SEED
from eldr.experiments.runner.experiment import main, plan_worker_group
from eldr.experiments.runner.fit import fit_centroids, training_dataset

RATES = (20, 40, 60, 80, 100)
CLUSTERINGS = ("balanced", "vanilla")


def configure(config, output, policies):
    training_dataset(Path(config["training_prompts"]), Path(config["dataset"]))
    folder = output / config["model"]
    variants = [("rr", "rr", {})] + [
        # Freeze both fits: balanced online refits would erase this distinction.
        (name, "eldr-static", dict(centroids=str(folder / f"{name}.json")))
        for name in CLUSTERINGS
    ]
    worker_group = plan_worker_group(
        config, folder, variants, inputs=("training_signatures", "training_prompts")
    )
    worker_group["site"]["centroids"] = str(folder / "balanced.json")
    worker_group["site"].pop("centroids_sha256", None)
    return [worker_group]


def prepare(worker_group):
    config, folder = worker_group["source_site"], Path(worker_group["directory"])
    for name in CLUSTERINGS:
        fit_centroids(
            Path(config["training_signatures"]),
            folder / f"{name}.json",
            config["model"],
            len(config["decoders"]),
            FIT_SEED,
            balanced=name == "balanced",
            transform_from=Path(config["centroids"]),
        )


if __name__ == "__main__":
    main("cluster_balance")
