# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Signature ablation: matched fits from newly captured calibration requests."""

from eldr.experiments.runner.experiment import main, plan_worker_group
from eldr.experiments.runner.fit import fit_signature

RATES = (60,)
POLICIES = ("rr", "eldr-static")
VARIANTS = ("count_idf", "gate_prob_all")


def configure(config, output, policies):
    if "activations" not in config:
        raise ValueError("Signature ablation requires request-aligned activations")
    groups = []
    for variant in VARIANTS:
        folder = output / f"{config['model']}-{variant}"
        group = plan_worker_group(
            config,
            folder,
            [("rr", "rr", {}), (variant, "eldr-static", {})],
            inputs=("activations",),
        )
        group["site"].update(
            centroids=str(folder / "centroids.json"), signature_variant=variant
        )
        group["site"].pop("centroids_sha256", None)
        groups.append(group)
    return groups


def prepare(worker_group):
    config = worker_group["site"]
    fit_signature(
        worker_group["source_site"]["activations"],
        config["centroids"],
        config["model"],
        config["signature_variant"],
        len(config["decoders"]),
    )


if __name__ == "__main__":
    main("signature_ablation")
