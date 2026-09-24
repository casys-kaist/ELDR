# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate environment configuration and define reproducible run profiles."""

import hashlib
import ipaddress
import json
import re
from pathlib import Path

from eldr.serving.clustering import load_centroid_file

IMAGE = (
    "rocm/vllm-dev@sha256:"
    "411f7ca9abeb57f69064e8bf60f46b5cd0b67f26264af0fc831f124fbc22f4b1"
)
MODEL_GEOMETRY = {
    "gptoss": (36, 128, 64),
    "qwen": (48, 128, 16),
    "gemma": (30, 128, 16),
}
INTERNAL_PORT_SPAN = 128
FIT_SEED = 1
RUN_PROFILES = {
    "paper": dict(
        recipe="full-jsq-min2-v1",
        seed=1234,
        shuffle=True,
        requests=7200,
        request_rate=60,
        output_tokens=512,
        initial_warmup_requests=14400,
        warmup_requests=3840,
        warmup_rate=64,
        window_seconds=5.0,
        refit_seconds=2.5,
    ),
    "smoke": dict(
        recipe="full-jsq-min2-v1",
        seed=1234,
        shuffle=True,
        requests=32,
        request_rate=2,
        output_tokens=32,
        warmup_requests=8,
        warmup_rate=2,
        window_seconds=5.0,
        refit_seconds=2.5,
    ),
}


def cpu_set(value):
    result = set()
    for part in value.split(","):
        limits = [int(v) for v in part.split("-")]
        if len(limits) > 2 or limits[0] > limits[-1] or limits[0] < 0:
            raise ValueError("Invalid CPU mask")
        result.update(range(limits[0], limits[-1] + 1))
    return result


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(
    path,
    *,
    profile="paper",
    check_files=True,
    require_centroids=True,
    prefix_cache_experiment=False,
):
    config = json.loads(Path(path).read_text())
    if config.get("schema_version") != 1 or config["model"] not in MODEL_GEOMETRY:
        raise ValueError("Unsupported configuration schema/model")
    if profile not in RUN_PROFILES:
        raise ValueError("Unknown profile")
    mode = config.setdefault("first_token_mode", "prefill")
    if mode != "prefill":
        raise ValueError("This revision requires prefill-first token handoff")
    if config.get("image", IMAGE) != IMAGE:
        raise ValueError("The tested serving image digest is required")
    if type(config.get("prefix_cache", True)) is not bool:
        raise ValueError("prefix_cache must be boolean")
    inputs = ("dataset", "centroids") if require_centroids else ("dataset",)
    paths = ("repo", "model_path", *inputs)
    if config.get("tiktoken_cache"):
        paths += ("tiktoken_cache",)
    for key in paths:
        value = config[key]
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or any(c in value for c in "\n\r:")
        ):
            raise ValueError(f"{key} must be an absolute, single-line path without ':'")
        if check_files and not Path(value).exists():
            raise ValueError(f"Missing {key}: {value}")
    for role in ("prefills", "decoders"):
        if not config[role]:
            raise ValueError("Nonempty prefill and decode worker groups required")
    allowed = ((8, 16),)
    if prefix_cache_experiment:
        allowed += ((1, 16),)
    if (
        profile == "paper"
        and (len(config["prefills"]), len(config["decoders"])) not in allowed
    ):
        expected = ", ".join(f"{p}P{d}D" for p, d in allowed)
        raise ValueError(f"The paper profile requires {expected} for this experiment")
    gpus, ports = set(), set()
    for worker in config["prefills"] + config["decoders"]:
        ipaddress.ip_address(worker["address"])
        host = worker["host"]
        if worker["address"] == config["controller_address"] and host != "local":
            raise ValueError("Use host='local' for the controller; never SSH to self")
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", host) or host.startswith("-"):
            raise ValueError("Invalid SSH alias (use 'local' on the controller)")
        if type(worker["gpu"]) is not int or worker["gpu"] < 0:
            raise ValueError("Invalid GPU")
        if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", worker["cpus"]):
            raise ValueError("Explicit GPU-local CPU list required")
        cpu_set(worker["cpus"])
        if not re.fullmatch(r"mlx5_\d+:1", worker["nic"]):
            raise ValueError("Explicit ROCm RDMA rail required")
        identity = (worker["address"], worker["gpu"])
        if identity in gpus:
            raise ValueError("A GPU cannot serve two workers")
        gpus.add(identity)
        for key in ("port", "side_channel_port", "internal_port"):
            value = worker[key]
            if type(value) is not int or not 1024 <= value <= 65000:
                raise ValueError("Invalid worker port")
            # Account for vLLM's additional ports; this is not a kernel reservation.
            span = INTERNAL_PORT_SPAN if key == "internal_port" else 1
            for port in range(value, value + span):
                endpoint = (worker["address"], port)
                if endpoint in ports:
                    raise ValueError("Worker ports overlap")
                ports.add(endpoint)
    ipaddress.ip_address(config["controller_address"])
    for key in ("proxy_cpus", "client_cpus"):
        if not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", config[key]):
            raise ValueError("Explicit controller CPU masks required")
    port = config["proxy_port"]
    if type(port) is not int or not 1024 <= port <= 65000:
        raise ValueError("Invalid proxy port")
    if (config["controller_address"], port) in ports:
        raise ValueError("Proxy port overlaps a worker")
    if check_files:
        if require_centroids:
            centroid_data = load_centroid_file(config["centroids"])
            layers, experts, _ = MODEL_GEOMETRY[config["model"]]
            if centroid_data["L"] != layers or centroid_data["e"] != experts:
                raise ValueError("Centroid/model geometry mismatch")
            if len(centroid_data["centroid_matrix"]) != len(config["decoders"]):
                raise ValueError("Require exactly one centroid per decoder")
            config["signature_variant"] = centroid_data.get("variant", "count")
            if profile == "paper" and config["signature_variant"] != "count_idf":
                raise ValueError(
                    "The paper recipe requires a frozen Full (count_idf) fit"
                )
        for key in inputs:
            actual = file_sha256(config[key])
            if config.get(key + "_sha256", actual) != actual:
                raise ValueError("Input SHA-256 mismatch: " + key)
            config[key + "_sha256"] = actual
    return config
