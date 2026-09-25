# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RR -> JSQ -> ELDR, once each, on one shared worker group."""

import json
import os
import signal
import time

import numpy as np

from eldr.experiments.runner.config import IMAGE, RUN_PROFILES, file_sha256
from eldr.experiments.runner.fit import read_capture
from eldr.experiments.runner.results import request_metrics
from eldr.experiments.runner.workers import (
    WorkerGroup,
    base_container,
    http,
    in_venv,
    write_new_json,
)


def endpoint(worker):
    return f"http://{worker['address']}:{worker['port']}"


def drain(worker_group, timeout=120):
    workers = worker_group.config["prefills"] + worker_group.config["decoders"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready = True
        for worker in workers:
            metrics = http(endpoint(worker) + "/metrics").decode()
            values = {
                "num_requests_running": [],
                "num_requests_waiting": [],
                "kv_cache_usage_perc": [],
            }
            for line in metrics.splitlines():
                for name in values:
                    aliases = (
                        (name, "gpu_cache_usage_perc")
                        if name == "kv_cache_usage_perc"
                        else (name,)
                    )
                    if any(
                        line.startswith("vllm:" + alias + "{")
                        or line.startswith("vllm:" + alias + " ")
                        for alias in aliases
                    ):
                        values[name].append(float(line.split()[-1]))
            if any(not v for v in values.values()):
                raise ValueError("Missing drain metrics")
            ready &= (
                sum(values["num_requests_running"]) == 0
                and sum(values["num_requests_waiting"]) == 0
                and max(values["kv_cache_usage_perc"]) < 0.01
            )
        if ready:
            return
        time.sleep(1)
    raise TimeoutError("Requests or retained KV failed to drain")


def reset(worker_group, mode):
    drain(worker_group)
    states = worker_group.control(mode)
    for worker in worker_group.config["decoders"]:
        http(endpoint(worker) + "/reset_prefix_cache", {})
    return states


def prime(worker_group):
    """Allocate every producer's count cache, then complete the real KV handoff."""
    config = worker_group.config
    for index, worker in enumerate(config["prefills"]):
        prompt = [100 + index] * 65
        request = dict(
            model="eldr",
            prompt=prompt,
            max_tokens=8,
            temperature=0,
            ignore_eos=True,
            return_token_ids=True,
            add_special_tokens=False,
            kv_transfer_params=dict(
                do_remote_decode=True, do_remote_prefill=False, prefill_first=True
            ),
        )
        pref = json.loads(
            http(endpoint(worker) + "/v1/completions", request, timeout=120)
        )
        params = pref["kv_transfer_params"]
        if not params.pop("eldr_sig", None):
            raise ValueError("Initial producer did not capture a signature")
        request.update(max_tokens=8, kv_transfer_params=params)
        decode = config["decoders"][
            index * len(config["decoders"]) // len(config["prefills"])
        ]
        result = json.loads(
            http(endpoint(decode) + "/v1/completions", request, timeout=120)
        )
        if result["usage"]["completion_tokens"] != 8:
            raise ValueError("KV handoff priming failed")
    drain(worker_group)
    if not all(s["resident_signature_cache"] for s in worker_group.control("status")):
        raise ValueError("All policies must share resident signature cache allocation")


def start_proxy(worker_group, policy, stage, *, label=None, protocol=None):
    config = worker_group.config
    protocol = protocol or {}
    name = f"{worker_group.run_id}-{label or policy}-{stage}-proxy"
    args = [
        "-m",
        "eldr.serving.proxy",
        "--host",
        config["controller_address"],
        "--port",
        str(config["proxy_port"]),
        "--prefill-router",
        "prefix-hash",
        "--prefill-prefix-bytes",
        "1024",
        "--prefill-load-factor",
        "1.25",
        "--seed",
        str(protocol.get("seed", 1234)),
        "--decode-router",
        policy,
    ]
    for role, flag in (("prefills", "prefiller"), ("decoders", "decoder")):
        args += [
            f"--{flag}-hosts",
            *[w["address"] for w in config[role]],
            f"--{flag}-ports",
            *[str(w["port"]) for w in config[role]],
        ]
    centroids = protocol.get("centroids", config.get("centroids"))
    if policy.startswith("eldr"):
        args += [
            "--eldr-centroids",
            centroids,
            "--eldr-tau",
            str(protocol.get("tau", 0.1)),
            "--eldr-online-window-seconds",
            str(protocol.get("window_seconds", 5.0)),
            "--eldr-online-refit-seconds",
            str(protocol.get("refit_seconds", 2.5)),
        ]
    if policy == "domain":
        args += ["--domain-mapping", protocol["domain_artifact"]]
    command = base_container(config, name, worker_group.run_id, config["proxy_cpus"])
    if policy.startswith("eldr"):
        command += ["-v", f"{centroids}:{centroids}:ro"]
    if policy == "domain":
        path = protocol["domain_artifact"]
        command += ["-v", f"{path}:{path}:ro"]
    if worker_group.prepared.get("capture"):
        args += ["--capture-output", str(worker_group.output / "signatures.bin")]
        command += ["-v", f"{worker_group.output}:{worker_group.output}"]
    command += [IMAGE] + in_venv(args)
    entry = worker_group.launch("local", name, command)
    url = f"http://{config['controller_address']}:{config['proxy_port']}"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if not worker_group.inspect(entry)["State"]["Running"]:
            raise RuntimeError("Proxy exited before readiness")
        try:
            http(url + "/admin/decoders", timeout=2)
            return entry, url
        except OSError:
            time.sleep(1)
    raise TimeoutError("Proxy readiness deadline")


def benchmark(worker_group, policy, stage, protocol, *, label=None):
    config = worker_group.config
    capture = worker_group.prepared.get("capture")
    dataset = capture["dataset"] if capture else config["dataset"]
    measured = stage == "measure"
    folder = worker_group.output / (label or policy) / stage
    folder.mkdir(parents=True, exist_ok=False)
    name = f"{worker_group.run_id}-{label or policy}-{stage}-client"
    args = [
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--host",
        config["controller_address"],
        "--port",
        str(config["proxy_port"]),
        "--model",
        config["model_path"],
        "--served-model-name",
        "eldr",
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        dataset,
        "--sharegpt-output-len",
        str(protocol["output_tokens"]),
        "--num-prompts",
        str(protocol["requests"] if measured else protocol["warmup_requests"]),
        "--request-rate",
        str(protocol["request_rate"] if measured else protocol["warmup_rate"]),
        "--ignore-eos",
        "--disable-tqdm",
        "--seed",
        str(protocol.get("seed", 1234)),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(folder),
        "--result-filename",
        "raw.json",
        "--percentile-metrics",
        "ttft,tpot",
        "--metric-percentiles",
        "50,95,99",
    ]
    # Match the final comparison: seeded shuffled traffic, no hidden readiness
    # generation. Captures preserve their input order.
    args += ["--ready-check-timeout-sec", "0"]
    if capture or not protocol.get("shuffle", True):
        args += ["--disable-shuffle"]
    if capture:
        # Exactly one signature per training prompt: no readiness generation,
        # no duplicated prompts if the benchmark's length filter drops a row.
        args += ["--no-oversample"]
    elif protocol.get("no_oversample"):
        args += ["--no-oversample"]
    command = base_container(config, name, worker_group.run_id, config["client_cpus"])
    command += [
        "-v",
        f"{dataset}:{dataset}:ro",
        "-v",
        f"{folder}:{folder}",
    ]
    if config.get("tiktoken_cache"):
        path = config["tiktoken_cache"]
        command += ["-v", f"{path}:{path}:ro", "-e", f"TIKTOKEN_ENCODINGS_BASE={path}"]
    entry = worker_group.launch("local", name, command + [IMAGE] + in_venv(args))
    worker_group.wait_client(entry)
    raw = json.loads((folder / "raw.json").read_text())
    count = protocol["requests"] if measured else protocol["warmup_requests"]
    tpot, ttft = request_metrics(raw, count, protocol["output_tokens"])
    return dict(
        policy=policy,
        tpot_ms={str(q): float(np.percentile(tpot, q)) for q in (50, 95, 99)},
        ttft50_ms=float(np.median(ttft)),
        requests=count,
    )


def execute(output, *, trials=None):
    worker_group = WorkerGroup(output)
    capture = worker_group.prepared.get("capture")
    if capture and trials is not None:
        raise ValueError("Capture does not accept measurement trials")
    # Exclusive creation prevents a second invocation from reusing this run.
    write_new_json(
        worker_group.output / "started.json",
        dict(pid=os.getpid(), time_ns=time.time_ns()),
    )
    previous = signal.signal(
        signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt())
    )
    results = []
    try:
        write_new_json(worker_group.output / "preflight.json", worker_group.preflight())
        protocol = (
            dict(requests=capture["requests"], request_rate=10, output_tokens=2)
            if capture
            else RUN_PROFILES[worker_group.prepared["profile"]]
        )
        if trials is None:
            trials = [
                (policy, policy, dict(protocol))
                for policy in (("rr",) if capture else ("rr", "jsq", "eldr"))
            ]
            if not capture:
                trials[0][2]["warmup_requests"] = protocol.get(
                    "initial_warmup_requests", protocol["warmup_requests"]
                )
        write_new_json(worker_group.output / "trials.json", trials)
        trial_inputs = {}
        for _, _, trial_protocol in trials:
            for key in ("centroids", "domain_artifact"):
                if key in trial_protocol:
                    path = trial_protocol[key]
                    if not os.path.isabs(path) or any(c in path for c in "\n\r:"):
                        raise ValueError(
                            "Trial input must be an absolute mountable path"
                        )
                    trial_inputs[path] = file_sha256(path)
        write_new_json(worker_group.output / "trial_inputs.json", trial_inputs)
        worker_group.start_engines()
        prime(worker_group)
        for label, policy, trial_protocol in trials:
            mode = "on" if capture or policy.startswith("eldr") else "off"
            for stage in ("measure",) if capture else ("warmup", "measure"):
                for path, expected in trial_inputs.items():
                    if file_sha256(path) != expected:
                        raise ValueError("Trial input changed: " + path)
                states = reset(worker_group, mode)
                proxy, url = start_proxy(
                    worker_group, policy, stage, label=label, protocol=trial_protocol
                )
                print(f"[runner] {label}/{stage}", flush=True)
                started = time.monotonic()
                result = benchmark(
                    worker_group, policy, stage, trial_protocol, label=label
                )
                result.update(
                    label=label,
                    protocol=trial_protocol,
                    elapsed_seconds=time.monotonic() - started,
                )
                folder = worker_group.output / label / stage
                write_new_json(folder / "metrics.json", result)
                drain(worker_group)
                if policy in ("eldr", "eldr-static"):
                    status = json.loads(http(url + "/admin/routing"))
                    expected_mode = {
                        "eldr": "online",
                        "eldr-static": "static",
                    }[policy]
                    if (
                        status.get("centroid_mode") != expected_mode
                        or status.get("selector") != "jsq"
                        or status.get("signature")
                        != worker_group.config["signature_variant"]
                        or status.get("tau") != trial_protocol.get("tau", 0.1)
                        or status.get("rule") != "locality_band_jsq_v1"
                    ):
                        raise ValueError(
                            "Proxy signature/routing differs from the frozen recipe"
                        )
                    write_new_json(folder / "routing.json", status)
                    if policy == "eldr":
                        refit = status["refit"]
                        if (
                            refit["last_error"]
                            or refit["window_seconds"]
                            != trial_protocol["window_seconds"]
                            or refit["refit_seconds"] != trial_protocol["refit_seconds"]
                            or (
                                stage == "measure"
                                and worker_group.prepared["profile"] == "paper"
                                and not refit["refits_completed"]
                            )
                        ):
                            raise ValueError(
                                "Online centroid refresh was not validated"
                            )
                write_new_json(folder / "capture.json", states)
                worker_group.stop(proxy)
                if stage == "measure":
                    results.append(result)
                    print(json.dumps(result), flush=True)
        if capture:
            counts = read_capture(
                (worker_group.output / "signatures.bin").read_bytes(),
                worker_group.config["model"],
            )
            if len(counts) != capture["requests"]:
                raise ValueError(
                    "Capture must contain exactly one signature per prompt"
                )
        write_new_json(
            worker_group.output / "results.json",
            dict(
                profile=worker_group.prepared["profile"],
                protocol=protocol,
                first_token_mode=worker_group.config["first_token_mode"],
                results=results,
            ),
        )
    except BaseException as error:
        write_new_json(
            worker_group.output / "failed.json",
            dict(type=type(error).__name__, error=str(error)),
        )
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        worker_group.cleanup()
    write_new_json(
        worker_group.output / "complete.json",
        dict(measurements=len(results), time_ns=time.time_ns()),
    )
    return results
