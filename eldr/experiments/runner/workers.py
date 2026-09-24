# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manage the experiment's worker containers without SSH reconnections."""

import json
import secrets
import shlex
import time
import urllib.request
from pathlib import Path

from eldr.experiments.runner.bootstrap import EXTENSIONS
from eldr.experiments.runner.config import (
    IMAGE,
    INTERNAL_PORT_SPAN,
    MODEL_GEOMETRY,
    cpu_set,
    file_sha256,
)
from eldr.experiments.runner.fit import training_dataset
from eldr.experiments.runner.remote import Remote

LABEL = "eldr.artifact.run"  # Stable Docker ownership label, not a module name.
VLLM_PACKAGE_PATH = "/usr/local/lib/python3.12/dist-packages/vllm"


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def http(url, body=None, timeout=10):
    request = urllib.request.Request(
        url,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def source_files(root):
    """Serving source, native extensions and runtime assets; no private capsules."""
    paths = []
    for prefix in (
        "vllm",
        "eldr/serving",
        "eldr/nixl_shim",
        "eldr/experiments",
    ):
        paths.extend(
            p
            for p in (root / prefix).rglob("*")
            if p.is_file()
            and (
                p.suffix in (".py", ".so")
                or prefix == "vllm"
                and p.suffix in (".json", ".jinja", ".sh", ".js", ".css")
            )
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
        )
    paths.extend(root / p for p in ("eldr/__init__.py", "eldr/serving/clustering.py"))
    paths.append(root / "eldr/experiments/__init__.py")
    return sorted(set(p for p in paths if p.exists() and "__pycache__" not in p.parts))


def check_ports(remote, host, ports):
    """Keep server ports outside automatic allocation; reject existing TCP users."""
    low, high = map(
        int,
        remote.run(
            host, ["cat", "/proc/sys/net/ipv4/ip_local_port_range"]
        ).stdout.split(),
    )
    overlap = sorted(p for p in ports if low <= p <= high)
    if overlap:
        raise ValueError(
            f"Server ports overlap the ephemeral range {low}-{high} on {host}: "
            f"{overlap[0]}..{overlap[-1]}; choose ports outside that range"
        )
    deadline = time.monotonic() + 90
    while True:
        sockets = {
            (int(fields[3].rsplit(":", 1)[1]), fields[0])
            for fields in map(
                str.split, remote.run(host, ["ss", "-Htan"]).stdout.splitlines()
            )
        }
        busy = sorted((port, state) for port, state in sockets if port in ports)
        if not busy:
            break
        if any(state != "TIME-WAIT" for _, state in busy):
            raise ValueError(f"Server ports already in use on {host}: {busy}")
        # Closed connections can outlive their containers. Wait for expiry,
        # not socket reuse; live owners and SSH failures still fail immediately.
        if time.monotonic() >= deadline:
            raise TimeoutError(f"TCP TIME_WAIT did not drain on {host}: {busy}")
        time.sleep(2)
    return dict(ephemeral_range=[low, high], checked_ports=len(ports))


def prepare(config, output, profile, *, training=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a fresh run directory")
    root = Path(config["repo"])
    # The source overlay must contain the extensions built for this image.
    for name, expected in EXTENSIONS.items():
        path = root / "vllm" / name
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected:
            raise ValueError("Run bootstrap: missing/incompatible extension " + name)
    sources = {str(p.relative_to(root)): file_sha256(p) for p in source_files(root)}
    rows, training_sha = (
        training_dataset(Path(training), Path(config["dataset"]))
        if training is not None
        else (None, None)
    )
    output.mkdir(parents=True, exist_ok=False)
    capture = None
    if rows is not None:
        path = output / "training.json"
        write_new_json(path, rows)
        capture = dict(
            dataset=str(path),
            dataset_sha256=file_sha256(path),
            source_sha256=training_sha,
            requests=len(rows),
        )
    write_new_json(
        output / "prepared.json",
        dict(
            site=config,
            profile=profile,
            capture=capture,
            sources=sources,
            image=IMAGE,
            run_id="eldr-" + secrets.token_hex(6),
            created_ns=time.time_ns(),
        ),
    )
    return output


def base_container(config, name, run_id, cpus):
    return [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--label",
        f"{LABEL}={run_id}",
        "--network",
        "host",
        "--ipc",
        "host",
        "--cpuset-cpus",
        cpus,
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "nofile=1048576:1048576",
        "-v",
        f"{config['repo']}:{config['repo']}:ro",
        "-v",
        f"{config['model_path']}:{config['model_path']}:ro",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "ELDR_BENCHMARK_GC=1",
        "-e",
        "OPENBLAS_NUM_THREADS=1",
        "-e",
        "OMP_NUM_THREADS=1",
        "-w",
        config["repo"],
    ]


def in_venv(argv):
    # Use the pinned image's native packages, with an ephemeral venv rather
    # than pip-installing dependencies into the image at every invocation.
    command = "uv venv --offline --system-site-packages /tmp/.venv && exec "
    command += shlex.join(["/tmp/.venv/bin/python", *argv])
    return ["bash", "-lc", command]


def engine_command(config, worker, producer, name, run_id, token):
    command = base_container(config, name, run_id, worker["cpus"])
    command += [
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--device",
        "/dev/infiniband",
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
    ]
    env = dict(
        HIP_VISIBLE_DEVICES=str(worker["gpu"]),
        UCX_TLS="all",
        UCX_NET_DEVICES=worker["nic"],
        UCX_MAX_RNDV_RAILS="1",
        UCX_IB_NUM_PATHS="1",
        VLLM_NIXL_SIDE_CHANNEL_HOST=worker["address"],
        VLLM_NIXL_SIDE_CHANNEL_PORT=str(worker["side_channel_port"]),
        VLLM_PORT=str(worker["internal_port"]),
        VLLM_SSM_CONV_STATE_LAYOUT="DS",
        VLLM_USE_LAYERNAME="0",
        VLLM_SERVER_DEV_MODE="1",
        PYTHONPATH=config["repo"] + "/eldr/nixl_shim",
        ELDR="1" if producer else "0",
        ELDR_SCRATCH_BUCKETS="1" if producer else "0",
    )
    if producer:
        env["ELDR_CONTROL_TOKEN"] = token
        if config.get("signature_variant") == "gate_prob_all":
            env["ELDR_SIGNATURE"] = "gate_prob_all"
    if config.get("tiktoken_cache"):
        path = config["tiktoken_cache"]
        command += ["-v", f"{path}:{path}:ro"]
        env["TIKTOKEN_ENCODINGS_BASE"] = path
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    serve = [
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        config["model_path"],
        "--served-model-name",
        "eldr",
        "--host",
        "0.0.0.0",
        "--port",
        str(worker["port"]),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
        "--block-size",
        str(MODEL_GEOMETRY[config["model"]][2]),
        "--enable-prefix-caching"
        if config.get("prefix_cache", True)
        else "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
        "--disable-hybrid-kv-cache-manager",
        "--distributed-executor-backend",
        "uni",
        "--compilation-config",
        '{"cudagraph_mode":"FULL_DECODE_ONLY"}',
        "--kv-transfer-config",
        json.dumps(
            dict(
                kv_connector="NixlConnector",
                kv_role="kv_producer" if producer else "kv_consumer",
                kv_load_failure_policy="fail",
            )
        ),
    ]
    if producer:
        serve += ["--enforce-eager", "--no-async-scheduling"]
    else:
        serve.append("--async-scheduling")
    if config["model"] == "gptoss":
        serve += ["--quantization", "mxfp4"]
    return command + [IMAGE] + in_venv(serve)


class WorkerGroup:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.prepared = json.loads((self.output / "prepared.json").read_text())
        self.config = self.prepared["site"]
        self.run_id = self.prepared["run_id"]
        self.remote = Remote()
        self.owned = []
        self.token = secrets.token_hex(32)

    def preflight(self):
        root, config = Path(self.config["repo"]), self.config
        for relative, expected in self.prepared["sources"].items():
            if file_sha256(root / relative) != expected:
                raise ValueError("Prepared source changed: " + relative)
        inputs = (
            ("dataset",) if self.prepared.get("capture") else ("dataset", "centroids")
        )
        for key in inputs:
            if file_sha256(config[key]) != config[key + "_sha256"]:
                raise ValueError("Prepared input changed: " + key)
        capture = self.prepared.get("capture")
        if capture and file_sha256(capture["dataset"]) != capture["dataset_sha256"]:
            raise ValueError("Prepared training input changed")
        workers = config["prefills"] + config["decoders"]
        hosts = sorted({w["host"] for w in workers} | {"local"})
        port_checks = {}
        for host in hosts:
            if self.remote.run(host, ["docker", "ps", "-q"]).stdout.strip():
                raise ValueError(
                    f"Running containers on {host}; exclusive nodes required"
                )
            ports = {config["proxy_port"]} if host == "local" else set()
            for worker in (w for w in workers if w["host"] == host):
                ports.update((worker["port"], worker["side_channel_port"]))
                start = worker["internal_port"]
                ports.update(range(start, start + INTERNAL_PORT_SPAN))
            port_checks[host] = check_ports(self.remote, host, ports)
            self.remote.run(host, ["docker", "image", "inspect", IMAGE])
            if host != "local":
                manifest = "".join(
                    f"{sha}  {root / rel}\n"
                    for rel, sha in self.prepared["sources"].items()
                )
                self.remote.run(
                    host,
                    ["sha256sum", "--check", "--status", "-"],
                    stdin=manifest,
                    timeout=120,
                )
            self.remote.run(host, ["test", "-f", config["model_path"] + "/config.json"])
            self.remote.run(host, ["test", "-d", config["model_path"]])
            if config.get("tiktoken_cache"):
                self.remote.run(
                    host,
                    ["test", "-f", config["tiktoken_cache"] + "/o200k_base.tiktoken"],
                )
            if any(w["host"] == host for w in workers):
                smi = self.remote.run(
                    host,
                    [
                        "sh",
                        "-c",
                        "command -v rocm-smi || command -v /opt/rocm/bin/rocm-smi",
                    ],
                ).stdout.strip()
                gpus = json.loads(
                    self.remote.run(
                        host, [smi, "--showbus", "--showuse", "--showmemuse", "--json"]
                    ).stdout
                )
                for worker in (w for w in workers if w["host"] == host):
                    card = gpus[f"card{worker['gpu']}"]
                    bus = card["PCI Bus"].lower()
                    import re

                    if not re.fullmatch(
                        r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bus
                    ):
                        raise ValueError("Invalid GPU PCI bus")
                    node = int(
                        self.remote.run(
                            host, ["cat", f"/sys/bus/pci/devices/{bus}/numa_node"]
                        ).stdout
                    )
                    if node < 0:
                        raise ValueError("Missing GPU NUMA locality")
                    local_cpus = self.remote.run(
                        host, ["cat", f"/sys/devices/system/node/node{node}/cpulist"]
                    ).stdout.strip()
                    if not cpu_set(worker["cpus"]) <= cpu_set(local_cpus):
                        raise ValueError("Worker CPU mask is not GPU-local")
                    if int(card["GPU use (%)"]) or int(
                        card["GPU Memory Allocated (VRAM%)"]
                    ):
                        raise ValueError(f"GPU already in use: {host}/{worker['gpu']}")
                    rail = worker["nic"].split(":")[0]
                    state = self.remote.run(
                        host, ["cat", f"/sys/class/infiniband/{rail}/ports/1/state"]
                    ).stdout
                    if state.strip() != "4: ACTIVE":
                        raise ValueError(f"RDMA rail is not active: {host}/{rail}")
        return dict(
            hosts=hosts,
            sources=len(self.prepared["sources"]),
            image=IMAGE,
            ports=port_checks,
        )

    def launch(self, host, name, command):
        # Record the exact owned NAME before launch. On transport failure the
        # result is uncertain; never reconnect or infer that launch failed.
        entry = dict(host=host, name=name, id=None)
        self.owned.append(entry)
        with (self.output / "ownership.jsonl").open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        result = self.remote.run(host, command, timeout=120)
        cid = result.stdout.strip()
        if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
            raise ValueError("Docker did not return a unique full container ID")
        entry["id"] = cid
        return entry

    def inspect(self, entry):
        result = self.remote.run(
            entry["host"], ["docker", "inspect", entry["id"] or entry["name"]]
        )
        (info,) = json.loads(result.stdout)
        if info["Config"]["Labels"].get(LABEL) != self.run_id:
            raise ValueError("Ownership mismatch; refusing container action")
        return info

    def stop(self, entry):
        info = self.inspect(entry)
        cid = info["Id"]
        self.remote.run(entry["host"], ["docker", "stop", "-t", "20", cid], timeout=45)
        result = self.remote.run(entry["host"], ["docker", "logs", cid], check=False)
        (self.output / (entry["name"] + ".log")).write_text(
            result.stdout + result.stderr
        )
        self.remote.run(entry["host"], ["docker", "rm", cid])
        self.owned.remove(entry)

    def cleanup(self):
        problems = []
        for entry in reversed(self.owned.copy()):
            if entry["host"] in self.remote.failed:
                problems.append(
                    dict(entry, error="Transport failed; manual recovery required")
                )
                continue
            try:
                self.stop(entry)
            except Exception as error:
                problems.append(dict(entry, error=str(error)))
        write_new_json(self.output / "cleanup.json", dict(remaining=problems))
        if problems:
            raise RuntimeError(
                "Cleanup incomplete: see cleanup.json; no automatic reconnect"
            )

    def start_engines(self):
        self.engines = []
        for role in ("prefills", "decoders"):
            for index, worker in enumerate(self.config[role]):
                name = f"{self.run_id}-{role}-{index}"
                entry = self.launch(
                    worker["host"],
                    name,
                    engine_command(
                        self.config,
                        worker,
                        role == "prefills",
                        name,
                        self.run_id,
                        self.token,
                    ),
                )
                self.engines.append((worker, entry, role))
        deadline = time.monotonic() + 900
        pending = self.engines.copy()
        while pending and time.monotonic() < deadline:
            for worker, entry, role in pending.copy():
                if not self.inspect(entry)["State"]["Running"]:
                    raise RuntimeError(
                        "Engine exited before readiness: " + entry["name"]
                    )
                try:
                    http(
                        f"http://{worker['address']}:{worker['port']}/health", timeout=2
                    )
                except (OSError, TimeoutError):
                    continue  # Expected HTTP readiness, not an SSH retry.
                pending.remove((worker, entry, role))
            if pending:
                time.sleep(2)
        if pending:
            raise TimeoutError("Engine readiness deadline")

    def control(self, mode):
        states = []
        for worker, entry, role in self.engines:
            if role != "prefills":
                continue
            # Credential is read inside the owning container, never in argv/logs.
            script = (
                "import os,urllib.request; r=urllib.request.Request('http://127.0.0.1:"
                + str(worker["port"])
                + "/eldr/capture?mode="
                + mode
                + "',data=b'',headers={'x-eldr-control-token':"
                "os.environ['ELDR_CONTROL_TOKEN']}); "
                "print(urllib.request.urlopen(r,timeout=30).read().decode())"
            )
            result = self.remote.run(
                worker["host"],
                ["docker", "exec", entry["id"], "/tmp/.venv/bin/python", "-c", script],
                timeout=45,
            )
            state = json.loads(result.stdout)
            if not state["generation_empty"] or (
                mode != "status" and state["enabled"] != (mode == "on")
            ):
                raise ValueError("Phase capture state did not match")
            states.append(state)
        return states

    def wait_client(self, entry, timeout=3600):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.inspect(entry)["State"]
            if not state["Running"]:
                if state["ExitCode"]:
                    raise RuntimeError("Client exited unsuccessfully: " + entry["name"])
                self.stop(entry)
                return
            time.sleep(2)
        raise TimeoutError("Client deadline")
