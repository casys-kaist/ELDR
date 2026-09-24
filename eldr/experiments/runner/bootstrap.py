# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore exact native extensions from the pinned image; never overwrite code."""

import json
import os
import re
import secrets
import shlex
import subprocess
import tempfile
from pathlib import Path

from eldr.experiments.runner.config import IMAGE, file_sha256
from eldr.experiments.runner.remote import SSH, Remote

EXTENSIONS = {
    "_C.abi3.so": "1035ffe242f81f92edba37f9ec285137f84e823b6bc21652c7f45dba8e602b91",
    "_moe_C.abi3.so": (
        "9cf2c07f6d05b5c6c131168fdd79379beef6825a2c39dfafadc1bff2d834155d"
    ),
    "_rocm_C.abi3.so": (
        "28063008e2fe7bca8fddff14a17c8db2db5cea7b43089e657d05a6418f7df46f"
    ),
    "cumem_allocator.abi3.so": (
        "b90eb29e4756864516b5bb1ec8149a7470bc2026b735e0be61d6e82eb1d6f16e"
    ),
}


def restore(root: Path):
    target = root.resolve() / "vllm"
    if not (target / "__init__.py").is_file():
        raise ValueError("Run bootstrap from an ELDR source tree")
    missing = []
    for name, expected in EXTENSIONS.items():
        path = target / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or file_sha256(path) != expected:
                raise ValueError(
                    "Existing extension differs; refusing overwrite: " + name
                )
        else:
            missing.append(name)
    if not missing:
        return dict(restored=[], verified=list(EXTENSIONS))
    # This temporary container is never started and cannot touch GPUs.
    subprocess.run(
        ["docker", "image", "inspect", IMAGE], check=True, stdout=subprocess.DEVNULL
    )
    name = "eldr-bootstrap-" + secrets.token_hex(6)
    cid = subprocess.run(
        [
            "docker",
            "create",
            "--network",
            "none",
            "--name",
            name,
            "--label",
            "eldr.bootstrap=" + name,
            IMAGE,
            "true",
        ],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if re.fullmatch(r"[0-9a-f]{64}", cid) is None:
        raise RuntimeError("Unknown container ID; inspect " + name + " manually")
    try:
        with tempfile.TemporaryDirectory(
            prefix=".eldr-native-", dir=target
        ) as directory:
            for filename in missing:
                temporary = Path(directory) / filename
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{cid}:/usr/local/lib/python3.12/dist-packages/vllm/{filename}",
                        str(temporary),
                    ],
                    check=True,
                    timeout=180,
                )
                if file_sha256(temporary) != EXTENSIONS[filename]:
                    raise ValueError("Pinned extension hash mismatch: " + filename)
            for filename in missing:
                # link() fails if another process installed the destination;
                # it never overwrites an existing file.
                os.link(Path(directory) / filename, target / filename)
    finally:
        subprocess.run(["docker", "rm", cid], check=True, stdout=subprocess.DEVNULL)
    return dict(restored=missing, verified=list(EXTENSIONS))


def deploy(config):
    """Prepare this checkout on the configured idle nodes; never replace a tree."""
    from eldr.experiments.runner.workers import source_files

    root = Path(config["repo"]).resolve()
    remote = Remote()
    interfaces = json.loads(remote.run("local", ["ip", "-j", "addr"]).stdout)
    addresses = {a["local"] for i in interfaces for a in i.get("addr_info", [])}
    if config["controller_address"] not in addresses:
        raise ValueError(
            "Run this script on controller " + config["controller_address"]
        )
    hosts = sorted(
        {w["host"] for w in config["prefills"] + config["decoders"]} | {"local"}
    )
    for host in hosts:
        if remote.run(host, ["docker", "ps", "-q"]).stdout.strip():
            raise ValueError(f"Running containers on {host}; exclusive nodes required")
        remote.run(host, ["docker", "image", "inspect", IMAGE])
    native = restore(root)
    files = source_files(root)
    manifest = "".join(f"{file_sha256(p)}  {p}\n" for p in files)
    for host in hosts:
        if host == "local":
            continue
        exists = remote.run(host, ["test", "-e", str(root)], check=False).returncode
        if exists not in (0, 1):
            raise RuntimeError("Cannot inspect worker checkout: " + host)
        if not exists:
            # A prior copy may be reused only if every serving file matches.
            # Never overwrite another reviewer's checkout or an interrupted copy.
            remote.run(host, ["sha256sum", "--check", "--status", "-"], stdin=manifest)
            continue
        remote.run(host, ["mkdir", "-p", "--", str(root)])
        subprocess.run(
            [
                "rsync",
                "-a",
                "--protect-args",
                "--ignore-existing",
                "--files-from=-",
                "-e",
                shlex.join(SSH),
                str(root) + "/",
                host + ":" + str(root) + "/",
            ],
            input="".join(str(p.relative_to(root)) + "\n" for p in files),
            text=True,
            check=True,
            timeout=300,
        )
        remote.run(host, ["sha256sum", "--check", "--status", "-"], stdin=manifest)
    return dict(hosts=hosts, sources=len(files), native=native)
