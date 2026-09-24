# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-attempt commands; never establish or repair SSH connections."""

import shlex
import subprocess

SSH = [
    "ssh",
    "-o",
    "ProxyCommand=false",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectionAttempts=1",
    "-o",
    "ConnectTimeout=10",
]


class Remote:
    def __init__(self):
        self.failed = set()

    def run(self, host, argv, *, stdin=None, timeout=60, check=True):
        if host in self.failed:
            raise RuntimeError(
                f"SSH transport failed earlier: {host}; explicit recovery required"
            )
        command = list(argv)
        if host != "local":
            if not host or host.startswith("-") or any(c.isspace() for c in host):
                raise ValueError("Use a configured SSH host alias")
            command = [*SSH, host, shlex.join(command)]
        try:
            result = subprocess.run(
                command,
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            if host != "local":
                self.failed.add(host)
            raise
        if host != "local" and result.returncode == 255:
            self.failed.add(host)
        if check and result.returncode:
            # argv may include per-run credentials. Do not print it on failure.
            raise RuntimeError(
                f"Command failed on {host} ({result.returncode}): "
                f"{result.stderr[-2000:]}"
            )
        return result
