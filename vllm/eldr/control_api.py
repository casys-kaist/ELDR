# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional loopback-only, authenticated experiment phase control."""

import hmac
import os
import re

from fastapi import HTTPException, Request


def authorized(host, supplied, token):
    return (
        host in ("127.0.0.1", "::1")
        and isinstance(supplied, str)
        and hmac.compare_digest(supplied, token)
    )


def add_routes(app):
    token = os.environ.get("ELDR_CONTROL_TOKEN")
    if token is None:
        return
    if re.fullmatch(r"[0-9a-f]{64}", token) is None:
        raise ValueError("ELDR_CONTROL_TOKEN must be a per-run 256-bit hex token")

    async def control(request: Request, mode: str):
        host = request.client.host if request.client else None
        if not authorized(host, request.headers.get("x-eldr-control-token"), token):
            raise HTTPException(403, "ELDR control access denied")
        if mode not in ("on", "off", "status"):
            raise HTTPException(400, "mode must be on, off, or status")
        return await app.state.engine_client.engine_core.call_utility_async(
            "eldr_capture", mode
        )

    app.post("/eldr/capture")(control)
