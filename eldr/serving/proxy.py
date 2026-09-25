#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""One P/D forwarding path for ELDR, its controls, and paper baselines.

Deliver the prefiller's first token, then resume its generated history on D.
No independently regenerated token is discarded. All policies share this path.
"""

import argparse
import hashlib
import json
import os
import random
import struct
import sys
import uuid
from contextlib import ExitStack

# Configure native libraries before NumPy imports, only in the standalone CLI.
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from eldr.serving.cpu_threads import configure_proxy_threads

    configure_proxy_threads()

import aiohttp
import numpy as np
import pybase64 as base64
from aiohttp import web

from eldr.serving.baselines import BaselineRouter
from eldr.serving.centroid_update import (
    REFIT_SECONDS,
    WINDOW_SECONDS,
    OnlineJSQRouter,
)
from eldr.serving.clustering import build_signature, load_centroid_file
from eldr.serving.prefill_first import (
    event,
    public_prefill,
    release_prefill,
    send_prefill,
    unpack_prefill,
    validate_request,
)
from eldr.serving.prefix_hash import PrefixHashRouter
from eldr.serving.routing import route_jsq

POLICIES = (
    "rr",
    "jsq",
    "random",
    "p2c",
    "domain",
    "eldr",
    "eldr-static",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10001)
    for role in ("prefiller", "decoder"):
        parser.add_argument(f"--{role}-hosts", nargs="+", required=True)
        parser.add_argument(f"--{role}-ports", nargs="+", type=int, required=True)
    parser.add_argument(
        "--prefill-router", choices=("prefix-hash", "jsq", "rr"), default="prefix-hash"
    )
    parser.add_argument("--decode-router", choices=POLICIES, default="rr")
    parser.add_argument("--eldr-centroids")
    parser.add_argument("--domain-mapping")
    parser.add_argument("--eldr-tau", type=float, default=0.1)
    parser.add_argument(
        "--eldr-online-window-seconds", type=float, default=WINDOW_SECONDS
    )
    parser.add_argument(
        "--eldr-online-refit-seconds", type=float, default=REFIT_SECONDS
    )
    parser.add_argument("--prefill-prefix-bytes", type=int, default=1024)
    parser.add_argument("--prefill-load-factor", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--capture-output", help="New binary training-capture file; never append"
    )
    args = parser.parse_args()
    if not np.isfinite(args.eldr_tau) or args.eldr_tau < 0:
        parser.error("--eldr-tau must be finite and nonnegative")
    return args


def prompt_key(payload):
    prompt = payload["prompt"]  # validate_request accepts one text completion only.
    if isinstance(prompt, list):
        prompt = "".join(str(x) for x in prompt)
    return str(prompt).encode("utf-8", errors="ignore")


def make_decode_router(name, decoders, seed):
    if name in ("rr", "jsq", "random", "p2c"):
        return BaselineRouter(name, decoders, seed)
    if name in POLICIES:
        return None
    raise ValueError(f"Unknown policy: {name}")


def decode_signature(app, encoded):
    if not encoded:
        raise web.HTTPInternalServerError(
            reason="ELDR requires a completed prefill signature"
        )
    centroid_data = app["centroid_data"]
    try:
        counts = np.frombuffer(
            base64.b64decode(encoded, validate=True),
            dtype=centroid_data.get("sig_dtype", "<i2"),
        )
        if np.any(counts < 0) or not np.isfinite(counts).all():
            raise ValueError("Invalid expert signal")
        return counts.reshape(centroid_data["L"], centroid_data["e"])
    except (ValueError, TypeError) as error:
        raise web.HTTPInternalServerError(reason="Invalid expert signature") from error


def choose_prefill(app, payload):
    urls = app["prefill"]
    loads = [app["prefill_inflight"][url] for url in urls]
    return urls[app["prefill_router"].select(prompt_key(payload), loads)]


def choose_decode(app, encoded_signature, payload):
    decoders = app["decode"]
    loads = [app["decode_inflight"][url] for url in decoders]
    policy = app["decode_router_name"]
    if policy == "eldr-static":
        index = route_jsq(
            app["centroid_data"],
            decode_signature(app, encoded_signature),
            loads,
            app["tau"],
        )
    elif policy == "eldr":
        vector = build_signature(
            app["centroid_data"], decode_signature(app, encoded_signature)
        )
        index = app["online_jsq_router"].route_transformed(vector, loads)
    elif policy == "domain":
        domain_mapping = app["domain_mapping"]
        key = hashlib.md5(prompt_key(payload)).hexdigest()[:16]
        domain = domain_mapping["hash2dom"].get(key)
        eligible = domain_mapping["dom2dec"].get(domain) if domain is not None else None
        eligible = eligible or domain_mapping["misc_dec"] or list(range(len(decoders)))
        minimum = min(loads[i] for i in eligible)
        index = app["domain_rng"].choice([i for i in eligible if loads[i] == minimum])
    else:
        index = app["decode_router"].select(b"", loads)
    return decoders[index]


async def handle_admin_decoders(request):
    app = request.app
    if request.method != "GET":
        raise web.HTTPBadRequest(
            reason="Fixed worker group: prepare a new run to change workers"
        )
    return web.json_response(
        {"decoders": app["decode"], "inflight": app["decode_inflight"]}
    )


async def handle(request):
    app = request.app
    payload = await request.json()
    validate_request(request.path, payload)
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex
    payload = dict(payload, request_id=request_id)
    headers = {"X-Request-Id": request_id}
    session = app["http"]
    prefiller = choose_prefill(app, payload)
    app["prefill_inflight"][prefiller] += 1
    try:
        prefill = dict(
            payload,
            stream=False,
            return_token_ids=True,
            kv_transfer_params={
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "transfer_id": f"xfer-{request_id}",
                "prefill_first": True,
            },
        )
        prefill.pop("stream_options", None)
        async with session.post(
            f"{prefiller}{request.path}", json=prefill, headers=headers
        ) as response:
            response.raise_for_status()
            result = await response.json()
    finally:
        app["prefill_inflight"][prefiller] -= 1

    try:
        transfer, prompt_ids = unpack_prefill(result)
    except web.HTTPBadGateway:
        pending = result.get("kv_transfer_params")
        if isinstance(pending, dict) and pending.get("do_remote_prefill"):
            await release_prefill(session, app["decode"][0], pending)
        raise
    signature = transfer.pop("eldr_sig", None)
    continues = transfer["prefill_continues"]
    stream_response = None
    decoder = None
    submitted = False
    try:
        if payload.get("stream"):
            stream_response = web.StreamResponse(
                headers={"Content-Type": "text/event-stream"}
            )
            await stream_response.prepare(request)
            # Flush before signature transformation, decoder selection, or KV
            # transfer: none of those can move the client TTFT boundary.
            await send_prefill(stream_response, result, payload, continues)
        dump = app.get("capture_file")
        if dump is not None:
            if not signature:
                raise web.HTTPInternalServerError(
                    reason="Training capture requires ELDR=1"
                )
            raw = base64.b64decode(signature, validate=True)
            dump.write(struct.pack("<I", len(raw)) + raw)
        if not continues:
            if stream_response is None:
                return web.json_response(public_prefill(result, payload, False))
            await stream_response.write_eof()
            return stream_response

        decoder = choose_decode(app, signature, payload)
        app["decode_inflight"][decoder] += 1
        # Reuse the exact original prompt IDs. The sampled token stays separate
        # as output history; it is never retokenized or appended to the prompt.
        payload["prompt"] = prompt_ids
        payload["add_special_tokens"] = False
        payload.pop("truncate_prompt_tokens", None)
        transfer["prefill_created"] = result["created"]
        payload["kv_transfer_params"] = transfer
        submitted = True
        async with session.post(
            f"{decoder}{request.path}", json=payload, headers=headers
        ) as response:
            response.raise_for_status()
            if stream_response is None:
                return web.json_response(await response.json())
            async for chunk in response.content.iter_any():
                await stream_response.write(chunk)
            await stream_response.write_eof()
            return stream_response
    except Exception:
        if stream_response is not None:
            # Never turn a partial completion into an apparently successful one.
            await stream_response.write(
                event({"error": {"message": "Decode continuation failed"}})
            )
            stream_response.force_close()
        raise
    finally:
        if decoder is not None:
            app["decode_inflight"][decoder] -= 1
        if not submitted:
            # EOS, a one-token budget, routing error, or an early disconnect:
            # release P's pinned KV through upstream's pre-aborted D request.
            await release_prefill(session, app["decode"][0], transfer)


async def routing_status(request):
    app = request.app
    policy = app["decode_router_name"]
    updater = app.get("online_jsq_router")
    status = dict(
        rule="locality_band_jsq_v1",
        centroid_mode="online" if updater is not None else "static",
        refit=updater.status() if updater is not None else None,
    )
    data = app["centroid_data"]
    return web.json_response(
        status
        | dict(
            policy=policy,
            selector="jsq",
            signature=data.get("variant", "count") if data is not None else None,
            tau=app["tau"] if data is not None else None,
            inflight_by_worker=[app["decode_inflight"][url] for url in app["decode"]],
        )
    )


def create_app(args):
    worker_groups = {}
    for role, key in (("prefiller", "prefill"), ("decoder", "decode")):
        hosts, ports = getattr(args, role + "_hosts"), getattr(args, role + "_ports")
        if len(hosts) != len(ports) or not hosts:
            raise ValueError(f"{role} hosts/ports mismatch")
        worker_groups[key] = [f"http://{h}:{p}" for h, p in zip(hosts, ports)]
        if len(set(worker_groups[key])) != len(worker_groups[key]):
            raise ValueError("Duplicate worker endpoint")
    app = web.Application(client_max_size=0)
    app.update(worker_groups)
    for key in ("prefill", "decode"):
        app[key + "_inflight"] = dict.fromkeys(app[key], 0)
    app["decode_router_name"], app["tau"], app["centroid_data"] = (
        args.decode_router,
        args.eldr_tau,
        None,
    )
    app["decode_router"] = make_decode_router(
        args.decode_router, app["decode"], args.seed
    )
    if args.prefill_router == "prefix-hash":
        app["prefill_router"] = PrefixHashRouter(
            app["prefill"],
            prefix_byte_count=args.prefill_prefix_bytes,
            load_factor=args.prefill_load_factor,
        )
    else:
        app["prefill_router"] = make_decode_router(
            args.prefill_router, app["prefill"], args.seed
        )
    if args.decode_router.startswith("eldr"):
        if not args.eldr_centroids:
            raise ValueError("ELDR requires --eldr-centroids")
        app["centroid_data"] = load_centroid_file(args.eldr_centroids)
        if len(app["centroid_data"]["centroid_matrix"]) != len(app["decode"]):
            raise ValueError("Require one fitted centroid per decoder")
        if args.decode_router == "eldr":
            app["online_jsq_router"] = OnlineJSQRouter(
                app["decode"],
                app["centroid_data"]["centroid_matrix"],
                args.eldr_tau,
                window_seconds=args.eldr_online_window_seconds,
                refit_seconds=args.eldr_online_refit_seconds,
            )
    elif args.decode_router == "domain":
        if not args.domain_mapping:
            raise ValueError("Domain baseline requires --domain-mapping")
        with open(args.domain_mapping) as stream:
            domain_mapping = json.load(stream)
        groups = [*domain_mapping["dom2dec"].values(), domain_mapping["misc_dec"]]
        if any(
            type(i) is not int or not 0 <= i < len(app["decode"])
            for g in groups
            for i in g
        ):
            raise ValueError("Domain mapping contains an invalid worker index")
        app["domain_mapping"], app["domain_rng"] = (
            domain_mapping,
            random.Random(args.seed),
        )
    if args.decode_router.startswith("eldr"):
        app.router.add_get("/admin/routing", routing_status)

    async def resources(app):
        try:
            with ExitStack() as files:
                app["capture_file"] = (
                    files.enter_context(open(args.capture_output, "xb"))
                    if args.capture_output
                    else None
                )
                connector = aiohttp.TCPConnector(
                    limit=0,
                    limit_per_host=0,
                    keepalive_timeout=2.0,
                    enable_cleanup_closed=True,
                )
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=6 * 3600), connector=connector
                ) as session:
                    app["http"] = session
                    yield
        finally:
            if app.get("online_jsq_router") is not None:
                app["online_jsq_router"].close()

    app.cleanup_ctx.append(resources)
    app.router.add_post("/v1/completions", handle)
    app.router.add_post("/v1/chat/completions", handle)
    app.router.add_get("/admin/decoders", handle_admin_decoders)
    return app


def main():
    args = parse_args()
    app = create_app(args)
    from vllm.eldr.benchmark import configure_gc

    configure_gc(freeze=True)
    try:
        import uvloop

        uvloop.install()
    except ImportError:
        pass
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
