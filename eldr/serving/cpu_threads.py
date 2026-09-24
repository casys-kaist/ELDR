# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native thread limits for the standalone CPU routing proxy only."""

import os
from collections.abc import MutableMapping

NATIVE_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_proxy_threads(environ: MutableMapping[str, str] | None = None) -> int:
    """Call before numerical imports, not around individual background fits.

    Per-fit threadpool changes would affect the routing thread too. Instead,
    standalone proxy processes use one native thread throughout their lifetime.
    ELDR_PROXY_NUM_THREADS explicitly overrides this default. GPU workers and
    offline fitting processes do not call this function.
    """
    environ = os.environ if environ is None else environ
    try:
        threads = int(environ.get("ELDR_PROXY_NUM_THREADS", "1"))
    except ValueError as error:
        raise ValueError("ELDR_PROXY_NUM_THREADS must be a positive integer") from error
    if threads <= 0:
        raise ValueError("ELDR_PROXY_NUM_THREADS must be a positive integer")
    for name in NATIVE_THREAD_VARIABLES:
        environ[name] = str(threads)
    return threads
