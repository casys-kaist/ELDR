# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit benchmark GC policy, shared by every compared routing policy."""

import gc
import os


def configure_gc(*, freeze=False):
    # Never change vanilla serving's GC policy just by importing ELDR.
    if os.environ.get("ELDR_BENCHMARK_GC") == "1":
        if freeze:
            gc.collect()
            gc.freeze()
        gc.disable()
