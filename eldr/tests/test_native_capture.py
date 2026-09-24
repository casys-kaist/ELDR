# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests without importing the GPU serving stack."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"vllm/eldr/{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class NativeCaptureTests(unittest.TestCase):
    def test_gc_only_changes_explicit_benchmark_processes(self):
        benchmark = module("benchmark")
        with (
            patch.dict(benchmark.os.environ, {}, clear=True),
            patch.object(benchmark.gc, "disable") as disable,
            patch.object(benchmark.gc, "collect") as collect,
            patch.object(benchmark.gc, "freeze") as freeze,
        ):
            benchmark.configure_gc(freeze=True)
            disable.assert_not_called()
            collect.assert_not_called()
            freeze.assert_not_called()
            benchmark.os.environ["ELDR_BENCHMARK_GC"] = "1"
            benchmark.configure_gc(freeze=True)
            disable.assert_called_once()
            collect.assert_called_once()
            freeze.assert_called_once()

    def test_actual_capture_exactly_once_and_exception_cleanup(self):
        capture = module("capture")
        records = []
        ids = object()
        with capture.observe(lambda *x: records.append(x)):
            capture.record(ids)
        self.assertEqual(records, [(ids,)])
        capture.record(ids)  # No observer: no work.
        for case in ("missing", "twice", "nested", "exception"):
            with (
                self.subTest(case=case),
                self.assertRaises(RuntimeError),
                capture.observe(lambda *x: None),
            ):
                if case == "twice":
                    capture.record(ids)
                    capture.record(ids)
                elif case == "nested":
                    with capture.observe(None):
                        pass
                elif case == "exception":
                    raise RuntimeError("test")
            self.assertIsNone(capture._active.get())

    def test_switch_preserves_cache_and_checks_external_mutation(self):
        control = module("control")

        class Router:
            def __init__(self):
                self.capture_fn = lambda *x: None

            def set_capture_fn(self, callback):
                self.capture_fn = callback

        routers = [Router(), Router()]
        cache = object()
        signature = NS(ENABLED=True, _L=2, _ESC=cache, _REFS=[], _NT=2)
        transport, env = NS(ENABLED=True), {"ELDR": "1"}
        switch = control.CaptureSwitch(signature, transport, routers, env)
        self.assertFalse(switch.set_enabled(False)["enabled"])
        self.assertEqual(switch.set_enabled(False)["transitions"], 1)
        self.assertIs(signature._ESC, cache)
        self.assertEqual(signature._NT, 0)
        self.assertTrue(switch.set_enabled(True)["enabled"])
        routers[0].capture_fn = None
        with self.assertRaises(ValueError):
            switch.set_enabled(False)

    def test_switch_rollback(self):
        control = module("control")

        class Router:
            capture_fn = None

            def set_capture_fn(self, callback):
                self.capture_fn = callback
                raise RuntimeError("setter failed")

        router = Router()
        original = router.capture_fn = object()
        sig = NS(ENABLED=True, _L=1, _ESC=None, _REFS=[original], _NT=2)
        switch = control.CaptureSwitch(sig, NS(ENABLED=True), [router], {"ELDR": "1"})
        with self.assertRaises(RuntimeError):
            switch.set_enabled(False)
        self.assertIs(router.capture_fn, original)
        self.assertTrue(sig.ENABLED)
        self.assertEqual(switch.transitions, 0)

    def test_empty_core_rejects_retained_kv(self):
        control = module("control")
        pool = NS(num_gpu_blocks=2, blocks=[None, None], get_num_free_blocks=lambda: 1)
        scheduler = NS(
            running=[],
            waiting=[],
            skipped_waiting=[],
            requests={},
            delayed_free_req_ids=set(),
            kv_cache_manager=NS(block_pool=pool),
        )
        core = NS(
            async_scheduling=False,
            batch_queue=None,
            is_scheduler_paused=lambda: False,
            scheduler=scheduler,
        )
        self.assertIs(control.require_empty_core(core), pool)
        scheduler.delayed_free_req_ids.add("retained")
        with self.assertRaises(ValueError):
            control.require_empty_core(core)

    def test_scratch_changes_only_bitmatrix_capacity(self):
        scratch = module("scratch")
        dtype = object()
        calls = []
        allocate = scratch.make_allocator(lambda *x: calls.append(x), dtype)
        allocate(0, (3, 65), dtype, "gpu", False)
        allocate(0, (3, 65), dtype, "gpu", True)
        self.assertEqual(calls[0][1], (3, 128))
        self.assertEqual(calls[1][1], (3, 65))


if __name__ == "__main__":
    unittest.main()
