# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pybase64 as base64

from eldr.serving.cpu_threads import NATIVE_THREAD_VARIABLES, configure_proxy_threads
from eldr.serving.proxy import choose_decode

ROOT = Path(__file__).resolve().parents[2]


class TestProxyCpuThreads(unittest.TestCase):
    def test_default_and_explicit_override(self):
        env = {"OPENBLAS_NUM_THREADS": "64", "UNRELATED": "preserved"}
        self.assertEqual(configure_proxy_threads(env), 1)
        self.assertTrue(all(env[name] == "1" for name in NATIVE_THREAD_VARIABLES))
        self.assertEqual(env["UNRELATED"], "preserved")
        env["ELDR_PROXY_NUM_THREADS"] = "2"
        self.assertEqual(configure_proxy_threads(env), 2)
        self.assertTrue(all(env[name] == "2" for name in NATIVE_THREAD_VARIABLES))

    def test_invalid_override_does_not_modify_environment(self):
        for value in ("0", "-1", "1.5", "", "invalid"):
            env = {"ELDR_PROXY_NUM_THREADS": value, "OPENBLAS_NUM_THREADS": "64"}
            before = env.copy()
            with self.assertRaisesRegex(ValueError, "positive integer"):
                configure_proxy_threads(env)
            self.assertEqual(env, before)

    def test_standalone_bootstrap_precedes_numpy_import(self):
        for entrypoint in ("module", "script"):
            code = textwrap.dedent(f"""
                import builtins, os, runpy, sys
                original = builtins.__import__
                checked = []
                def intercept(name, *args, **kwargs):
                    if name == 'numpy' and not checked:
                        for key in {NATIVE_THREAD_VARIABLES!r}:
                            assert os.environ[key] == '2', (key, os.environ.get(key))
                        checked.append(True)
                    return original(name, *args, **kwargs)
                builtins.__import__ = intercept
                sys.argv = ['proxy', '--help']
                try:
                    if {entrypoint!r} == 'module':
                        runpy.run_module('eldr.serving.proxy', run_name='__main__')
                    else:
                        runpy.run_path('eldr/serving/proxy.py', run_name='__main__')
                except SystemExit as error:
                    assert error.code == 0
                assert checked
            """)
            env = {
                **os.environ,
                "OPENBLAS_NUM_THREADS": "64",
                "ELDR_PROXY_NUM_THREADS": "2",
            }
            result = subprocess.run(
                [sys.executable, "-B", "-c", code],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_library_import_does_not_change_thread_settings(self):
        code = textwrap.dedent("""
            import os
            before = os.environ.copy()
            import eldr.serving.proxy
            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                        'MKL_NUM_THREADS', 'BLIS_NUM_THREADS'):
                assert os.environ.get(key) == before.get(key), key
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_proxy_uses_owned_transformed_route(self):
        router = Mock()
        router.route_transformed.return_value = 1
        app = {
            "decode_router_name": "eldr",
            "decode": ["d0", "d1"],
            "decode_inflight": {"d0": 2, "d1": 0},
            "online_jsq_router": router,
            "centroid_data": {"L": 1, "e": 2, "_mask": np.asarray([0])},
        }
        encoded = base64.b64encode(np.asarray([[3, 4]], dtype="<i2").tobytes())
        self.assertEqual(choose_decode(app, encoded, {}), "d1")
        signature, loads = router.route_transformed.call_args.args
        np.testing.assert_allclose(signature, [0.6, 0.8])
        self.assertEqual(signature.dtype, np.float32)
        self.assertEqual(loads, [2, 0])
        router.route.assert_not_called()
