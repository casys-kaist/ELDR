# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-fleet baselines preserve deterministic placement and load guards."""

import unittest
from unittest.mock import patch

from eldr.serving.baselines import BaselineRouter
from eldr.serving.prefix_hash import PrefixHashRouter
from eldr.serving.proxy import create_app, parse_args


class BaselineTests(unittest.TestCase):
    def test_prefill_default_and_explicit_overrides(self):
        argv = [
            "proxy",
            "--prefiller-hosts",
            "one",
            "--prefiller-ports",
            "20001",
            "--decoder-hosts",
            "two",
            "--decoder-ports",
            "22001",
        ]
        with patch(
            "sys.argv",
            argv
            + ["--capture-output", "counts.bin", "--domain-mapping", "domains.json"],
        ):
            args = parse_args()
            self.assertEqual(args.capture_output, "counts.bin")
            self.assertEqual(args.domain_mapping, "domains.json")
        for extra, policy in (
            ([], "prefix-hash"),
            (["--prefill-router", "jsq"], "jsq"),
            (["--prefill-router", "rr"], "rr"),
        ):
            with self.subTest(policy=policy), patch("sys.argv", argv + extra):
                args = parse_args()
                self.assertEqual(args.prefill_router, policy)
                router = create_app(args)["prefill_router"]
                if policy == "prefix-hash":
                    self.assertIsInstance(router, PrefixHashRouter)
                    self.assertEqual(router.prefix_byte_count, 1024)
                    self.assertEqual(router.load_factor, 1.25)
                else:
                    self.assertEqual(router.policy, policy)

    def test_round_robin_and_seeded_policies(self):
        workers = ["a", "b", "c"]
        rr = BaselineRouter("rr", workers)
        self.assertEqual(
            [rr.select(b"", [0] * 3) for _ in range(7)], [0, 1, 2, 0, 1, 2, 0]
        )
        for policy in ("jsq", "random", "p2c"):
            a, b = (BaselineRouter(policy, workers, seed=58) for _ in range(2))
            self.assertEqual(
                [a.select(b"", [1, 1, 1]) for _ in range(30)],
                [b.select(b"", [1, 1, 1]) for _ in range(30)],
            )
        self.assertEqual(BaselineRouter("jsq", workers).select(b"", [3, 1, 2]), 1)

    def test_prefix_hash_keeps_ring_and_load_gate(self):
        router = PrefixHashRouter(["a", "b", "c"])
        chosen = router.select(b"same prefix", [0, 0, 0])
        self.assertEqual(router.select(b"same prefix", [0, 0, 0]), chosen)
        loads = [0, 0, 0]
        loads[chosen] = 100
        self.assertEqual(
            router.select(b"same prefix", loads),
            next(i for i in range(3) if i != chosen),
        )
        self.assertEqual(router.select(b"", [2, 1, 1]), 1)
        with self.assertRaises(ValueError):
            router.select(b"prompt", [0])
