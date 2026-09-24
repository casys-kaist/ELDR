# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four fixed-worker load baselines; seeded ties preserve measured policies."""

import random


class BaselineRouter:
    def __init__(self, policy, workers, seed=0):
        if policy not in ("rr", "jsq", "random", "p2c") or not workers:
            raise ValueError("Known baseline and nonempty worker group required")
        self.policy = policy
        self.workers = tuple(range(len(workers)))
        self.rng = random.Random(seed)
        self.next = 0

    def select(self, key, loads):
        if len(loads) != len(self.workers):
            raise ValueError("Worker count and load count differ")
        if self.policy == "rr":
            chosen = self.next
            self.next = (chosen + 1) % len(self.workers)
            return chosen
        if self.policy == "random":
            return self.rng.choice(self.workers)
        if self.policy == "p2c":
            if len(self.workers) == 1:
                return 0
            a, b = self.rng.sample(self.workers, 2)
            return a if loads[a] <= loads[b] else b
        minimum = min(loads)
        tied = [i for i in self.workers if loads[i] == minimum]
        return tied[0] if len(tied) == 1 else self.rng.choice(tied)
