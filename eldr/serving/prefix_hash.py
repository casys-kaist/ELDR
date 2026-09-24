"""Stateless prefix-hash router for prefill workers.

Algorithm:
  1. Hash the first N bytes of the request's prompt key (deterministic).
  2. Consistent-hash-ring lookup over worker URLs picks an "initial" worker.
  3. Keep that worker if its load <= avg_load * load_factor.
  4. Otherwise select the least-loaded worker under the threshold.
  5. If all exceed the threshold, retain the initial worker.

Properties:
  - Stateless: same prompt bytes -> same worker, every run, every variant.
  - O(prefix_bytes) hash + O(log n) ring lookup. No tree, no LRU.
  - Two knobs only: prefix_byte_count, load_factor.

Reference: SGLang's prefix-hash policy (sgl-model-gateway/src/policies/prefix_hash.rs).
"""

import bisect
import hashlib
from collections.abc import Sequence


def _hash64(data: bytes) -> int:
    """Stable 64-bit hash using stdlib blake2b. Deterministic across processes
    (unlike built-in `hash()`, which is salted per interpreter)."""
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


class PrefixHashRouter:
    """Consistent-hash router with a 1-knob load gate.

    Parameters
    ----------
    worker_urls : list[str]
        URLs of the prefill workers, in stable order. Used as the ring identity.
    prefix_byte_count : int
        Number of leading bytes of the prompt key to hash. ~1024 ≈ 256 tokens.
    load_factor : float
        Imbalance gate. A worker is "overloaded" if its load > avg_load * load_factor.
    vnodes_per_worker : int
        Virtual nodes per worker on the hash ring. Higher = smoother distribution.

    Notes
    -----
    The router is intentionally stateless. The caller passes current per-worker
    load counts on every `select()`; the router holds no per-request state.
    """

    def __init__(
        self,
        worker_urls: Sequence[str],
        prefix_byte_count: int = 1024,
        load_factor: float = 1.25,
        vnodes_per_worker: int = 160,
    ):
        if not worker_urls:
            raise ValueError("worker_urls must be non-empty")
        if prefix_byte_count <= 0:
            raise ValueError("prefix_byte_count must be positive")
        if load_factor < 1.0:
            raise ValueError("load_factor must be >= 1.0")

        self.worker_urls = list(worker_urls)
        self.prefix_byte_count = int(prefix_byte_count)
        self.load_factor = float(load_factor)

        # Build the consistent hash ring: (position, worker_idx), sorted by position.
        entries = []
        for w_idx, url in enumerate(self.worker_urls):
            for v in range(vnodes_per_worker):
                entries.append((_hash64(f"{url}#{v}".encode()), w_idx))
        entries.sort()
        self._ring_pos = [p for p, _ in entries]
        self._ring_idx = [i for _, i in entries]

    # ----- core API -----

    def select(
        self,
        key: bytes,
        loads: Sequence[int],
    ) -> int:
        """Return the index (into `worker_urls`) of the chosen prefill worker.

        Parameters
        ----------
        key : bytes
            Prompt key to hash. Caller chooses encoding (raw prompt text bytes,
            or serialized token IDs). The first `prefix_byte_count` bytes are used.
        loads : Sequence[int]
            Per-worker pending-request counts, indexed parallel to `worker_urls`.
        """
        if len(loads) != len(self.worker_urls):
            raise ValueError("loads length must equal number of workers")
        workers = range(len(self.worker_urls))
        if not key:
            return min(workers, key=lambda i: (loads[i], i))

        # 1) Hash prefix, 2) ring lookup.
        prefix = bytes(key[: self.prefix_byte_count])
        h = _hash64(prefix)
        initial = self._ring_lookup(h)

        # 3) Load gate. Threshold simulates "one more incoming" via the +1.
        total = sum(loads) + 1
        avg = total / len(workers)
        threshold = avg * self.load_factor

        if loads[initial] <= threshold:
            return initial

        # 4) Initial overloaded -> least-loaded among workers still under threshold.
        candidates = [i for i in workers if loads[i] <= threshold]
        if candidates:
            return min(candidates, key=lambda i: (loads[i], i))

        # 5) All overloaded -> stick with the initial.
        return initial

    # ----- helpers -----

    def _ring_lookup(self, h: int) -> int:
        """First ring entry with position >= h; wrap around if at end."""
        i = bisect.bisect_left(self._ring_pos, h)
        if i >= len(self._ring_pos):
            i = 0
        return self._ring_idx[i]
