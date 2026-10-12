# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build one ordered Task-domain workload with gradual composition drift."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "task_domain_gradual.json"
SCHEDULE_ID = "three_phase"
DOMAINS = ("code", "math", "medical", "legal")
BASELINE_MIX = {
    "code": 0.24,
    "math": 0.25,
    "medical": 0.19,
    "legal": 0.32,
}
DRIFT_MIX = {
    "code": 0.45,
    "math": 0.35,
    "medical": 0.08,
    "legal": 0.12,
}
MIX_PROFILES = {"moderate": (BASELINE_MIX, DRIFT_MIX)}

# Three equal conceptual phases for the default 120-second experiment:
# baseline [0, 40), drift [40, 80), recovery [80, 120). The drift and
# recovery phases begin with a short ramp occupying 1/12 of total time
# (10 seconds at the default duration).
BASELINE_PHASE_END = 1 / 3
DRIFT_RAMP_END = 5 / 12
DRIFT_PHASE_END = 2 / 3
RETURN_RAMP_END = 3 / 4


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _prompt(record: dict[str, Any]) -> str:
    turns = record.get("conversations")
    if not isinstance(turns, list):
        raise ValueError("record has no ShareGPT conversations list")
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value") or turn.get("content")
        if role in {"human", "user"} and isinstance(value, str) and value:
            return value
    raise ValueError("record has no non-empty human turn")


def _prompt_hash(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()[:16]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Any) -> str:
    payload = (
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()
    with path.open("xb") as stream:
        stream.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _lerp(
    left: dict[str, float],
    right: dict[str, float],
    amount: float,
) -> dict[str, float]:
    return {
        domain: left[domain] + amount * (right[domain] - left[domain])
        for domain in DOMAINS
    }


def mix_at(
    elapsed_s: float,
    duration_s: int,
    baseline_mix: dict[str, float] = BASELINE_MIX,
    drift_mix: dict[str, float] = DRIFT_MIX,
) -> dict[str, float]:
    """Return the target domain mix at one point in the normalized schedule."""
    progress = elapsed_s / duration_s
    if progress < BASELINE_PHASE_END:
        return dict(baseline_mix)
    if progress < DRIFT_RAMP_END:
        amount = (progress - BASELINE_PHASE_END) / (DRIFT_RAMP_END - BASELINE_PHASE_END)
        return _lerp(baseline_mix, drift_mix, amount)
    if progress < DRIFT_PHASE_END:
        return dict(drift_mix)
    if progress < RETURN_RAMP_END:
        amount = (progress - DRIFT_PHASE_END) / (RETURN_RAMP_END - DRIFT_PHASE_END)
        return _lerp(drift_mix, baseline_mix, amount)
    return dict(baseline_mix)


def _allocate_quotas(
    quotas: dict[str, float],
    total: int,
) -> dict[str, int]:
    """Round cumulative quotas while carrying prior rounding error forward."""
    counts = {domain: int(quotas[domain]) for domain in DOMAINS}
    remaining = total - sum(counts.values())
    order = sorted(
        DOMAINS,
        key=lambda domain: (
            -round(quotas[domain] - counts[domain], 12),
            DOMAINS.index(domain),
        ),
    )
    for domain in order[:remaining]:
        counts[domain] += 1
    return counts


def _stable_pool(
    records: list[tuple[str, dict[str, Any]]],
    domain: str,
    seed: int,
) -> list[tuple[str, dict[str, Any]]]:
    prefix = f"{seed}\0{domain}\0".encode()
    return sorted(
        records,
        key=lambda item: (
            hashlib.sha256(prefix + item[0].encode()).digest(),
            item[0],
        ),
    )


def _interleaved_domains(
    counts: dict[str, int],
    second: int,
    seed: int,
) -> list[str]:
    tagged: list[tuple[bytes, str]] = []
    for domain in DOMAINS:
        for ordinal in range(counts[domain]):
            key = f"{seed}\0{second}\0{domain}\0{ordinal}".encode()
            tagged.append((hashlib.sha256(key).digest(), domain))
    tagged.sort()
    return [domain for _, domain in tagged]


def build(
    task_path: Path,
    domain_map_path: Path,
    output_dir: Path,
    rate: int,
    duration_s: int,
    seed: int,
    profile: str = "moderate",
    guard_s: int = 0,
) -> dict[str, Any]:
    if rate < 1 or duration_s < 1 or guard_s < 0:
        raise ValueError(
            "rate and duration must be positive; guard must be non-negative"
        )
    if profile not in MIX_PROFILES:
        raise ValueError(f"unknown mixture profile: {profile}")
    baseline_mix, drift_mix = MIX_PROFILES[profile]
    for path in (task_path, domain_map_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing input: {path}")

    raw_records = json.loads(task_path.read_text(encoding="utf-8"))
    domain_map = json.loads(domain_map_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list) or not isinstance(domain_map, dict):
        raise ValueError("task dataset must be a list and domain map a mapping")

    pools: dict[str, list[tuple[str, dict[str, Any]]]] = {
        domain: [] for domain in DOMAINS
    }
    seen: set[str] = set()
    for index, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise ValueError(f"{task_path}: record {index} is not an object")
        prompt = _prompt(record)
        if prompt in seen:
            raise ValueError(f"{task_path}: duplicate prompt at record {index}")
        seen.add(prompt)
        domain = domain_map.get(_prompt_hash(prompt))
        if domain not in pools:
            raise ValueError(
                f"{task_path}: record {index} has no recognized domain label"
            )
        pools[domain].append((prompt, record))

    for domain in DOMAINS:
        if not pools[domain]:
            raise ValueError("Missing evaluation domain: " + domain)
        pools[domain] = _stable_pool(pools[domain], domain, seed)

    schedule: list[dict[str, Any]] = []
    requested: Counter[str] = Counter()
    target_totals: Counter[str] = Counter()
    traffic_duration_s = duration_s + 2 * guard_s
    for second in range(traffic_duration_s):
        experiment_second = second - guard_s
        target = (
            dict(baseline_mix)
            if experiment_second < 0 or experiment_second >= duration_s
            else mix_at(
                experiment_second + 0.5,
                duration_s,
                baseline_mix,
                drift_mix,
            )
        )
        for domain in DOMAINS:
            target_totals[domain] += target[domain] * rate
        quotas = {
            domain: target_totals[domain] - requested[domain] for domain in DOMAINS
        }
        counts = _allocate_quotas(quotas, rate)
        requested.update(counts)
        schedule.append(
            {
                "second": experiment_second,
                "target_mix": {domain: round(target[domain], 8) for domain in DOMAINS},
                "counts": counts,
            }
        )

    offsets = {domain: 0 for domain in DOMAINS}
    output_records: list[dict[str, Any]] = []
    for item in schedule:
        second = int(item["second"])
        domains = _interleaved_domains(item["counts"], second, seed)
        for domain in domains:
            offset = offsets[domain]
            _, source = pools[domain][offset % len(pools[domain])]
            offsets[domain] += 1
            record = dict(source)
            record["_eldr_domain"] = domain
            record["_eldr_second"] = second
            output_records.append(record)

    expected = rate * traffic_duration_s
    if len(output_records) != expected:
        raise RuntimeError(
            f"internal error: generated {len(output_records)} != {expected}"
        )
    unique_records = sum(
        min(requested[domain], len(pools[domain])) for domain in DOMAINS
    )
    reused_records = expected - unique_records
    reuse_by_domain = {
        domain: max(0, requested[domain] - len(pools[domain])) for domain in DOMAINS
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = output_dir / DATASET_NAME
    dataset_sha = _write_json(dataset_path, output_records)
    manifest = {
        "schema_version": 4,
        "schedule_id": SCHEDULE_ID,
        "seed": seed,
        "ordered": True,
        "rate": rate,
        "duration_s": duration_s,
        "traffic_duration_s": traffic_duration_s,
        "guard_s": guard_s,
        "records": expected,
        "domains": list(DOMAINS),
        "profile": profile,
        "baseline_mix": baseline_mix,
        "drift_mix": drift_mix,
        "segments": [
            {
                "start": 0,
                "end": duration_s * BASELINE_PHASE_END,
                "kind": "baseline",
            },
            {
                "start": duration_s * BASELINE_PHASE_END,
                "end": duration_s * DRIFT_RAMP_END,
                "kind": "transition_to_drift",
            },
            {
                "start": duration_s * DRIFT_RAMP_END,
                "end": duration_s * DRIFT_PHASE_END,
                "kind": "drift",
            },
            {
                "start": duration_s * DRIFT_PHASE_END,
                "end": duration_s * RETURN_RAMP_END,
                "kind": "transition_to_baseline",
            },
            {
                "start": duration_s * RETURN_RAMP_END,
                "end": duration_s,
                "kind": "baseline_return",
            },
        ],
        "domain_counts": {domain: requested[domain] for domain in DOMAINS},
        "available_domain_counts": {domain: len(pools[domain]) for domain in DOMAINS},
        "sampling": {
            "method": "without_replacement_then_deterministic_cycle",
            "unique_records": unique_records,
            "reused_records": reused_records,
            "reuse_by_domain": reuse_by_domain,
        },
        "source": {
            "task_eval": {
                "path": _display_path(task_path),
                "sha256": _sha256(task_path),
            },
            "domain_map": {
                "path": _display_path(domain_map_path),
                "sha256": _sha256(domain_map_path),
            },
        },
        "dataset": {
            "file": DATASET_NAME,
            "sha256": dataset_sha,
        },
        "schedule": schedule,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest
