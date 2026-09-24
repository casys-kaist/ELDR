#!/usr/bin/env python3
"""Build disjoint Task fit/evaluation prompts from 12 pinned benchmarks.

Use code, math, medical and legal sources, deduplicate across domains, and
keep prompts of 4–1024 tokens. Write task.fit.json, task.eval.json and label
sidecars under --output. Source failures abort rather than reduce coverage.
Run with: .venv/bin/python -m eldr.experiments.datasets.build_task --help
"""

import hashlib
import random
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from eldr.experiments.datasets.common import arguments, write_json

MAXCH = 6000
CALIB_N = 1000

ARGS = arguments(__doc__)
D, CAPTURE_DIR = ARGS.output, ARGS.output / "captures"
MODEL = ARGS.tokenizer
MAX_PROMPT, MAX_TOTAL, OUTPUT_LEN = 1024, 2048, 512
TOK = AutoTokenizer.from_pretrained(MODEL)
PROMPT_CAP = min(MAX_PROMPT, MAX_TOTAL - OUTPUT_LEN)
rng = random.Random(0)

REVISIONS = {
    "openai/openai_humaneval": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    "bigcode/bigcodebench": "b74c0d0bf70d2c0bc459be537895cca163007f1a",
    "xlangai/DS-1000": "4416080ac5cb80bdf7576aefb8f9a0b4d5426a44",
    "google-research-datasets/mbpp": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "HuggingFaceH4/MATH-500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    "Hothan/OlympiadBench": "91184b52131e7fc9455fef848035173aea8cc01a",
    "deepmind/aqua_rat": "33301c6a050c96af81f63cad5562cb5363e88971",
    "GBaker/MedQA-USMLE-4-options": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
    "qiaojin/PubMedQA": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "hails/mmlu_no_train": "b2e1ec9aa795adafe68e8e983248dbd4b52a1c60",
    "coastalcph/lex_glue": "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c",
}


def load_pinned(dataset_id, **kwargs):
    return load_dataset(dataset_id, revision=REVISIONS[dataset_id], **kwargs)


def servable(p):
    n = len(TOK(p, add_special_tokens=False)["input_ids"])
    return 4 <= n <= PROMPT_CAP


# FAM: each value is a list of (sub_source_name, generator_callable) tuples.
# Sub-source name is the human-readable label used in the purity sidecar.
FAM = {
    "code": [
        (
            "HumanEval",
            lambda: (
                r["prompt"]
                for r in load_pinned("openai/openai_humaneval", split="test")
            ),
        ),
        (
            "BigCodeBench",
            lambda: (
                r["instruct_prompt"]
                for r in load_pinned("bigcode/bigcodebench", split="v0.1.4")
            ),
        ),
        (
            "DS-1000",
            lambda: (r["prompt"] for r in load_pinned("xlangai/DS-1000", split="test")),
        ),
        (
            "MBPP-full",
            lambda: (
                r["text"]
                for r in load_pinned(
                    "google-research-datasets/mbpp", name="full", split="test"
                )
            ),
        ),
    ],
    "math": [
        (
            "GSM8K",
            lambda: (
                r["question"]
                for r in load_pinned("openai/gsm8k", name="main", split="test")
            ),
        ),
        (
            "MATH-500",
            lambda: (
                r["problem"]
                for r in load_pinned("HuggingFaceH4/MATH-500", split="test")
            ),
        ),
        (
            "OlympiadBench",
            lambda: (
                r["question"]
                for r in load_pinned(
                    "Hothan/OlympiadBench", name="OE_TO_maths_en_COMP", split="train"
                )
            ),
        ),
        (
            "AQuA-RAT",
            lambda: (
                r["question"]
                for r in load_pinned("deepmind/aqua_rat", split="test")
                if r.get("question")
            ),
        ),
    ],
    "medical": [
        (
            "MedQA-USMLE",
            lambda: (
                r["question"]
                for r in load_pinned("GBaker/MedQA-USMLE-4-options", split="test")
            ),
        ),
        (
            "PubMedQA-L",
            lambda: (
                (
                    (r.get("context") or {}).get("contexts", [""])[0]
                    + "\nQuestion: "
                    + r["question"]
                )
                for r in load_pinned(
                    "qiaojin/PubMedQA", name="pqa_labeled", split="train"
                )
            ),
        ),
        (
            "MMLU-prof-med",
            lambda: (
                r["question"]
                for r in load_pinned(
                    "hails/mmlu_no_train", name="professional_medicine", split="test"
                )
            ),
        ),
    ],
    "legal": [
        (
            "LexGLUE-CaseHold",
            lambda: (
                r["context"]
                for r in load_pinned(
                    "coastalcph/lex_glue", name="case_hold", split="test"
                )
            ),
        ),
    ],
}

# Build doms[domain] = [(subsource_name, [prompts...]), ...]  with cross-domain dedup
# Order: domain order from FAM, sub-source order from FAM[domain], prompts sorted within
seen_global = set()
doms = {}
source_errors = []
for fam, sub_sources in FAM.items():
    sub_lists = []
    for src_name, gen in sub_sources:
        pool = set()
        try:
            for p in gen():
                if isinstance(p, str):
                    p = p.strip()[:MAXCH]
                    if len(p) >= 16 and servable(p) and p not in seen_global:
                        seen_global.add(p)
                        pool.add(p)
        except Exception as e:
            source_errors.append((fam, src_name, str(e)))
            print(f"  [{fam}/{src_name}] FAILED: {str(e)[:80]}", flush=True)
        sub_lists.append((src_name, sorted(pool)))
        print(f"  [{fam}/{src_name}] {len(pool)}", flush=True)
    doms[fam] = sub_lists
    total = sum(len(p) for _, p in sub_lists)
    print(f"[{fam}] total {total}", flush=True)

if source_errors:
    details = "; ".join(f"{fam}/{src}: {error}" for fam, src, error in source_errors)
    raise RuntimeError(
        "Refusing to write an incomplete task dataset; source failures: " + details
    )

TASK_ORDER = ["code", "math", "medical", "legal"]

# Domain-grouped full list used by the counts and hash sidecars. Each domain's
# prompts appear in sub-source order (HumanEval first, BigCodeBench next, etc.).
fullitems = []
domain_counts = []
subsource_counts = []  # [[domain, subsource, count], ...]
for fam in TASK_ORDER:
    fam_total = 0
    for src_name, prompts in doms[fam]:
        for p in prompts:
            fullitems.append([p, fam])
        subsource_counts.append([fam, src_name, len(prompts)])
        fam_total += len(prompts)
    domain_counts.append([fam, fam_total])

write_json(domain_counts, D / "task.counts.json")
write_json(subsource_counts, D / "task.subsources.json")
write_json(
    {hashlib.md5(p.encode()).hexdigest()[:16]: fam for p, fam in fullitems},
    CAPTURE_DIR / "task_unbal_full_hash2dom.json",
)

# Calib split: random 1000 from the full shuffled allitems
allitems = list(fullitems)
rng.shuffle(allitems)
calib, serve = allitems[:CALIB_N], allitems[CALIB_N:]
write_json(calib, D / "task.fit.json")
write_json(
    {hashlib.md5(p.encode()).hexdigest()[:16]: fam for p, fam in calib},
    CAPTURE_DIR / "calibserve_hash2dom.json",
)
write_json(
    [
        {
            "conversations": [
                {"from": "human", "value": p},
                {"from": "gpt", "value": "ok"},
            ]
        }
        for p, _ in serve
    ],
    D / "task.eval.json",
)

print()
print(f"full={len(fullitems)} calib={len(calib)} serve={len(serve)}")
print(f"per-domain sizes: {dict(domain_counts)}")
print(f"calib_mix: {dict(Counter(d for _, d in calib))}")
print(
    f"subsource sidecar -> task.subsources.json  ({len(subsource_counts)} subsources)"
)
