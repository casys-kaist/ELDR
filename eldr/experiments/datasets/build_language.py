#!/usr/bin/env python3
"""Build disjoint Language fit/evaluation prompts from pinned WildChat-1M.

Extract the first user turn, deduplicate, and keep prompts of 4–1024 tokens.
Write lang.fit.json, lang.eval.json and label sidecars under --output.
Run with: .venv/bin/python -m eldr.experiments.datasets.build_lang --help
"""

import hashlib
import os
import random
from collections import Counter

from datasets import load_dataset
from transformers import AutoTokenizer

from eldr.experiments.datasets.common import arguments, write_json

ARGS = arguments(__doc__)
D, CAPTURE_DIR = ARGS.output, ARGS.output / "captures"
MODEL = ARGS.tokenizer
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
MAXCH = 6000
CALIB_N = 1000
SERVE_N = 13000  # cap serve (avoid huge eval.json)
MAX_PROMPT = 1024
TOK = AutoTokenizer.from_pretrained(MODEL)
rng = random.Random(0)


def first_user(conv):
    """First user-role turn from a WildChat conversation."""
    if not isinstance(conv, list):
        return None
    for t in conv:
        r = (t.get("role") or t.get("from") or "").lower()
        c = t.get("content") or t.get("value") or ""
        if r in ("user", "human") and isinstance(c, str) and len(c.strip()) >= 16:
            return c.strip()
    return None


print("loading WildChat-1M (streaming)...", flush=True)
ds = load_dataset(
    "allenai/WildChat-1M",
    split="train",
    streaming=True,
    revision=WILDCHAT_REVISION,
    token=os.environ.get("HF_TOKEN"),
)
pool = []  # [(prompt, lang), ...]
seen = set()
TARGET = CALIB_N + SERVE_N + 5000  # over-pull to allow dedup loss
for i, r in enumerate(ds):
    if len(pool) >= TARGET:
        break
    if i % 5000 == 0:
        print(f"  scanned {i}  kept {len(pool)}", flush=True)
    p = first_user(r.get("conversation") or r.get("conversations") or [])
    if not p:
        continue
    p = p[:MAXCH]
    if p in seen:
        continue
    n = len(TOK(p, add_special_tokens=False)["input_ids"])
    if not (4 <= n <= MAX_PROMPT):
        continue
    seen.add(p)
    pool.append([p, (r.get("language") or "unknown")])

print(
    f"\nservable={len(pool)}  lang mix (top 8): "
    f"{dict(Counter(lang for _, lang in pool).most_common(8))}",
    flush=True,
)

# Build the language-grouped order used by the counts and hash sidecars.
# Languages are ordered by frequency in the pool.
lang_counts = Counter(lang for _, lang in pool)
LANG_ORDER = [lang for lang, _ in lang_counts.most_common()]
by_lang = {lang: [] for lang in LANG_ORDER}
for p, lang in pool:
    by_lang[lang].append([p, lang])
fullitems = []
for lang in LANG_ORDER:
    fullitems += by_lang[lang]
# Sidecar -- positional partition matching the file's lang order.
write_json([[lang, lang_counts[lang]] for lang in LANG_ORDER], D / "lang.counts.json")
# Hash -> lang map for the FULL set (figures + general lookup).
write_json(
    {hashlib.md5(p.encode()).hexdigest()[:16]: lang for p, lang in fullitems},
    CAPTURE_DIR / "wildchat_full_hash2lang.json",
)

rng.shuffle(pool)
calib = pool[:CALIB_N]
serve = pool[CALIB_N : CALIB_N + SERVE_N]
print(
    f"calib={len(calib)} serve={len(serve)}  "
    f"calib lang mix: {dict(Counter(lang for _, lang in calib).most_common(6))}",
    flush=True,
)
write_json(calib, D / "lang.fit.json")
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
    D / "lang.eval.json",
)
# Hash -> lang map for the training subset only.
write_json(
    {hashlib.md5(p.encode()).hexdigest()[:16]: lang for p, lang in calib},
    CAPTURE_DIR / "wildchat_hash2lang.json",
)
print("wrote lang.{fit,eval,counts}.json + hash2lang")
print(
    f"full domain-grouped: {len(fullitems)} prompts, top-6 lang sidecar="
    f"{[[lang, lang_counts[lang]] for lang in LANG_ORDER[:6]]}"
)
