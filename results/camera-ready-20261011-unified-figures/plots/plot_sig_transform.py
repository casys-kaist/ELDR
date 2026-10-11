#!/usr/bin/env python3
"""sig_transform figure (new 6-variant version): ρ for plain count and 5
alternative constructions of f, averaged over 3 models × 2 workloads on the
full layer mask. Validates the count choice for §sig_validation.

Reads rho_transform.json (produced by rho_transform.py).

  python3 plots/plot_sig_transform.py
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import figstyle as fs
fs.apply()

IN = fs.DATA / "rho_transform.json"
OUT = fs.OUTPUT / "04_design/sig_transform"

# Comparison transforms first; ELDR's choice is red and rightmost.
VARIANTS = [
    ("count",          "count"),
    ("sqrt",           "$\\sqrt{\\,}$count"),
    ("gate_prob_all",  "gate prob"),
    ("gate_logit_all", "gate logit"),
    ("binary",         "binary"),
    ("count_idf",      "count$\\cdot$idf"),
]


def main():
    if not os.path.exists(IN):
        print(f"missing {IN}; run rho_transform.py first")
        sys.exit(1)
    rows = json.load(open(IN))

    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r["rho"])

    names = [label for _, label in VARIANTS]
    data = [[x for x in by_var.get(name, [])
             if not (isinstance(x, float) and np.isnan(x))]
            for name, _ in VARIANTS]

    # Preserve every transform's values while following the display order above.
    means = [np.mean(d) if d else float("nan") for d in data]
    # Compute mean/min/max for the bar chart
    mean = np.array(means)
    lo = np.array([min(d) if d else float("nan") for d in data])
    hi = np.array([max(d) if d else float("nan") for d in data])

    fig, ax = plt.subplots(figsize=(fs.WIDTH, 2.8))
    x = np.arange(len(names))
    cols = [fs.ELDR_COLOR if names[i] == "count$\\cdot$idf" else fs.RR_COLOR
            for i in range(len(names))]
    ax.bar(x, mean, color=cols, edgecolor="k", lw=0.8, width=0.66, zorder=3)
    ax.errorbar(x, mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor="#111", capsize=5, capthick=1.7, lw=1.7, zorder=4)
    for i, mv in enumerate(mean):
        if not np.isnan(mv):
            ax.text(i, hi[i] + 0.014, f"{mv:.2f}", ha="center", fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=14, rotation=10, ha="right")
    ax.set_ylabel(r"Signature quality $\rho$", fontsize=16)
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.tick_params(axis="y", labelsize=13)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.25)
    fs.save(fig, OUT)


if __name__ == "__main__":
    main()
