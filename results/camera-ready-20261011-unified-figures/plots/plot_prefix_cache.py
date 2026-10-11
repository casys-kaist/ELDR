"""Prefix cache on GPT-OSS Task: 1P16D, 100 req/s, online centroids.

Four validated October measurement cells in data/prefix_cache_100reqs.csv.
"""

import argparse
import csv
import math
from pathlib import Path

import figstyle as plot_style

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
POLICIES = (("rr", "RR", plot_style.RR_COLOR), ("eldr", "ELDR", plot_style.ELDR_COLOR))
METRICS = (
    ("ttft50_ms", "P50 TTFT", 240, (0, 60, 120, 180, 240), "%.0f"),
    ("tpot50_ms", "P50 TPOT", 68, (0, 20, 40, 60), "%.0f"),
    ("tpot99_ms", "P99 TPOT", 68, (0, 20, 40, 60), "%.0f"),
)


def load(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cells = {(r["setting"], r["policy"]): r for r in rows}
    expected = {(f"gptoss-cache-{c}", p) for c in (0, 1) for p, _, _ in POLICIES}
    if len(rows) != 4 or set(cells) != expected:
        raise ValueError("Exactly one RR/online-ELDR measurement per cache state required")
    for row in rows:
        if int(row["rate"]) != 100 or int(row["requests"]) != 12000:
            raise ValueError("Expected 12,000 requests at 100 req/s in every cell")
        for key, _, limit, _, _ in METRICS:
            value = float(row[key])
            if not math.isfinite(value) or not 0 < value < limit:
                raise ValueError(f"Invalid or clipped metric: {key}={value}")
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=plot_style.DATA / "prefix_cache_100reqs.csv")
    parser.add_argument(
        "--output", type=Path, default=plot_style.OUTPUT / "06_evaluation/fig_prefix_cache_gptoss_task"
    )
    args = parser.parse_args()
    cells = load(args.data)
    plot_style.apply()
    plt.rcParams.update({
        "axes.labelsize": 15, "axes.titlesize": 15, "axes.titlepad": 4,
        "legend.fontsize": 15, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(plot_style.WIDTH, 2.8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.78, wspace=0.22)
    x = np.array([0.0, 2.4])
    bar_width = 0.85
    for ax, (key, title, limit, ticks, fmt) in zip(axes, METRICS):
        for i, (policy, label, color) in enumerate(POLICIES):
            values = [float(cells[f"gptoss-cache-{c}", policy][key]) for c in (0, 1)]
            bars = ax.bar(
                x + (2 * i - 1) * (bar_width / 2 + 0.04), values, width=bar_width,
                color=color, edgecolor="black", linewidth=0.5, label=label, zorder=2,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    fmt % value, (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=12, clip_on=False,
                )
        ax.set(title=title, xlabel="Prefix cache", ylim=(0, limit), yticks=ticks)
        ax.set_xlim(-1.05, x[-1] + 1.05)
        ax.set_xticks(x, ("Off", "On"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="lower center", frameon=True,
               bbox_to_anchor=(0.5, 0.87), handletextpad=0.55,
               columnspacing=1.7, handlelength=2.0, handleheight=1.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_style.save(fig, args.output)
    plt.close(fig)


if __name__ == "__main__":
    main()
