"""Validated October main curves, retaining the approved paired axes/ticks."""
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply, save, WIDTH, DATA, OUTPUT, RR_COLOR, ELDR_COLOR
from matplotlib.ticker import MaxNLocator

OUT = OUTPUT / "06_evaluation"
RATES = [20, 40, 60, 80, 100]
MODELS = ["qwen", "gptoss", "gemma"]
PRETTY = {"qwen": "Qwen3-30B-A3B",
          "gptoss": "GPT-OSS-120B",
          "gemma": "Gemma-4-26B-A4B"}

ROUTERS = [
    ("random", "Random",       "#7f7f7f", "-", "v"),
    ("rr",     "RR",           RR_COLOR,  "-", "s"),
    ("jsq",    "JSQ",          "#2ca02c", "-", "^"),
    ("p2c",    "P2C",          "#ff7f0e", "-", "D"),
    ("domain", "Domain",       "#9467bd", "-", "P"),
    ("eldr",   "ELDR",         ELDR_COLOR, "-", "o"),
]

METRICS = [
    ("median_tpot_ms", "P50 TPOT (ms)"),
    ("p99_tpot_ms",    "P99 TPOT (ms)"),
    ("median_ttft_ms", "P50 TTFT (ms)"),
]

def plot_dataset(dataset, out_basename):
    apply()
    with (DATA / f"main_{dataset}.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    table = {(r['setting'], r['variant'], int(r['rate'])): r for r in rows}
    assert len(rows) == len(table) == 90
    assert set(table) == {(m, p, r) for m in MODELS for p, *_ in ROUTERS for r in RATES}
    keys = dict(median_tpot_ms='tpot50_ms', p99_tpot_ms='tpot99_ms', median_ttft_ms='ttft50_ms')
    data = {m: {p: {k: [float(table[m, p, r][v]) for r in RATES]
                     for k, v in keys.items()} for p, *_ in ROUTERS} for m in MODELS}

    nrows, ncols = len(METRICS), len(MODELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH * 0.85, 5.6),
                             sharex=True, squeeze=False)
    xtick_pos = [20, 60, 100]

    for ci, model in enumerate(MODELS):
        for ri, (m, ylabel) in enumerate(METRICS):
            ax = axes[ri][ci]
            any_data = False
            for rkey, rlabel, color, ls, mk in ROUTERS:
                ys = data[model][rkey][m]
                if any(not np.isnan(y) for y in ys):
                    any_data = True
                ax.plot(RATES, ys, color=color, ls=ls, marker=mk,
                        markersize=5.0, lw=1.5, label=rlabel,
                        markeredgewidth=0.5)
            if any_data:
                all_ys = [y for rkey, *_ in ROUTERS for y in data[model][rkey][m]
                          if not np.isnan(y)]
                if all_ys:
                    mn, mx = min(all_ys), max(all_ys)
                    spread = max(mx - mn, 1e-6)
                    ax.set_ylim(mn - 0.05 * spread, mx + 0.08 * spread)
            ax.grid(True, alpha=0.30, lw=0.5)
            if ri == 0:
                ax.set_title(PRETTY[model], fontsize=13, pad=6)
            if ri == nrows - 1:
                ax.set_xlabel("Request rate (req/s)", fontsize=12)
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=12)
                ax.yaxis.set_label_coords(-0.25, 0.5)
            ax.tick_params(axis="both", labelsize=10)
            ax.set_xticks(xtick_pos)
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)

    handles, labels = [], []
    for rkey, rlabel, color, ls, mk in ROUTERS:
        line, = axes[0][0].plot([], [], color=color, ls=ls, marker=mk,
                                markersize=6.0, lw=1.6, label=rlabel,
                                markeredgewidth=0.5)
        handles.append(line); labels.append(rlabel)

    LEFT, RIGHT, TOP, BOTTOM = 0.075, 0.995, 0.84, 0.09
    fig.subplots_adjust(top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT,
                        wspace=0.22, hspace=0.20)
    fig.legend(handles, labels, ncol=len(ROUTERS), loc="lower center",
               bbox_to_anchor=(LEFT, 0.90, RIGHT - LEFT, 0.05),
               mode="expand", frameon=True, fontsize=12,
               handletextpad=0.5, columnspacing=1.0, borderaxespad=0.2)
    for p50, p99 in zip(axes[0], axes[1]):
        limits = (min(p50.get_ylim()[0], p99.get_ylim()[0]),
                  max(p50.get_ylim()[1], p99.get_ylim()[1]))
        p99.sharey(p50)
        p50.set_ylim(*limits)
    for ax in axes[2]:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, integer=True))
    fig.canvas.draw()
    for p50, p99, ttft in zip(*axes):
        assert p50.get_ylim() == p99.get_ylim()
        assert np.array_equal(p50.get_yticks(), p99.get_yticks())
        lo, hi = ttft.get_ylim()
        ticks = [v for v in ttft.get_yticks() if lo <= v <= hi]
        assert len(ticks) >= 3 and all(float(t).is_integer() for t in ticks)
    for ax in fig.axes:
        lo, hi = ax.get_ylim()
        for line in ax.lines:
            values = line.get_ydata()
            assert not len(values) or lo <= min(values) <= max(values) <= hi
    save(fig, f"{OUT}/{out_basename}")
    plt.close(fig)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("lang", "both"):
        print("=== lang ===")
        plot_dataset("lang", "fig_main_lang")
    if which in ("task", "both"):
        print("=== task ===")
        plot_dataset("task", "fig_main_task")


if __name__ == "__main__":
    main()
