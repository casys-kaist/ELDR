"""Clustering ablation: equal-weight means over 20/40/60/80/100 req/s.

Same format as plot_sig_transform_ablation.py:
  Rows = dataset (Task, Lang). Cols = model (Qwen, GPT-OSS, Gemma).
  Each panel: Δ vs RR (%) for TPOT P50 and TPOT P99.
  Two bars per metric: vanilla (blue, left) vs balanced ELDR (red, right).

Source: 90 validated October cells; matched RR controls at every rate.
"""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from figstyle import apply, save, WIDTH, DATA, OUTPUT, RR_COLOR, ELDR_COLOR

MODELS = ["qwen", "gptoss", "gemma"]
DATASETS = ["task", "lang"]
PRETTY = {
    "qwen":   "Qwen3-30B-A3B",
    "gptoss": "GPT-OSS-120B",
    "gemma":  "Gemma-4-26B-A4B",
}
DATASET_PRETTY = {"lang": "Language", "task": "Task"}
METRIC_KEYS = [("p50_tpot_ms", "P50"), ("p99_tpot_ms", "P99")]


def main():
    apply()
    data = {}
    rates = (20, 40, 60, 80, 100)
    for dataset in DATASETS:
        workload = 'language' if dataset == 'lang' else dataset
        rows = json.loads((DATA / f'cluster_balance_five_rates-{workload}.json').read_text())
        table = {(r['setting'], r['variant'], r['rate']): r for r in rows}
        assert len(rows) == len(table) == 45
        assert set(table) == {(m, v, r) for m in MODELS
                             for v in ('rr', 'balanced', 'vanilla') for r in rates}
        for model in MODELS:
            data[model, dataset] = [[float(np.mean([
                100 * (table[model, variant, rate][metric] / table[model, 'rr', rate][metric] - 1)
                for rate in rates])) for metric in ('tpot50_ms', 'tpot99_ms')]
                for variant in ('balanced', 'vanilla')]

    fig, axes = plt.subplots(
        len(DATASETS), len(MODELS),
        figsize=(WIDTH, 4.6), sharey="row", squeeze=False,
    )

    c_kmbal, c_kmpure = ELDR_COLOR, RR_COLOR
    bar_w = 0.55
    x = np.array([0.0, 1.55])
    off_l = -bar_w / 2 - 0.03
    off_r = +bar_w / 2 + 0.03

    for i, dataset in enumerate(DATASETS):
        # Row-shared ylim from present cells.
        row_vals = []
        for model in MODELS:
            d = data[(model, dataset)]
            if d:
                row_vals.extend(d[0] + d[1])
        if row_vals:
            mn, mx = min(row_vals), max(max(row_vals), 0)
            span = max(mx - mn, 4.0)
            vmin = mn - max(span * 0.25, 5.0)
            vmax = mx + max(span * 0.22, 5.5)
        else:
            vmin, vmax = -10, 2

        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            d = data[(model, dataset)]
            if d is None:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, fontsize=13, color="#888")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
                if i == 0:
                    ax.set_title(PRETTY[model], fontsize=16, pad=4)
                continue

            vb, vp = d                        # kmbal, kmpure
            ax.bar(x + off_r, vb, bar_w, color=c_kmbal,
                   edgecolor="black", linewidth=0.5, zorder=2)
            ax.bar(x + off_l, vp, bar_w, color=c_kmpure,
                   edgecolor="black", linewidth=0.5, zorder=2)
            for xi, v in zip(x + off_r, vb):
                ax.annotate(f"{v:+.1f}", xy=(xi, v), xycoords="data",
                            xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=13, clip_on=False)
            for xi, v in zip(x + off_l, vp):
                ax.annotate(f"{v:+.1f}", xy=(xi, v), xycoords="data",
                            xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=13, clip_on=False)

            ax.axhline(0, color="black", linewidth=0.6, zorder=1)
            ax.set_xticks(x)
            if i == len(DATASETS) - 1:
                ax.set_xticklabels([f"{lab} TPOT" for _, lab in METRIC_KEYS],
                                    fontsize=15)
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=14, labelleft=(j == 0))
            ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.70, x[-1] + 0.70)
            ax.set_ylim(vmin, vmax)
            if i == 0:
                ax.set_title(PRETTY[model], fontsize=16, pad=4)

    for i, dataset in enumerate(DATASETS):
        axes[i, 0].set_ylabel(
            f"{DATASET_PRETTY[dataset]}\n" + r"$\Delta$ vs RR (%)",
            fontsize=15)
    fig.align_ylabels(axes[:, 0])

    handles = [
        Patch(facecolor=c_kmpure, edgecolor="black", linewidth=0.5,
              label="$K$-means"),
        Patch(facecolor=c_kmbal,  edgecolor="black", linewidth=0.5,
              label="balanced $K$-means"),
    ]
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.10,
                        left=0.085, right=0.99,
                        wspace=0.08, hspace=0.12)
    fig.legend(handles=handles, ncol=2, fontsize=15, frameon=True,
               loc="lower center", bbox_to_anchor=(0.5, 0.88),
               handletextpad=0.55, columnspacing=1.7,
               handlelength=2.0, handleheight=1.1)
    save(fig, OUTPUT / "06_evaluation/fig_cluster_ablation")


if __name__ == "__main__":
    main()
