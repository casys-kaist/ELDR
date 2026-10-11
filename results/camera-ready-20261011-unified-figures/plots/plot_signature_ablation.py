"""§Expert Signature Transform ablation — paper-ready 2×3 figure at r=60.

Layout: rows = dataset (Task, Lang). Cols = model (Qwen, GPT-OSS, Gemma).
Each panel plots Δ vs RR (%) for TPOT P50 and TPOT P99, with gate-prob
(blue, left) and count·idf ELDR (red, right). Uses figstyle.py conventions.
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


def main():
    apply()
    metrics = [("median_tpot_ms", "P50"), ("p99_tpot_ms", "P99")]
    c_count, c_gate = ELDR_COLOR, RR_COLOR

    stored = json.loads((DATA / "signature_ablation.json").read_text())
    data = {(m, d): stored[d][m] for m in MODELS for d in DATASETS}

    fig, axes = plt.subplots(
        len(DATASETS), len(MODELS),
        figsize=(WIDTH, 4.6), sharey="row", squeeze=False,
    )

    bar_w = 0.55                           # thicker bars (visually prominent)
    x = np.array([0.0, 1.55])              # P50 ↔ P99 well separated
    off_l = -bar_w / 2 - 0.03              # tight pair gap; bars dominate,
    off_r = +bar_w / 2 + 0.03              # labels sit above/below not beside

    for i, dataset in enumerate(DATASETS):
        # Row-shared ylim with generous headroom on BOTH sides so the value
        # labels (placed in points-from-bar) never collide with the axes box.
        row_vals = []
        for model in MODELS:
            d = data[(model, dataset)]
            if d:
                row_vals.extend(d[0] + d[1])
        if row_vals:
            mn, mx = min(row_vals), max(max(row_vals), 0)
            span = max(mx - mn, 4.0)
            # ~10pt label + 3pt offset above bar tip ≈ 4 data units of room
            # at this figure scale; floor the headroom so labels never hit
            # the axes frame even when bars sit near zero.
            vmin = mn - max(span * 0.20, 3.0)
            vmax = mx + max(span * 0.20, 4.0)
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

            vc, vg = d
            bars_c = ax.bar(x + off_r, vc, bar_w, color=c_count,
                            edgecolor="black", linewidth=0.5, zorder=2)
            bars_g = ax.bar(x + off_l, vg, bar_w, color=c_gate,
                            edgecolor="black", linewidth=0.5, zorder=2)

            # Value labels: offset 3 points OUTSIDE the bar tip (in display
            # space, not data) so a label is always at a fixed visual distance
            # from the bar regardless of bar height, and the row's vmin/vmax
            # headroom guarantees it never touches the axes frame.
            for xi, v in zip(x + off_r, vc):
                ax.annotate(f"{v:+.1f}", xy=(xi, v), xycoords="data",
                            xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=13, clip_on=False)
            for xi, v in zip(x + off_l, vg):
                ax.annotate(f"{v:+.1f}", xy=(xi, v), xycoords="data",
                            xytext=(0, 2 if v >= 0 else -2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom" if v >= 0 else "top",
                            fontsize=13, clip_on=False)

            ax.axhline(0, color="black", linewidth=0.6, zorder=1)
            ax.set_xticks(x)
            # Only the bottom row shows TPOT P50 / P99 x-tick labels — the
            # top row inherits the meaning visually, and dropping them avoids
            # the "P50 / P99" labels colliding at this column width.
            if i == len(DATASETS) - 1:
                ax.set_xticklabels([f"{lab} TPOT" for _, lab in metrics], fontsize=15)
            else:
                ax.set_xticklabels([])
            # Y-tick labels only on the leftmost column (sharey="row" default,
            # explicit for clarity). Inner panels inherit the same scale.
            ax.tick_params(axis="y", labelsize=14,
                           labelleft=(j == 0))
            ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.70, x[-1] + 0.70)
            ax.set_ylim(vmin, vmax)
            if i == 0:
                ax.set_title(PRETTY[model], fontsize=16, pad=4)

    # Row labels.
    for i, dataset in enumerate(DATASETS):
        axes[i, 0].set_ylabel(
            f"{DATASET_PRETTY[dataset]}\n" + r"$\Delta$ vs RR (%)",
            fontsize=15)

    # Legend boxed, snug on top.
    handles = [
        Patch(facecolor=c_gate, edgecolor="black", linewidth=0.5,
              label="gate-prob"),
        Patch(facecolor=c_count, edgecolor="black", linewidth=0.5,
              label=r"count$\cdot$idf"),
    ]
    # Compute layout with tight_layout first, then explicitly nudge the axes
    # top down and anchor the legend just above it — this guarantees the
    # legend sits flush over the column titles with no extra whitespace.
    plt.tight_layout()
    plt.subplots_adjust(top=0.80, bottom=0.10,
                        left=0.085, right=0.99,
                        wspace=0.08, hspace=0.12)
    fig.legend(handles=handles, ncol=2, fontsize=15, frameon=True,
               loc="lower center", bbox_to_anchor=(0.5, 0.88),
               handletextpad=0.55, columnspacing=1.7,
               handlelength=2.0, handleheight=1.1)
    save(fig, OUTPUT / "06_evaluation/fig_transform_ablation")

    print()
    print("§Sig Transform Ablation @ r=60 (paper 2×3):")
    for dataset in DATASETS:
        for model in MODELS:
            d = data[(model, dataset)]
            if d is None:
                print(f"  {model}+{dataset}  PENDING")
                continue
            vc, vg = d
            for (_, lab), c, g in zip(metrics, vc, vg):
                print(f"  {model:6}+{dataset:4}  TPOT {lab}  "
                      f"count={c:+6.1f}%  gate={g:+6.1f}%")


if __name__ == "__main__":
    main()
