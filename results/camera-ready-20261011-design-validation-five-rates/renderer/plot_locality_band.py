import json

import numpy as np
import matplotlib.pyplot as plt

from figstyle import apply, save, WIDTH, DATA, OUTPUT


# (cell_dir_name, display_label) — 6 cells in figure order.
# Convention (locked 2026-06-03): row order Qwen -> GPT-OSS -> Gemma;
# GPT-OSS always rendered in ALL CAPS in labels.
CELLS = [
    ("qwen_task",   "Qwen Task"),
    ("qwen_lang",   "Qwen Language"),
    ("gptoss_task", "GPT-OSS Task"),
    ("gptoss_lang", "GPT-OSS Language"),
    ("gemma_task",  "Gemma Task"),
    ("gemma_lang",  "Gemma Language"),
]
RATES = [20, 40, 60, 80, 100]
TAUS = [0.0, 0.1, 0.2, 0.3]
METRICS = [
    ("tpot50_ms", "P50 TPOT"),
    ("tpot99_ms", "P99 TPOT"),
]


def plot(data, outpath):
    apply()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH * 0.85, 3.4), squeeze=False)
    cmap = plt.cm.RdBu_r
    all_vals = [data[ck][m][t]
                for ck, _ in CELLS
                for m, _ in METRICS
                for t in TAUS
                if not np.isnan(data[ck][m][t])]
    vmax = max(abs(v) for v in all_vals) if all_vals else 10.0
    cell_labels = [c[1] for c in CELLS]

    for mi, (m, mlabel) in enumerate(METRICS):
        ax = axes[0][mi]
        M = np.array([[data[ck][m][t] for t in TAUS] for ck, _ in CELLS])
        # Mask NaN so they show as gray
        masked = np.ma.masked_invalid(M)
        cmap.set_bad("#dddddd")
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=11, color="#888")
                else:
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                            fontsize=12,
                            color="white" if abs(v) > 0.5 * vmax else "black")
        ax.set_xticks(range(len(TAUS)))
        ax.set_xticklabels([f"{t}" for t in TAUS], fontsize=13)
        ax.set_xlabel(r"Locality band width ($\tau$)", fontsize=13)
        ax.set_yticks(range(len(CELLS)))
        if mi == 0:
            ax.set_yticklabels(cell_labels, fontsize=13)
        else:
            ax.set_yticklabels([])
        ax.set_title(mlabel + r" ($\Delta$ vs RR, %)", fontsize=13, pad=4)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

    fig.subplots_adjust(top=0.91, bottom=0.18, left=0.15, right=0.89, wspace=0.04)
    # Place colorbar AFTER subplots_adjust so its y extent matches the heatmap exactly.
    cax = fig.add_axes([0.905, 0.18, 0.014, 0.91 - 0.18])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    save(fig, outpath)
    plt.close(fig)


if __name__ == "__main__":
    data = {}
    for workload in ('task', 'language'):
        rows = json.loads((DATA / f'locality_band_five_rates-{workload}.json').read_text())
        table = {(r['setting'], r['variant'], r['rate']): r for r in rows}
        variants = ['rr'] + [f'tau-{tau:g}' for tau in TAUS]
        models = ('qwen', 'gptoss', 'gemma')
        assert len(rows) == len(table) == 75
        assert set(table) == {(m, v, r) for m in models for v in variants for r in RATES}
        for model in models:
            cell = f"{model}_{'lang' if workload == 'language' else workload}"
            data[cell] = {metric: {tau: float(np.mean([
                100 * (table[model, f'tau-{tau:g}', rate][metric] / table[model, 'rr', rate][metric] - 1)
                for rate in RATES])) for tau in TAUS} for metric, _ in METRICS}
    plot(data, OUTPUT / "06_evaluation/fig_tau_ablation")
