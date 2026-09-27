# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Paper renderers, reused with validated AE measurements.

Main, signature/cluster and tau layouts come from the submission plotters
plot_iter1_main.py, plot_sig_transform_ablation.py,
plot_cluster_ablation_6cell.py and plot_tau_new.py (c1e8ac22d).
Prefix cache uses the paper's updated 1P16D plot_prefix_cache.py.
The two original bar layouts share one renderer; no historical run selection.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from eldr.experiments import plot_style

MODELS = ("qwen", "gptoss", "gemma")
PRETTY = {
    "qwen": "Qwen3-30B-A3B",
    "gptoss": "GPT-OSS-120B",
    "gemma": "Gemma-4-26B-A4B",
}
WORKLOAD_NAMES = {"task": "Task", "language": "Language"}
ROUTERS = [
    ("random", "Random", "#7f7f7f", "-", "v"),
    ("rr", "RR", "#ff7f0e", "-", "s"),
    ("jsq", "JSQ", "#2ca02c", "-", "^"),
    ("p2c", "P2C", "#1f77b4", "-", "D"),
    ("domain", "Domain", "#9467bd", "-", "P"),
    ("eldr-static", "ELDR", "#d62728", "-", "o"),
]
MAIN_METRICS = [
    ("tpot50_ms", "P50 TPOT (ms)"),
    ("tpot99_ms", "P99 TPOT (ms)"),
    ("ttft50_ms", "P50 TTFT (ms)"),
]
TPOT_METRICS = [("tpot50_ms", "P50"), ("tpot99_ms", "P99")]
TAUS = [0.0, 0.1, 0.2, 0.3]
PREFIX_POLICIES = (("rr", "RR", "#08519c"), ("eldr-static", "ELDR", "#6baed6"))
PREFIX_METRICS = (
    ("ttft50_ms", "TTFT P50", 300, (0, 100, 200, 300), "%.0f"),
    ("tpot50_ms", "TPOT P50", 68, (0, 20, 40, 60), "%.0f"),
    ("tpot99_ms", "TPOT P99", 68, (0, 20, 40, 60), "%.0f"),
)


def draw(panels, outpath, experiment):
    """Adapt saved rows, preserving matched RR and mean per-rate percent changes."""
    tables = {}
    for workload, rows in panels.items():
        table = {(r["setting"], r["variant"], r["rate"]): r for r in rows}
        if not rows or len(table) != len(rows):
            raise ValueError("Empty or duplicate plot cells")
        tables[workload] = table
    present = {r["setting"].split("-")[0] for rows in panels.values() for r in rows}
    if not present or present - set(MODELS):
        raise ValueError("Unknown model")
    models = [m for m in MODELS if m in present]
    data = {}
    for workload, table in tables.items():
        rates = sorted({key[2] for key in table})
        variants = {key[1] for key in table}

        def delta(setting, variant, table=table, rates=rates):
            baseline, candidate = np.array(
                [
                    [
                        [table[setting, v, rate][m] for m, _ in TPOT_METRICS]
                        for rate in rates
                    ]
                    for v in ("rr", variant)
                ]
            )
            return (100 * (candidate - baseline) / baseline).mean(axis=0).tolist()

        if experiment in ("main_task", "main_language"):
            routers = [r for r in ROUTERS if r[0] in variants]
            if "eldr" in variants:
                routers.append(
                    (
                        "eldr",
                        "ELDR (online)" if "eldr-static" in variants else "ELDR",
                        "#d62728",
                        "--",
                        "o",
                    )
                )
            if variants != {r[0] for r in routers} or len(panels) != 1:
                raise ValueError("Unknown main policy or multiple main workloads")
            data = {
                model: {
                    policy: {
                        metric: [table[model, policy, rate][metric] for rate in rates]
                        for metric, _ in MAIN_METRICS
                    }
                    for policy, *_ in routers
                }
                for model in models
            }
            plot_main(data, outpath, rates, models, routers)
            return
        if experiment == "prefix_cache":
            expected = {
                (f"gptoss-cache-{cache}", policy, rates[0])
                for cache in (0, 1)
                for policy, _, _ in PREFIX_POLICIES
            }
            if len(panels) != 1 or len(rates) != 1 or set(table) != expected:
                raise ValueError("Prefix plot requires four GPT-OSS cells at one rate")
            plot_prefix_cache(
                {(setting, policy): row for (setting, policy, _), row in table.items()},
                outpath,
            )
            return
        for model in models:
            if experiment == "signature_ablation":
                data[model, workload] = [
                    delta(f"{model}-{variant}", variant)
                    for variant in ("count_idf", "gate_prob_all")
                ]
            elif experiment == "cluster_balance":
                data[model, workload] = [
                    delta(model, v) for v in ("balanced", "vanilla")
                ]
            elif experiment == "locality_band":
                data[model, workload] = {
                    metric: {tau: delta(model, f"tau-{tau:g}")[i] for tau in TAUS}
                    for i, (metric, _) in enumerate(TPOT_METRICS)
                }
            else:
                raise ValueError("Unknown experiment")
    if experiment == "locality_band":
        cells = [
            (
                (m, w),
                f"{dict(qwen='Qwen', gptoss='GPT-OSS', gemma='Gemma')[m]} "
                f"{'lang' if w == 'language' else w}",
            )
            for m in models
            for w in panels
        ]
        plot_locality_band(data, outpath, cells)
    else:
        plot_ablation(
            data,
            outpath,
            models,
            list(panels),
            signature=experiment == "signature_ablation",
        )


def plot_main(data, outpath, rates, models, routers):
    plot_style.apply()

    nrows, ncols = len(MAIN_METRICS), len(models)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(plot_style.WIDTH * 0.85, 5.6), sharex=True, squeeze=False
    )
    xtick_pos = [20, 60, 100] if rates == [20, 40, 60, 80, 100] else rates

    for ci, model in enumerate(models):
        for ri, (m, ylabel) in enumerate(MAIN_METRICS):
            ax = axes[ri][ci]
            for rkey, rlabel, color, ls, mk in routers:
                ys = data[model][rkey][m]
                ax.plot(
                    rates,
                    ys,
                    color=color,
                    ls=ls,
                    marker=mk,
                    markersize=5.0,
                    lw=1.5,
                    label=rlabel,
                    markeredgewidth=0.5,
                )
            all_ys = [y for rkey, *_ in routers for y in data[model][rkey][m]]
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
    for rkey, rlabel, color, ls, mk in routers:
        (line,) = axes[0][0].plot(
            [],
            [],
            color=color,
            ls=ls,
            marker=mk,
            markersize=6.0,
            lw=1.6,
            label=rlabel,
            markeredgewidth=0.5,
        )
        handles.append(line)
        labels.append(rlabel)

    LEFT, RIGHT, TOP, BOTTOM = 0.075, 0.995, 0.84, 0.09
    fig.subplots_adjust(
        top=TOP, bottom=BOTTOM, left=LEFT, right=RIGHT, wspace=0.22, hspace=0.20
    )
    fig.legend(
        handles,
        labels,
        ncol=len(routers),
        loc="lower center",
        bbox_to_anchor=(LEFT, 0.90, RIGHT - LEFT, 0.05),
        mode="expand",
        frameon=True,
        fontsize=12,
        handletextpad=0.5,
        columnspacing=1.0,
        borderaxespad=0.2,
    )
    plot_style.save(fig, outpath)
    plt.close(fig)


def plot_ablation(data, outpath, models, workloads, signature):
    plot_style.apply()
    fig, axes = plt.subplots(
        len(workloads),
        len(models),
        figsize=(plot_style.WIDTH, 4.6 * len(workloads) / 2),
        sharey="row",
        squeeze=False,
    )

    c_first, c_second = plot_style.BLUES[0], plot_style.BLUES[2]
    bar_w = 0.55
    x = np.array([0.0, 1.55])
    off_l = -bar_w / 2 - 0.03
    off_r = +bar_w / 2 + 0.03

    for i, dataset in enumerate(workloads):
        row_vals = [v for m in models for values in data[m, dataset] for v in values]
        mn, mx = min(row_vals), max(max(row_vals), 0)
        span = max(mx - mn, 4.0)
        vmin = mn - max(span * (0.20 if signature else 0.25), 3.0 if signature else 5.0)
        vmax = mx + max(span * (0.20 if signature else 0.22), 4.0 if signature else 5.5)

        for j, model in enumerate(models):
            ax = axes[i, j]
            for values, offset, color in zip(
                data[model, dataset], (off_l, off_r), (c_first, c_second)
            ):
                ax.bar(
                    x + offset,
                    values,
                    bar_w,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    zorder=2,
                )
                for xi, v in zip(x + offset, values):
                    ax.annotate(
                        f"{v:+.1f}",
                        xy=(xi, v),
                        xycoords="data",
                        xytext=(0, 2 if v >= 0 else -2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=13,
                        clip_on=False,
                    )

            ax.axhline(0, color="black", linewidth=0.6, zorder=1)
            ax.set_xticks(x)
            if i == len(workloads) - 1:
                ax.set_xticklabels(
                    [f"TPOT {lab}" for _, lab in TPOT_METRICS], fontsize=15
                )
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=14, labelleft=(j == 0))
            ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.70, x[-1] + 0.70)
            ax.set_ylim(vmin, vmax)
            if i == 0:
                ax.set_title(PRETTY[model], fontsize=16, pad=4)

    for i, dataset in enumerate(workloads):
        axes[i, 0].set_ylabel(
            f"{WORKLOAD_NAMES.get(dataset, dataset)}\n" + r"$\Delta$ vs RR (%)",
            fontsize=15,
        )
    if not signature:
        fig.align_ylabels(axes[:, 0])

    handles = [
        Patch(
            facecolor=c_first,
            edgecolor="black",
            linewidth=0.5,
            label=r"count$\cdot$idf" if signature else "balanced $K$-means",
        ),
        Patch(
            facecolor=c_second,
            edgecolor="black",
            linewidth=0.5,
            label="gate-prob" if signature else "$K$-means",
        ),
    ]
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.80, bottom=0.10, left=0.085, right=0.99, wspace=0.08, hspace=0.12
    )
    fig.legend(
        handles=handles,
        ncol=2,
        fontsize=15,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.88),
        handletextpad=0.55,
        columnspacing=1.7,
        handlelength=2.0,
        handleheight=1.1,
    )
    plot_style.save(fig, outpath)
    plt.close(fig)


def plot_locality_band(data, outpath, cells):
    plot_style.apply()
    fig, axes = plt.subplots(
        1, 2, figsize=(plot_style.WIDTH * 0.85, 3.4), squeeze=False
    )
    cmap = plt.cm.RdBu_r
    all_vals = [
        data[ck][m][t] for ck, _ in cells for m, _ in TPOT_METRICS for t in TAUS
    ]
    vmax = max(abs(v) for v in all_vals) or 1.0
    cell_labels = [c[1] for c in cells]

    for mi, (m, mlabel) in enumerate(TPOT_METRICS):
        ax = axes[0][mi]
        M = np.array([[data[ck][m][t] for t in TAUS] for ck, _ in cells])
        im = ax.imshow(
            M, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest"
        )
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(
                    j,
                    i,
                    f"{v:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if abs(v) > 0.5 * vmax else "black",
                )
        ax.set_xticks(range(len(TAUS)))
        ax.set_xticklabels([f"{t}" for t in TAUS], fontsize=13)
        ax.set_xlabel(r"Locality band width ($\tau$)", fontsize=13)
        ax.set_yticks(range(len(cells)))
        if mi == 0:
            ax.set_yticklabels(cell_labels, fontsize=13)
        else:
            ax.set_yticklabels([])
        ax.set_title("TPOT " + mlabel + r" (% $\Delta$ vs RR)", fontsize=13, pad=4)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)

    fig.subplots_adjust(top=0.91, bottom=0.18, left=0.15, right=0.89, wspace=0.04)
    # Place colorbar AFTER subplots_adjust so its y extent matches the heatmap exactly.
    cax = fig.add_axes([0.905, 0.18, 0.014, 0.91 - 0.18])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    plot_style.save(fig, outpath)
    plt.close(fig)


def plot_prefix_cache(cells, outpath):
    plot_style.apply()
    plt.rcParams.update(
        {
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "axes.titlepad": 4,
            "legend.fontsize": 15,
            "pdf.fonttype": 3,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(plot_style.WIDTH, 2.8))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.78, wspace=0.22)
    x = np.array([0.0, 2.4])
    bar_width = 0.85
    for ax, (key, title, limit, ticks, fmt) in zip(axes, PREFIX_METRICS):
        for i, (policy, label, color) in enumerate(PREFIX_POLICIES):
            values = [float(cells[f"gptoss-cache-{c}", policy][key]) for c in (0, 1)]
            bars = ax.bar(
                x + (2 * i - 1) * (bar_width / 2 + 0.04),
                values,
                width=bar_width,
                color=color,
                edgecolor="black",
                linewidth=0.5,
                label=label,
                zorder=2,
            )
            for bar, value in zip(bars, values):
                ax.annotate(
                    fmt % value,
                    (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    clip_on=False,
                )
        maximum = max(float(row[key]) for row in cells.values())
        ax.set(title=title, xlabel="Prefix cache", ylim=(0, max(limit, maximum * 1.15)))
        if maximum * 1.15 <= limit:
            ax.set_yticks(ticks)
        ax.set_xlim(-1.05, x[-1] + 1.05)
        ax.set_xticks(x, ("Off", "On"))
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Latency (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=2,
        loc="lower center",
        frameon=True,
        bbox_to_anchor=(0.5, 0.87),
        handletextpad=0.55,
        columnspacing=1.7,
        handlelength=2.0,
        handleheight=1.1,
    )
    plot_style.save(fig, outpath, dpi=300)
    plt.close(fig)
