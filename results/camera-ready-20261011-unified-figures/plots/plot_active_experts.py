#!/usr/bin/env python3
"""Opportunity figure from the REAL measurement (run_opportunity.py output):
active experts per decode step, mixed vs same-domain batch, across a batch-size
sweep (32/64/128), for task and language domains, 3 models. 2x3 grid
(rows=task/language, cols=models).
  python3 plots/plot_active_experts.py
"""
import json
import figstyle as fs
import matplotlib.pyplot as plt
fs.apply()
R = str(fs.DATA)
MODELS = [("qwen", "Qwen3-30B-A3B"), ("gptoss", "GPT-OSS-120B"), ("gemma", "Gemma-4-26B-A4B")]
BS = [32, 64, 128]
MIX, PURE = fs.RR_COLOR, fs.ELDR_COLOR   # blue (mixed), red (same-domain)

data = {m: json.load(open(f"{R}/opportunity_{m}.json")) for m, _ in MODELS}


def vals(d, ph):
    return [d[ph].get(str(b), d[ph].get(b, 0)) for b in BS]


fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH, 4.0), sharex=True, sharey=True)
rows = [("Task", "task_mixed", "task_pure"), ("Language", "lang_mixed", "lang_pure")]
for i, (rlab, mph, pph) in enumerate(rows):
    for j, (m, name) in enumerate(MODELS):
        ax = axes[i, j]; d = data[m]
        mx, pu = vals(d, mph), vals(d, pph)
        ax.plot(BS, mx, "-s", color=MIX, lw=2, ms=6, label="Mixed-domain batch")
        ax.plot(BS, pu, "-o", color=PURE, lw=2, ms=6, label="Same-domain batch")
        for bi, b in enumerate(BS):                       # reduction % below each same-domain dot
            red = 100 * (mx[bi] - pu[bi]) / mx[bi]
            ax.annotate(f"$-${red:.0f}%", (b, pu[bi]), textcoords="offset points",
                        xytext=(0, -12), ha="center", va="top", fontsize=11,
                        fontweight="bold", color=PURE)
        if i == 0: ax.set_title(name, fontsize=16)
        ax.set_xticks(BS); ax.tick_params(labelsize=14); ax.grid(alpha=0.25, lw=0.5)
        ax.set_ylim(28, 124); ax.set_xlim(12, 148)   # x-margin so edge %-labels don't collide
for j in range(3): axes[1, j].set_xlabel("Batch size", fontsize=14)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.03), fontsize=14)
fig.subplots_adjust(left=0.13, right=0.91, top=0.85, bottom=0.10, wspace=0.08, hspace=0.18)
fig.supylabel("Active experts per decode step", fontsize=14, x=0.045)   # units on the LEFT
for i, rlab in [(0, "Task"), (1, "Language")]:                   # row identifiers on the RIGHT
    p = axes[i, 0].get_position()
    fig.text(0.938, (p.y0 + p.y1) / 2, rlab, rotation=270, fontsize=15, va="center", ha="center")
fs.save(fig, fs.OUTPUT / "03_motivation/active_experts_sweep")
print("wrote active_experts_sweep")
