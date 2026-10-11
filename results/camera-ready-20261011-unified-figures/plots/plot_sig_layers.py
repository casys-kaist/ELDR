"""Original layer-selection curves, with embedded TrueType text."""
import json

import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

data = json.loads((fs.DATA / "sig_layers.json").read_text())
fs.apply()
fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.9), sharey=True)
for ax, (tag, name) in zip(axes, [
    ("qwen", "Qwen3-30B-A3B"), ("gptoss", "GPT-OSS-120B"),
    ("gemma", "Gemma-4-26B-A4B"),
]):
    for key, label, color in [
        ("task", "Task", fs.RR_COLOR), ("lang", "Language", fs.ELDR_COLOR),
    ]:
        cum = data[tag][key]
        ax.plot(np.arange(1, len(cum) + 1), cum, "-", color=color,
                lw=1.8, label=label)
        k = int(np.argmax(cum))
        ax.plot(k + 1, cum[k], "*", color=color, ms=22, mec="white",
                mew=1.0, zorder=5)
    ax.set_title(name, fontsize=fs.LABEL, pad=4)
    ax.set_xlabel("Layers kept")
    ax.grid(alpha=0.3, ls=":", lw=0.6)
    ax.set_xlim(0, None)
    ax.margins(x=0.02)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
axes[0].set_ylabel(r"Cumulative $\rho$")
axes[0].set_ylim(0.58, 0.90)
axes[0].set_yticks([0.6, 0.7, 0.8, 0.9])
h, labels = axes[0].get_legend_handles_labels()
fig.subplots_adjust(top=0.78, bottom=0.17, left=0.085, right=0.99, wspace=0.08)
fig.legend(h, labels, ncol=2, fontsize=fs.LEG, frameon=True,
           loc="lower center", bbox_to_anchor=(0.54, 0.88),
           handletextpad=0.5, columnspacing=1.6)
fs.save(fig, fs.OUTPUT / "04_design/sig_layers")
