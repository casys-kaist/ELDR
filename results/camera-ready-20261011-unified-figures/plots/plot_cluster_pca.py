"""Original count-IDF signature PCA and balanced-cluster visualization."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import figstyle as fs

fs.apply()
MODELS = [("qwen", "Qwen3-30B-A3B"), ("gptoss", "GPT-OSS-120B"),
          ("gemma", "Gemma-4-26B-A4B")]
DOMS = ["code", "math", "medical", "legal"]
DOM_PRETTY = dict(zip(DOMS, ["Code", "Math", "Medical", "Legal"]))
DOM_COL = dict(zip(DOMS, [fs.BLUES[0], "#e8702a", "#2ca02c", "#9467bd"]))
LANGS = ["English", "Chinese", "Russian", "French"]
LANG_COL = dict(zip(LANGS, ["#08519c", "#e8702a", "#2ca02c", "#9467bd"]))
OTHER = "#cccccc"
data = json.loads((fs.DATA / "cluster_pca.json").read_text())

fig, axes = plt.subplots(2, 3, figsize=(fs.WIDTH, 5.2))
for i, workload in enumerate(("task", "lang")):
    for j, (tag, pretty) in enumerate(MODELS):
        ax = axes[i, j]
        cell = data[tag][workload]
        P2 = np.asarray(cell["points"])
        cen2 = np.asarray(cell["centers"])
        lab_true = cell["labels"]
        colmap = DOM_COL if i == 0 else LANG_COL
        cols = [colmap.get(l, OTHER) for l in lab_true]
        ax.scatter(P2[:, 0], P2[:, 1], c=cols, s=4, alpha=0.55, lw=0)
        for k in range(len(cen2)):
            ax.scatter(cen2[k, 0], cen2[k, 1], marker="X", s=80, color="k",
                       edgecolor="w", linewidth=0.8, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(pretty, fontsize=16)


fig.subplots_adjust(left=0.075, right=0.90, top=0.87, bottom=0.18, wspace=0.06, hspace=0.10)
for i, rlab in [(0, "Task"), (1, "Language")]:
    pp = axes[i, 0].get_position()
    fig.text(0.915, (pp.y0 + pp.y1) / 2, rlab, rotation=270, fontsize=14, va="center", ha="center")
fig.text(0.035, 0.52, "PC 2", fontsize=14, va="center", rotation=90)
fig.text(0.485, 0.13, "PC 1", fontsize=14, ha="center")

xmark = plt.Line2D([], [], marker="X", color="k", mfc="k", mec="w", ls="", ms=13, label="centroid")
h1 = [Patch(color=DOM_COL[d], label=DOM_PRETTY[d]) for d in DOMS] + [xmark]
fig.legend(handles=h1, loc="lower center", bbox_to_anchor=(0.48, 0.94), ncol=5,
           fontsize=14, frameon=True, columnspacing=1.0, handletextpad=0.5)
h2 = [Patch(color=LANG_COL[l], label=l) for l in LANGS] + [Patch(color=OTHER, label="other")]
fig.legend(handles=h2, loc="upper center", bbox_to_anchor=(0.48, 0.10), ncol=5,
           fontsize=14, frameon=True, columnspacing=1.0, handletextpad=0.5)
fs.save(fig, fs.OUTPUT / "04_design/cluster_pca")
