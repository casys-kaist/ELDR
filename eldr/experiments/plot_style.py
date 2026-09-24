"""Shared figure style for ELDR paper plots. Import + apply().

Every paper figure must use this for a consistent look:
  - width 7.4in (2-column paper width); height per content
  - font.family DejaVu Sans; font.size 16, axes.labelsize 16, xtick/ytick 14
  - axes.linewidth 0.7
  - legend: BOXED (frameon=True), compact, CENTERED, fontsize 14, placed ABOVE
    the plot (the breakdown/moe_u convention)
  - blue sequential palette (BLUES, dark->light) + GREYS for muted buckets
  - save BOTH .pdf and .png (dpi 300), bbox_inches="tight"
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt

WIDTH = 7.4  # 2-column paper width (inches)
BLUES = ["#08519c", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]  # dark -> light
GREYS = ["#d9d9d9", "#f0f0f0"]
FONT, LABEL, TICK, LEG = 16, 16, 14, 14  # the size convention


def apply():
    plt.rcParams.update(
        {
            "font.size": FONT,
            "axes.labelsize": LABEL,
            "xtick.labelsize": TICK,
            "ytick.labelsize": TICK,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
        }
    )


def legend_top(obj, ncol, handles=None, labels=None, x0=0.0, w=1.0, y=1.02):
    """Frameless, full-width-expanded legend above the plot (the convention).
    obj = an Axes (single panel, y in axes coords) or Figure (multi-panel, y in
    figure coords -- set y just above subplots `top` to control the gap)."""
    xc = x0 + w / 2.0  # center of the plot span
    kw = dict(
        ncol=ncol,
        fontsize=LEG,
        frameon=True,
        loc="lower center",
        bbox_to_anchor=(xc, y),
        handletextpad=0.5,
        columnspacing=1.4,
    )
    if handles is not None:
        obj.legend(handles, labels, **kw)
    else:
        obj.legend(**kw)


def save(fig, stem):
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    # dpi controls the resolution of rasterized images (imshow); text/lines stay vector
    fig.savefig(f"{stem}.pdf", bbox_inches="tight", dpi=600)
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    print(f"wrote {stem}.{{pdf,png}}")
