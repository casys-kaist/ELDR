# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared style from the original ELDR paper plotters."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 7.4
BLUES = ["#08519c", "#4292c6", "#6baed6", "#9ecae1", "#c6dbef"]


def apply():
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
        }
    )


def save(fig, stem, dpi=600):
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(f"{stem}{suffix}", bbox_inches="tight", dpi=dpi)
    print(f"wrote {stem}.{{pdf,png}}")
