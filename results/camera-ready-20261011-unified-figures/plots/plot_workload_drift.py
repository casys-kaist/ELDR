#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Render a compact RR/static/online timeline from an audited comparison."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import figstyle as plot_style

METHODS = ("rr", "static", "online")
LABELS = ("RR", "Static ELDR", "Online ELDR")
# Match the paper's main plots: RR blue, ELDR red.
# Static ELDR is the additional neutral comparator in this figure.
COLORS = (plot_style.RR_COLOR, "#7f7f7f", plot_style.ELDR_COLOR)
MARKERS = ("s", "D", "o")
STYLES = ("-", "--", "-")
DOMAINS = ("code", "math", "medical", "legal")
DOMAIN_COLORS = ("#6baed6", "#fdae6b", "#a1d99b", "#bcbddc")


def validate(comparison: dict[str, Any]) -> None:
    windows = comparison["windows"]
    expected_start = 0
    for window in windows:
        if window["start_s"] != expected_start:
            raise ValueError("Windows must be contiguous and start at zero")
        if window["end_s"] - window["start_s"] != comparison["window_s"]:
            raise ValueError("Unexpected analysis window width")
        expected_start = window["end_s"]
        for method in METHODS:
            value = window[method]["tpot_p50_ms"]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {method} TPOT")
            if window[method]["requests"] <= 0:
                raise ValueError(f"Empty {method} window")
        mix = [window["domain_percent"][domain] for domain in DOMAINS]
        if any(not np.isfinite(value) or value < 0 for value in mix):
            raise ValueError("Invalid workload mix")
        if not np.isclose(sum(mix), 100):
            raise ValueError("Workload shares must sum to 100%")
    if not windows or expected_start != comparison["duration_s"]:
        raise ValueError("Windows must cover the analyzed interval")
    if comparison["provenance"]["runs_per_method"] != dict.fromkeys(METHODS, 1):
        raise ValueError("This renderer shows one run per method, without CIs")


def plot(
    comparison: dict[str, Any],
    output: Path,
    *,
    larger_mix: bool = False,
    shade_transitions: bool = True,
    static_color: str = COLORS[1],
) -> Path:
    validate(comparison)
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "eldr-matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot_style.apply()
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.labelsize": 15,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, (mix_axis, latency_axis) = plt.subplots(
        2,
        1,
        figsize=(plot_style.WIDTH, 4.5 if larger_mix else 4.1),
        sharex=True,
        gridspec_kw={
            "height_ratios": (1.0 if larger_mix else 0.65, 2.25),
            "hspace": 0.65,
        },
    )
    figure.subplots_adjust(left=0.13, right=0.985, bottom=0.14, top=0.865)
    windows = comparison["windows"]
    centers = np.array([(w["start_s"] + w["end_s"]) / 2 for w in windows])
    edges = np.array([w["start_s"] for w in windows] + [windows[-1]["end_s"]])
    mix = np.array([[w["domain_percent"][d] for w in windows] for d in DOMAINS])
    base = np.zeros(len(windows))
    for values, color in zip(mix, DOMAIN_COLORS):
        mix_axis.stairs(
            base + values,
            edges,
            baseline=base,
            fill=True,
            color=color,
            linewidth=0,
        )
        # Matplotlib retains the baseline array; keep each band's copy immutable.
        base = base + values
    mix_axis.set_ylim(0, 100)
    mix_axis.set_yticks([0, 100])
    mix_axis.set_ylabel("Mix (%)")
    mix_axis.tick_params(axis="x", bottom=False, labelbottom=False)
    mix_axis.legend(
        handles=[
            Patch(facecolor=c, label=d.title()) for d, c in zip(DOMAINS, DOMAIN_COLORS)
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=True,
        borderpad=0.28,
        handlelength=1.05,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    for method, label, color, marker, style in zip(
        METHODS, LABELS, (COLORS[0], static_color, COLORS[2]), MARKERS, STYLES
    ):
        values = np.array([w[method]["tpot_p50_ms"] for w in windows])
        latency_axis.plot(
            centers,
            values,
            color=color,
            linestyle=style,
            marker=marker,
            linewidth=2.0 if method == "online" else 1.6,
            markersize=4.5,
            markeredgewidth=0.7,
            markerfacecolor="white" if method == "static" else color,
            label=label,
            zorder=4 if method == "online" else 3,
        )

    duration = comparison["duration_s"]
    for axis in (mix_axis, latency_axis):
        for boundary in comparison["phase_boundaries_s"][1:-1]:
            axis.axvline(
                boundary, color="#777777", linestyle=":", linewidth=0.8, zorder=2
            )
        axis.set_xlim(0, duration)
        axis.yaxis.set_label_coords(-0.095, 0.5)
        for spine in axis.spines.values():
            spine.set_linewidth(0.7)
    if shade_transitions:
        for start, end in comparison["transition_intervals_s"]:
            latency_axis.axvspan(start, end, color="#ededed", zorder=0)
    latency_axis.grid(color="#b0b0b0", alpha=0.25, linewidth=0.6, zorder=1)
    latency_axis.set_xlabel("Request arrival time (s)")
    latency_axis.set_ylabel("P50 TPOT (ms)")
    latency_axis.margins(y=0.10)
    latency_axis.set_xticks(np.arange(0, duration + 1, 20))
    latency_axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=3,
        frameon=True,
        borderpad=0.3,
        handlelength=1.8,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    figure.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={
            "Title": "Workload drift: RR, static ELDR, and online ELDR",
            "Subject": "One run per method; 5-second request-arrival bins; "
            "whole-generation request TPOT; no confidence intervals",
        },
    )
    figure.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.04, dpi=200)
    plt.close(figure)
    return pdf



if __name__ == "__main__":
    comparison = json.loads((plot_style.DATA / "workload_drift.json").read_text())
    plot(comparison, plot_style.OUTPUT / "06_evaluation/fig_workload_drift",
         larger_mix=True, shade_transitions=False, static_color="#595959")
