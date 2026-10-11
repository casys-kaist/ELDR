#!/usr/bin/env python3
"""MoE expert latency vs # active experts, 3 models x 3 batch sizes (line plot).
Data: plots/data/u2_bf16.log (qwen,gemma) + plots/data/u2_gptoss.log.
Run: python3 plots/plot_moe_latency.py
"""
import re
import figstyle as fs
import matplotlib.pyplot as plt
fs.apply()

D = fs.DATA
OUT = fs.OUTPUT / "03_motivation/moe_latency"


def parse_bf16(path):
    d = {}; model = None; B = None
    for ln in open(path):
        m = re.search(r"=== (\w+).*B=(\d+)", ln)
        if m:
            model, B = m.group(1), int(m.group(2)); d.setdefault(model, {})[B] = ([], []); continue
        r = ln.split()
        if model and len(r) == 4 and r[0].lstrip('-').isdigit():
            d[model][B][0].append(int(r[1])); d[model][B][1].append(float(r[3]))
    return d


def parse_gpt(path):
    d = {}
    for ln in open(path):
        m = re.match(r"B\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", ln)
        if m:
            B = int(m.group(1)); d.setdefault(B, ([], []))
            d[B][0].append(int(m.group(3))); d[B][1].append(float(m.group(5)))
    return d


bf = parse_bf16(f"{D}/u2_bf16.log"); gp = parse_gpt(f"{D}/u2_gptoss.log")
panels = [("Qwen3-30B-A3B", bf.get("qwen3", {})), ("GPT-OSS-120B", gp),
          ("Gemma-4-26B-A4B", bf.get("gemma", {}))]
BATCHES = [32, 64, 128]
col = {32: fs.BLUES[3], 64: fs.BLUES[1], 128: fs.BLUES[0]}   # light->dark with B

fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 3.2))
for ax, (lab, bd) in zip(axes, panels):
    for B in BATCHES:
        if B in bd:
            U, g = bd[B]; ax.plot(U, g, "-o", color=col[B], lw=1.6, ms=3.3, label=f"B={B}")
    ax.grid(True, alpha=.3, lw=.5); ax.set_xlim(0, 132); ax.set_ylim(0, None)
    ax.set_title(lab, fontsize=16, pad=4)
axes[0].set_ylabel("MoE expert latency (ms)")
fig.supxlabel("Number of active experts", fontsize=fs.LABEL, y=0.015)
h, l = axes[0].get_legend_handles_labels()
fig.subplots_adjust(left=0.10, right=0.985, top=0.78, bottom=0.20, wspace=0.30)
fs.legend_top(fig, ncol=3, handles=h, labels=l, x0=0.10, w=0.885, y=0.91)
fs.save(fig, OUT)
