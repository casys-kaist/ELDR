"""fig_qwen235_2p8d — Qwen3-235B-A22B 2P8D × TP=4 × EP=4 WildChat τ=0.1.

Two-panel TPOT P50 / P99, matching the original paper figure:
WIDTH × 2.8 flat, DejaVu 15/14, RR=blue square, ELDR=red circle,
lw=2 ms=6, boxed centered legend above, 600dpi.
"""
import json
import sys
import matplotlib.pyplot as plt
from figstyle import apply, save, WIDTH, DATA, OUTPUT, RR_COLOR, ELDR_COLOR

OUT = OUTPUT / "06_evaluation"


def main():
    data = json.loads((DATA / "qwen235.json").read_text())
    rr = {int(r): v for r, v in data["baselines"].items()}
    el = {int(r): v for r, v in data["eldr"].items()}
    rates = sorted(rr)
    if not rates:
        sys.exit("no common rates")
    print(f"rates: {rates}")

    rr_p50  = [rr[r]["median_tpot_ms"] for r in rates]
    el_p50  = [el[r]["median_tpot_ms"] for r in rates]
    rr_p99  = [rr[r]["p99_tpot_ms"]    for r in rates]
    el_p99  = [el[r]["p99_tpot_ms"]    for r in rates]
    apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 2.8))

    for ax, rr_y, el_y, title in [
        (ax1, rr_p50,  el_p50,  "P50 TPOT"),
        (ax2, rr_p99,  el_p99,  "P99 TPOT"),
    ]:
        ax.plot(rates, rr_y, color=RR_COLOR, lw=2, marker="s", ms=6.0,
                markeredgewidth=0.5, label="RR")
        ax.plot(rates, el_y, color=ELDR_COLOR, lw=2, marker="o", ms=6.0,
                markeredgewidth=0.5, label="ELDR")
        ax.set_xlabel("Request rate (req/s)", fontsize=15)
        ax.set_title(title, fontsize=15, pad=4)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, alpha=0.30, lw=0.5)
        ax.set_xticks(rates)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        ymax = max(rr_y + el_y); ymin = min(rr_y + el_y)
        spread = ymax - ymin
        ax.set_ylim(ymin - 0.10 * spread, ymax + 0.10 * spread)
    ax1.set_ylabel("Latency (ms)", fontsize=15)

    handles = [
        plt.Line2D([], [], color=RR_COLOR, lw=2, marker="s", ms=7.0,
                   markeredgewidth=0.5, label="RR"),
        plt.Line2D([], [], color=ELDR_COLOR, lw=2, marker="o", ms=7.0,
                   markeredgewidth=0.5, label="ELDR"),
    ]
    fig.legend(handles=handles, labels=["RR", "ELDR"],
               ncol=2, fontsize=14, frameon=True,
               loc="lower center", bbox_to_anchor=(0.5, 0.90),
               handletextpad=0.5, columnspacing=1.4)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.08, right=0.99,
                        wspace=0.22)
    save(fig, f"{OUT}/fig_qwen235_2p8d")
    plt.close(fig)

    print()
    print(f"{'rate':>5}  {'metric':>10}  {'RR':>8}  {'ELDR':>9}  {'Δ':>7}")
    for r in rates:
        for key, lbl in [("median_tpot_ms","TPOT P50"),
                         ("p99_tpot_ms","TPOT P99"),
                         ("median_ttft_ms","TTFT P50")]:
            d = 100*(el[r][key]-rr[r][key])/rr[r][key]
            print(f"  r={r:>2}  {lbl:>10}  {rr[r][key]:>8.1f}  "
                  f"{el[r][key]:>9.1f}  {d:>+6.1f}%")


if __name__ == "__main__":
    main()
