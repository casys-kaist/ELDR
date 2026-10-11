"""fig_U_reduction — design validation: ELDR locality routing reduces
the per-decode-step active-expert union U vs RR.

Single panel (U vs B), paper convention: figstyle.apply(), DejaVu Sans
16/14, WIDTH=7.4in, BLUES palette, boxed centered legend above, 600dpi.

Live runtime measurement collected via ELDR_COUNT_UNIQUE on every decoder
of the qwen3-30b-a3b × task × 8P16D pipeline (paper's main eval topology).

Inputs: data/u_dumps/{rr,eldr}/ per-worker JSON measurements.
"""
import glob, json, os
import numpy as np
import matplotlib.pyplot as plt
from figstyle import apply, save, WIDTH, legend_top, DATA, OUTPUT, RR_COLOR, ELDR_COLOR

OUT = OUTPUT / "06_evaluation"
# Original per-worker measurements collected via ELDR_COUNT_UNIQUE.
_DATA = str(DATA / "u_dumps")
DIR_RR = os.path.join(_DATA, "rr")
DIR_EL = os.path.join(_DATA, "eldr")


def aggregate(d):
    accum = {}                          # B -> [sum, n]
    for f in glob.glob(f"{d}/*.json"):
        j = json.load(open(f))
        for k, (s, n) in j.items():
            B = int(k)
            cur = accum.setdefault(B, [0.0, 0])
            cur[0] += s; cur[1] += n
    Bs = sorted(accum.keys())
    means = np.array([accum[b][0] / accum[b][1] if accum[b][1] else np.nan
                      for b in Bs])
    counts = np.array([accum[b][1] for b in Bs])
    return np.array(Bs), means, counts


def main():
    Br, Mr, Cr = aggregate(DIR_RR)
    Be, Me, Ce = aggregate(DIR_EL)
    print(f"RR   B-range {Br.min()}..{Br.max()}  "
          f"total_steps={Cr.sum()}  worker_jsons={len(glob.glob(f'{DIR_RR}/*.json'))}")
    print(f"ELDR B-range {Be.min()}..{Be.max()}  "
          f"total_steps={Ce.sum()}  worker_jsons={len(glob.glob(f'{DIR_EL}/*.json'))}")

    # Common B-buckets with ≥20 steps per cell (stable mean).
    common = sorted(set(Br) & set(Be))
    sel = [b for b in common
           if Cr[list(Br).index(b)] >= 20 and Ce[list(Be).index(b)] >= 20]
    rr_vals = np.array([Mr[list(Br).index(b)] for b in sel])
    el_vals = np.array([Me[list(Be).index(b)] for b in sel])
    sel = np.array(sel)
    red = 100 * (rr_vals - el_vals) / rr_vals
    print(f"Common B-buckets (≥20 steps each): {len(sel)}  "
          f"mean U reduction = {float(np.mean(red)):.2f}%")

    # --- Single panel, wide-and-flat ---
    baseline_color, eldr_color = RR_COLOR, ELDR_COLOR

    apply()
    # Author at full WIDTH with canonical 16/14 fonts — LaTeX
    # \includegraphics[width=\columnwidth] downscales by ~0.46 to render
    # at ~7.4pt visible (the §sig_transform convention).
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, 2.8))

    ax.plot(sel, rr_vals, color=baseline_color, lw=2, ls="--", label="RR")
    ax.plot(sel, el_vals, color=eldr_color, lw=2, label="ELDR")

    ax.set_xlabel("Decode batch size", fontsize=15)
    ax.set_ylabel("Mean active experts", fontsize=15)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, alpha=0.30, lw=0.5)
    ymax = max(rr_vals.max(), el_vals.max())
    ax.set_ylim(0, ymax * 1.10)
    ax.set_xlim(0, sel.max() + 2)
    for sp in ax.spines.values(): sp.set_linewidth(0.7)

    handles = [
        plt.Line2D([], [], color=baseline_color, lw=2, ls="--", label="RR"),
        plt.Line2D([], [], color=eldr_color, lw=2, label="ELDR"),
    ]
    legend_top(ax, ncol=2, handles=handles, labels=["RR", "ELDR"], y=1.02)

    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.10, right=0.99)
    save(fig, f"{OUT}/fig_expert_reduction")
    plt.close(fig)


if __name__ == "__main__":
    main()
