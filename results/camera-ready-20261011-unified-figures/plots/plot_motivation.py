"""Original heatmap/scatter layout, using compact plot data or paired captures."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("captures", type=Path, nargs="?", default=fs.DATA / "motivation.json",
                    help="Compact plot JSON, original capture directory, or fresh run root")
parser.add_argument("--fresh", action="store_true", help="Read fresh inputs/ and results/ NPZ captures")
parser.add_argument("--output", type=Path, default=fs.OUTPUT, help="Figure root; use a separate path for previews")
parser.add_argument("--save-data", type=Path, help="Save compact plot inputs without request text or raw captures")
args = parser.parse_args()
fs.apply()
E = 128
MODELS = [("qwen", "Qwen3-30B-A3B"), ("gptoss", "GPT-OSS-120B"),
          ("gemma", "Gemma-4-26B-A4B")]
# Reject the later calibration captures that reused these filenames.
ORIGINAL_BYTES = {
    "task_unbal_qwen.bin": 572809764, "task_unbal_gptoss.bin": 429974160,
    "task_unbal_gemma.bin": 357997068, "wildchat_qwen.bin": 934066836,
    "wildchat_gptoss.bin": 700607124, "wildchat_gemma.bin": 583877268,
}
if args.captures.is_file() and args.fresh:
    parser.error("--fresh requires a capture directory, not compact plot JSON")
if not args.fresh and not args.captures.is_file():
    for name, size in ORIGINAL_BYTES.items():
        path = args.captures / name
        if not path.is_file() or path.stat().st_size != size:
            parser.error(f"Original May 29 capture missing or replaced: {path} ({size} bytes)")

capture_metadata = {}


def read_capture(name, model):
    if args.fresh:
        workload = {"task_unbal": "task", "wildchat": "language"}[name]
        raw = (args.captures / "inputs" / f"{workload}.json").read_bytes()
        pairs = json.loads(raw)
        labels = np.asarray([label for _, label in pairs])
        ids = np.asarray([hashlib.sha256(prompt.encode()).hexdigest() for prompt, _ in pairs])
        path = args.captures / "results" / f"{model}-{workload}-full.npz"
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if (metadata["model"] != model
                    or metadata["training_sha256"] != hashlib.sha256(raw).hexdigest()
                    or not np.array_equal(data["request_ids"], ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError(f"Capture/input request alignment mismatch: {path}")
            pre, dec = data["prefill_counts"], data["decode_counts"]
        layers = {"qwen": 48, "gptoss": 36, "gemma": 30}[model]
        if (pre.shape != dec.shape or pre.shape != (len(pairs), layers, E)
                or any(not np.issubdtype(a.dtype, np.integer) or np.any(a < 0)
                       for a in (pre, dec))):
            raise ValueError(f"Invalid paired expert counts: {path}")
        capture_metadata[path.name] = dict(metadata, requests=len(pairs))
        return pre.astype(np.float32), dec.astype(np.float32), labels, layers
    raw = (args.captures / f"{name}_{model}.bin").read_bytes()
    pc, pp, dc = struct.unpack_from("<III", raw)
    layers = pc // (E * 2)
    offset, pre, dec = 0, [], []
    while offset < len(raw):
        sizes = struct.unpack_from("<III", raw, offset)
        assert sizes == (layers * E * 2, layers * E * 4, layers * E * 2)
        offset += 12
        pre.append(np.frombuffer(raw, "<i2", layers * E, offset).reshape(layers, E))
        offset += pc + pp
        dec.append(np.frombuffer(raw, "<i2", layers * E, offset).reshape(layers, E))
        offset += dc
    counts = json.loads((args.captures / f"{name}_counts_{model}.json").read_text())
    labels = [label for label, count in counts for _ in range(count)]
    n = min(len(pre), len(labels))
    return (np.asarray(pre[:n], np.float32), np.asarray(dec[:n], np.float32),
            np.asarray(labels[:n]), layers)

def best_layer(dec,labs,names,L):
    best,bl=9,0
    for l in range(L):
        P=np.stack([(lambda a:a/max(a.sum(),1))(dec[labs==c,l].sum(0)) for c in names])
        Pn=P/np.clip(np.linalg.norm(P,axis=1,keepdims=True),1e-9,None); S=Pn@Pn.T
        od=S[~np.eye(len(names),dtype=bool)].mean()
        if od<best: best,bl=od,l
    return bl

def enrich_at(dec,labs,names,l):
    P=np.stack([(lambda a:a/max(a.sum(),1))(dec[labs==c,l].sum(0)) for c in names])
    Enr=100*(np.clip(P,1e-9,None)/np.clip(P.mean(0,keepdims=True),1e-9,None)-1)
    order=np.argsort(Enr.argmax(0)*1e6-Enr.max(0))
    return Enr[:,order], Enr.argmax(0)[order]   # sorted enrichment + dominant-domain per column


LANG_PRIO = ["English", "Chinese"]
TASK_ORDER = ["code", "math", "medical", "legal"]
TPRETTY = dict(zip(TASK_ORDER, ["Code", "Math", "Medical", "Legal"]))
if args.captures.is_file():
    prepared = json.loads(args.captures.read_text())
else:
    captured = {name: {m: read_capture(name, m) for m, _ in MODELS}
                for name in ("task_unbal", "wildchat")}
    task = {m: (d, labs, layers) for m, (_, d, labs, layers) in captured["task_unbal"].items()}
    lang = {m: (d, labs, layers) for m, (_, d, labs, layers) in captured["wildchat"].items()}
    lc = set.intersection(*[{c for c in set(lg) if (lg == c).sum() >= 80}
                           for _, lg, _ in lang.values()])
    lfreq = {c: sum(int((lang[m][1] == c).sum()) for m, _ in MODELS) for c in lc}
    lnames = ([c for c in LANG_PRIO if c in lc]
              + sorted(lc - set(LANG_PRIO), key=lambda c: -lfreq[c]))[:4]
    tnames = [c for c in TASK_ORDER if all((task[m][1] == c).sum() >= 50 for m, _ in MODELS)]
    prepared = dict(
        source="fresh capture" if args.fresh else "original May 29 capture",
        capture_metadata=capture_metadata,
        heatmap_task_domains=tnames,
        heatmap_languages=lnames,
        heatmap_layers={}, heatmap_enrichment={}, scatter={},
    )
    for m, _ in MODELS:
        dt, lt, Lt = task[m]; dl, ll, Ll = lang[m]
        bl_t = best_layer(dt, lt, tnames, Lt); bl_l = best_layer(dl, ll, lnames, Ll)
        et, _ = enrich_at(dt, lt, tnames, bl_t); el, _ = enrich_at(dl, ll, lnames, bl_l)
        prepared["heatmap_layers"][m] = dict(task=int(bl_t), language=int(bl_l))
        prepared["heatmap_enrichment"][m] = dict(task=et.tolist(), language=el.tolist())
        xs, ys = [], []
        for prefix in ("task_unbal", "wildchat"):
            pre, dec, labs, _ = captured[prefix][m]
            for d in [c for c in dict.fromkeys(labs) if (labs == c).sum() >= 80]:
                pe = pre[labs == d].sum(0).sum(0); de = dec[labs == d].sum(0).sum(0)
                xs += list(pe / pe.sum()); ys += list(de / de.sum())
        prepared["scatter"][m] = dict(prefill=np.asarray(xs).tolist(), decode=np.asarray(ys).tolist())

if args.save_data:
    args.save_data.write_text(json.dumps(prepared, indent=2) + "\n")
tnames = prepared["heatmap_task_domains"]
lnames = prepared["heatmap_languages"]

fig,axes=plt.subplots(2,3,figsize=(7.16,2.5))
im=None
for j,(m,title) in enumerate(MODELS):
    bl_t = prepared["heatmap_layers"][m]["task"]
    bl_l = prepared["heatmap_layers"][m]["language"]
    et = np.asarray(prepared["heatmap_enrichment"][m]["task"], dtype=np.float32)
    el = np.asarray(prepared["heatmap_enrichment"][m]["language"], dtype=np.float32)
    im=axes[0,j].imshow(et,aspect="auto",cmap="Blues",vmin=0,vmax=250,interpolation="nearest")
    axes[0,j].set_title(f"{title}",fontsize=9,pad=4)
    axes[1,j].imshow(el,aspect="auto",cmap="Blues",vmin=0,vmax=250,interpolation="nearest")
    for i in (0,1):
        axes[i,j].set_xticks([])
        if j>0: axes[i,j].set_yticks([])
    axes[0,j].set_xlabel(f"layer {bl_t}",fontsize=7); axes[1,j].set_xlabel(f"layer {bl_l}",fontsize=7)
axes[0,0].set_yticks(range(len(tnames))); axes[0,0].set_yticklabels([TPRETTY.get(c,c) for c in tnames],fontsize=7)
axes[1,0].set_yticks(range(len(lnames))); axes[1,0].set_yticklabels(lnames,fontsize=7)
axes[0,0].set_ylabel("Task",fontsize=9,labelpad=4); axes[1,0].set_ylabel("Language",fontsize=9,labelpad=4)
for _a in (axes[0,0],axes[1,0]): _a.yaxis.set_label_coords(-0.34,0.5)  # align Task/Language vertically
fig.supxlabel("Expert (sorted by dominant domain)",y=0.02,fontsize=9)
fig.subplots_adjust(left=0.095,right=0.84,top=0.91,bottom=0.17,wspace=0.09,hspace=0.38)
cax=fig.add_axes([0.875,0.135,0.016,0.795]); cb=fig.colorbar(im,cax=cax)
cb.set_label("Activation vs. average",fontsize=9); cb.ax.tick_params(labelsize=7)
cb.ax.yaxis.set_major_formatter(lambda x,_:f"{x:+.0f}%")
fs.save(fig, args.output / "03_motivation/domain_2x3_layer")
plt.close(fig)

summary = {key: value for key, value in prepared.items()
           if key not in ("heatmap_enrichment", "scatter")}
summary["pooled_correlation"] = {}
fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.7), sharex=True, sharey=True)
for ax, (m, name) in zip(axes, MODELS):
    xs = np.asarray(prepared["scatter"][m]["prefill"], dtype=np.float32)
    ys = np.asarray(prepared["scatter"][m]["decode"], dtype=np.float32)
    r = np.corrcoef(xs, ys)[0, 1]
    summary["pooled_correlation"][m] = dict(r=float(r), points=len(xs))
    ax.scatter(xs, ys, color="#1f4e8c", s=6, alpha=0.4, linewidths=0)
    lim = max(xs.max(), ys.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.6)
    ax.set_title(f"{name}\n($r={r:.2f}$)", fontsize=16)
    ax.set_xlabel("Prefill activation", fontsize=15); ax.tick_params(labelsize=14)
axes[0].set_ylabel("Decode activation", fontsize=15)
fig.subplots_adjust(left=0.10, right=0.99, top=0.83, bottom=0.20, wspace=0.08)
fs.save(fig, args.output / "03_motivation/pd_scatter")
plt.close(fig)
with (args.output / "03_motivation/summary.json").open("w") as stream:
    json.dump(summary, stream, indent=2)
print("Pooled r:", {model: row["r"] for model, row in summary["pooled_correlation"].items()})
