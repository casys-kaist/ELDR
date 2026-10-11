"""WildChat language distribution from the original plot's counts."""
import json
import matplotlib.pyplot as plt
import figstyle as fs

fs.apply()
counts = json.loads((fs.DATA / "wc_langdist.json").read_text())
total = sum(counts.values())
items = sorted(counts.items(), key=lambda x: -x[1])
labels = [label for label, _ in items[:6]] + ["Other"]
values = [100 * count / total for _, count in items[:6]]
values.append(100 * sum(count for _, count in items[6:]) / total)
fig, ax = plt.subplots(figsize=(fs.WIDTH, 2.4))
ys = range(len(labels))
ax.barh(list(ys), values, color=fs.BLUES[1], edgecolor="white", height=0.7)
for y, value in zip(ys, values):
    ax.text(value + 0.7, y, f"{value:.1f}%", va="center", fontsize=13)
ax.set_yticks(list(ys))
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlim(0, 52)
ax.set_xlabel("Share of WildChat requests (%)")
ax.tick_params(axis="y", length=0)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
fig.subplots_adjust(left=0.22, right=0.97, top=0.97, bottom=0.20)
fs.save(fig, fs.OUTPUT / "03_motivation/wc_langdist")
