"""Figures for e17_dinov2_frozen_crossvendor.py (docs/128 §2):
(1) temp/dinov2_probe/summary.png  — per-pair Dice dots + mean bars, 3 conditions x 3 methods
(2) temp/dinov2_probe/examples.png — worst / median / best pair by cross-subject DINOv2 RV Dice
Run from the repo root after e17: micromamba run -n svr python tools/interp/e17_viz.py
"""
import os
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "temp/dinov2_probe")
df = pd.read_csv(f"{OUT}/results.csv")
maps = torch.load(f"{OUT}/maps.pt")
METH = {"DINOv2": "#2a78d6", "raw pixels": "#eb6834", "same location": "#1baf7a"}
COND = {"cross": "Cross-subject, cross-vendor\n(B = LV-volume-matched M&Ms)", "ctrl": "Same slice, contrast remapped", "ctrl_rot": "Same slice, contrast remapped\n+ rotated 90°"}
CLS = ["LV", "MYO", "RV"]

# ---- 1) summary: every pair as a dot, mean as a bar
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, (cond, title) in zip(axs, COND.items()):
    d = df[df.cond == cond]
    for ci, c in enumerate(CLS):
        for mi, (m, col) in enumerate(METH.items()):
            v = d[d.method == m][c].values; x = ci + (mi - 1) * 0.27
            ax.bar(x, v.mean(), 0.22, color=col, alpha=.25, edgecolor=col, linewidth=1.5)
            ax.scatter(x + rng.uniform(-.07, .07, len(v)), v, s=10, color=col, edgecolors="white", linewidths=.4, zorder=3)
            ax.text(x, 1.03, f"{v.mean():.2f}", ha="center", fontsize=8, color="#333")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLS, fontsize=11); ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.1); ax.grid(axis="y", color="#ddd", lw=.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("Dice of transferred patch labels vs A's own labels")
axs[0].legend(handles=[Patch(color=c, label=m) for m, c in METH.items()], loc="lower left", fontsize=9, frameon=False)
fig.suptitle(f"Does frozen DINOv2 match heart anatomy across contrast?  ({df.pair.nunique()} pairs; dot = one pair, bar = mean)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT}/summary.png", dpi=110, bbox_inches="tight")

# ---- 2) gallery: worst / median / best pairs by cross-subject DINOv2 RV Dice
rv = df[(df.cond == "cross") & (df.method == "DINOv2")].set_index("pair")["RV"].sort_values()
pick = {"worst": rv.index[0], "median": rv.index[len(rv) // 2], "best": rv.index[-1]}
lab = ListedColormap([(0, 0, 0, 0), (.89, .29, .28, .65), (.11, .69, .48, .65), (.16, .47, .84, .65)])  # bg transparent, LV red, MYO green, RV blue
cols = ["A (Siemens)", "B (other vendor)", "A: own labels", "B→A: DINOv2", "B→A: same location", "A′ rotated", "A′rot→A: DINOv2"]
fig, ax = plt.subplots(len(pick), len(cols), figsize=(2.5 * len(cols), 2.75 * len(pick)))
def dice_txt(k, cond, m): return " ".join(f"{c} {df[(df.pair == k) & (df.cond == cond) & (df.method == m)][c].item():.2f}" for c in CLS)
for r, (tag, k) in enumerate(pick.items()):
    m = maps[k]; A = m["a"].numpy()
    def over(a, lbl, title=None):
        a.imshow(A, cmap="gray", extent=(0, 37, 37, 0)); a.imshow(lbl.reshape(37, 37).numpy(), cmap=lab, vmin=0, vmax=3, extent=(0, 37, 37, 0), interpolation="nearest")
        if title: a.set_xlabel(title, fontsize=7.5)
    ax[r, 0].imshow(A, cmap="gray"); ax[r, 0].set_ylabel(f"{tag} RV\n{m['names'][0]}", fontsize=9)
    ax[r, 1].imshow(m["b"].numpy(), cmap="gray"); ax[r, 1].set_xlabel(f"{m['names'][1]} ({m['names'][2]})", fontsize=7.5)
    over(ax[r, 2], m["la"])
    over(ax[r, 3], m["cross/DINOv2"], dice_txt(k, "cross", "DINOv2"))
    over(ax[r, 4], m["cross/same location"], dice_txt(k, "cross", "same location"))
    ax[r, 5].imshow(m["r"].numpy(), cmap="gray")
    over(ax[r, 6], m["ctrl_rot/DINOv2"], dice_txt(k, "ctrl_rot", "DINOv2"))
for c, t in enumerate(cols): ax[0, c].set_title(t, fontsize=10)
for a in ax.flat: a.set_xticks([]); a.set_yticks([])
fig.legend(handles=[Patch(color=lab(i), label=n) for i, n in [(1, "LV"), (2, "MYO"), (3, "RV")]], loc="lower center", ncol=3, frameon=False, fontsize=10)
fig.suptitle("Examples (labels painted on A's patches; Dice under each map). Rows = worst / median / best cross-subject DINOv2 RV Dice", fontsize=11)
fig.tight_layout(rect=(0, .03, 1, 1)); fig.savefig(f"{OUT}/examples.png", dpi=100, bbox_inches="tight")
print(f"saved {OUT}/summary.png, {OUT}/examples.png", pick)
