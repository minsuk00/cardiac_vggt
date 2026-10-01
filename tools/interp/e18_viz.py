"""Figures for e18_aggregator_anatomy_probe.py (docs/128 §3):
(1) temp/aggft_probe/dice_vs_layer.png — Dice vs depth, every pair (thin) + mean (thick) + same-location baseline
(2) temp/aggft_probe/pairNN_layers.png — per saved pair: rows = DINOv2 / blocks 3, 11, 19, 23; col 1 = labels
    copied B->A by 1-NN; cols 2-4 = cos-sim of every B patch to A's LV / MYO / RV query patch, zoomed to the heart.
Run from the repo root after e18: micromamba run -n svr python tools/interp/e18_viz.py
"""
import glob, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "temp/aggft_probe")
G, LAYERS = 37, ["dino", "g3", "g11", "g19", "g23"]
LNAME = {"dino": "DINOv2 (frozen)", "g3": "block 3", "g11": "block 11", "g19": "block 19", "g23": "block 23"}

# ---- 1) Dice vs depth
df = pd.read_csv(f"{D}/results.csv"); df = df[df.slot == "ref"]
order = ["dino", "g0", "g3", "g7", "g11", "g15", "g19", "g23"]
x = np.arange(len(order))
fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, c, col in zip(axs, ["LV", "MYO", "RV"], ["#e34948", "#1baf7a", "#2a78d6"]):
    p = df[df.layer.isin(order)].pivot(index="pair", columns="layer", values=c)[order]
    for _, r in p.iterrows(): ax.plot(x, r.values, color=col, alpha=.18, lw=1)
    ax.plot(x, p.mean().values, color=col, lw=2.5, marker="o", label=f"mean of {len(p)} pairs")
    ax.axhline(df[df.layer == "same location"][c].mean(), color="#888", ls="--", lw=1.2, label="same-location baseline")
    ax.set_xticks(x); ax.set_xticklabels(["DINOv2"] + [f"blk {o[1:]}" for o in order[1:]], fontsize=9)
    ax.set_title(c, fontsize=12); ax.set_ylim(0, 1); ax.grid(axis="y", color="#ddd", lw=.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].set_ylabel("Dice of 1-NN label transfer B→A"); axs[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("Trained model (final518): anatomy in patch tokens vs depth — thin line = one cross-dataset pair, thick = mean (test split, mid slice, ED)", fontsize=11)
fig.tight_layout(); fig.savefig(f"{D}/dice_vs_layer.png", dpi=110, bbox_inches="tight")

# ---- 2) per-pair progression
lab_cmap = ListedColormap([(0, 0, 0, 0), (.89, .29, .28, .7), (.11, .69, .48, .7), (.16, .47, .84, .7)])

def outline(ax, lab):
    for c, col in [(3, "c"), (2, "lime"), (1, "r")]:  # LV last: its edge coincides with MYO's inner edge
        ax.contour(np.arange(G) + .5, np.arange(G) + .5, (lab.reshape(G, G) == c).numpy().astype(float), levels=[.5], colors=col, linewidths=.9)

def query(lab, c):  # patch of class c nearest to that class's centroid
    pts = torch.nonzero(lab.reshape(G, G) == c).float()
    q = pts[(pts - pts.mean(0)).norm(dim=1).argmin()]
    return int(q[0]) * G + int(q[1])

def dice(pred, gt, c): return (2 * ((pred == c) & (gt == c)).sum() / ((pred == c).sum() + (gt == c).sum()).clamp(min=1)).item()

def box(lab, pad=7):  # heart-centred square window in patch coords (x0, x1, y_bottom, y_top)
    yx = torch.nonzero(lab.reshape(G, G) > 0); c = yx.float().mean(0)
    h = max((yx.max(0).values - yx.min(0).values).max().item() / 2 + pad, 8)
    return (c[1] - h + .5, c[1] + h + .5, c[0] + h + .5, c[0] - h + .5)

def zoom(a, b): a.set_xlim(b[0], b[1]); a.set_ylim(b[2], b[3])

for path in sorted(glob.glob(f"{D}/pair*.pt")):
    d = torch.load(path); la, lb = d["la"], d["lb"]
    ia, ib = d["fa"]["_img"].float(), d["fb"]["_img"].float()
    qa = {c: query(la, c) for c in (1, 2, 3)}
    ba, bb = box(la), box(lb)
    fig, ax = plt.subplots(1 + len(LAYERS), 4, figsize=(12, 2.9 * (1 + len(LAYERS))))
    fig.suptitle(f"A = {d['A'].replace('mri_axial_', '')}   B = {d['B'].replace('mri_axial_', '')}\n"
                 "rows = depth in the trained model. Col 1: labels copied B→A by 1-NN (Dice below). "
                 "Cols 2-4: cosine sim of every B patch to A's ★ LV / MYO / RV patch (bright = similar). Zoomed to the heart.", fontsize=9)
    ax[0, 0].imshow(ia, cmap="gray", extent=(0, G, G, 0)); ax[0, 0].set_title("A (full slice)", fontsize=10)
    ax[0, 1].imshow(ia, cmap="gray", extent=(0, G, G, 0)); ax[0, 1].imshow(la.reshape(G, G), cmap=lab_cmap, vmin=0, vmax=3, extent=(0, G, G, 0), interpolation="nearest")
    ax[0, 1].set_title("A: true labels (answer key)", fontsize=10)
    for c, mk in [(1, "r"), (2, "lime"), (3, "c")]:
        ax[0, 1].plot(qa[c] % G + .5, qa[c] // G + .5, "*", color=mk, ms=15, mec="k")
    zoom(ax[0, 1], ba)
    ax[0, 2].imshow(ib, cmap="gray", extent=(0, G, G, 0)); ax[0, 2].set_title("B (full slice)", fontsize=10)
    ax[0, 3].imshow(ib, cmap="gray", extent=(0, G, G, 0)); ax[0, 3].imshow(lb.reshape(G, G), cmap=lab_cmap, vmin=0, vmax=3, extent=(0, G, G, 0), interpolation="nearest")
    ax[0, 3].set_title("B: true labels", fontsize=10); zoom(ax[0, 3], bb)
    for r, L in enumerate(LAYERS, 1):
        fa, fb = d["fa"][L].float(), d["fb"][L].float()
        m = torch.cat([fa, fb]).mean(0); na, nb = F.normalize(fa - m, dim=-1), F.normalize(fb - m, dim=-1)
        pred = lb[(na @ nb.T).argmax(1)]
        ax[r, 0].imshow(ia, cmap="gray", extent=(0, G, G, 0)); ax[r, 0].imshow(pred.reshape(G, G), cmap=lab_cmap, vmin=0, vmax=3, extent=(0, G, G, 0), interpolation="nearest")
        ax[r, 0].set_ylabel(LNAME[L], fontsize=11); zoom(ax[r, 0], ba)
        ax[r, 0].set_xlabel("Dice " + "  ".join(f"{n} {dice(pred, la, c):.2f}" for c, n in [(1, "LV"), (2, "MYO"), (3, "RV")]), fontsize=8)
        for j, (c, n) in enumerate([(1, "LV"), (2, "MYO"), (3, "RV")]):
            im = ax[r, j + 1].imshow((nb @ na[qa[c]]).reshape(G, G).numpy(), cmap="magma", vmin=-.2, vmax=1, extent=(0, G, G, 0), interpolation="nearest")
            outline(ax[r, j + 1], lb); zoom(ax[r, j + 1], bb)
            if r == 1: ax[r, j + 1].set_title(f"on B: sim to A's {n} ★", fontsize=10)
    for a in ax.flat: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[1:, 1:4], shrink=.4, label="cosine similarity")
    fig.legend(handles=[Patch(color=lab_cmap(i), label=n) for i, n in [(1, "LV"), (2, "MYO"), (3, "RV")]], loc="lower center", ncol=3, frameon=False)
    out = f"{D}/{os.path.basename(path)[:-3]}_layers.png"
    fig.savefig(out, dpi=85, bbox_inches="tight"); plt.close(fig); print("saved", out)
