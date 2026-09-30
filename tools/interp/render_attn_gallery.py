"""Render E12 attention gallery (exploration figures, figs/interp/gallery/).

atlas_<sub>_<tag>_<q>.png : one query point, all 24 global layers (rows) x all slices sorted by z (cols),
                            mean of 16 heads, each row on its own colour scale, % = share of attention.
mass_<sub>.png            : for every query point, a (24 layers x slices) matrix of attention share per slice.
"""
import glob
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figs/interp/gallery")
G, SC = 37, 14 * 256 / 518


def atlas(r, tag, qname, fn):
    q = next(q for q in r["queries"] if q[0] == tag and q[2] == qname)
    _, qs, _, qr, qc = q
    S, z, imgs = r["S"], r["slot_z"], r["images"].astype(np.float32)
    order = list(np.argsort(z))
    fig, axs = plt.subplots(24, S, figsize=(0.62 * S, 0.62 * 24), gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    for l in range(24):
        A = r["maps"][(l, qs, qr, qc)].astype(np.float32).mean(0)
        vmax = np.percentile(A[:, 5:], 99.8) + 1e-12
        for j, s in enumerate(order):
            ax = axs[l, j]
            ax.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
            ax.imshow(zoom(A[s, 5:].reshape(G, G), 256 / G, order=1), cmap="inferno", alpha=0.6, vmin=0, vmax=vmax)
            if s == qs:
                ax.plot((qc + .5) * SC, (qr + .5) * SC, "+", color="#3fd0ff", ms=5, mew=1.2)
            ax.set_xticks([]); ax.set_yticks([])
            if s == 0:
                for sp in ax.spines.values():
                    sp.set_color("#eb6834"); sp.set_linewidth(1.5)
            ax.text(.03, .04, f"{100 * A[s].sum():.0f}", transform=ax.transAxes, color="w", fontsize=5)
            if l == 0:
                ax.set_title(("REF " if s == 0 else "") + ("QUERY " if s == qs else "") + f"z{z[s]}", fontsize=6,
                             color="#eb6834" if s == 0 else ("#3fd0ff" if s == qs else "k"))
            if j == 0:
                ax.set_ylabel(f"L{l}", fontsize=6, rotation=0, labelpad=8)
    fig.suptitle(f"s{r['idx']:03d}  query '{qname}' on {tag} slice (z{z[qs]})  — each row own colour scale; number = % attention to that slice",
                 fontsize=8, y=0.995)
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)


def mass(r, fn):
    S, z = r["S"], r["slot_z"]
    order = list(np.argsort(z))
    Q = r["queries"]
    tags = [t for t in ("ref", "near", "far") if any(q[0] == t for q in Q)]
    names = ["lv", "edge_l", "edge_r", "body", "air"]
    fig, axs = plt.subplots(len(tags), len(names), figsize=(2.0 * len(names), 2.6 * len(tags)), squeeze=False)
    for i, t in enumerate(tags):
        for j, n in enumerate(names):
            ax = axs[i, j]
            q = next((q for q in Q if q[0] == t and q[2] == n), None)
            if q is None:
                ax.axis("off"); continue
            _, qs, _, qr, qc = q
            M = np.stack([r["maps"][(l, qs, qr, qc)].astype(np.float32).mean(0).sum(1) for l in range(24)])[:, order]
            ax.imshow(M * S, cmap="magma", vmin=0, vmax=5, aspect="auto")
            ax.set_xticks(range(S)); ax.set_xticklabels([("R" if s == 0 else ("Q" if s == qs else "")) for s in order], fontsize=6)
            ax.set_yticks([0, 3, 8, 12, 17, 23]); ax.tick_params(labelsize=6)
            ax.set_title(f"{t} slice, '{n}'", fontsize=7)
            if j == 0:
                ax.set_ylabel("global layer", fontsize=7)
    fig.suptitle(f"s{r['idx']:03d}: attention share per slice × fair share (1 = even; 5+ = bright). "
                 "Columns = slices ordered by z; R = reference, Q = query's own slice", fontsize=8)
    fig.savefig(fn, dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in sorted(glob.glob(os.path.join(ROOT, "scratch/interp/e12/s*.pkl"))):
        r = pickle.load(open(f, "rb"))
        sub = f"s{r['idx']:03d}"
        mass(r, f"{OUT}/mass_{sub}.png")
        for tag, qn in [("near", "lv"), ("near", "body"), ("ref", "lv")] + (sys.argv[1:] and [] or []):
            if any(q[0] == tag and q[2] == qn for q in r["queries"]):
                atlas(r, tag, qn, f"{OUT}/atlas_{sub}_{tag}_{qn}.png")
        print("rendered", sub)
