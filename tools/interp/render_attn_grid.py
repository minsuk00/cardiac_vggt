"""Render E12 grid-query attention (figs/interp/gallery/grid_<sub>_L<l>.png).

Left: the query slice with every query point. Right: one tile per query point, arranged like the points on
the slice, showing that point's attention over the REFERENCE slice at global layer l (mean of 16 heads,
one shared colour scale). Cyan + = the query's own (x,y); white x = attention centroid on the reference.
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


def render(r, layer, fn):
    Q = r["queries"]; imgs = r["images"].astype(np.float32)
    qs = Q[0][1]
    rows = sorted({q[3] for q in Q}); cols = sorted({q[4] for q in Q})
    maps = {q: r["maps"][(layer, q[1], q[3], q[4])].astype(np.float32).mean(0)[0] for q in Q}   # attention on ref slot
    vmax = np.percentile(np.stack([m[5:] for m in maps.values()]), 99.9)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    cy, cx = (r0 + r1 + 1) / 2 * SC, (c0 + c1 + 1) / 2 * SC
    h = max(r1 - r0, c1 - c0) * SC / 2 + 35
    nR, nC = len(rows), len(cols)
    t = 0.62
    fig = plt.figure(figsize=(2.3 + nC * t, max(2.3, nR * t) + 0.4))
    W, Hh = fig.get_size_inches()
    a0 = fig.add_axes([0.02 / W * W / W, 0.4 / Hh, 2.1 / W, 2.1 / Hh])
    a0.imshow(imgs[qs], cmap="gray", vmax=np.percentile(imgs[qs], 99.5))
    for q in Q:
        a0.plot((q[4] + .5) * SC, (q[3] + .5) * SC, "+", color="#3fd0ff", ms=4, mew=1)
    a0.set_xlim(cx - h, cx + h); a0.set_ylim(cy + h, cy - h); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f"query slice (z{r['slot_z'][qs]}): {len(Q)} points", fontsize=7)
    dists, shares = [], []
    for q in Q:
        i, j = rows.index(q[3]), cols.index(q[4])
        ax = fig.add_axes([(2.25 + j * t) / W, (Hh - 0.3 - (i + 1) * t) / Hh, t * 0.97 / W, t * 0.97 / Hh])
        m = maps[q][5:].reshape(G, G)
        ax.imshow(imgs[0], cmap="gray", vmax=np.percentile(imgs[0], 99.5))
        ax.imshow(zoom(m, 256 / G, order=1), cmap="inferno", alpha=.65, vmin=0, vmax=vmax)
        yy, xx = np.mgrid[0:G, 0:G]
        my, mx = (m * yy).sum() / m.sum(), (m * xx).sum() / m.sum()
        ax.plot((q[4] + .5) * SC, (q[3] + .5) * SC, "+", color="#3fd0ff", ms=4, mew=1)
        ax.plot((mx + .5) * SC, (my + .5) * SC, "x", color="w", ms=3.5, mew=1)
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h); ax.set_xticks([]); ax.set_yticks([])
        ax.text(.03, .03, f"{100 * maps[q].sum():.0f}%", transform=ax.transAxes, color="w", fontsize=4.5)
        dists.append(np.hypot(my - q[3], mx - q[4]) * 14 * 256 / 518 * 1.4)
        shares.append(maps[q].sum())
    fig.text(2.25 / W, (Hh - 0.12) / Hh, f"s{r['idx']:03d}, global layer {layer}: each tile = that point's attention on the REFERENCE "
             f"slice (shared scale). + = point's own (x,y), x = attention centroid. median share {100*np.median(shares):.0f}%, "
             f"median |centroid − own (x,y)| {np.median(dists):.1f} mm", fontsize=6.5, va="top")
    fig.savefig(fn, dpi=170); plt.close(fig)
    return np.median(shares), np.median(dists)


if __name__ == "__main__":
    layers = [int(a) for a in sys.argv[1:]] or [3, 5]
    for f in sorted(glob.glob(os.path.join(ROOT, "scratch/interp/e12_grid/s*.pkl"))):
        r = pickle.load(open(f, "rb"))
        for l in layers:
            s, d = render(r, l, f"{OUT}/grid_s{r['idx']:03d}_L{l}.png")
            print(f"s{r['idx']:03d} L{l}: median share to ref {100*s:.0f}%  median centroid offset {d:.1f} mm")
