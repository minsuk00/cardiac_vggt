#!/usr/bin/env python
"""LAX candidate sheets (stacked input vs ours) from tools/run_vggt_rt_stacked.py outputs.

For each subject and each cut angle, one PNG: a grid of every saved frame, each cell = the stacked
input (top) over ours (bottom), cut through the LV centre at that in-plane angle (0 deg = x-z, 90 deg =
y-z), in-plane linear sampling, z nearest-stretched to physical size, base at top. Same contrast
for input and ours. No model run.

    PYTHONPATH=training:. python tools/render_rt_lax_candidates.py --dir <run_vggt_rt_stacked out> \
        --angles 0 45 90 135 --out <dir>
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_vggt_rt as rt                                  # noqa: E402
from render_rt_orthoviews import lv_centre                # noqa: E402

HALF = 45          # half-length of the cut, px (1.4 mm)


def lax_cut(vol, x, y, deg):
    """vol (X, Y, Z) -> (Z, 2*HALF) cut through (x, y) at in-plane angle deg, base at top."""
    s = np.arange(-HALF, HALF, dtype=np.float32)
    px, py = x + s * np.cos(np.deg2rad(deg)), y + s * np.sin(np.deg2rad(deg))
    rows = [map_coordinates(vol[:, :, z], [px, py], order=1) for z in range(vol.shape[2])]
    return np.stack(rows)[::-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    ap.add_argument("--angles", type=float, nargs="+", default=[0, 45, 90, 135])
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", nargs="+", default=None, help="default: every MIITT_* dir")
    ap.add_argument("--per-page", type=int, default=0, help="frames per PNG (0 = all in one)")
    ap.add_argument("--frame-cols", type=int, default=1, help="frames side by side per figure row")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    load = lambda p: np.asarray(nib.load(p).dataobj, np.float32)[..., 0]          # (X, Y, Z)

    for sdir in sorted(d for d in glob.glob(os.path.join(args.dir, "MIITT_*")) if os.path.isdir(d)):
        subj = os.path.basename(sdir)
        if args.subjects and subj not in args.subjects:
            continue
        x, y, _ = lv_centre(os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/heart_seg.nii.gz"))
        fdirs = sorted(glob.glob(os.path.join(sdir, "frame_*")))
        vols = [(int(os.path.basename(fd)[6:]), load(os.path.join(fd, "input.nii.gz")),
                 load(os.path.join(fd, "ours.nii.gz"))) for fd in fdirs]
        dz = float(nib.load(os.path.join(fdirs[0], "input.nii.gz")).header.get_zooms()[2])
        zs = dz / rt.rv.INPLANE_MM
        vmax = float(np.percentile(np.concatenate([v[1].ravel() for v in vols]), 99.5))
        # One row per frame: input at every angle, then ours at every angle, side by side.
        # Each frame = one group of panels: input at every angle, then ours at every angle.
        # --frame-cols groups per figure row; --per-page frames per PNG.
        cols = [("input", a) for a in args.angles] + [("ours", a) for a in args.angles]
        nc = len(cols)
        per = args.per_page or len(vols)
        for p0 in range(0, len(vols), per):
            page = vols[p0:p0 + per]
            nrow = int(np.ceil(len(page) / args.frame_cols))
            fig, ax = plt.subplots(nrow, nc * args.frame_cols, figsize=(nc * args.frame_cols * 1.9, nrow * 2.0),
                                   squeeze=False)
            for a in ax.ravel():
                a.axis("off")
            for i, (f, inp, ours) in enumerate(page):
                r, c0 = i // args.frame_cols, (i % args.frame_cols) * nc
                for j, (lab, deg) in enumerate(cols):
                    a = ax[r, c0 + j]
                    a.imshow(lax_cut(inp if lab == "input" else ours, x, y, deg), cmap="gray", vmin=0,
                             vmax=vmax, aspect=zs, interpolation="nearest")
                    cut = 'x-z' if deg == 0 else 'y-z' if deg == 90 else f'{deg:.0f}deg'
                    a.set_title(f"f{f} {lab} {cut}", fontsize=7, color="C0" if lab == "input" else "C3")
            fig.suptitle(f"{subj}: stacked input (blue) vs ours (red), orthogonal LAX cuts through the LV "
                         f"centre, base at top — frames {page[0][0]}-{page[-1][0]}", fontsize=11)
            plt.tight_layout(rect=(0, 0, 1, 1 - 0.4 / (nrow * 2.0)))
            out = os.path.join(args.out, f"{subj}_f{page[0][0]:03d}-{page[-1][0]:03d}.png"
                               if per < len(vols) else f"{subj}.png")
            plt.savefig(out, dpi=100); plt.close(fig)
            print(f"-> {out}")


if __name__ == "__main__":
    main()
