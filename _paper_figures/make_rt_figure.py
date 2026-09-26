#!/usr/bin/env python
"""Paper figure: real-time free-breathing MIITT, naive stack vs ours (half text width).

One block per subject. Left: one LAX cut (x-z at 0 deg = "view 1", y-z at 90 deg = "view 2") through the LV
centre of the naively stacked input (frame f of every slice, top) and of ours at the same f (below)
(tools/run_vggt_rt_stacked.py outputs). Right: LV volume over the 4.5 s recording, naive stack vs ours
(tools/rt_lv_curve_2d.py). Display window = p99.5 of the subject's two shown cuts (shared), then gamma.

    PYTHONPATH=training:. python _paper_figures/make_rt_figure.py --root scratch/temp/miitt_afib_rt \
        --row MIITT_Volunteer1:20:volunteer1:"Healthy volunteer":0 \
              MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:"AF patient":90 \
        --out scratch/temp/miitt_afib_rt/figure/rt_figure
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_vggt_rt as rt                                  # noqa: E402
from render_rt_orthoviews import lv_centre                # noqa: E402
from render_rt_lax_candidates import lax_cut              # noqa: E402
from render_runtime_accuracy import TEXTW, paper_rc       # noqa: E402

FRAME_MS = 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="dir holding stacked_vs_ours_180/ and lvseg2d/")
    ap.add_argument("--row", nargs="+", required=True, help="SUBJECT:frame:curve_name:label:lax_deg")
    ap.add_argument("--gamma", type=float, default=1.0, help="display gamma of the LAX cuts")
    ap.add_argument("--legend-vertical", action="store_true", help="stack the legend entries vertically")
    ap.add_argument("--out", required=True, help="output path without extension (.png + .pdf)")
    args = ap.parse_args()
    curves = json.load(open(os.path.join(args.root, "lvseg2d", "lv_curves.json")))
    load = lambda p: np.asarray(nib.load(p).dataobj, np.float32)[..., 0]          # (X, Y, Z)

    paper_rc()
    n = len(args.row)
    # manual layout in inches (constrained layout collapses the image column): per subject, the two cuts
    # stacked (naive over ours) | LV curve spanning both; height matches the reference-conditioning figure
    W, H = TEXTW / 2, 1.93
    BOT, TOP, GAP_BLOCK, GAP_IMG, IMG_X0, CURVE_X1 = 0.27, 0.03, 0.06, 0.02, 0.30, W - 0.04
    bh = (H - BOT - TOP - GAP_BLOCK * (n - 1)) / n
    ih = (bh - GAP_IMG) / 2
    fig = plt.figure(figsize=(W, H))
    box = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    curve_x0 = IMG_X0 + 1.05 * ih + 0.36
    for i, spec in enumerate(args.row):
        yb = H - TOP - (i + 1) * bh - i * GAP_BLOCK          # bottom of this subject's block
        subj, f, cname, label, deg = spec.split(":", 4)
        f, deg = int(f), float(deg)
        fd = os.path.join(args.root, "stacked_vs_ours_180", subj, f"frame_{f:03d}")
        inp, ours = load(os.path.join(fd, "input.nii.gz")), load(os.path.join(fd, "ours.nii.gz"))
        x, y, _ = lv_centre(os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/heart_seg.nii.gz"))
        zs = float(nib.load(os.path.join(fd, "input.nii.gz")).header.get_zooms()[2]) / rt.rv.INPLANE_MM
        cuts = [lax_cut(v, x, y, deg) for v in (inp, ours)]
        vmax = float(np.percentile(np.concatenate([c.ravel() for c in cuts]), 99.5))
        for j, (c, head) in enumerate(zip(cuts, ("Naive", "Ours"))):
            a = box(IMG_X0, yb + (1 - j) * (ih + GAP_IMG), ih * c.shape[1] / (c.shape[0] * zs), ih)
            a.imshow(np.clip(c / vmax, 0, 1) ** args.gamma, cmap="gray", vmin=0, vmax=1, aspect=zs,
                     interpolation="nearest")
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
            a.set_ylabel(head, fontsize=6.5, labelpad=1)
            if j == 0:   # subject name, rotated, left of both of its cuts
                fig.text(0.06 / W, (yb + bh / 2) / H, label, rotation=90, ha="center", va="center",
                         fontsize=6.5, fontweight="bold")
        a = box(curve_x0, yb, CURVE_X1 - curve_x0, bh)
        for kind, style in (("naive", dict(color="0.55", lw=0.6, label="Naive stack")),
                            ("ours", dict(color="C3", lw=0.8, label="Ours"))):
            v = curves.get(f"{cname}_{kind}")
            if v is None:
                continue
            a.plot(np.arange(len(v)) * FRAME_MS / 1000, v, **style)
        a.set_xlim(0, 180 * FRAME_MS / 1000)
        a.tick_params(labelsize=6, pad=1)
        a.set_ylabel("LV vol. (mL)", fontsize=6.5, labelpad=1)
        if i == n - 1:
            a.set_xlabel("Time (s)", fontsize=6.5, labelpad=1)
            # short legend box inside the bottom plot. Horizontal: y-range extended downward to leave a free
            # strip. Vertical (--legend-vertical): stacked in the lower-right corner, over part of the curves.
            if not args.legend_vertical:
                lo, hi = a.get_ylim()
                a.set_ylim(lo - 0.28 * (hi - lo), hi)
            leg = a.legend(loc="lower right", ncol=1 if args.legend_vertical else 2, fontsize=6, handlelength=1.4, columnspacing=0.8,
                           handletextpad=0.4, borderpad=0.3, borderaxespad=0.2, frameon=True, framealpha=0.6 if args.legend_vertical else 1,
                           fancybox=False, edgecolor="0.6")
            leg.get_frame().set_linewidth(0.4)
        else:
            a.tick_params(labelbottom=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"-> {args.out}.{ext}")


if __name__ == "__main__":
    main()
