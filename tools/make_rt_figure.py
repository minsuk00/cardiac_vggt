#!/usr/bin/env python
"""Paper figure: real-time free-breathing MIITT, naive stack vs ours.

One row per subject. Left: orthogonal LAX cuts (x-z, y-z) through the LV centre of the naively
stacked input (frame f of every slice) and of ours at the same f (tools/run_vggt_rt_stacked.py
outputs). Right: LV volume over the 4.5 s recording, naive stack vs ours (tools/rt_lv_curve_2d.py).

    PYTHONPATH=training:. python tools/make_rt_figure.py --root scratch/temp/miitt_afib_rt \
        --row MIITT_Volunteer1:20:volunteer1:"Volunteer (sinus rhythm)" \
        --row MIITT_Patient_2024Jan04_Cardiomyopathy_AFib:170:afib:"Patient (atrial fibrillation)" \
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

FRAME_MS = 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="dir holding stacked_vs_ours_180/ and lvseg2d/")
    ap.add_argument("--row", nargs="+", required=True, help="SUBJECT:frame:curve_name:label")
    ap.add_argument("--out", required=True, help="output path without extension (.png + .pdf)")
    args = ap.parse_args()
    curves = json.load(open(os.path.join(args.root, "lvseg2d", "lv_curves.json")))
    noref = os.path.join(args.root, "lvseg2d", "lv_curves_noref.json")   # ours without the reference slab
    if os.path.exists(noref):
        curves.update(json.load(open(noref)))
    load = lambda p: np.asarray(nib.load(p).dataobj, np.float32)[..., 0]          # (X, Y, Z)

    # Per subject: two grid rows (LAX view 1 on top, LAX view 2 below); columns = naive stack | ours |
    # spacer | LV curve (spanning both LAX rows).
    fig = plt.figure(figsize=(12.5, 3.9 * len(args.row)))
    outer = fig.add_gridspec(len(args.row), 1, hspace=0.18)
    for i, spec in enumerate(args.row):
        gs = outer[i].subgridspec(2, 4, width_ratios=[1, 1, 0.3, 4.0], wspace=0.06, hspace=0.12)
        subj, f, cname, label = spec.split(":", 3)
        f = int(f)
        fd = os.path.join(args.root, "stacked_vs_ours_180", subj, f"frame_{f:03d}")
        inp, ours = load(os.path.join(fd, "input.nii.gz")), load(os.path.join(fd, "ours.nii.gz"))
        x, y, _ = lv_centre(os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/heart_seg.nii.gz"))
        vmax = float(np.percentile(inp, 99.5))
        zs = float(nib.load(os.path.join(fd, "input.nii.gz")).header.get_zooms()[2]) / rt.rv.INPLANE_MM
        for r, deg in enumerate((0, 90)):
            for j, (v, head) in enumerate(((inp, "Naive stack"), (ours, "Ours"))):
                a = fig.add_subplot(gs[r, j])
                a.imshow(lax_cut(v, x, y, deg), cmap="gray", vmin=0, vmax=vmax, aspect=zs, interpolation="nearest")
                a.set_xticks([]); a.set_yticks([])
                if i == 0 and r == 0:
                    a.set_title(head, fontsize=11)
                if j == 0:
                    a.set_ylabel(f"LAX view {r + 1}", fontsize=9)
                    if r == 0:
                        a.annotate(label, xy=(-0.28, 0.0), xycoords="axes fraction", rotation=90,
                                   ha="center", va="center", fontsize=10)
        a = fig.add_subplot(gs[:, 3])
        for kind, style in (("naive", dict(color="0.55", lw=1.2, label="Naive stack")),
                            ("ours", dict(color="C3", lw=1.8, label="Ours")),
                            ("ours_noref", dict(color="C3", lw=1.4, ls="--", label="Ours, reference slice excluded"))):
            v = curves.get(f"{cname}_{kind}")
            if v is None:
                continue
            a.plot(np.arange(len(v)) * FRAME_MS / 1000, v, **style)
        a.set_ylabel("LV volume (mL)", fontsize=9)
        a.set_xlim(0, 180 * FRAME_MS / 1000)
        a.tick_params(labelsize=8)
        if i == 0:
            a.set_title("LV volume over time", fontsize=10, pad=22)
            a.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
        if i == len(args.row) - 1:
            a.set_xlabel("time (s)", fontsize=9)
        else:
            a.tick_params(labelbottom=False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
        print(f"-> {args.out}.{ext}")


if __name__ == "__main__":
    main()
