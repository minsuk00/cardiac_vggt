#!/usr/bin/env python
"""Naive stack of a real-time (RT) MIITT recording at candidate frames: frame f of EVERY slice,
stacked with no correction — the input to pick for the stacked-vs-ours misalignment figure.

Each slice is filmed separately (no ECG / breathing sync), so frame f of different slices sits at
a different breathing + cardiac state. Rows: SAX at the LV-centre plane | x-z cut | y-z cut
(through the LV centre, base at top, z nearest-stretched to physical size). One column per f.
Same canonical normalisation as the model input (run_vggt_rt.build_rt_bundle).

    PYTHONPATH=training:. python tools/render_rt_stacked_frames.py \
        --subjects MIITT_Patient_2024Jan04_Cardiomyopathy_AFib --out <dir>
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_vggt_rt as rt                                  # noqa: E402
from render_rt_orthoviews import lv_centre                # noqa: E402

HALF = 45          # crop half-width, px (1.4 mm)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=list(range(0, 180, 15)))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for subj in args.subjects:
        sd = os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax")
        b = rt.build_rt_bundle(os.path.join(sd, "4d_recon.nii.gz"))      # (F, D, H, W)
        x, y, zc = lv_centre(os.path.join(sd, "heart_seg.nii.gz"))
        zs = 10.0 / rt.rv.INPLANE_MM
        vmax = float(np.percentile(b[::10], 99.5))
        fig, ax = plt.subplots(3, len(args.frames), figsize=(1.9 * len(args.frames), 6.2), squeeze=False)
        for j, f in enumerate(args.frames):
            v = b[f]                                                     # (D, H, W)
            ya, yb, xa, xb = y - HALF, y + HALF, x - HALF, x + HALF
            for i, (img, asp) in enumerate([(v[zc, ya:yb, xa:xb], 1),
                                             (v[::-1, y, xa:xb], zs),    # x-z, base at top
                                             (v[::-1, ya:yb, x], zs)]):  # y-z, base at top
                ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=asp, interpolation="nearest")
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            ax[0, j].set_title(f"frame {f}\n({f * 25} ms)", fontsize=9)
        for i, lab in enumerate([f"SAX z{zc}", "x-z cut", "y-z cut"]):
            ax[i, 0].set_ylabel(lab)
        fig.suptitle(f"{subj}: frame f of every slice, naively stacked (no correction)", fontsize=10)
        plt.tight_layout()
        out = os.path.join(args.out, f"{subj}.png")
        plt.savefig(out, dpi=110); plt.close(fig)
        print(f"{subj}: D={b.shape[1]} LV centre (x={x}, y={y}, z={zc}) -> {out}")


if __name__ == "__main__":
    main()
