#!/usr/bin/env python
"""M-mode (kymograph) view of a real-time 4D recon (run_vggt_rt.py output): a horizontal line
through the LV of one SAX plane, stacked over the query frames (time runs left to right).

Rows: the reference plane's raw RT (the real beats that set the target time) | for each other
plane in --planes: model input (that plane's ONE fixed frame, so it is flat over time) and ours
(should beat in step with the reference). Needs the companion-frame json written for this run.

    PYTHONPATH=training:. python tools/render_rt_mmode.py --rt-input <subj>/rt_input.nii.gz \
        --recon <subj>/<arm>/recon_rt.nii.gz --frames-json slot_frames.json \
        --seg <RT heart_seg.nii.gz> --planes 3 8 --out mmode.png
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
sys.path.insert(0, os.path.join(ROOT, "tools"))
from render_rt_orthoviews import lv_centre                # noqa: E402

HALF = 45          # line half-length, px (1.4 mm)
FRAME_MS = 25.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rt-input", required=True)
    ap.add_argument("--recon", required=True)
    ap.add_argument("--frames-json", required=True, help="{ref_z, frame_of_z} of this run")
    ap.add_argument("--seg", required=True)
    ap.add_argument("--planes", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = np.asarray(nib.load(args.rt_input).dataobj, np.float32)       # (X, Y, Z, F)
    ours = np.asarray(nib.load(args.recon).dataobj, np.float32)
    sf = json.load(open(args.frames_json))
    ref = sf["ref_z"]
    x, y, _ = lv_centre(args.seg)
    xs = slice(x - HALF, x + HALF)
    F = raw.shape[3]
    vmax = float(np.percentile(raw[..., ::10], 99.5))

    def line(v, z):                          # (space, time): line at row y of plane z
        return v[xs, y, z, :]

    rows = [(f"z{ref} reference\nraw RT (real beats)", line(raw, ref))]
    for z in args.planes:
        f = sf["frame_of_z"][str(z)]
        rows.append((f"z{z} model input\n(frame {f}, frozen)", np.repeat(raw[xs, y, z, f][:, None], F, 1)))
        rows.append((f"z{z} ours", line(ours, z)))

    fig, ax = plt.subplots(len(rows), 1, figsize=(10, 1.25 * len(rows)), sharex=True)
    ext = [0, F * FRAME_MS / 1000, 2 * HALF * 1.4, 0]
    for a, (lab, img) in zip(ax, rows):
        a.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect="auto", extent=ext)
        a.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=8)
        a.set_yticks([])
    ax[-1].set_xlabel("time (s)")
    fig.suptitle(f"M-mode through the LV (row y={y}), {F} frames x {FRAME_MS:.0f} ms", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
