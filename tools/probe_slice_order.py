"""Which anatomical end sits at z=0? Measure base-vs-apex ordering from the segmentations.

Provenance for docs/58 §10a. Nothing in the training pipeline checks this — `plan_reframe()` in
`tools/convert_to_sax_layout.py` decides the axis-2 flip from the **affine**, which is real only for
M&Ms (ACDC has sform=qform=0; CMRxRecon's recon writes the SimpleITK default). This probe decides it
from **anatomy** instead, which is the only trustworthy source when the header is fabricated.

Method: on the ED frame of `heart_seg.nii.gz`, count labeled voxels per z-plane and fit area vs z.
The LV tapers to a point at the apex and is widest near the base, so

    slope > 0  ->  area GROWS with z  ->  apex stored FIRST (z0 = apex)
    slope < 0  ->  area SHRINKS with z ->  base stored FIRST (z0 = base)

Why it matters (docs/58 §10a): `respiratory.py` applies a ONE-SIDED displacement
(`A.clamp_min(0.0)`, fixed sign convention) along array axis D and never checks what D means
anatomically. So the physical direction of the simulated breathing is set entirely by the storage
order — inferior (correct, heart follows the descending diaphragm on inspiration) if z0 is the apex,
superior (backwards) if z0 is the base.

`--min-planes` guards the estimate: a stack with only 2-3 labeled planes cannot support a slope fit.

Usage:
    python tools/probe_slice_order.py                     # all sources
    python tools/probe_slice_order.py --per-source 0      # no subsampling (slow, exact)
    python tools/probe_slice_order.py --csv out.csv       # per-subject, for a per-subject fix
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")

SOURCES = [
    ("CMRx", os.path.join(DATA, "CMRxRecon202*/Cine_combined/*/sax/heart_seg.nii.gz")),
    ("ACDC", os.path.join(DATA, "ACDC_sax/*/sax/heart_seg.nii.gz")),
    ("M&Ms", os.path.join(DATA, "MNMs_sax/*/sax/heart_seg.nii.gz")),
]


def slice_order(seg_path, min_planes=4):
    """-> (order, slope, n_labeled) with order in {'apex-first','base-first',None}."""
    seg = np.asarray(nib.load(seg_path).dataobj)
    if seg.ndim == 4:
        seg = seg[..., 0]                       # ED frame (frame 0 is ED by construction)
    if seg.ndim != 3:
        return None, float("nan"), 0
    area = np.array([(seg[..., z] > 0).sum() for z in range(seg.shape[2])], dtype=float)
    idx = np.flatnonzero(area)
    if idx.size < min_planes:                   # too few planes to fit a slope
        return None, float("nan"), int(idx.size)
    a = area[idx]
    slope = float(np.polyfit(np.arange(len(a)), a, 1)[0])
    return ("apex-first" if slope > 0 else "base-first"), slope, int(idx.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=60,
                    help="subsample N per source for speed; 0 = all")
    ap.add_argument("--min-planes", type=int, default=4)
    ap.add_argument("--csv", default="", help="write per-subject rows here")
    args = ap.parse_args()

    rows = []
    for label, pattern in SOURCES:
        fs = sorted(glob.glob(pattern))
        if not fs:
            print(f"  {label:6} no segmentations found yet")
            continue
        if args.per_source:
            fs = fs[:: max(1, len(fs) // args.per_source)][: args.per_source]
        res = []
        for f in fs:
            subj = os.path.basename(os.path.dirname(os.path.dirname(f)))
            order, slope, n = slice_order(f, args.min_planes)
            rows.append(dict(source=label, subject=subj, order=order or "undetermined",
                             slope=round(slope, 2) if slope == slope else "", n_labeled=n))
            if order:
                res.append(order)
        if not res:
            print(f"  {label:6} n={len(fs)}  all undetermined (<{args.min_planes} labeled planes)")
            continue
        apex = res.count("apex-first")
        base = res.count("base-first")
        maj = max(apex, base)
        print(f"  {label:6} n={len(res):4d}   apex-first {apex:4d}   base-first {base:4d}"
              f"   -> {100*maj/len(res):5.1f}% consistent"
              f"   ({len(fs)-len(res)} undetermined)")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source", "subject", "order", "slope", "n_labeled"])
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}  ({len(rows)} subjects)")

    print("\napex-first (z0 = apex) is what the stamped LPS affine (+z = superior) implies,")
    print("and is the ordering under which respiratory.py's one-sided shift moves the heart")
    print("INFERIORLY on inspiration -- the physiological direction. See docs/58 §10a.")
    print("A fix must be driven PER SUBJECT from these slopes: CMRx measured 59/60, not 60/60.")


if __name__ == "__main__":
    raise SystemExit(main())
