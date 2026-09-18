"""Write <subject>/mask_heart_pad10.nii.gz next to every eval bundle's mask_heart.nii.gz (docs/104, docs/107).

The segmentation crop in `evaluation/src/score/ef_dice.py` and the classical baselines' recon
mask now use this padded ROI: `mask_heart` (seg union + 6 mm in-plane + z±1) dilated a further
`--pad_mm` (default 10 mm -> 7 px at 1.4 mm) IN-PLANE ONLY (z-extent unchanged), then clamped to
the native FOV mask so it never covers zero-padded X/Y (GT is 0 there; the seg crop is identical
with or without the clamp, it only keeps the baselines from reconstructing empty pixels).
Tight `mask_heart` stays as-is — it remains the image-metric scoring ROI.

    PYTHONPATH=training:. python tools/build_padded_heart_mask.py            # all cohorts
    PYTHONPATH=training:. python tools/build_padded_heart_mask.py --datasets cmrx2024 --dry_run
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402


def pad_mask(roi, fov, pad_px):
    se = np.ones((2 * pad_px + 1, 2 * pad_px + 1, 1), bool)     # in-plane only, z untouched
    return ndimage.binary_dilation(roi, se) & fov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(paths.DATASETS))
    ap.add_argument("--pad_mm", type=float, default=10.0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    n_done = n_skip = 0
    for ds in args.datasets:
        for subj in paths.subjects(ds):
            src = paths.heart_mask(ds, subj)
            dst = paths.heart_mask_pad(ds, subj)
            if not src.is_file():
                print(f"{ds}/{subj}: no mask_heart.nii.gz, skipped"); n_skip += 1; continue
            if dst.is_file() and not args.overwrite:
                n_skip += 1; continue
            img = nib.load(str(src))
            sx = float(img.header.get_zooms()[0])
            pad_px = max(1, int(round(args.pad_mm / sx)))
            roi = np.asarray(img.dataobj) > 0.5
            fov = np.asarray(nib.load(str(paths.fov_mask(ds, subj))).dataobj) > 0.5
            out = pad_mask(roi, fov, pad_px)
            zr, zo = np.where(roi.any((0, 1)))[0], np.where(out.any((0, 1)))[0]
            assert zr.min() == zo.min() and zr.max() == zo.max(), f"{ds}/{subj}: z-extent changed"
            assert not (roi & ~out).any(), f"{ds}/{subj}: padded mask does not contain the tight one"
            print(f"{ds}/{subj}: sx={sx:.2f} pad={pad_px}px voxels {int(roi.sum())} -> {int(out.sum())}")
            if not args.dry_run:
                nib.save(nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header), str(dst))
            n_done += 1
    print(f"done: {n_done} written, {n_skip} skipped")


if __name__ == "__main__":
    main()
