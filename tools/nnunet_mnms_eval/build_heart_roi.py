"""Derive a binary whole-heart ROI mask from a Task114 3-class seg (LV/MYO/RV).

The ROI is a GENEROUS ventricle-centered cardiac region (SVR recon-FOV + common metric ROI) --
NOT true whole-heart (Task114 has no atria/vessel class; dilation only pulls in adjacent tissue).

Recipe (physical, spacing-aware so it's consistent across datasets):
  union(seg>0) -> per-slice fill-holes + closing (bridge LV-RV into one blob)
              -> dilate `in_mm` in-plane -> extend `z_extend` planes through-plane -> fill.

Usage:
  python build_heart_roi.py --seg_dir <segs> --out_dir <rois> [--in_mm 6] [--z_extend 1]
Writes <out_dir>/<case>.nii.gz (uint8), same geometry as the seg. Keeps the seg untouched.
"""
import argparse, glob, os
import numpy as np
import nibabel as nib
from scipy import ndimage


def _disk(r):
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r


def build_roi(seg, spacing, in_mm=6.0, z_extend=1):
    """seg: (X,Y,Z) int; spacing: (sx,sy,sz) mm. Returns uint8 (X,Y,Z)."""
    heart = seg > 0
    sx = float(spacing[0])                                   # in-plane mm/voxel (sx≈sy)
    r = max(1, int(round(in_mm / sx)))
    se, se_close = _disk(r), _disk(max(2, r // 2))
    out = np.zeros_like(heart)
    for z in range(heart.shape[-1]):
        s = heart[..., z]
        if not s.any():
            continue
        s = ndimage.binary_fill_holes(s)
        s = ndimage.binary_closing(s, se_close)              # bridge LV-RV septal gap
        s = ndimage.binary_dilation(s, se)
        out[..., z] = ndimage.binary_fill_holes(s)
    if z_extend > 0:                                          # grow ±z_extend planes (apex tip + toward base)
        out = ndimage.binary_dilation(out, np.ones((1, 1, 2 * z_extend + 1), bool))
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--in_mm", type=float, default=6.0)
    ap.add_argument("--z_extend", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    segs = sorted(glob.glob(os.path.join(args.seg_dir, "*.nii.gz")))
    print(f"{len(segs)} segs | in_mm={args.in_mm} z_extend={args.z_extend}")
    for f in segs:
        im = nib.load(f)
        seg = np.asarray(im.dataobj).astype(np.int16)
        spacing = nib.affines.voxel_sizes(im.affine)
        roi = build_roi(seg, spacing, args.in_mm, args.z_extend)
        case = os.path.basename(f)[:-7]
        nib.save(nib.Nifti1Image(roi, im.affine),
                 os.path.join(args.out_dir, f"{case}.nii.gz"))
        frac_grow = roi.sum() / max(1, (seg > 0).sum())
        print(f"  {case:44} seg_vox={(seg>0).sum():>7} roi_vox={int(roi.sum()):>7}  x{frac_grow:.2f}")
    print("ROIs ->", args.out_dir)


if __name__ == "__main__":
    main()
