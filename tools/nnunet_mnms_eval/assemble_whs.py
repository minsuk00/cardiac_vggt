"""Assemble one work unit's nnU-Net segs into the whole-heart products, written as siblings.

Reads per-frame 3-class segs `<seg_dir>/f{ff}.nii.gz` (native affine, preserved by nnU-Net) and writes,
next to the recon:
  heart_seg.nii.gz   (X,Y,Z,T) 3-class stack
  heart_roi.nii.gz   binary ROI -- GATED: static 3D = union-over-phases dilated (6mm in-plane + z+/-1);
                                   RTFB: per-frame 4D = each frame dilated in-plane only (z_extend=0),
                                   NO temporal union / NO z-dilation (RTFB frames aren't resp-aligned).
CMRx also gets canonical-space siblings (resampled through preprocess's spatial transforms, nearest):
  heart_seg_canonical.nii.gz (256,256,12,12)   heart_roi_canonical.nii.gz (256,256,12)

Appends one manifest row to <manifest_dir>/<unit>.csv (per-unit file -> no concurrent-append race).

Usage: assemble_whs.py --dataset D --regime R --path P --seg_dir DIR --manifest_dir DIR
"""
import argparse, glob, os, sys
import numpy as np
import nibabel as nib

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "training"))
from build_heart_roi import build_roi                                  # noqa: E402

LOW_LABELED = 4.0    # mean labeled-planes/frame below this -> quality_flag "low" (flags failed OCMR-rtfb)


def is_gated(regime):
    return regime in ("native", "gated")


def unit_id(dataset, regime, path):
    if dataset == "cmrx":
        subj = os.path.basename(os.path.dirname(path))
        return f"cmrx_{subj}", subj, path                              # out dir = the sax dir
    out = os.path.dirname(path)
    if dataset == "acdc":
        subj = os.path.basename(out)                                   # patientNNN dir
        return f"acdc_{subj}", subj, out
    if dataset == "miitt":
        subj = path.split("/nifti/")[1].split("/")[0]
        return f"miitt_{subj}_{regime}", subj, out
    if dataset == "ocmr":
        rel = os.path.relpath(out, os.path.join(DATA, "ocmr/recon"))
        return f"ocmr_{rel.replace('/', '__')}", rel, out
    if dataset == "goettingen":
        subj = os.path.basename(out)
        return f"goettingen_{subj}", subj, out
    raise ValueError(dataset)


def _load_preprocess():
    """Load training/data/preprocess.py standalone (it only needs torch+monai), bypassing
    data/__init__.py which would pull the whole MRIDataset/vggt chain (not on the svr subprocess path)."""
    import importlib.util
    p = os.path.join(ROOT, "training/data/preprocess.py")
    spec = importlib.util.spec_from_file_location("_preprocess_standalone", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _canon_tfm():
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd,
                                  Orientationd, Spacingd, ResizeWithPadOrCropd)
    pp = _load_preprocess()
    TARGET_SPACING, TARGET_SHAPE = pp.TARGET_SPACING, pp.TARGET_SHAPE
    k = ["seg"]
    return Compose([
        LoadImaged(keys=k, image_only=True),
        EnsureChannelFirstd(keys=k),
        Orientationd(keys=k, axcodes="LPS"),
        Spacingd(keys=k, pixdim=TARGET_SPACING, mode="nearest"),
        ResizeWithPadOrCropd(keys=k, spatial_size=TARGET_SHAPE, mode="constant", value=0),
    ]), TARGET_SPACING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--manifest_dir", required=True)
    ap.add_argument("--in_mm", type=float, default=6.0)
    ap.add_argument("--out_dir", default=None, help="override sibling dir (dry-run/testing)")
    args = ap.parse_args()

    uid, subj, out_dir = unit_id(args.dataset, args.regime, args.path)
    if args.out_dir:
        out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)

    seg_files = sorted(glob.glob(os.path.join(args.seg_dir, "f*.nii.gz")))
    if not seg_files:
        raise SystemExit(f"no segs in {args.seg_dir}")
    im0 = nib.load(seg_files[0])
    affine = im0.affine
    spacing = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
    segs = [np.asarray(nib.load(f).dataobj).astype(np.uint8) for f in seg_files]  # each (X,Y,Z)
    seg4d = np.stack(segs, axis=-1)                                     # (X,Y,Z,T)
    X, Y, Z, T = seg4d.shape

    # --- 4D seg sibling ---
    nib.save(nib.Nifti1Image(seg4d, affine), os.path.join(out_dir, "heart_seg.nii.gz"))

    # --- ROI (native space) ---
    gated = is_gated(args.regime)
    if gated:
        union = (seg4d > 0).any(axis=-1).astype(np.uint8)              # aligned frames -> union OK
        roi = build_roi(union, spacing, in_mm=args.in_mm, z_extend=1)  # static 3D
    else:
        roi = np.stack([build_roi(seg4d[..., t], spacing, in_mm=args.in_mm, z_extend=0)
                        for t in range(T)], axis=-1)                     # per-frame 4D, in-plane only
    nib.save(nib.Nifti1Image(roi.astype(np.uint8), affine), os.path.join(out_dir, "heart_roi.nii.gz"))
    roi_vox = int(roi.sum())

    # --- CMRx canonical siblings ---
    if args.dataset == "cmrx":
        tfm, cspacing = _canon_tfm()
        cseg = np.stack([np.asarray(tfm({"seg": f})["seg"])[0].astype(np.uint8) for f in seg_files],
                        axis=-1)                                        # (256,256,12,T)
        caffine = np.diag([*cspacing, 1.0])
        nib.save(nib.Nifti1Image(cseg, caffine), os.path.join(out_dir, "heart_seg_canonical.nii.gz"))
        cunion = (cseg > 0).any(axis=-1).astype(np.uint8)
        croi = build_roi(cunion, cspacing, in_mm=args.in_mm, z_extend=1)
        nib.save(nib.Nifti1Image(croi.astype(np.uint8), caffine),
                 os.path.join(out_dir, "heart_roi_canonical.nii.gz"))

    # --- manifest row ---
    n_lab = [int((seg4d[..., t] > 0).any(axis=(0, 1)).sum()) for t in range(T)]  # labeled z-planes/frame
    mean_lab = float(np.mean(n_lab))
    lv, myo, rv = (int((seg4d == c).sum()) for c in (1, 2, 3))
    flag = "low" if mean_lab < LOW_LABELED else "ok"
    row = [uid, args.dataset, args.regime, subj, f"{X}x{Y}x{Z}x{T}",
           "x".join(f"{s:.2f}" for s in spacing), str(T), f"{mean_lab:.2f}",
           str(lv), str(myo), str(rv), str(roi_vox), flag]
    with open(os.path.join(args.manifest_dir, uid + ".csv"), "w") as f:
        f.write(",".join(row) + "\n")
    print(f"assembled {uid}: seg {seg4d.shape} roi {roi.shape} mean_lab={mean_lab:.1f} flag={flag} -> {out_dir}")


if __name__ == "__main__":
    main()
