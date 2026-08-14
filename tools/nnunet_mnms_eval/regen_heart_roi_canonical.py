"""Regenerate heart_seg_canonical.nii.gz / heart_roi_canonical.nii.gz on the native-z (256,256,D)
grid for all already-segmented cmrx/acdc_sax/mnms_sax units (docs/58 A6).

The per-frame nnU-Net outputs used by sbatch/_archive/whs_segment.sh live on node-local $TMPDIR and are
deleted after each unit -- this reads the persisted native-space heart_seg.nii.gz sibling instead
(same affine as the per-frame segs it was stacked from) and rebuilds the canonical siblings from
it. No GPU / nnU-Net rerun needed. CPU-only; run directly, not via sbatch.

Usage: python regen_heart_roi_canonical.py [--worklist PATH] [--dry-run]
"""
import argparse, os, sys
import numpy as np
import nibabel as nib

ROOT = "/home/minsukc/vggt"
sys.path.insert(0, os.path.dirname(__file__))
from assemble_whs import unit_id, is_gated, build_canonical_siblings   # noqa: E402

TRAINING_DATASETS = ("cmrx", "acdc_sax", "mnms_sax")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=os.path.join(ROOT, "scratch/data/whs/worklist.txt"))
    ap.add_argument("--in_mm", type=float, default=6.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lines = [l.split() for l in open(args.worklist) if len(l.split()) == 3]
    n_ok = n_missing = n_skip = n_fail = 0
    for ds, regime, p in lines:
        if ds not in TRAINING_DATASETS or not is_gated(regime):
            n_skip += 1
            continue
        uid, subj, out_dir = unit_id(ds, regime, p)
        seg_f = os.path.join(out_dir, "heart_seg.nii.gz")
        if not os.path.exists(seg_f):
            print(f"MISSING heart_seg: {seg_f}")
            n_missing += 1
            continue
        if args.dry_run:
            n_ok += 1
            continue
        try:
            im = nib.load(seg_f)
            seg4d = np.asarray(im.dataobj).astype(np.uint8)            # (X,Y,Z,T)
            cseg, croi, cspacing = build_canonical_siblings(seg4d, im.affine, in_mm=args.in_mm)
            caffine = np.diag([*cspacing, 1.0])
            nib.save(nib.Nifti1Image(cseg, caffine), os.path.join(out_dir, "heart_seg_canonical.nii.gz"))
            nib.save(nib.Nifti1Image(croi.astype(np.uint8), caffine),
                      os.path.join(out_dir, "heart_roi_canonical.nii.gz"))
            print(f"  {uid}: canonical {cseg.shape} roi_vox={int(croi.sum())} -> {out_dir}")
            n_ok += 1
        except Exception as e:
            print(f"  FAIL {uid}: {e}")
            n_fail += 1
    print(f"done: ok={n_ok} missing={n_missing} skip={n_skip} fail={n_fail}")


if __name__ == "__main__":
    main()
