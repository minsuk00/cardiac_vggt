"""Convert VGGT SAX-stack volumes -> nnU-Net v1 M&Ms (Task114) inputs.

Our val_volumes NIfTIs are saved in splat order (Z,Y,X) with an identity affine
(1mm isotropic placeholder). nnU-Net resamples by header spacing, so we rewrite
each volume into nibabel (X,Y,Z) order with the true canonical spacing
(1.4, 1.4, 12.0) mm. nnU-Net does its own per-image z-score normalization, so the
[-1,1]/percentile-normalized intensities are fine to pass through.

Files are named <case>_0000.nii.gz (single modality 0000) as nnU-Net requires.
"""
import argparse, glob, os
import numpy as np
import nibabel as nib

CANON_SPACING = (1.4, 1.4, 12.0)  # x, y, z mm — true CMRx pitch (preprocess.TARGET_SPACING); docs/18


def convert(src, dst):
    im = nib.load(src)
    arr = np.asarray(im.dataobj)              # (Z, Y, X) splat order
    arr = np.transpose(arr, (2, 1, 0))        # -> (X, Y, Z) nibabel order
    affine = np.diag([*CANON_SPACING, 1.0])
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=6, help="number of subjects")
    ap.add_argument("--kinds", default="gt,pred")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gts = sorted(glob.glob(os.path.join(args.val_dir, "*_sax_gt.nii.gz")))[: args.n]
    kinds = args.kinds.split(",")
    n = 0
    for g in gts:
        base = os.path.basename(g).replace("_sax_gt.nii.gz", "")  # subjXX_tYY_Split_PNNN
        for k in kinds:
            src = g if k == "gt" else g.replace("_sax_gt.nii.gz", "_sax_pred.nii.gz")
            if not os.path.exists(src):
                print("MISSING", src); continue
            case = f"{base}_{k}"
            convert(src, os.path.join(args.out_dir, f"{case}_0000.nii.gz"))
            n += 1
    print(f"wrote {n} cases to {args.out_dir}")


if __name__ == "__main__":
    main()
