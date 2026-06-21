"""Convert native CMRxRecon SAX cine phases -> nnU-Net v1 Task114 inputs.

Each subject has 12 on-disk phase volumes `sax/3d_recon/sax_frame_{tt:02d}.nii.gz`
already in nibabel (X,Y,Z) order with the subject's TRUE per-subject spacing in the
header. We rewrite each to `<subj>_t{tt}_0000.nii.gz` with a clean positive-diagonal
affine carrying that same spacing (matching the validated prep_inputs.py convention,
which is what produced the 0.949 LV Dice). Intensities pass through (nnU-Net does its
own per-image z-score).

One case per (subject, phase) => 12 cases/subject.
"""
import argparse, os, glob
import numpy as np
import nibabel as nib

NUM_PHASES = 12


def convert(src, dst):
    im = nib.load(src)
    arr = np.asarray(im.dataobj).astype(np.float32)   # (X, Y, Z) nibabel order
    sx, sy, sz = im.header.get_zooms()[:3]             # subject's TRUE spacing
    affine = np.diag([float(sx), float(sy), float(sz), 1.0])
    nib.save(nib.Nifti1Image(arr, affine), dst)
    return arr.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--subjects", default=None,
                    help="comma-separated subject dir names; default = all")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.subjects:
        subs = args.subjects.split(",")
    else:
        subs = sorted(d for d in os.listdir(args.data_root)
                      if os.path.isdir(os.path.join(args.data_root, d)))
    if args.limit:
        subs = subs[: args.limit]

    n = 0
    missing = 0
    for s in subs:
        rd = os.path.join(args.data_root, s, "sax", "3d_recon")
        for t in range(NUM_PHASES):
            src = os.path.join(rd, f"sax_frame_{t:02d}.nii.gz")
            if not os.path.exists(src):
                print("MISSING", src); missing += 1; continue
            dst = os.path.join(args.out_dir, f"{s}_t{t:02d}_0000.nii.gz")
            convert(src, dst)
            n += 1
    print(f"wrote {n} cases ({len(subs)} subjects) to {args.out_dir}; missing={missing}")


if __name__ == "__main__":
    main()
