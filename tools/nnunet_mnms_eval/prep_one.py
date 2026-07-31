"""Prep ONE work unit (subject-regime, all phases/frames) into nnU-Net v1 Task114 inputs.

Each phase/frame -> a 3D SAX volume `<out_dir>/f{ff:03d}_0000.nii.gz` (X,Y,Z, real affine). nnU-Net
does its own per-image z-score, so raw intensities pass through. Called by `sbatch/whs_segment.sh`.

  cmrx (native):  path = subject `sax/` dir; reads 12 `3d_recon/sax_frame_{tt}.nii.gz` (each 3D).
  others:         path = 4D recon NIfTI (X,Y,Z,T); one input per frame t.

Usage: prep_one.py --dataset D --regime R --path P --out_dir DIR
"""
import argparse, os
import numpy as np
import nibabel as nib


def _save3d(arr_xyz, affine, out_dir, fidx):
    nib.save(nib.Nifti1Image(np.asarray(arr_xyz).astype(np.float32), affine),
             os.path.join(out_dir, f"f{fidx:03d}_0000.nii.gz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # `acdc_sax` / `mnms_sax` are ACDC and M&Ms-1 converted into the CMRx on-disk layout by
    # tools/convert_to_sax_layout.py (docs/58): 12 ED-anchored 3D frames under <ID>/sax/3d_recon/.
    # Identical path convention => identical code path. The legacy `acdc` branch below still reads
    # the raw 4D download and is kept for the native-T seg used as an independent QC reference.
    if args.dataset in ("cmrx", "acdc_sax", "mnms_sax"):
        n = 0
        for tt in range(12):
            f = os.path.join(args.path, "3d_recon", f"sax_frame_{tt:02d}.nii.gz")
            im = nib.load(f)
            _save3d(np.asarray(im.dataobj), im.affine, args.out_dir, tt)  # (X,Y,Z)
            n += 1
        print(f"prep cmrx {os.path.basename(os.path.dirname(args.path))}: {n} phases -> {args.out_dir}")
    elif args.dataset == "acdc":
        # ACDC stores real spacing in the header, NOT the affine (which is identity) -> build a
        # spacing-correct diagonal affine from header zooms so nnU-Net resamples right.
        im = nib.load(args.path)
        arr = np.asarray(im.dataobj)            # (X,Y,Z,T)
        assert arr.ndim == 4, f"expected 4D ACDC cine, got {arr.shape}"
        sp = [float(z) for z in im.header.get_zooms()[:3]]
        aff = np.diag([sp[0], sp[1], sp[2], 1.0])
        for t in range(arr.shape[-1]):
            _save3d(arr[..., t], aff, args.out_dir, t)
        print(f"prep acdc {os.path.basename(os.path.dirname(args.path))}: {arr.shape[-1]} phases -> {args.out_dir}")
    else:
        im = nib.load(args.path)
        arr = np.asarray(im.dataobj)            # (X,Y,Z,T)
        assert arr.ndim == 4, f"expected 4D recon, got {arr.shape}"
        T = arr.shape[-1]
        for t in range(T):
            _save3d(arr[..., t], im.affine, args.out_dir, t)
        print(f"prep {args.dataset}/{args.regime}: {T} frames -> {args.out_dir}")


if __name__ == "__main__":
    main()
