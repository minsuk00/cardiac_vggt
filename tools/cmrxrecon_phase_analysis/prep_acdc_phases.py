"""Resample each ACDC 4D cine to 12 ED-aligned fractional phases -> nnU-Net inputs.

To compare ACDC to CMRxRecon with the IDENTICAL measurement method, we put ACDC on
the same axis: 12 cardiac phases tiling the R-R, phase 0 = ED. ACDC ships the full
cine (NbFrame frames, 14-35) + ED frame index in Info.cfg. We roll so ED is first,
then linearly resample the time axis to 12 samples at k/12 of the cycle.

Writes <pid>_t{tt}_0000.nii.gz (X,Y,Z) with the patient's true spacing.
"""
import argparse, glob, os
import numpy as np
import nibabel as nib

NUM_PHASES = 12


def cfg_ed(path):
    for line in open(path):
        if line.startswith("ED:"):
            return int(line.split(":")[1].strip())
    raise KeyError("ED")


def resample_time(vol4d, ed, n_out=NUM_PHASES):
    """vol4d: (X,Y,Z,T). Roll so frame `ed` (1-indexed) -> index 0, linearly resample
    T -> n_out at fractional positions k/n_out of the cycle (periodic)."""
    T = vol4d.shape[-1]
    rolled = np.roll(vol4d, -(ed - 1), axis=-1)        # ED-aligned, periodic
    pos = np.arange(n_out) * (T / n_out)               # 0, T/12, ... in source-frame units
    i0 = np.floor(pos).astype(int) % T
    i1 = (i0 + 1) % T
    w = (pos - np.floor(pos))[None, None, None, :]
    return rolled[..., i0] * (1 - w) + rolled[..., i1] * w   # (X,Y,Z,n_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pats = []
    for split in ("training", "testing"):
        pats += sorted(glob.glob(os.path.join(args.acdc_root, split, "patient*")))

    n = 0
    for p in pats:
        pid = os.path.basename(p)
        f4 = os.path.join(p, f"{pid}_4d.nii.gz")
        if not os.path.exists(f4):
            print("MISSING", f4); continue
        im = nib.load(f4)
        arr = np.asarray(im.dataobj).astype(np.float32)   # (X,Y,Z,T)
        ed = cfg_ed(os.path.join(p, "Info.cfg"))
        phases = resample_time(arr, ed, NUM_PHASES)        # (X,Y,Z,12)
        sx, sy, sz = im.header.get_zooms()[:3]
        affine = np.diag([float(sx), float(sy), float(sz), 1.0])
        for t in range(NUM_PHASES):
            dst = os.path.join(args.out_dir, f"{pid}_t{t:02d}_0000.nii.gz")
            nib.save(nib.Nifti1Image(np.ascontiguousarray(phases[..., t]), affine), dst)
            n += 1
    print(f"wrote {n} cases ({len(pats)} patients) to {args.out_dir}")


if __name__ == "__main__":
    main()
