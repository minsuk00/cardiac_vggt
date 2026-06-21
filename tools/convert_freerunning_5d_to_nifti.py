"""Convert the free-running 5D demo recon (Zenodo 15033956) to per-respiratory 4D NIfTI.

Source of truth stays the original `recon_nufft_5D1.mat` (complex, shape
(20 cardiac, 5 respiratory, 200, 186, 234)). This emits one 4D NIfTI per
respiratory phase: `resp{r}_4d.nii.gz` with shape (X, Y, Z, cardiac=20),
magnitude, 1.5 mm isotropic. The 5 files together hold the full data.

Spacing: FOV 350x280x300 mm over matrix (array axes 200,186,234) -> ~1.5 mm
isotropic (300/200, 280/186, 350/234 = 1.500, 1.505, 1.496). Orientation is
nominal RAS-diagonal; true anatomical orientation is set at the SAX-reslice step.

Run in the `elastix` env:  micromamba run -n elastix python tools/convert_freerunning_5d_to_nifti.py
"""
import os
import argparse
import numpy as np
import h5py
import nibabel as nib

DEFAULT_SRC = "/home/minsukc/vggt/scratch/data/FRF/recon_nufft_5D1.mat"
DEFAULT_OUT = "/home/minsukc/vggt/scratch/data/FRF/nifti"
# voxel spacing for array axes (200, 186, 234), from FOV/matrix
SPACING = (300.0 / 200, 280.0 / 186, 350.0 / 234)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC, help="input 5D recon .mat (v7.3/HDF5)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for resp*_4d.nii.gz")
    args = ap.parse_args()
    SRC, OUT = args.src, args.out

    os.makedirs(OUT, exist_ok=True)
    affine = np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])

    print(f"reading {SRC} ...")
    with h5py.File(SRC, "r") as f:
        key = [k for k in f.keys() if not k.startswith("#")][0]
        print(f"dataset key: {key}")
        # chunking is (20,5,1,62,1): each chunk holds all cardiac+resp for a spatial
        # location, so one full read touches every chunk exactly once -> read it all once.
        raw = f[key][:]  # (20, 5, 200, 186, 234) compound real/imag
    n_card, n_resp = raw.shape[0], raw.shape[1]
    print(f"loaded: {n_card} cardiac x {n_resp} respiratory x {raw.shape[2:]}")

    for r in range(n_resp):
        block = raw[:, r]  # (cardiac, 200, 186, 234) compound
        mag = np.sqrt(block["real"].astype(np.float32) ** 2
                      + block["imag"].astype(np.float32) ** 2)
        vol4d = np.moveaxis(mag, 0, -1)  # (200, 186, 234, cardiac) -> (X,Y,Z,T)
        img = nib.Nifti1Image(vol4d, affine)
        img.header.set_zooms((SPACING[0], SPACING[1], SPACING[2], 1.0))
        path = os.path.join(OUT, f"resp{r}_4d.nii.gz")
        nib.save(img, path)
        print(f"wrote {path}  shape={vol4d.shape}  max={vol4d.max():.3f}")


if __name__ == "__main__":
    main()
