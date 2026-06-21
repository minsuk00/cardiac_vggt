"""Reslice the free-running 5D demo recon (Zenodo 15033956) to short-axis (SAX).

The recon ships as whole-heart near-isotropic volumes in a nominal (non-anatomical)
frame; the model needs SAX slices. The LV long axis was marked by hand in 3D Slicer
(apex + mitral-valve-center fiducials, saved as *.mrk.json in LPS). This tool builds
the oblique SAX basis from those two points and resamples every respiratory x cardiac
volume onto it, emitting two products per respiratory phase:

  resp{r}_sax_native_4d.nii.gz : ~1.5 mm isotropic SAX (high-res master)
  resp{r}_sax_8mm_4d.nii.gz    : through-plane slab-averaged to 8 mm (true partial-
                                 volume slice profile, matches CMRxRecon cine), then
                                 fed to the canonical pipeline like any SAX subject.

Both carry a diagonal affine in the SAX frame (spacing = in-plane, in-plane, slice),
so downstream Orientation/Spacing transforms treat SAX as the reference frame.

Usage:
  python tools/reslice_freerunning_to_sax.py \
      --recon-dir scratch/data/FRF/nifti_rlr \
      --apex .../apex.mrk.json --base .../base.mrk.json \
      --out  scratch/data/FRF
"""
import os, json, glob, argparse
import numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates, gaussian_filter1d

SP = np.array([1.5, 1.50537634, 1.49572647])   # source voxel size (X,Y,Z) mm


def lps_point_to_voxel(mrk_json):
    d = json.load(open(mrk_json))["markups"][0]
    p = np.array(d["controlPoints"][0]["position"], float)
    if d.get("coordinateSystem", "LPS").upper() == "LPS":
        ras = np.array([-p[0], -p[1], p[2]])     # LPS -> RAS
    else:
        ras = p
    return ras / SP                              # RAS -> voxel (diagonal affine, origin 0)


def sax_basis(apex_vox, base_vox):
    A = apex_vox * SP; B = base_vox * SP         # to world (mm)
    n = B - A; n /= np.linalg.norm(n)            # slice normal (apex -> base)
    ref = np.array([1.0, 0, 0])
    if abs(ref @ n) > 0.9:
        ref = np.array([0, 1.0, 0])
    e1 = ref - (ref @ n) * n; e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    C = 0.5 * (A + B)
    return C, n, e1, e2


def reslice(v, C, n, e1, e2, npx, dxy, nslc, dz):
    s = (np.arange(nslc) - nslc / 2 + 0.5) * dz
    u = (np.arange(npx) - npx / 2 + 0.5) * dxy
    w = (np.arange(npx) - npx / 2 + 0.5) * dxy
    U, W, S = np.meshgrid(u, w, s, indexing="ij")   # out shape (npx, npx, nslc) = (H,W,slice)
    world = C[None, None, None, :] + U[..., None] * e1 + W[..., None] * e2 + S[..., None] * n
    vox = (world / SP).reshape(-1, 3).T
    out = map_coordinates(v, vox, order=1, mode="constant").reshape(npx, npx, nslc)
    return out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-dir", required=True)
    ap.add_argument("--apex", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dxy", type=float, default=1.5)
    ap.add_argument("--npx", type=int, default=160)
    ap.add_argument("--dz-native", type=float, default=1.5)
    ap.add_argument("--nslc-native", type=int, default=96)
    ap.add_argument("--slice-mm", type=float, default=8.0)   # target thick-slice
    args = ap.parse_args()

    apex = lps_point_to_voxel(args.apex)
    base = lps_point_to_voxel(args.base)
    C, n, e1, e2 = sax_basis(apex, base)
    print(f"apex(vox) {apex.round(1)}  base(vox) {base.round(1)}  |axis|={np.linalg.norm((base-apex)*SP):.1f}mm")
    print(f"n={n.round(3)} e1={e1.round(3)} e2={e2.round(3)}")

    nat_dir = os.path.join(args.out, "sax_native"); os.makedirs(nat_dir, exist_ok=True)
    mm_dir = os.path.join(args.out, "sax_8mm"); os.makedirs(mm_dir, exist_ok=True)
    aff_nat = np.diag([args.dxy, args.dxy, args.dz_native, 1.0])
    aff_mm = np.diag([args.dxy, args.dxy, args.slice_mm, 1.0])

    sigma_vox = (args.slice_mm / 2.355) / args.dz_native     # FWHM=slice_mm slab profile
    step = int(round(args.slice_mm / args.dz_native))

    resp_files = sorted(glob.glob(os.path.join(args.recon_dir, "resp*_4d.nii.gz")))
    for rf in resp_files:
        r = os.path.basename(rf).split("_")[0]               # resp0..4
        vol4d = np.asarray(nib.load(rf).dataobj)             # (X,Y,Z,T)
        T = vol4d.shape[-1]
        nat = np.empty((args.npx, args.npx, args.nslc_native, T), np.float32)
        for t in range(T):
            nat[..., t] = reslice(vol4d[..., t].astype(np.float32), C, n, e1, e2,
                                  args.npx, args.dxy, args.nslc_native, args.dz_native)
        # 8 mm = slab-average (Gaussian along slice axis) then subsample
        blur = gaussian_filter1d(nat, sigma_vox, axis=2, mode="nearest")
        mm = blur[:, :, step // 2::step, :]
        nib.save(nib.Nifti1Image(nat, aff_nat),
                 os.path.join(nat_dir, f"{r}_sax_native_4d.nii.gz"))
        nib.save(nib.Nifti1Image(mm, aff_mm),
                 os.path.join(mm_dir, f"{r}_sax_8mm_4d.nii.gz"))
        print(f"{r}: native {nat.shape} -> 8mm {mm.shape}")
    print("done. axis from:", args.apex, args.base)


if __name__ == "__main__":
    main()
