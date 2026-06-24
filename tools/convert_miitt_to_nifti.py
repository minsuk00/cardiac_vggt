#!/usr/bin/env python
"""Convert the MIITT (U-Michigan, J. Hamilton) cardiac cine .mat files to NIfTI.

The MIITT `.mat` files are ALREADY reconstructed images (not k-space), so this is a
pure format conversion — read, (RT) take magnitude, fix axis order, attach an affine,
write NIfTI. No reconstruction.

Per volunteer there are two scans:
  imagesStandardCine.mat  ->  (T=30 phases, Z=13, 180, 224)  float32 magnitude  (ECG-gated breath-hold)
  imagesRT.mat            ->  (F=180 frames, Z=13, 128, 128)  complex            (free-breathing real-time)

Output layout mirrors CMRxRecon so the gated scan is a drop-in for training/data/preprocess.py:
  MIITT/nifti/VolunteerN/gated/sax/3d_recon/sax_frame_{00..29}.nii.gz   # one 3D (X,Y,Z) per gated phase
  MIITT/nifti/VolunteerN/gated/sax/4d_recon.nii.gz                       # (X,Y,Z,30)
  MIITT/nifti/VolunteerN/realtime/sax/4d_recon.nii.gz                    # (X,Y,Z,180) magnitude (ungated time series)

================================  SPACING IS A PLACEHOLDER  ================================
The .mat files carry NO spatial metadata. The spacings below are literature/CMRxRecon-based
ESTIMATES, not the real protocol values. They are good enough to develop, visualize, and
compute PSNR against, but DO NOT compute physical volumes (EF in mL), true distances, or
cross-modal overlays off them.

To finalize: get FOVx/FOVy and SpacingBetweenSlices (or SliceThickness+gap) per scan from
the data authors, then edit SPACING below and re-run. Only the affine diagonal changes; the
pixel data / axis order / file structure are unaffected.
  dx = FOVx / matrix_x ,  dy = FOVy / matrix_y ,  dz = slice_thickness + gap
==========================================================================================
"""
import argparse
import logging
import os

import h5py
import nibabel as nib
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── PLACEHOLDER spacing (mm) — swap with real FOV-derived values when available ──
SPACING_IS_PLACEHOLDER = True
SPACING = {
    # (dx, dy, dz). gated ~CMRxRecon (1.46 mm); rt 2.1 mm is LV-size-calibrated (docs/23 follow-up:
    # 2.6 mm gave ~20% oversized LV vs in-dist GT; 2.6 × 49/59 ≈ 2.1 mm). NOTE: changing this does
    # NOT rewrite already-converted NIfTIs, and the inference adapter (eval/adapters/miitt.py) reads
    # its own INPLANE_MM constant, not the affine — keep the two in sync.
    "gated":    (1.5, 1.5, 8.0),
    "realtime": (2.1, 2.1, 8.0),
}


def _affine(dx, dy, dz):
    """Diagonal RAS-ish affine matching CMRxRecon sign convention (-dx, -dy, +dz)."""
    a = np.eye(4, dtype=np.float64)
    a[0, 0] = -dx
    a[1, 1] = -dy
    a[2, 2] = dz
    return a


def _load_mat(path, key, is_complex):
    """Read an HDF5-v7.3 .mat array -> numpy. RT is stored as a compound (real, imag)."""
    with h5py.File(path, "r") as f:
        arr = f[key][...]
    if is_complex:
        arr = np.abs(arr["real"] + 1j * arr["imag"])  # magnitude per frame
    return np.asarray(arr, dtype=np.float32)


def _to_xyzt(vol_tzab):
    """(T|F, Z, A, B) MATLAB/HDF5 order -> (X=B, Y=A, Z, T) NIfTI order."""
    return np.transpose(vol_tzab, (3, 2, 1, 0))  # (B, A, Z, T)


def convert_volunteer(in_dir, out_dir, save_phase_files=True):
    vol_id = os.path.basename(in_dir.rstrip("/"))

    # ── gated (ECG-gated breath-hold cine) ───────────────────────────────────
    g = _load_mat(os.path.join(in_dir, "imagesStandardCine.mat"),
                  "imagesStandardCine", is_complex=False)      # (30, 13, 180, 224)
    g = _to_xyzt(g)                                            # (224, 180, 13, 30)
    aff_g = _affine(*SPACING["gated"])
    gsax = os.path.join(out_dir, vol_id, "gated", "sax")
    os.makedirs(os.path.join(gsax, "3d_recon"), exist_ok=True)
    nib.save(nib.Nifti1Image(g, aff_g), os.path.join(gsax, "4d_recon.nii.gz"))
    if save_phase_files:
        for t in range(g.shape[3]):
            nib.save(nib.Nifti1Image(np.ascontiguousarray(g[..., t]), aff_g),
                     os.path.join(gsax, "3d_recon", f"sax_frame_{t:02d}.nii.gz"))

    # ── realtime (free-breathing ungated, magnitude) ─────────────────────────
    r = _load_mat(os.path.join(in_dir, "imagesRT.mat"),
                  "imagesRT", is_complex=True)                 # (180, 13, 128, 128)
    r = _to_xyzt(r)                                            # (128, 128, 13, 180)
    aff_r = _affine(*SPACING["realtime"])
    rsax = os.path.join(out_dir, vol_id, "realtime", "sax")
    os.makedirs(rsax, exist_ok=True)
    nib.save(nib.Nifti1Image(r, aff_r), os.path.join(rsax, "4d_recon.nii.gz"))

    logging.info(f"  {vol_id}: gated {g.shape} @ {SPACING['gated']} mm | "
                 f"realtime {r.shape} @ {SPACING['realtime']} mm")
    return g.shape, r.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/minsukc/MIITT",
                    help="MIITT dir containing VolunteerN/ with the raw .mat files")
    ap.add_argument("--out", default=None, help="output dir (default: <root>/nifti)")
    ap.add_argument("--no-phase-files", action="store_true",
                    help="skip per-phase gated sax_frame_NN.nii.gz, write only 4d_recon")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.root, "nifti")
    vols = sorted(d for d in os.listdir(args.root)
                  if d.lower().startswith("volunteer")
                  and os.path.isfile(os.path.join(args.root, d, "imagesStandardCine.mat")))

    if SPACING_IS_PLACEHOLDER:
        logging.warning("*** SPACING IS A PLACEHOLDER — swap with real FOV-derived values before "
                        "computing physical volumes/distances. ***")
    logging.info(f"Converting {len(vols)} volunteers -> {out_dir}")
    for v in vols:
        convert_volunteer(os.path.join(args.root, v), out_dir,
                          save_phase_files=not args.no_phase_files)
    logging.info("Done.")


if __name__ == "__main__":
    main()
