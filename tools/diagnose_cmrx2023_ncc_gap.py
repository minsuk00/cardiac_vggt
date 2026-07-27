"""Why do some CONFIRMED CMRxRecon2023<->CMRxRecon-300 matches score NCC ~0.57 instead of ~0.99?

Candidate explanations, each with a discriminating test:
  A. cardiac phase   -> frame 0 of the challenge (12 frames) != frame 0 of the donor (13-19 frames).
                        Test: best NCC over ALL donor frames at the same slice.
  B. slice indexing  -> nz//2 is not the same anatomical plane in both.
                        Test: best NCC over ALL donor slices at frame 0.
  C. spatial shift   -> different FOV/centring; a fixed centre crop misaligns by a few pixels.
                        Test: best NCC over +-8 px translations.
  D. recon algorithm -> donor is R=3 CS-SENSE + POCS, ours is a plain fully-sampled RSS; differs in
                        noise, sharpness, coil shading (all HIGH-frequency / smooth-field).
                        Test: low-pass both and re-correlate. If NCC jumps, the disagreement lives
                        in detail the two algorithms render differently, not in anatomy.

Run: python tools/diagnose_cmrx2023_ncc_gap.py
"""

import importlib.util
import os

import h5py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("v", os.path.join(REPO, "tools", "verify_cmrx2023_donor_identity.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
D = v.D

CASES = [
    ("TestSet", "P099", "LOW  0.566"),
    ("TrainingSet", "P052", "LOW  0.573"),
    ("ValidationSet", "P011", "LOW  0.671"),
    ("TestSet", "P076", "HIGH 0.998 (control)"),
    ("TrainingSet", "P007", "HIGH 0.997 (control)"),
]


def challenge_stack(section, pid):
    """All slices, frame 0, as RSS magnitude images."""
    with h5py.File(os.path.join(D, v.SECTIONS[section], pid, "cine_sax.mat"), "r") as f:
        d = f["kspace_full"]
        nz = d["real"].shape[1]
        return [np.sqrt((np.abs(v.centered_ifft2(d["real"][0, z] + 1j * d["imag"][0, z])) ** 2).sum(0))
                for z in range(nz)]


def best_shift(a, b, rad=8):
    """Max NCC over integer translations of +-rad, after centre-matching shapes."""
    n0, n1 = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    ay, ax = (a.shape[0] - n0) // 2, (a.shape[1] - n1) // 2
    by, bx = (b.shape[0] - n0) // 2, (b.shape[1] - n1) // 2
    A = a[ay: ay + n0, ax: ax + n1]
    B = b[by: by + n0, bx: bx + n1]
    best = -1.0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            As = A[max(0, dy): n0 + min(0, dy), max(0, dx): n1 + min(0, dx)]
            Bs = B[max(0, -dy): n0 + min(0, -dy), max(0, -dx): n1 + min(0, -dx)]
            if As.size == 0:
                continue
            best = max(best, v.ncc(As, Bs))
    return best


def main():
    print(f"{'subject':<22}{'base':>7}{'+frames':>9}{'+slices':>9}{'+shift':>8}"
          f"{'blur2':>7}{'blur4':>7}{'blur8':>7}   note")
    for section, pid, note in CASES:
        q = challenge_stack(section, pid)
        a = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(D, "CMRxRecon-300", section, pid, "reconstruction", "sax_4d.nii.gz")))
        T, Z = a.shape[0], a.shape[1]
        zc, zd = len(q) // 2, Z // 2
        base = v.ncc(q[zc], a[0, zd])
        over_frames = max(v.ncc(q[zc], a[t, zd]) for t in range(T))
        over_slices = max(v.ncc(q[zc], a[0, z]) for z in range(Z))
        over_shift = best_shift(q[zc], a[0, zd])
        blurs = [v.ncc(gaussian_filter(q[zc], s), gaussian_filter(a[0, zd], s)) for s in (2, 4, 8)]
        print(f"{section+'/'+pid:<22}{base:>7.3f}{over_frames:>9.3f}{over_slices:>9.3f}"
              f"{over_shift:>8.3f}{blurs[0]:>7.3f}{blurs[1]:>7.3f}{blurs[2]:>7.3f}   {note}")
        print(f"{'':<22}   (challenge nz={len(q)}, donor T={T} Z={Z})")


if __name__ == "__main__":
    main()
