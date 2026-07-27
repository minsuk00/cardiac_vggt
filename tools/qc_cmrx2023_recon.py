"""QC the CMRxRecon2023 reconstruction.

Two independent checks:

1. STRUCTURAL AUDIT (every subject): 12 frames present, 4D shape consistent with the 3D frames,
   LPS orientation, expected in-plane spacing = FOVx/ReconMatrix_X from the borrowed CSV, expected
   slice spacing (12.0 mm, or 10.0 mm for the three 6 mm variants), finite non-empty data.

2. CROSS-CHECK vs CMRxRecon-300's shipped recon of the SAME person. This has real discriminating
   power because it is an independent reconstruction of the same acquisition.

   ⚠️ Expected value is NOT ~0.99. Two confounds, both measured earlier and neither a defect:
     * the two releases differ by a CYCLIC SLICE ROLL for some subjects -> take the best over all
       donor slices, never the naive z->z;
     * our recon feeds EspiritCalib image-domain data, producing near-flat maps and hence a
       different smooth coil-shading field than CMRxRecon-300's iterative CS-SENSE. On 2024 that
       cost NCC ~0.65 vs a plain RSS, but ~82% of the discrepancy is absorbed by a smooth
       multiplicative field. So we report BOTH raw and shading-corrected NCC.

   The QC signal is therefore the DISTRIBUTION plus outliers, not an absolute threshold: a genuinely
   broken or mis-joined subject lands near 0 on both, far from the pack.

Usage: python tools/qc_cmrx2023_recon.py
"""

import csv
import json
import os

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D23 = os.path.join(REPO, "scratch", "data", "CMRxRecon2023")
CINE = os.path.join(D23, "Cine_combined")
P300 = os.path.join(REPO, "scratch", "data", "CMRxRecon-300")
SIX_MM = {"CMRx23_Train_P040", "CMRx23_Train_P046", "CMRx23_Test_P116"}


def ncc(a, b):
    if a.shape != b.shape:
        n0, n1 = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        ay, ax = (a.shape[0] - n0) // 2, (a.shape[1] - n1) // 2
        by, bx = (b.shape[0] - n0) // 2, (b.shape[1] - n1) // 2
        a, b = a[ay:ay + n0, ax:ax + n1], b[by:by + n0, bx:bx + n1]
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float((a * b).mean())


def shading_corrected_ncc(a, b, sigma=20):
    """Divide out a smooth multiplicative field before comparing, so the known coil-shading
    difference between the two recon algorithms doesn't masquerade as an anatomy mismatch."""
    if a.shape != b.shape:
        n0, n1 = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
        ay, ax = (a.shape[0] - n0) // 2, (a.shape[1] - n1) // 2
        by, bx = (b.shape[0] - n0) // 2, (b.shape[1] - n1) // 2
        a, b = a[ay:ay + n0, ax:ax + n1], b[by:by + n0, bx:bx + n1]
    fa = gaussian_filter(a, sigma) + 1e-9
    fb = gaussian_filter(b, sigma) + 1e-9
    return ncc(a / fa, b / fb)


def main():
    with open(os.path.join(D23, "SUBJECT_MANIFEST.csv")) as f:
        man = {r["combined_id"]: r for r in csv.DictReader(f) if r["reconstruct"] == "1"}

    rows, problems = [], []
    for cid in sorted(man):
        r = man[cid]
        sax = os.path.join(CINE, cid, "sax")
        rec = {"combined_id": cid, "six_mm": r["six_mm"]}
        f4 = os.path.join(sax, "4d_recon.nii.gz")
        if not os.path.exists(f4):
            problems.append((cid, "MISSING 4d_recon.nii.gz"))
            continue

        n3 = len([x for x in os.listdir(os.path.join(sax, "3d_recon")) if x.endswith(".nii.gz")])
        im = nib.load(f4)
        zoom = tuple(round(float(z), 4) for z in im.header.get_zooms())
        ax = "".join(nib.aff2axcodes(im.affine))
        arr = np.asanyarray(im.dataobj)

        want_x = round(float(r["FOVx"]) / int(r["ReconMatrix_X"]), 4)
        want_y = round(float(r["FOVy"]) / int(r["ReconMatrix_Y"]), 4)
        want_z = 10.0 if cid in SIX_MM else 12.0

        rec.update(n_frames_3d=n3, shape=list(arr.shape), zooms=list(zoom), axcodes=ax,
                   want=[want_x, want_y, want_z])
        if n3 != 12:                      problems.append((cid, f"{n3} 3d frames, expected 12"))
        if arr.shape[3] != 12:            problems.append((cid, f"4D t={arr.shape[3]}, expected 12"))
        if ax != "LPS":                   problems.append((cid, f"axcodes {ax}, expected LPS"))
        if abs(zoom[0] - want_x) > 1e-3:  problems.append((cid, f"x spacing {zoom[0]} != {want_x}"))
        if abs(zoom[1] - want_y) > 1e-3:  problems.append((cid, f"y spacing {zoom[1]} != {want_y}"))
        if abs(zoom[2] - want_z) > 1e-3:  problems.append((cid, f"z spacing {zoom[2]} != {want_z}"))
        if not np.isfinite(arr).all():    problems.append((cid, "non-finite voxels"))
        if float(arr.max()) <= 0:         problems.append((cid, "all-zero volume"))

        # cross-check against CMRxRecon-300's independent recon of the same person
        section, pid = r["section"], r["pid"]
        donor = os.path.join(P300, section, pid, "reconstruction", "sax_4d.nii.gz")
        if os.path.exists(donor):
            try:
                d = sitk.GetArrayFromImage(sitk.ReadImage(donor))  # (t,z,y,x)
                ours = np.asanyarray(nib.load(f4).dataobj)[:, :, :, 0].transpose(2, 1, 0)  # ->(z,y,x)
                zc = ours.shape[0] // 2
                best = max(range(d.shape[1]), key=lambda j: ncc(ours[zc], d[0, j]))
                rec["ncc_raw"] = round(ncc(ours[zc], d[0, best]), 4)
                rec["ncc_shadecorr"] = round(shading_corrected_ncc(ours[zc], d[0, best]), 4)
                rec["best_donor_z"] = best
                rec["roll"] = best - zc
            except Exception as e:
                rec["ncc_error"] = str(e)[:60]
        rows.append(rec)

    out = os.path.join(REPO, "result", "cmrx2023_recon_qc")
    os.makedirs(out, exist_ok=True)
    json.dump(rows, open(os.path.join(out, "qc.json"), "w"), indent=1)

    print(f"audited {len(rows)} subjects")
    print(f"STRUCTURAL PROBLEMS: {len(problems)}")
    for cid, why in problems[:30]:
        print(f"   !! {cid}: {why}")

    v = [r["ncc_raw"] for r in rows if "ncc_raw" in r]
    s = [r["ncc_shadecorr"] for r in rows if "ncc_shadecorr" in r]
    if v:
        v, s = np.array(v), np.array(s)
        print(f"\nCROSS-CHECK vs CMRxRecon-300 (n={len(v)}), best-over-slices:")
        print(f"  raw NCC        : median {np.median(v):.3f}  p05 {np.percentile(v,5):.3f}  min {v.min():.3f}")
        print(f"  shading-corr   : median {np.median(s):.3f}  p05 {np.percentile(s,5):.3f}  min {s.min():.3f}")
        lo = sorted(rows, key=lambda r: r.get("ncc_shadecorr", 9))[:8]
        print("  lowest 8 (investigate if far from the pack):")
        for r in lo:
            if "ncc_shadecorr" in r:
                print(f"    {r['combined_id']:24s} raw {r['ncc_raw']:.3f}  corr {r['ncc_shadecorr']:.3f}  roll {r.get('roll')}")
        rolls = [r["roll"] for r in rows if "roll" in r]
        from collections import Counter
        print(f"  slice roll distribution: {dict(Counter(rolls))}")
    print(f"\njson -> {os.path.join(out, 'qc.json')}")


if __name__ == "__main__":
    main()
