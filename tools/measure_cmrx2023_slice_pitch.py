"""Measure the CMRxRecon2023 SAX slice PITCH from the expert SegmentROI labels.

The dataset ships slice THICKNESS (8 mm) but never a slice gap, so the canonical
Z = 12 mm pitch is inherited from the CMRxRecon2024 protocol doc and is unverified
(docs/27). This measures it instead of assuming it.

Idea (needs no reconstruction, no slice positions):
  * LAX labels give the LV blood pool in a long-axis view, and the LAX in-plane
    spacing is known (FOV / ReconMatrix) -> measure the LV base-to-apex CALIPER
    length in mm.
  * SAX labels give the LV blood pool per slice -> count how many slices contain LV.
  * Those slices span the same base-to-apex distance, so
        (n_lv - 1) * pitch  <=  LV_length  <=  (n_lv + 1) * pitch
    which brackets the pitch tightly enough to separate 8 mm from 12 mm.

Geometry note: the *_forlabel/_label NIfTIs are on the UNCROPPED readout-oversampled
grid (512 wide), whose spacing is still FOVx/ReconMatrix_X, because 2x readout
oversampling doubles both the FOV and the sample count.

Usage:  python tools/measure_cmrx2023_slice_pitch.py [--limit N]
"""

import argparse
import csv
import os

import nibabel as nib
import numpy as np

CH_ROOT = "scratch/data/CMRxRecon2023"
PA_ROOT = "scratch/data/CMRxRecon-300"

SPLITS = [
    ("ChallengeData", "TrainingSet", "TrainingSet"),
    ("ChallengeData_validation", "ValidationSet", "ValidationSet"),
    ("ChallengeData_test", "TestSet", "TestSet"),
]

SAX_LV = 1  # SAX: 1=LV blood pool, 2=LV myocardium, 3=RV
LAX_LV = 3  # LAX: 1=LA, 2=RA, 3=LV, 4=RV


def read_info(path):
    if not os.path.exists(path):
        return None
    return {r[0]: r[1] for r in csv.reader(open(path)) if len(r) == 2}


def inplane_spacing(info):
    """(sx, sy) mm on the reconstructed grid."""
    return (
        float(info["FOVx"]) / int(info["ReconMatrix_X"]),
        float(info["FOVy"]) / int(info["ReconMatrix_Y"]),
    )


def lv_caliper_mm(label_vol, sx, sy):
    """Max LV-to-LV distance (base-to-apex) over each long-axis view, in mm."""
    best = 0.0
    n_view = label_vol.shape[2] if label_vol.ndim == 3 else 1
    for v in range(n_view):
        sl = label_vol[:, :, v] if label_vol.ndim == 3 else label_vol
        ys, xs = np.where(sl == LAX_LV)
        if len(ys) < 20:
            continue
        pts = np.stack([ys * sx, xs * sy], 1)
        # caliper via the principal axis (exact max-pair is O(n^2); projection is
        # equivalent for an elongated convex-ish blob and far cheaper)
        c = pts - pts.mean(0)
        u = np.linalg.svd(c, full_matrices=False)[2][0]
        proj = c @ u
        best = max(best, float(proj.max() - proj.min()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--only", nargs="+", default=None,
                    help="restrict to these 'Set/P###' ids, e.g. TrainingSet/P046 (used to probe the "
                         "6 mm-thickness protocol variants, whose pitch is undocumented)")
    args = ap.parse_args()

    rows = []
    for split, setn, pasub in SPLITS:
        seg = f"{CH_ROOT}/{split}/SingleCoil/Cine/{setn}/SegmentROI"
        if not os.path.isdir(seg):
            continue
        for pid in sorted(os.listdir(seg)):
            if args.only is not None:
                if f"{pasub}/{pid}" not in args.only:
                    continue
            elif len(rows) >= args.limit:
                break
            sax_l = f"{seg}/{pid}/cine_sax_label.nii.gz"
            lax_l = f"{seg}/{pid}/cine_lax_label.nii.gz"
            sax_i = read_info(f"{PA_ROOT}/{pasub}/{pid}/cine_sax_info.csv")
            lax_i = read_info(f"{PA_ROOT}/{pasub}/{pid}/cine_lax_info.csv")
            if not (os.path.exists(sax_l) and os.path.exists(lax_l) and sax_i and lax_i):
                continue
            try:
                sax = np.asanyarray(nib.load(sax_l).dataobj)
                lax = np.asanyarray(nib.load(lax_l).dataobj)
            except Exception:
                continue

            # SAX: how many slices contain LV blood pool?
            n_lv = int(sum((sax[:, :, z] == SAX_LV).sum() > 20 for z in range(sax.shape[2])))
            if n_lv < 3:
                continue
            lsx, lsy = inplane_spacing(lax_i)
            lv_mm = lv_caliper_mm(lax, lsx, lsy)
            if lv_mm < 30:
                continue
            rows.append(
                dict(
                    subj=f"{pasub}/{pid}",
                    n_slices=sax.shape[2],
                    n_lv=n_lv,
                    lv_mm=lv_mm,
                    thk=float(sax_i["SliceThickness"]),
                    pitch_lo=lv_mm / (n_lv + 1),
                    pitch_mid=lv_mm / n_lv,
                    pitch_hi=lv_mm / max(n_lv - 1, 1),
                )
            )

    if not rows:
        print("no usable subjects")
        return

    print(f"n = {len(rows)} subjects\n")
    print(f"{'subject':24s} {'Zt':>3s} {'nLV':>4s} {'LVlen_mm':>9s} {'pitch_lo':>9s} {'pitch_mid':>10s} {'pitch_hi':>9s}")
    for r in rows[:25]:
        print(
            f"{r['subj']:24s} {r['n_slices']:3d} {r['n_lv']:4d} {r['lv_mm']:9.1f} "
            f"{r['pitch_lo']:9.2f} {r['pitch_mid']:10.2f} {r['pitch_hi']:9.2f}"
        )

    mid = np.array([r["pitch_mid"] for r in rows])
    lo = np.array([r["pitch_lo"] for r in rows])
    hi = np.array([r["pitch_hi"] for r in rows])
    print(f"\npitch estimate (mm), n={len(rows)}")
    print(f"  lower bound  LV/(n+1): median {np.median(lo):5.2f}   IQR {np.percentile(lo,25):.2f}-{np.percentile(lo,75):.2f}")
    print(f"  central      LV/n    : median {np.median(mid):5.2f}   IQR {np.percentile(mid,25):.2f}-{np.percentile(mid,75):.2f}")
    print(f"  upper bound  LV/(n-1): median {np.median(hi):5.2f}   IQR {np.percentile(hi,25):.2f}-{np.percentile(hi,75):.2f}")
    print(f"\n  vs candidates: thickness-only 8.0 mm   |   assumed 2024 pitch 12.0 mm")
    print(f"  central estimate is closer to: {'8 mm' if abs(np.median(mid)-8) < abs(np.median(mid)-12) else '12 mm'}")

    # DIAGNOSTIC ONLY -- DO NOT QUOTE THIS SLOPE AS THE PITCH.
    # The idea was that across subjects LV_length ~= pitch * n_lv + c, so the slope
    # would give the pitch with the intercept absorbing the endpoint bias. It does
    # NOT work here: n_lv is an integer spanning only ~8-10 (a quantized, narrow-range
    # regressor) while LV_length varies widely for biological reasons, so the slope is
    # severely attenuated by regression dilution -- it returns ~4 mm and "excludes"
    # every physically plausible pitch, including the true one. Kept only to record
    # that this approach was tried and why it fails. The per-subject ratio above is
    # the estimator to trust.
    n = np.array([r["n_lv"] for r in rows], float)
    y = np.array([r["lv_mm"] for r in rows], float)
    A = np.stack([n, np.ones_like(n)], 1)
    (slope, icept), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ np.array([slope, icept])
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid**2).sum() / ss_tot if ss_tot > 0 else float("nan")
    # bootstrap CI on the slope
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(n), len(n))
        try:
            s, _c = np.linalg.lstsq(np.stack([n[idx], np.ones(len(idx))], 1), y[idx], rcond=None)[0]
            boots.append(s)
        except Exception:
            pass
    lo_ci, hi_ci = np.percentile(boots, [2.5, 97.5])
    print("\n  [DIAGNOSTIC, NOT A PITCH ESTIMATE] regression LV_len = slope * n_lv + c")
    print(f"    slope = {slope:.2f} mm  95% CI [{lo_ci:.2f}, {hi_ci:.2f}]  intercept {icept:+.1f} mm")
    print(f"    R^2 = {r2:.3f}   corr(n_lv, LV_len) = {np.corrcoef(n,y)[0,1]:.3f}")
    print("    ^ attenuated by regression dilution (n_lv is integer, range ~8-10) --")
    print("      it under-estimates badly and is NOT usable. Trust the ratio above.")


if __name__ == "__main__":
    main()
