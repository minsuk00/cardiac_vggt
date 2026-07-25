"""Do two models with very different breathing error actually produce different VOLUMES?

The paradox this settles: at matched epoch, no_gather has 2.6x gather05's breathing EPE
(3.49 vs 1.36 mm) on IDENTICAL val breathing, yet identical heart PSNR -- while an independent
GT-shift test (tools/zshift_sensitivity.py) says a 2 mm z-error should cost several dB.

Three outcomes, three different stories:
  (a) recons nearly identical (PSNR(A,B) very high)  -> the Dz difference never reaches the
      volume: the breathing metric is measuring something the splat doesn't act on.
  (b) recons differ a lot, both score ~equal vs GT   -> they are DIFFERENTLY wrong; the score
      is saturated/insensitive in the direction that separates them.
  (c) recons differ and scores differ                -> no paradox; the earlier read was wrong.

Scored on the same heart&FOV ROI the harness uses.
"""
import glob
import json
import os
import sys
import numpy as np
import nibabel as nib

# EVAL_DATASET=miitt to run the same decomposition on the real OOD cohort.
ROOT = "/home/minsukc/vggt/scratch/eval/%s/out" % os.environ.get("EVAL_DATASET", "cmrxrecon")
FOV_MASK = os.environ.get("FOV_MASK", "mask.nii.gz")


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float32)


def psnr(a, b, m):
    mse = float(((a[m] - b[m]) ** 2).mean())
    peak = float(b[m].max())
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-10)))


def mse(a, b, m):
    return float(((a[m] - b[m]) ** 2).mean())


def main():
    A = sys.argv[1] if len(sys.argv) > 1 else "vggt_20260715_1f_gather05"
    B = sys.argv[2] if len(sys.argv) > 2 else "vggt_20260715_1f_no_gather"
    var = sys.argv[3] if len(sys.argv) > 3 else "breath"
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/*/{A}/recon_{var}/vol_t00.nii.gz")):
        subj = f.split("/")[-4]
        sd = f"{ROOT}/{subj}"
        fb = f"{sd}/{B}/recon_{var}/vol_t00.nii.gz"
        if not os.path.exists(fb):
            continue
        heart = load(f"{sd}/mask_heart.nii.gz") > 0.5
        fov = load(f"{sd}/{FOV_MASK}") > 0.5   # MIITT bundles carry mask_fov.nii.gz, not mask.nii.gz
        m = heart & fov
        if not m.any():
            continue
        va, vb = load(f), load(fb)
        gt = load(f"{sd}/gt/gt_t00.nii.gz")
        rows.append({"subject": subj,
                     "psnr_A_vs_B": psnr(va, vb, m),      # how different are the two recons?
                     "psnr_A_vs_GT": psnr(va, gt, m),
                     "psnr_B_vs_GT": psnr(vb, gt, m),
                     # Raw MSE for the shared/unique decomposition. DO NOT derive the decomposition
                     # from PSNR differences: psnr() uses peak=b.max, so PSNR(A,B) and PSNR(A,GT) carry
                     # DIFFERENT peaks and their difference is contaminated by (GT.max/B.max)^2. This bit
                     # once (OOD shared reported 74.5% vs the correct ~79%). Use these MSEs directly.
                     "mse_A_vs_GT": mse(va, gt, m),
                     "mse_B_vs_GT": mse(vb, gt, m),
                     "mse_A_vs_B": mse(va, vb, m)})
    if not rows:
        print("no overlapping subjects yet"); return
    ab = np.array([r["psnr_A_vs_B"] for r in rows])
    ag = np.array([r["psnr_A_vs_GT"] for r in rows])
    bg = np.array([r["psnr_B_vs_GT"] for r in rows])
    print(f"n={len(rows)} subjects, variant={var}, ROI=heart&FOV, ED phase")
    print(f"  A = {A}")
    print(f"  B = {B}")
    print(f"  PSNR(A, GT) = {ag.mean():6.2f} +- {ag.std():.2f} dB")
    print(f"  PSNR(B, GT) = {bg.mean():6.2f} +- {bg.std():.2f} dB     [A-B = {ag.mean()-bg.mean():+.2f} dB]")
    print(f"  PSNR(A, B)  = {ab.mean():6.2f} +- {ab.std():.2f} dB     <- how different the two recons are")
    print()
    # Shared/unique decomposition from DIRECT MSE (peak-safe). Two views:
    #  (i)  unique fraction = (MSE(A,B)/2)/MSE(A,GT), per subject then averaged (assumes symmetric,
    #       uncorrelated unique errors);
    #  (ii) error-field correlation rho = cov(eA,eB)/sqrt(Var eA * Var eB), assumption-free.
    mAG = np.array([r["mse_A_vs_GT"] for r in rows]); mBG = np.array([r["mse_B_vs_GT"] for r in rows])
    mAB = np.array([r["mse_A_vs_B"] for r in rows])
    uniq = (mAB / 2) / mAG                       # per-subject unique fraction
    cov = (mAG + mBG - mAB) / 2                   # cov(eA, eB) since eA=A-GT, eB=B-GT
    rho = cov / np.sqrt(mAG * mBG)               # error-field correlation = shared fraction (assumption-free)
    print(f"  shared-error decomposition (direct MSE, peak-safe):")
    print(f"    unique {100*uniq.mean():.1f}%  shared {100*(1-uniq).mean():.1f}%   "
          f"(error-field correlation rho = {100*rho.mean():.1f}% shared)")
    print()
    if ab.mean() > max(ag.mean(), bg.mean()) + 6:
        print("  => (a) The two recons agree with EACH OTHER far better than either agrees with GT.")
        print("        The breathing-Dz difference is NOT reaching the reconstructed volume.")
    elif abs(ag.mean() - bg.mean()) < 0.3:
        print("  => (b) The recons genuinely DIFFER, yet score equally vs GT: differently wrong.")
    else:
        print("  => (c) Scores differ -- no paradox.")
    p = "/home/minsukc/vggt/result/1frame_series/recon_compare.json"
    json.dump({"A": A, "B": B, "variant": var, "rows": rows,
               "mean": {"A_vs_GT": float(ag.mean()), "B_vs_GT": float(bg.mean()), "A_vs_B": float(ab.mean())}},
              open(p, "w"), indent=1)
    print(f"-> {p}")


if __name__ == "__main__":
    main()
