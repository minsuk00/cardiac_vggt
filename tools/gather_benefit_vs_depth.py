"""Does the gather aux loss pay off specifically on DEEP breathers?

Hypothesis (docs/44): the coverage-division splat (V = sum(w*I)/sum(w)) largely ABSORBS a
sub-pitch through-plane error. With one slice per plane at 12 mm pitch, a 1-4 mm Dz error usually
keeps a slice in the same plane's basin, and the coverage division then normalizes the value back
-- so the reconstructed volume barely moves. A breath deeper than ~half the pitch (>6 mm, and
certainly >12 mm) pushes content ACROSS a plane boundary, where the error becomes real.

If that is why gather05 and no_gather score identically despite a 2.6x breathing-EPE gap, then
the per-subject PSNR advantage of gather05 over no_gather must GROW with the subject's breath
depth. If the advantage is flat in breath depth, the hypothesis is wrong.

Pure post-hoc read of the eval outputs. No GPU.
"""
import glob
import json
import os
import sys
import numpy as np
from scipy import stats

ROOT = "/home/minsukc/vggt/scratch/eval/cmrxrecon/out"
# Method names may be passed as argv[1], argv[2]; default to the ep39 set.
A = sys.argv[1] if len(sys.argv) > 1 else "vggt_20260715_1f_gather05"
B = sys.argv[2] if len(sys.argv) > 2 else "vggt_20260715_1f_no_gather"


def main():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/*/{A}/metrics.json")):
        subj = f.split("/")[-3]
        fb = f"{ROOT}/{subj}/{B}/metrics.json"
        rda = f"{ROOT}/{subj}/{A}/resp_diag.json"
        rdb = f"{ROOT}/{subj}/{B}/resp_diag.json"
        if not all(os.path.exists(p) for p in (fb, rda, rdb)):
            continue
        ma, mb = json.load(open(f)), json.load(open(fb))
        da, db = json.load(open(rda)), json.load(open(rdb))
        appl = np.abs(np.array(da["breath"]["applied_dz_mm"]))
        rows.append({
            "subject": subj,
            "max_disp": ma["breath_max_disp_mm"],
            "mean_disp": ma["breath_mean_disp_mm"],
            "frac_deep": float((appl >= 12).mean()),          # fraction of slots past a full pitch
            "frac_over_half_pitch": float((appl >= 6).mean()),
            "gain_breath": ma["breath_psnr_mean"] - mb["breath_psnr_mean"],   # gather05 - no_gather
            "gain_clean": ma["clean_psnr_mean"] - mb["clean_psnr_mean"],      # control: no breathing
            "epe_a": da["breath"]["epe_dz_mm"], "epe_b": db["breath"]["epe_dz_mm"],
        })
    if len(rows) < 5:
        print(f"only {len(rows)} paired subjects -- eval not far enough along yet"); return

    g = np.array([r["gain_breath"] for r in rows])
    gc = np.array([r["gain_clean"] for r in rows])
    md = np.array([r["max_disp"] for r in rows])
    fd = np.array([r["frac_deep"] for r in rows])
    de = np.array([r["epe_b"] - r["epe_a"] for r in rows])   # how much better gather's Dz is

    print(f"n={len(rows)} paired CMRx subjects   A={A}  B={B}")
    print(f"  breathing EPE:  gather05 {np.mean([r['epe_a'] for r in rows]):.2f} mm   "
          f"no_gather {np.mean([r['epe_b'] for r in rows]):.2f} mm")
    print(f"  PSNR gain (gather05 - no_gather) under BREATHING: {g.mean():+.3f} +- {g.std():.3f} dB")
    print(f"  PSNR gain under CLEAN (control, no breathing)   : {gc.mean():+.3f} +- {gc.std():.3f} dB")
    print()
    print("  Does the gather advantage grow with breath depth?  (the splat-quantization prediction)")
    for name, x in [("max applied |SI| (mm)", md), ("frac slots >=12 mm", fd),
                    ("Dz-accuracy gap (mm)", de)]:
        r, p = stats.pearsonr(x, g)
        print(f"    gain_breath vs {name:22s}  r={r:+.3f}  p={p:.3f}")
    print()
    lo, hi = md < np.median(md), md >= np.median(md)
    print(f"  shallow half (max<{np.median(md):.1f} mm, n={lo.sum()}): gain {g[lo].mean():+.3f} dB")
    print(f"  deep    half (max>={np.median(md):.1f} mm, n={hi.sum()}): gain {g[hi].mean():+.3f} dB")
    t, p = stats.ttest_ind(g[hi], g[lo])
    print(f"  deep vs shallow difference: {g[hi].mean()-g[lo].mean():+.3f} dB  (t={t:.2f}, p={p:.3f})")

    out = "/home/minsukc/vggt/result/1frame_series/gather_benefit_vs_depth.json"
    json.dump({"A": A, "B": B, "rows": rows,
               "gain_breath_mean": float(g.mean()), "gain_clean_mean": float(gc.mean()),
               "r_gain_vs_maxdisp": float(stats.pearsonr(md, g)[0]),
               "p_gain_vs_maxdisp": float(stats.pearsonr(md, g)[1])},
              open(out, "w"), indent=1)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
