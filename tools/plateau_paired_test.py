"""Paired epoch-by-epoch test over the training PLATEAU, instead of one cherry-picked epoch.

Why: a single-epoch read (ep26) is noise-dominated and, as an adversarial review found, mildly
flatters gather05. Both runs are evaluated on byte-identical val data at every epoch (verified:
identical identity/oracle MSE and identical applied breathing), so epoch e is a legitimate PAIR.
Pairing over the plateau removes the epoch-to-epoch common-mode noise and gives a t-test.

Usage: plateau_paired_test.py [variantA] [variantB] [lo_epoch] [hi_epoch]
"""
import json
import sys
import numpy as np
from scipy import stats

H = json.load(open("/home/minsukc/vggt/result/1frame_series/history.json"))

METRICS = [
    ("val/resp/epe_dz_mm", "resp EPE mm", "lower"),
    ("val/resp/slope_dz", "resp slope", "higher"),
    ("val/resp/corr_dz", "resp corr", "higher"),
    ("val/metric/recov_frac_heart", "recov_frac_heart", "higher"),
    ("val/metric/hole_frac_heart", "hole_frac_heart", "lower"),
    ("val/metric/coverage_frac", "coverage_frac", "higher"),
    ("val/psnr/motion/mean", "PSNR motion", "higher"),
    ("val/psnr/bbox_mean", "PSNR bbox", "higher"),
    ("val/psnr/static", "PSNR static", "higher"),
]


def by_epoch(v, key):
    return {s // 1000: x for s, x in H[v]["series"].get(key, [])}


def main():
    A = sys.argv[1] if len(sys.argv) > 1 else "gather05"
    B = sys.argv[2] if len(sys.argv) > 2 else "no_gather"
    lo = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    hi = int(sys.argv[4]) if len(sys.argv) > 4 else 38

    print(f"PAIRED over plateau epochs {lo}..{hi}   A={A}  B={B}   (diff = A - B)\n")
    print(f"{'metric':18s} {'A mean':>8s} {'B mean':>8s} {'diff':>8s} {'t':>7s} {'p':>8s} "
          f"{'A wins':>7s}  verdict")
    out = {}
    for key, lab, better in METRICS:
        a, b = by_epoch(A, key), by_epoch(B, key)
        eps = sorted(set(a) & set(b) & set(range(lo, hi + 1)))
        if len(eps) < 5:
            continue
        xa = np.array([a[e] for e in eps]); xb = np.array([b[e] for e in eps])
        d = xa - xb
        t, p = stats.ttest_rel(xa, xb)
        wins = int((d > 0).sum() if better == "higher" else (d < 0).sum())
        sig = p < 0.05
        favors = ("A" if (d.mean() > 0) == (better == "higher") else "B") if sig else "—"
        verdict = f"{favors} (p<0.05)" if sig else "coin-flip"
        print(f"{lab:18s} {xa.mean():8.3f} {xb.mean():8.3f} {d.mean():+8.3f} {t:7.2f} {p:8.1e} "
              f"{wins:3d}/{len(eps):<3d}  {verdict}")
        out[key] = {"n_epochs": len(eps), "A_mean": float(xa.mean()), "B_mean": float(xb.mean()),
                    "diff": float(d.mean()), "t": float(t), "p": float(p),
                    "A_wins": wins, "favors": favors}

    # Is the val data really identical at each epoch? (a pair is only valid if so)
    print("\nPairing validity — these must match to ~fp noise for the pairing to be legitimate:")
    for key in ["val/metric/mse_heart_identity", "val/metric/mse_heart_oracle",
                "val/resp/disp_mm_mean", "val/resp/disp_mm_max"]:
        a, b = by_epoch(A, key), by_epoch(B, key)
        eps = sorted(set(a) & set(b) & set(range(lo, hi + 1)))
        if not eps:
            continue
        dm = max(abs(a[e] - b[e]) for e in eps)
        print(f"  {key:34s} max|A-B| over {len(eps)} epochs = {dm:.3e}  "
              f"{'IDENTICAL' if dm < 1e-6 else 'DIFFER'}")

    # Oracle headroom: is the appearance metric saturated?
    print("\nIs the appearance metric saturated? (headroom between model and the splat-achievable oracle)")
    for v in (A, B):
        mi, mm, mo = (by_epoch(v, f"val/metric/mse_heart_{k}") for k in ("identity", "model", "oracle"))
        eps = sorted(set(mi) & set(mm) & set(mo) & set(range(lo, hi + 1)))
        if not eps:
            continue
        i, m, o = (np.mean([d[e] for e in eps]) for d in (mi, mm, mo))
        db = lambda x: 10 * np.log10(1.0 / x)
        print(f"  {v:10s} identity {db(i):.2f} dB -> model {db(m):.2f} dB -> oracle {db(o):.2f} dB   "
              f"HEADROOM {db(o)-db(m):.2f} dB   recov_frac {(i-m)/(i-o):.3f}")

    p = f"/home/minsukc/vggt/result/1frame_series/plateau_{A}_vs_{B}.json"
    json.dump({"A": A, "B": B, "lo": lo, "hi": hi, "metrics": out}, open(p, "w"), indent=1)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
