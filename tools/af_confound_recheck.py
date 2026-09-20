#!/usr/bin/env python
"""docs/111 open item #1: re-run the pilot's confounded correlation at full cohort scale.

THE PROBLEM (docs/111 s3a-bis). The pilot argued the oracle arm loses PSNR because true AF timing
leaves the engine's 12 phase bins unevenly filled, evidenced by `r(bin imbalance, oracle gap) =
-0.832` at n=12. But a plain "is this the af cohort" dummy predicted the SAME gap BETTER
(r = -0.891), and fitting both collapsed imbalance's coefficient from -3.51 to -1.123. At n=12 the
two regressors are near-collinear by construction (imbalance is ~0 for both regular arms, small for
hrv, large for af), so the design simply cannot separate "imbalance causes the loss" from "af is
worse". The mechanism was subsequently settled by INTERVENTION (s3d: fixing coverage while keeping
true timing recovers and beats the gap), so this analysis cannot overturn it -- it only decides
whether the correlational evidence is reportable alongside it.

THE FIX AT SCALE. With ~33 paired subjects per cohort the cohort dummy can be held CONSTANT: run
the correlation WITHIN each cohort separately. Within `af`, every subject shares the same cohort
label, so any surviving association is imbalance-vs-gap and nothing else. Reported alongside a
fixed-effects fit (gap ~ imbalance + cohort dummies), which is the same test in one equation.

WHAT IS MEASURED, per (cohort, subject):
  gap          = psnr_unit_peak(fetal_cmr_4d_oracle) - psnr_unit_peak(fetal_cmr_4d), the quantity
                 being explained. Within-cohort, same input/GT/engine -- only the gate differs.
  imbalance    = std/mean of the 12 phase bins' image counts under the ORACLE gate (the pilot's
                 s3a definition: hard-binned counts). Unit-tested against the pilot's published
                 P017/af occupancy.
  imbalance_w  = the same on the engine's actual Gaussian temporal PSF weights (s3c's refinement:
                 the engine does not hard-bin).
  empty_frac   = fraction of (slice, output-phase) cells receiving ~no weight. s3c argued this is
                 the sharper quantity and that n=12 could not separate it from imbalance.

Usage:  PYTHONPATH=training:. python tools/af_confound_recheck.py [--json OUT.json]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

T = 12
COHORTS = ["cmrx2024_regular_frozen", "cmrx2024_regular", "cmrx2024_hrv", "cmrx2024_af"]
SELF, ORACLE = "fetal_cmr_4d", "fetal_cmr_4d_oracle"
PSNR_KEY = "breath_psnr_unit_peak_mean"
DEGEN_DUP_PAIRS = 5
EMPTY_W = 1e-2               # a (slice, phase) cell below this total weight has effectively no data

# The engine's temporal PSF (ReconstructionCardiac4D.cc:567-586, Gaussian branch):
#   dtrad = 2*pi*dt/rr, and the gate writes dt = RR_NOMINAL_S/T with rr = RR_NOMINAL_S, so
#   dtrad = 2*pi/T regardless of the subject.  sigma = dtrad/2.355 (~FWHM one cine frame).
SIGMA = (2 * np.pi / T) / 2.355


def ang(x):
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def read_theta(cohort, subj, arm):
    p = paths.subject_dir(cohort, subj) / arm / "gate" / "cardphase.txt"
    if not p.is_file():
        return None
    return np.array(open(p).read().split(), float).reshape(-1, T)      # (D, T)


def occupancy(theta):
    """Hard bin counts (the pilot's s3a definition) -> (T,) counts over all D*T images."""
    b = np.round(np.asarray(theta).ravel() * T / (2 * np.pi)).astype(int) % T
    return np.bincount(b, minlength=T).astype(float)


def weights(theta):
    """(D, T_out) total Gaussian PSF weight each (slice, output phase) cell receives."""
    out = 2 * np.pi * np.arange(T) / T
    d = ang(theta[:, :, None] - out[None, None, :])                    # (D, T_in, T_out)
    return np.exp(-(d ** 2) / (2 * SIGMA ** 2)).sum(1)                 # sum over that slice's images


def cv(x):
    x = np.asarray(x, float)
    return float(x.std() / x.mean()) if x.mean() > 0 else float("nan")


def read_psnr(cohort, subj, arm):
    p = paths.subject_dir(cohort, subj) / arm / "metrics.json"
    if not p.is_file():
        return None
    return json.load(open(p)).get(PSNR_KEY)


def degenerate(cohort, subj):
    man = json.load(open(paths.manifest(cohort, subj)))
    n = sum(len(g) * (len(g) - 1) // 2 for g in man.get("rhythm", {}).get("duplicate_targets", []))
    return n > DEGEN_DUP_PAIRS


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), float("nan"), n
    r = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import pearsonr
        return r, float(pearsonr(x, y)[1]), n
    except Exception:
        return r, float("nan"), n


def ols(X, y, names):
    """Least squares with t-tests. X excludes the intercept, which is added here."""
    X = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    try:
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(np.abs(t), dof)
    except Exception:
        p = np.full_like(t, np.nan)
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()
    return list(zip(["intercept"] + names, beta, se, t, p)), r2, dof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="do NOT drop subjects degenerate in any cohort (default drops them, "
                         "matching every other table in docs/111)")
    args = ap.parse_args()

    subs, _ = paths.filter_by_split("cmrx2024", paths.subjects("cmrx2024"), "test")
    degen = {s for s in subs for c in COHORTS if degenerate(c, s)}
    keep = subs if args.keep_degenerate else [s for s in subs if s not in degen]
    print(f"subjects: {len(keep)} used, {len(subs) - len(keep)} dropped as degenerate "
          f"{sorted(x.replace('CMRx24_', '') for x in degen)}")

    rows = []
    for c in COHORTS:
        for s in keep:
            th_o, th_s = read_theta(c, s, ORACLE), read_theta(c, s, SELF)
            p_o, p_s = read_psnr(c, s, ORACLE), read_psnr(c, s, SELF)
            if th_o is None or th_s is None or p_o is None or p_s is None:
                continue
            w = weights(th_o)
            rows.append({"cohort": c, "subject": s, "gap": p_o - p_s,
                         "imbalance": cv(occupancy(th_o)),
                         "imbalance_w": cv(w.sum(0)),
                         "empty_frac": float((w < EMPTY_W).mean()),
                         "min_cell_w": float(w.min()),
                         "self_min_cell_w": float(weights(th_s).min()),
                         "af": float(c.endswith("_af"))})
    print(f"rows: {len(rows)}  ({', '.join(c.replace('cmrx2024_','') + '=' + str(sum(r['cohort']==c for r in rows)) for c in COHORTS)})\n")

    g = lambda k, sub=None: [r[k] for r in rows if sub is None or r["cohort"] == sub]  # noqa: E731

    # ---- unit test against the pilot's published numbers -------------------------------
    th = read_theta("cmrx2024_af", "CMRx24_Test_P017", ORACLE)
    if th is not None:
        occ = occupancy(th).astype(int)
        print(f"[check] af/P017 oracle occupancy {list(occ)}  std/mean {cv(occ):.3f}")
        print(f"        docs/111 s3a published  [12 10 9 25 9 8 11 12 13 11 11 13]  0.350")
    print(f"[check] self-gated min cell weight {min(g('self_min_cell_w')):.3f} "
          f"(docs/111 s3c: 1.125); self-gated cells below {EMPTY_W}: "
          f"{sum(r['self_min_cell_w'] < EMPTY_W for r in rows)} of {len(rows)} subjects (s3c: 0.0%)")

    print("\n=== cohort means ===\n")
    print(f"  {'cohort':16s} {'n':>3s} {'imbalance':>10s} {'imbal_w':>8s} {'empty%':>7s} {'gap dB':>8s}")
    for c in COHORTS:
        if not g("gap", c):
            continue
        print(f"  {c.replace('cmrx2024_',''):16s} {len(g('gap', c)):3d} "
              f"{np.mean(g('imbalance', c)):10.3f} {np.mean(g('imbalance_w', c)):8.3f} "
              f"{100*np.mean(g('empty_frac', c)):7.1f} {np.mean(g('gap', c)):+8.2f}")

    print("\n=== A. the pilot's correlation, recomputed at full n (pooled across cohorts) ===\n")
    for k in ("imbalance", "imbalance_w", "empty_frac"):
        r, p, n = pearson(g(k), g("gap"))
        print(f"  r({k:11s}, gap) = {r:+.3f}   p = {p:.3g}   n = {n}")
    r, p, n = pearson(g("af"), g("gap"))
    print(f"  r({'af_dummy':11s}, gap) = {r:+.3f}   p = {p:.3g}   n = {n}   <- the confound")

    print("\n=== B. THE DECISIVE TEST: within each cohort, where the dummy is constant ===\n")
    for c in COHORTS:
        if len(g("gap", c)) < 3:
            continue
        line = f"  {c.replace('cmrx2024_',''):16s} n={len(g('gap', c)):2d}  "
        for k in ("imbalance", "imbalance_w", "empty_frac"):
            r, p, _ = pearson(g(k, c), g("gap", c))
            line += f"{k}: r={r:+.3f} p={p:6.3g}   "
        print(line)

    # A within-cohort null is only informative if the regressor actually VARIES there and the gap
    # it must explain is large enough to see -- otherwise it is a restriction-of-range artifact.
    print("\n  spread available to detect (within-cohort min/max, and the r this n could resolve):")
    try:
        from scipy.stats import t as tdist
        rcrit = lambda n: float(np.sqrt(  # noqa: E731  smallest |r| significant at .05, two-sided
            (v := tdist.ppf(0.975, n - 2) ** 2) / (v + n - 2)))
    except Exception:
        rcrit = lambda n: float("nan")   # noqa: E731
    for c in COHORTS:
        if len(g("gap", c)) < 3:
            continue
        line = f"  {c.replace('cmrx2024_',''):16s} "
        for k in ("imbalance", "empty_frac", "gap"):
            v = np.asarray(g(k, c), float)
            line += f"{k}: [{v.min():+.3f}, {v.max():+.3f}] sd {v.std():.3f}   "
        print(line + f"|r| resolvable >= {rcrit(len(g('gap', c))):.3f}")

    print("\n=== C. fixed-effects fit: gap ~ imbalance + cohort dummies (same test, one equation) ===\n")
    for k in ("imbalance", "imbalance_w", "empty_frac"):
        d = [[float(r["cohort"] == c) for r in rows] for c in COHORTS[1:]]   # frozen = reference
        res, r2, dof = ols([g(k)] + d, g("gap"),
                           [k] + [c.replace("cmrx2024_", "d_") for c in COHORTS[1:]])
        print(f"  --- {k} (R2 = {r2:.3f}, dof = {dof}) ---")
        for name, b, se, t, p in res:
            star = " *" if p < 0.05 else ""
            print(f"    {name:18s} {b:+8.3f}  se {se:6.3f}  t {t:+6.2f}  p {p:8.3g}{star}")

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
