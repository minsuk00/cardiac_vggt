#!/usr/bin/env python
"""docs/111 open item #2: confidence intervals and paired tests for the headline AF deltas.

Every number in docs/111 s10 is a bare descriptive mean over n=33 -- no CI, no p-value anywhere.
The load-bearing claim is "Fetal CMR 4D's EF advantage survives AF but collapses from 3.61 pp to
1.01 pp". Without an interval, "still wins, barely" cannot be distinguished from "no longer wins",
and those are different sentences in the paper.

Everything here is PAIRED on the same 33 subjects (degenerate windows excluded exactly as in
af_pilot_report.py), so each contrast is a one-sample test on per-subject differences:
  - mean difference with a 95% percentile bootstrap CI (20k resamples of the subject index),
  - Wilcoxon signed-rank (distribution-free; n=33 is small and EF errors are skewed),
  - paired t as a cross-check.

The headline advantage is a DIFFERENCE OF DIFFERENCES: how much the (VGGT - Fetal) EF-error gap
shrinks from `regular_frozen` to `af`. Testing the two cohort gaps separately would answer a
different, weaker question (it can only show one is significant and the other is not, which is not
the same as showing they differ).

Usage:  PYTHONPATH=training:. python tools/af_significance.py [--boot 20000]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

COHORTS = ["cmrx2024_regular_frozen", "cmrx2024_regular", "cmrx2024_hrv", "cmrx2024_af"]
SELF, ORACLE = "fetal_cmr_4d", "fetal_cmr_4d_oracle"
BALANCED, VGGT = "fetal_cmr_4d_oracle_balanced", "vggt"
EF_FILE = {SELF: "temp/afsim_full_ef_fetal_cmr_4d.json",
           ORACLE: "temp/afsim_full_ef_fetal_cmr_4d_oracle.json",
           BALANCED: "temp/afsim_full_ef_fetal_cmr_4d_oracle_balanced.json",
           VGGT: "temp/afsim_full_ef_vggt.json"}
PSNR_KEY = "breath_psnr_unit_peak_mean"
DEGEN_DUP_PAIRS = 5
RNG = np.random.default_rng(0)


def degenerate(cohort, subj):
    man = json.load(open(paths.manifest(cohort, subj)))
    n = sum(len(g) * (len(g) - 1) // 2 for g in man.get("rhythm", {}).get("duplicate_targets", []))
    return n > DEGEN_DUP_PAIRS


def ef_err(arm):
    """-> {(cohort, subject): |pred EF - GT EF| in percentage points}."""
    p = os.path.join(ROOT, EF_FILE[arm])
    if not os.path.isfile(p):
        return {}
    d = json.load(open(p))
    return {(r["cohort"], r["subject"]): abs(r["ef_breath"] - r["ef_gt"])
            for r in d.get("per_subject", [])
            if r.get("ef_breath") is not None and r.get("ef_gt") is not None}


def psnr(cohort, subj, arm):
    p = paths.subject_dir(cohort, subj) / arm / "metrics.json"
    return json.load(open(p)).get(PSNR_KEY) if p.is_file() else None


def paired(name, d, boot, alpha=0.05):
    """One-sample inference on the per-subject differences `d`."""
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    n = len(d)
    if n < 3:
        print(f"  {name:52s}  n={n} -- too few")
        return
    m = d.mean()
    idx = RNG.integers(0, n, size=(boot, n))
    bs = d[idx].mean(1)
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    try:
        from scipy.stats import ttest_rel, wilcoxon
        w = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        t = ttest_rel(d, np.zeros(n)).pvalue
    except Exception:
        w = t = float("nan")
    star = "*" if hi < 0 or lo > 0 else " "
    print(f"  {name:52s}  n={n:2d}  {m:+7.3f}  [{lo:+7.3f}, {hi:+7.3f}] {star}  "
          f"wilcoxon p={w:8.3g}  t p={t:8.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=20000)
    args = ap.parse_args()

    subs, _ = paths.filter_by_split("cmrx2024", paths.subjects("cmrx2024"), "test")
    keep = [s for s in subs if not any(degenerate(c, s) for c in COHORTS)]
    print(f"paired subject set: n={len(keep)} (degenerate windows dropped from every cohort)")
    print("\nAll rows: mean paired difference, 95% percentile bootstrap CI, '*' = CI excludes 0.\n")

    ef = {a: ef_err(a) for a in EF_FILE}
    frozen, af = "cmrx2024_regular_frozen", "cmrx2024_af"
    G = lambda a, c: np.array([ef[a].get((c, s), np.nan) for s in keep])  # noqa: E731

    print("=== 1. THE HEADLINE: does Fetal CMR 4D's EF advantage over VGGT survive AF? ===\n")
    for c in COHORTS:
        d = G(VGGT, c) - G(SELF, c)          # >0 means Fetal is better (smaller EF error)
        paired(f"EF advantage (VGGT err - Fetal err), {c.replace('cmrx2024_','')}", d, args.boot)
    print()
    dd = (G(VGGT, af) - G(SELF, af)) - (G(VGGT, frozen) - G(SELF, frozen))
    paired("COLLAPSE of that advantage, af - regular_frozen", dd, args.boot)

    print("\n=== 2. Whose EF error moves under AF? ===\n")
    for a, lbl in [(SELF, "Fetal self-gated"), (VGGT, "VGGT")]:
        for c in COHORTS[1:]:
            paired(f"{lbl} EF |err|, {c.replace('cmrx2024_','')} - regular_frozen",
                   G(a, c) - G(a, frozen), args.boot)

    print("\n=== 3. The mechanism arms, under af (EF |err|; negative = better than self-gated) ===\n")
    paired("oracle-raw  - self-gated, af", G(ORACLE, af) - G(SELF, af), args.boot)
    paired("oracle-BAL  - self-gated, af", G(BALANCED, af) - G(SELF, af), args.boot)
    paired("oracle-BAL  - oracle-raw, af", G(BALANCED, af) - G(ORACLE, af), args.boot)

    print("\n=== 4. PSNR: the oracle gap (oracle - self-gated, dB) ===\n")
    P = lambda a, c: np.array([  # noqa: E731
        (lambda x: x if x is not None else np.nan)(psnr(c, s, a)) for s in keep])
    for c in COHORTS:
        paired(f"PSNR oracle gap, {c.replace('cmrx2024_','')}", P(ORACLE, c) - P(SELF, c), args.boot)
    print()
    paired("REVERSAL of the gap, af - regular_frozen",
           (P(ORACLE, af) - P(SELF, af)) - (P(ORACLE, frozen) - P(SELF, frozen)), args.boot)
    paired("PSNR oracle-BALANCED gap, af", P(BALANCED, af) - P(SELF, af), args.boot)

    print("\n=== 5. PSNR cost of AF itself (vs regular_frozen) ===\n")
    for a, lbl in [(SELF, "Fetal self-gated"), ("vggt_final518_diff1000_ep300", "VGGT")]:
        paired(f"{lbl} PSNR, af - regular_frozen", P(a, af) - P(a, frozen), args.boot)


if __name__ == "__main__":
    main()
