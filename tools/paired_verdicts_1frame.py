"""Paired per-subject C1-C5 verdicts at ep100, across all 4 cohorts.

Reads the per_subject arrays from each cohort/method _summary.json (offline, no GPU) and runs a
paired (by subject) t-test of breath PSNR for each single-knob comparison vs the gather05 hub:
  C1 no_gather-hub | C2 aug-hub | C3 contz-hub | C4 dino-hub | C5 lowdiff-hub.
Prints mean paired diff, paired SEM, t-test p, Wilcoxon p, n_win/n per (comparison, cohort).
"""
import json
import glob
import os
import re
import numpy as np
from scipy import stats

E = "/home/minsukc/vggt/scratch/eval"
HUB = "gather05"
COMPARISONS = [("C1", "no_gather"), ("C2", "aug_moderate"), ("C3", "contz"),
               ("C4", "dino_ft"), ("C5", "lowdiff100")]
COHORTS = ["cmrxrecon", "miitt", "ocmr", "acdc"]
METRIC = "breath_psnr"


def load(cohort, variant):
    # contz on OOD cohorts carries a _contz suffix (continuous-z run); cmrx does not.
    for suf in ("", "_contz"):
        f = glob.glob(f"{E}/{cohort}/out/vggt_20260719_1f_{variant}_ep99{suf}_summary.json")
        if f:
            ps = json.load(open(f[0]))["per_subject"]
            return {r["subject"]: r[METRIC] for r in ps}
    return {}


def main():
    print(f"Paired per-subject {METRIC} vs {HUB} hub (mean diff = variant - hub; >0 favors variant)\n")
    hub = {c: load(c, HUB) for c in COHORTS}
    for cid, var in COMPARISONS:
        print(f"===== {cid}: {var} - {HUB} =====")
        print(f"  {'cohort':10s} {'n':>3s} {'meanΔ':>7s} {'SEM':>6s} {'t-p':>7s} {'wilcox':>7s} {'win':>7s}")
        ood_d = []
        for c in COHORTS:
            v = load(c, var)
            subs = [s for s in hub[c] if s in v]
            if len(subs) < 3:
                print(f"  {c:10s}  (n<3)"); continue
            d = np.array([v[s] - hub[c][s] for s in subs])
            if c != "cmrxrecon":
                ood_d.extend(d.tolist())
            sem = d.std(ddof=1) / np.sqrt(len(d))
            tp = stats.ttest_rel([v[s] for s in subs], [hub[c][s] for s in subs]).pvalue
            wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            print(f"  {c:10s} {len(d):>3d} {d.mean():+7.3f} {sem:6.3f} {tp:7.3f} {wp:7.3f} {int((d>0).sum()):>3d}/{len(d)}")
        if len(ood_d) >= 3:                                    # pooled OOD (miitt+ocmr+acdc)
            d = np.array(ood_d)
            sem = d.std(ddof=1) / np.sqrt(len(d))
            tp = stats.ttest_1samp(d, 0).pvalue
            wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else float("nan")
            print(f"  {'OOD-pool':10s} {len(d):>3d} {d.mean():+7.3f} {sem:6.3f} {tp:7.3f} {wp:7.3f} {int((d>0).sum()):>3d}/{len(d)}")
        print()


if __name__ == "__main__":
    main()
