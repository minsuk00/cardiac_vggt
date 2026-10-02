#!/usr/bin/env python
"""Paired per-subject comparison of two VGGT arms on the rhythm test cohorts (docs/130).

Image metrics from <cohort>/out/<subj>/<arm>/metrics.json, EF from
evaluation/metric_results/test/<cohort>/ef/<arm>.json. Pools the 5 sources per rhythm arm.

    python tools/frame_selector_compare.py --a vggt_final518_diff1000_ep300 \
        --b vggt_final518_diff1000_ep300_sel_oracle
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wilcoxon

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
IMG = ("breath_psnr_unit_peak_mean", "breath_ssim_mean", "breath_ncc_mean")


def load(arm, cohort):
    rows = {}
    out = os.path.join("scratch/eval", cohort, "out")
    for s in sorted(os.listdir(out)):
        p = os.path.join(out, s, arm, "metrics.json")
        if os.path.exists(p):
            m = json.load(open(p))
            rows[s] = {k: m[k] for k in IMG}
    ef = os.path.join("evaluation/metric_results/test", cohort, "ef", f"{arm}.json")
    if os.path.exists(ef):
        for r in json.load(open(ef))["per_subject"]:
            if r["subject"] in rows and r.get("ef_breath") is not None:
                rows[r["subject"]]["ef_abs_err"] = abs(r["ef_breath"] - r["ef_gt"])
                rows[r["subject"]]["dice_lv_es"] = r.get("dice_breath_LV_ES")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--arms", nargs="+", default=["af12", "hrv12"])
    args = ap.parse_args()
    for arm in args.arms:
        A, B = {}, {}
        for s in SOURCES:
            A.update({(s, k): v for k, v in load(args.a, f"{s}_{arm}").items()})
            B.update({(s, k): v for k, v in load(args.b, f"{s}_{arm}").items()})
        print(f"\n== {arm}: {args.a} (a) vs {args.b} (b)")
        for key in IMG + ("ef_abs_err", "dice_lv_es"):
            common = [k for k in A if k in B and A[k].get(key) is not None and B[k].get(key) is not None]
            if not common:
                print(f"  {key:28s} no paired subjects"); continue
            a = np.array([A[k][key] for k in common]); b = np.array([B[k][key] for k in common])
            p = wilcoxon(b - a).pvalue if np.any(b != a) else 1.0
            print(f"  {key:28s} n={len(common):3d}  a {a.mean():8.3f}  b {b.mean():8.3f}  "
                  f"b-a {np.mean(b - a):+8.3f}  p={p:.1e}")


if __name__ == "__main__":
    main()
