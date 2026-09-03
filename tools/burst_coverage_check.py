#!/usr/bin/env python
"""burst_coverage_check.py — is the burst-k5 NCC/PSNR gain a coverage (hole) effect?

For every val subject and each of the 4 burst-2x2 arms (breath variant), over the scoring ROI
(GT heart seg ∩ FOV, same as image_metrics.score_subject):
  hole_frac   fraction of ROI voxels the splat left EMPTY (recon == 0), mean over the 12 phases
  ncc_roi     NCC over the full ROI (what the harness reports)
  ncc_common  NCC over ROI ∩ (covered by ALL 4 arms at that phase) — equal-coverage re-score
  ncc_filled  NCC over ROI ∩ (this arm's own covered voxels)
If ncc_common still separates the arms, the gain is in the rendered voxels, not in fewer holes.
Writes temp/burst_eval/coverage.csv and prints pooled medians.

  PYTHONPATH=training:. python tools/burst_coverage_check.py [--limit N]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "evaluation" / "src" / "score"))
import paths  # noqa: E402
from image_metrics import load_canon, subject_grid, ncc  # noqa: E402

ARMS = ["vggt_burst5noreg224_ep300", "vggt_burst5noreg224_k1_ep300",
        "vggt_noreg224_ep300", "vggt_noreg224_k5_ep300"]
SOURCES = ["cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms", "miitt", "ocmr"]
EPS = 1e-6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "temp/burst_eval/coverage.csv"))
    a = ap.parse_args()
    rows = []
    n = 0
    for ds in SOURCES:
        for subj in paths.subjects(ds):
            if not all(paths.recon(ds, subj, arm, "breath", 0).is_file() for arm in ARMS):
                continue
            shape, aff = subject_grid(ds, subj)
            content = load_canon(str(paths.fov_mask(ds, subj)), shape, aff) > 0.5
            heart = load_canon(str(paths.heart_mask(ds, subj)), shape, aff) > 0.5
            roi = heart & content
            import json
            T = json.load(open(paths.manifest(ds, subj)))["T"]
            acc = {arm: {"hole": [], "ncc_roi": [], "ncc_common": [], "ncc_filled": []} for arm in ARMS}
            for t in range(T):
                gt = load_canon(str(paths.bundle_stack(ds, subj, "gt", t)), shape, aff)
                rec = {arm: load_canon(str(paths.recon(ds, subj, arm, "breath", t)), shape, aff) for arm in ARMS}
                cov = {arm: rec[arm] > EPS for arm in ARMS}
                common = roi.copy()
                for arm in ARMS:
                    common &= cov[arm]
                for arm in ARMS:
                    acc[arm]["hole"].append(float((roi & ~cov[arm]).sum() / max(roi.sum(), 1)))
                    acc[arm]["ncc_roi"].append(ncc(rec[arm], gt, roi))
                    acc[arm]["ncc_common"].append(ncc(rec[arm], gt, common))
                    acc[arm]["ncc_filled"].append(ncc(rec[arm], gt, roi & cov[arm]))
            for arm in ARMS:
                rows.append({"source": ds, "subject": subj, "arm": arm,
                             **{k: float(np.nanmean(v)) for k, v in acc[arm].items()}})
            n += 1
            print(f"[{n}] {ds}/{subj}: " + "  ".join(
                f"{arm.split('_')[1]}: hole {np.mean(acc[arm]['hole']):.3f} ncc {np.nanmean(acc[arm]['ncc_roi']):.3f}"
                f"/{np.nanmean(acc[arm]['ncc_common']):.3f}" for arm in ARMS), flush=True)
            if a.limit and n >= a.limit:
                break
        if a.limit and n >= a.limit:
            break
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n-> {a.out}  ({n} subjects)\n")
    print(f"{'arm':34s} {'hole_frac':>9s} {'ncc_roi':>8s} {'ncc_common':>10s} {'ncc_filled':>10s}")
    for arm in ARMS:
        r = [x for x in rows if x["arm"] == arm]
        print(f"{arm:34s} " + " ".join(f"{np.median([x[k] for x in r]):>{w}.3f}"
                                      for k, w in (("hole", 9), ("ncc_roi", 8), ("ncc_common", 10), ("ncc_filled", 10))))


if __name__ == "__main__":
    main()
