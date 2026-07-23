"""Aggregate per-subject metrics.json across a dataset into a cohort summary.

Reads  volumes/<dataset>/out/<subject>/<method>/metrics.json  (produced by
assemble_and_gif.py) and reports cohort clean/breath PSNR/SSIM + breathing-cost
(clean - breath) mean+-std, plus a volunteer-vs-patient split (subjects whose
name contains 'Patient' are pathology; everything else is treated as volunteer/
healthy). CMRxRecon is all-volunteer so it collapses to one group; MIITT is
mixed (10 volunteers + 3 patients).

Run: micromamba run -n svr python evaluation/engine/aggregate.py [dataset=cmrxrecon] [method=svrtk3d]

Paths/naming go through evaluation/paths.py (the single source of truth).
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


def group_of(name):
    return "patient" if "patient" in name.lower() else "volunteer"


def stat(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std()), int(xs.size)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cmrxrecon"
    method = sys.argv[2] if len(sys.argv) > 2 else "svrtk3d"
    root = paths.dataset_root(dataset)
    files = sorted(glob.glob(str(root / "*" / method / "metrics.json")))
    if not files:
        sys.exit(f"no metrics found at {root}/*/{method}/metrics.json")

    rows = []
    for f in files:
        d = json.load(open(f))
        s = d["subject"]
        rows.append({
            "subject": s, "group": group_of(s),
            "clean_psnr": d["clean_psnr_mean"], "clean_ssim": d["clean_ssim_mean"],
            "breath_psnr": d["breath_psnr_mean"], "breath_ssim": d["breath_ssim_mean"],
            "cost_psnr": d["clean_psnr_mean"] - d["breath_psnr_mean"],
            "breath_disp_mm": d["breath_mean_disp_mm"],
        })

    # per-subject table (sorted by group then subject)
    print(f"\n=== {dataset} / {method}  (n={len(rows)}) ===")
    hdr = f"{'subject':<40}{'grp':<10}{'clean':>8}{'breath':>8}{'cost':>7}{'|disp|mm':>9}"
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["group"], r["subject"])):
        print(f"{r['subject']:<40}{r['group']:<10}{r['clean_psnr']:>8.2f}"
              f"{r['breath_psnr']:>8.2f}{r['cost_psnr']:>7.2f}{r['breath_disp_mm']:>9.2f}")

    # cohort + per-group summaries
    def summarize(subset, label):
        if not subset:
            return None
        cp = stat([r["clean_psnr"] for r in subset]); cs = stat([r["clean_ssim"] for r in subset])
        bp = stat([r["breath_psnr"] for r in subset]); bs = stat([r["breath_ssim"] for r in subset])
        ct = stat([r["cost_psnr"] for r in subset]); dz = stat([r["breath_disp_mm"] for r in subset])
        print(f"\n[{label}]  n={cp[2]}")
        print(f"  clean : PSNR {cp[0]:6.2f} +- {cp[1]:.2f} dB   SSIM {cs[0]:.3f} +- {cs[1]:.3f}")
        print(f"  breath: PSNR {bp[0]:6.2f} +- {bp[1]:.2f} dB   SSIM {bs[0]:.3f} +- {bs[1]:.3f}")
        print(f"  breathing cost (clean-breath): {ct[0]:.2f} +- {ct[1]:.2f} dB   |disp| {dz[0]:.2f} +- {dz[1]:.2f} mm")
        return {"n": cp[2],
                "clean_psnr": cp[:2], "clean_ssim": cs[:2],
                "breath_psnr": bp[:2], "breath_ssim": bs[:2],
                "cost_psnr": ct[:2], "breath_disp_mm": dz[:2]}

    summary = {"dataset": dataset, "method": method, "n": len(rows),
               "all": summarize(rows, "ALL")}
    for g in ("volunteer", "patient"):
        sub = [r for r in rows if r["group"] == g]
        if sub:
            summary[g] = summarize(sub, g.upper())
    summary["per_subject"] = rows

    out = paths.summary(dataset, method)              # git-tracked results/<ds>/<arm>.json
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
