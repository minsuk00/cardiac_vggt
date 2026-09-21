#!/usr/bin/env python
"""EF / Dice comparison table for a 24-frame rhythm arm (docs/115), from the per-(method, cohort)
ef_dice score jsons that sbatch/rhythm24_seg.sh writes to temp/rhythm24_ef/<arm>/.

Reports per method: n, EF MAE / bias / slope / r (predicted vs GT EF), EDV/ESV MAE, LV + MYO Dice,
then each VGGT arm PAIRED against fetal_cmr_4d on the subjects both scored (|EF err|, Wilcoxon),
and a per-source breakdown of the paired contrast.

Usage:  python tools/rhythm24_ef_table.py af24 [--json temp/rhythm24_ef/af24_summary.json]
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import linregress, wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = "fetal_cmr_4d"


def short(m):
    return m.replace("vggt_final518_", "").replace("_ep300", "").replace("fetal_cmr_4d", "fetal")


def load(arm):
    D = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "temp", "rhythm24_ef", arm, "*__*.json"))):
        m = os.path.basename(f)[:-5].split("__")[0]
        if m == "gt":
            continue
        for s in json.load(open(f))["per_subject"]:
            if s.get("ef_breath") is None or s.get("ef_gt") is None:
                continue
            D.setdefault(m, {})[(s["cohort"], s["subject"])] = s
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    D = load(a.arm)
    methods = [REF] + sorted(m for m in D if m != REF)
    out = {"arm": a.arm, "methods": {}, "paired_vs_fetal": {}}

    print(f"=== {a.arm} ===")
    print(f"{'method':10}{'n':>5}{'EF MAE':>8}{'bias':>8}{'slope':>7}{'r':>6}{'predEF':>8}{'EDV MAE':>9}"
          f"{'ESV MAE':>9}{'DiceLV ED/ES':>15}{'DiceMYO ED/ES':>15}")
    for m in methods:
        v = list(D.get(m, {}).values())
        if not v:
            continue
        g = np.array([s["ef_gt"] for s in v]); p = np.array([s["ef_breath"] for s in v])
        lr = linregress(g, p)
        mean = lambda k: float(np.nanmean([s.get(k, np.nan) for s in v]))          # noqa: E731
        r = {"n": len(v), "ef_mae": float(np.abs(p - g).mean()), "ef_bias": float((p - g).mean()),
             "slope": float(lr.slope), "r": float(lr.rvalue), "pred_ef_mean": float(p.mean()),
             "gt_ef_mean": float(g.mean()),
             "edv_mae": float(np.mean([abs(s["edv_breath"] - s["edv_gt"]) for s in v])),
             "esv_mae": float(np.mean([abs(s["esv_breath"] - s["esv_gt"]) for s in v])),
             "dice_lv_ed": mean("dice_breath_LV_ED"), "dice_lv_es": mean("dice_breath_LV_ES"),
             "dice_myo_ed": mean("dice_breath_MYO_ED"), "dice_myo_es": mean("dice_breath_MYO_ES")}
        out["methods"][m] = r
        print(f"{short(m):10}{r['n']:5d}{r['ef_mae']:8.2f}{r['ef_bias']:+8.2f}{r['slope']:7.2f}{r['r']:6.2f}"
              f"{r['pred_ef_mean']:8.1f}{r['edv_mae']:9.1f}{r['esv_mae']:9.1f}"
              f"{r['dice_lv_ed']:9.3f}/{r['dice_lv_es']:.3f}{r['dice_myo_ed']:9.3f}/{r['dice_myo_es']:.3f}")

    F = D.get(REF, {})
    print(f"\npaired vs {short(REF)} -- |EF err| (pp), subjects both methods scored:")
    for m in methods[1:]:
        k = [x for x in D[m] if x in F]
        if not k:
            continue
        ef = np.array([abs(F[x]["ef_breath"] - F[x]["ef_gt"]) for x in k])
        ev = np.array([abs(D[m][x]["ef_breath"] - D[m][x]["ef_gt"]) for x in k])
        d = ev - ef
        res = {"n": len(k), "fetal": float(ef.mean()), "vggt": float(ev.mean()), "diff": float(d.mean()),
               "p_wilcoxon": float(wilcoxon(d).pvalue), "vggt_better": int((d < 0).sum()), "per_source": {}}
        print(f"  {short(m):9} n={len(k):3}  fetal {ef.mean():6.2f} | vggt {ev.mean():6.2f} | "
              f"vggt-fetal {d.mean():+6.2f}  (p={res['p_wilcoxon']:.2e}; VGGT better in {res['vggt_better']}/{len(k)})")
        for src in sorted({x[0] for x in k}):
            i = [j for j, x in enumerate(k) if x[0] == src]
            res["per_source"][src] = {"n": len(i), "fetal": float(ef[i].mean()), "vggt": float(ev[i].mean())}
            print(f"      {src:16} n={len(i):3}  fetal {ef[i].mean():6.2f} | vggt {ev[i].mean():6.2f}")
        out["paired_vs_fetal"][m] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
