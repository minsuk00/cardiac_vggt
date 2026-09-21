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
            # Keep a subject whose EF is undefined (ef_dice: the LV vanished in >= 1 frame, so ESV
            # and EF are None). Its EDV and Dice ARE defined and are part of the method's record --
            # dropping it here would quietly remove the method's worst cases from every column.
            if s.get("ef_gt") is None or s.get("edv_breath") is None:
                continue
            D.setdefault(m, {})[(s["cohort"], s["subject"])] = s
    return D


def image_metrics(arm):
    """method -> mean unit-peak PSNR / SSIM / NCC over every scored subject of the arm's 5 cohorts."""
    out = {}
    for f in glob.glob(os.path.join(ROOT, "scratch", "eval", f"*_{arm}", "out", "*", "*", "metrics.json")):
        j = json.load(open(f))
        if "breath_psnr_unit_peak_mean" in j:
            out.setdefault(f.split("/")[-2], []).append(
                (j["breath_psnr_unit_peak_mean"], j["breath_ssim_mean"], j["breath_ncc_mean"]))
    return {m: dict(zip(("psnr", "ssim", "ncc"), np.mean(v, axis=0).tolist()), n_img=len(v))
            for m, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    D = load(a.arm)
    methods = [REF] + sorted(m for m in D if m != REF)
    out = {"arm": a.arm, "methods": {}, "paired_vs_fetal": {}}

    img = image_metrics(a.arm)
    rp = os.path.join(ROOT, "temp", "rhythm24_ef", f"{a.arm}_resp_epe.json")     # tools/rhythm24_resp_epe.py
    resp = json.load(open(rp)) if os.path.exists(rp) else {}
    print(f"=== {a.arm} ===   (motion EPE = breathing through-plane EPE vs the TRUE frame-wise shift; "
          f"VGGT only -- Fetal predicts no per-slice shift)")
    print(f"{'method':10}{'n':>5}{'PSNR':>7}{'SSIM':>7}{'NCC':>7}{'EF MAE':>8}{'bias':>8}{'r':>6}{'EDV MAE':>9}"
          f"{'ESV MAE':>9}{'DiceLV ED/ES':>15}{'DiceMYO ED/ES':>15}{'motionEPE mm':>14}")
    for m in methods:
        v = list(D.get(m, {}).values())
        if not v:
            continue
        e = [s for s in v if s.get("ef_breath") is not None]      # EF/ESV: only where they are defined
        g = np.array([s["ef_gt"] for s in e]); p = np.array([s["ef_breath"] for s in e])
        lr = linregress(g, p)
        mean = lambda k: float(np.nanmean([np.nan if s.get(k) is None else s[k] for s in v]))   # noqa: E731
        r = {"n": len(v), "n_ef": len(e), "ef_mae": float(np.abs(p - g).mean()), "ef_bias": float((p - g).mean()),
             "slope": float(lr.slope), "r": float(lr.rvalue), "pred_ef_mean": float(p.mean()),
             "gt_ef_mean": float(g.mean()),
             "edv_mae": float(np.mean([abs(s["edv_breath"] - s["edv_gt"]) for s in v])),
             "esv_mae": float(np.mean([abs(s["esv_breath"] - s["esv_gt"]) for s in e])),
             "dice_lv_ed": mean("dice_breath_LV_ED"), "dice_lv_es": mean("dice_breath_LV_ES"),
             "dice_myo_ed": mean("dice_breath_MYO_ED"), "dice_myo_es": mean("dice_breath_MYO_ES")}
        r.update(img.get(m, {}))
        r["motion_epe_mm"] = resp.get(m, {}).get("epe")
        out["methods"][m] = r
        epe = f"{r['motion_epe_mm']:14.2f}" if r["motion_epe_mm"] is not None else f"{'n/a':>14}"
        print(f"{short(m):10}{r['n']:5d}{r.get('psnr', float('nan')):7.2f}{r.get('ssim', float('nan')):7.3f}"
              f"{r.get('ncc', float('nan')):7.3f}{r['ef_mae']:8.2f}{r['ef_bias']:+8.2f}{r['r']:6.2f}"
              f"{r['edv_mae']:9.1f}{r['esv_mae']:9.1f}"
              f"{r['dice_lv_ed']:9.3f}/{r['dice_lv_es']:.3f}{r['dice_myo_ed']:9.3f}/{r['dice_myo_es']:.3f}{epe}")

    F = D.get(REF, {})
    print(f"\npaired vs {short(REF)} -- |EF err| (pp), subjects both methods scored:")
    for m in methods[1:]:
        k = [x for x in D[m] if x in F and F[x].get("ef_breath") is not None and D[m][x].get("ef_breath") is not None]
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
