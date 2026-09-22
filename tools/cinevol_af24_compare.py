"""Paired comparison on a 24-frame rhythm arm: CiNeVol vs Fetal CMR 4D vs VGGT, READ-ONLY on results.

Sources (all written by the standing scorers, identical code for every arm):
  image   <subject>/<arm>/metrics.json                         breath_{ncc,ssim,psnr}_mean
  EF      temp/rhythm24_ef/<ARM>/<arm>__<cohort>.json           per_subject EF / EDV / ESV / mass / Dice / HD95
  curve   scratch/cinevol/segs/<ARM>/cinevol__<cohort>/         kept per-frame segs (CiNeVol only)
Paired on the subjects where EVERY listed arm has the value. Writes one json to temp/.

    PYTHONPATH=baselines/cinevol:training:.:tools:evaluation:evaluation/src/engine \
        python tools/cinevol_af24_compare.py --arm af24
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy.stats import wilcoxon

import cinevol_prepare as cp
import paths

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
ARMS = {"CiNeVol": "cinevol", "Fetal CMR 4D": "fetal_cmr_4d", "VGGT base": "vggt_final518_base_ep300",
        "VGGT diff1000": "vggt_final518_diff1000_ep300"}


def load(arm_ds):
    img, ef = {}, {}
    for name, arm in ARMS.items():
        img[name], ef[name] = {}, {}
        for src in SOURCES:
            ds = f"{src}_{arm_ds}"
            for f in glob.glob(str(paths.dataset_root(ds) / "*" / arm / "metrics.json")):
                m = json.load(open(f))
                img[name][(ds, f.split("/")[-3])] = m
            p = f"temp/rhythm24_ef/{arm_ds}/{arm}__{ds}.json"
            for r in json.load(open(p))["per_subject"]:
                ef[name][(ds, r["subject"])] = r
    return img, ef


def table(title, rows_by_arm, fmt="{:8.3f}", ref="VGGT base"):
    keys = set.intersection(*[set(k for k, v in d.items() if v is not None and np.isfinite(v)) for d in rows_by_arm.values()])
    keys = sorted(keys)
    out = {"n": len(keys)}
    line = f"{title:34s} n={len(keys):3d} |"
    for name, d in rows_by_arm.items():
        v = np.array([d[k] for k in keys])
        out[name] = {"mean": float(v.mean()), "median": float(np.median(v))}
        line += f" {name}: " + fmt.format(v.mean())
    base = np.array([rows_by_arm["CiNeVol"][k] for k in keys])
    for other in rows_by_arm:
        if other == "CiNeVol":
            continue
        o = np.array([rows_by_arm[other][k] for k in keys])
        p = wilcoxon(base, o).pvalue if np.any(base != o) else 1.0
        out[f"p_CiNeVol_vs_{other}"] = float(p)
    print(line + "   | p(CiNeVol vs " + ", ".join(f"{o.split()[0]} {out[f'p_CiNeVol_vs_{o}']:.1e}" for o in rows_by_arm if o != "CiNeVol") + ")")
    return out


def curve_metrics(arm_ds):
    """CiNeVol only: per-frame LV volume error (% of GT EDV) and curve correlation, from kept segs."""
    res = {}
    for src in SOURCES:
        ds = f"{src}_{arm_ds}"
        d = f"scratch/cinevol/segs/{arm_ds}/cinevol__{ds}"
        subs = json.load(open(f"{d}/ef_manifest.json"))["subjects"]
        subs = subs if isinstance(subs, list) else list(subs.values())
        for sidx, s in enumerate(subs):
            def cur(tag):
                return np.array([(np.asarray(nib.load(f"{d}/{ds}__s{sidx:03d}__{tag}__t{t:02d}.nii.gz").dataobj) == 1).sum()
                                 for t in range(24)], float)
            gt, pr = cur("gt"), cur("breath")
            if gt.max() <= 0 or pr.std() == 0:
                continue
            res[(ds, s["subject"] if isinstance(s, dict) else s)] = (100 * np.abs(pr - gt).mean() / gt.max(),
                                                                       float(np.corrcoef(gt, pr)[0, 1]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="af24")
    a = ap.parse_args()
    img, ef = load(a.arm)
    out = {"arm": a.arm, "arms": ARMS}
    print(f"=== {a.arm}: paired on subjects where every arm has the value; p = paired Wilcoxon, CiNeVol vs each ===")
    for key, lab in (("breath_ncc_mean", "NCC"), ("breath_ssim_mean", "SSIM"), ("breath_psnr_mean", "PSNR dB")):
        out[lab] = table(lab, {n: {k: m.get(key) for k, m in d.items()} for n, d in img.items()}, "{:7.3f}")
    g = lambda r, k: r.get(k)
    def derived(fn):
        return {n: {k: fn(r) for k, r in d.items()} for n, d in ef.items()}
    ok = lambda r, *ks: all(r.get(k) is not None for k in ks)
    out["EF abs err (pts)"] = table("EF abs error (points)", derived(lambda r: abs(r["ef_breath"] - r["ef_gt"]) if ok(r, "ef_breath", "ef_gt") else None), "{:6.2f}")
    out["EF bias (pts)"] = table("EF signed error (points)", derived(lambda r: r["ef_breath"] - r["ef_gt"] if ok(r, "ef_breath", "ef_gt") else None), "{:+6.2f}")
    out["EDV abs err %"] = table("EDV abs error (% of GT)", derived(lambda r: 100 * abs(r["edv_breath"] - r["edv_gt"]) / r["edv_gt"] if ok(r, "edv_breath", "edv_gt") else None), "{:6.2f}")
    out["ESV abs err %"] = table("ESV abs error (% of GT)", derived(lambda r: 100 * abs(r["esv_breath"] - r["esv_gt"]) / r["esv_gt"] if ok(r, "esv_breath", "esv_gt") else None), "{:6.2f}")
    sv = lambda r, t: r[f"edv_{t}"] - r[f"esv_{t}"]
    out["SV abs err %"] = table("Stroke vol abs error (% of GT)", derived(lambda r: 100 * abs(sv(r, "breath") - sv(r, "gt")) / sv(r, "gt") if ok(r, "edv_breath", "esv_breath") and sv(r, "gt") > 0 else None), "{:6.2f}")
    out["SV bias %"] = table("Stroke vol signed error (%)", derived(lambda r: 100 * (sv(r, "breath") - sv(r, "gt")) / sv(r, "gt") if ok(r, "edv_breath", "esv_breath") and sv(r, "gt") > 0 else None), "{:+6.2f}")
    out["LV mass abs err %"] = table("LV mass abs error (% of GT)", derived(lambda r: 100 * abs(r["lvm_breath"] - r["lvm_gt"]) / r["lvm_gt"] if ok(r, "lvm_breath", "lvm_gt") and r["lvm_gt"] > 0 else None), "{:6.2f}")
    for k in ("dice_breath_LV_ED", "dice_breath_LV_ES", "dice_breath_MYO_ED", "dice_breath_MYO_ES", "dice_breath_RV_ED", "dice_breath_RV_ES"):
        out[k] = table(k, derived(lambda r, k=k: g(r, k)), "{:6.3f}")
    for k in ("hd95_breath_LV_ED", "hd95_breath_LV_ES"):
        out[k] = table(k + " (mm)", derived(lambda r, k=k: g(r, k)), "{:6.2f}")

    # ---- CiNeVol mechanism: does its error follow the label misplacement? ----
    cm = curve_metrics(a.arm)
    mis, eferr, cerr, ccor = [], [], [], []
    for (ds, s), (ce, cc) in cm.items():
        man = json.load(open(paths.manifest(ds, s)))
        phi = cp.cardiac_labels(man); pos = np.array(man["rhythm"]["pos_per_plane"]) % 12
        d = np.abs(phi * 12 - pos); d = np.minimum(d, 12 - d)
        r = ef["CiNeVol"].get((ds, s))
        if r and ok(r, "ef_breath", "ef_gt"):
            mis.append(d.mean()); eferr.append(abs(r["ef_breath"] - r["ef_gt"])); cerr.append(ce); ccor.append(cc)
    mis, eferr, cerr, ccor = map(np.array, (mis, eferr, cerr, ccor))
    print(f"\nCiNeVol only (kept segs, n={len(mis)}): LV volume-curve error {cerr.mean():.1f}% of EDV (median {np.median(cerr):.1f}) | curve corr mean {ccor.mean():.3f} (median {np.median(ccor):.3f})")
    from scipy.stats import spearmanr
    print(f"Spearman vs label misplacement: EF abs err rho={spearmanr(mis, eferr)[0]:+.2f} | curve error rho={spearmanr(mis, cerr)[0]:+.2f} | curve corr rho={spearmanr(mis, ccor)[0]:+.2f}")
    t = np.percentile(mis, [33.3, 66.7]); grp = np.digitize(mis, t)
    for gi, nm in enumerate(("mild third", "middle third", "severe third")):
        m = grp == gi
        print(f"  {nm:13s} misplacement {mis[m].mean():.2f} ph | EF abs err {eferr[m].mean():5.2f} | curve err {cerr[m].mean():5.1f}% | curve corr {ccor[m].mean():.3f}")
    out["cinevol_curve"] = {"n": int(len(mis)), "curve_err_pct_edv_mean": float(cerr.mean()), "curve_corr_mean": float(ccor.mean()),
                            "rho_mis_eferr": float(spearmanr(mis, eferr)[0]), "rho_mis_curveerr": float(spearmanr(mis, cerr)[0])}
    dst = f"temp/cinevol_trial/compare_{a.arm}.json"
    if os.path.exists(dst):
        dst = dst.replace(".json", f"_{int(__import__('time').time())}.json")     # never overwrite
    json.dump(out, open(dst, "w"), indent=1)
    print("saved ->", dst)


if __name__ == "__main__":
    main()
