"""Geometry-vs-appearance diagnosis of the ES under-contraction bias (no-reg ep300).

Per subject, at the GT ES phase (argmin GT LV curve):
  A) annulus test  : annulus = GT LV(ED) & ~GT LV(ES). Normalized intensity
                     a = (mean(img[annulus]) - myo_ref) / (blood_ref - myo_ref)
                     GT should be low (myocardium moved in); pred high => blood still
                     fills the annulus => cavity geometrically too big (GEOMETRY).
  B) sharpness test: mean |grad| on the GT LV-ES boundary shell, pred vs GT, and the
                     same at ED. ES-specific extra blur in pred => APPEARANCE.
  C) threshold test: pseudo-ESV = voxels above blood/myo mid-threshold inside a dilated
                     GT LV-ED region, threshold swept 0.35..0.65. Inflated-but-stable
                     => GEOMETRY; steep threshold slope => blurry boundary (APPEARANCE).
  D) effective-phase: pseudo-LV-volume curve over all 12 pred phases vs the GT curve
                     (same matched-filter measurement) -> how far down the GT contraction
                     curve does the pred minimum reach (contraction fraction)?
Outputs a per-subject CSV + printed group stats (worst-20 shared-EF-error vs rest).
"""
import json, sys, csv
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from scipy import stats

ROOT = "temp/dice_ef_sweep/mirror_full/volumes"
ARM = "vggt_noreg_ep300"
LV, MYO = 1, 2

cov = list(csv.DictReader(open("temp/dice_ef_sweep/difficulty_covariates.csv")))
subj2coh = {r["subj"]: r["coh"] for r in cov}
err = {r["subj"]: float(r["err"]) for r in cov}
worst20 = set(sorted(err, key=lambda s: -err[s])[:20])

ef = {r["subject"]: r for r in json.load(
    open(f"temp/dice_ef_sweep/mirror_full/results/_ef/{ARM}.json"))["per_subject"]}

def norm01(v):
    lo, hi = np.percentile(v[v > 0], [1, 99.5]) if (v > 0).any() else (0, 1)
    return np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)

rows = []
for r in cov:
    subj, coh = r["subj"], r["coh"]
    d = f"{ROOT}/{coh}/out/{subj}"
    try:
        seg = np.asarray(nib.load(f"{d}/heart_seg.nii.gz").dataobj)          # (X,Y,Z,T)
        gt = np.asarray(nib.load(f"{d}/cine_gt.nii.gz").dataobj).astype(np.float32)
        zoom = nib.load(f"{d}/cine_gt.nii.gz").header.get_zooms()[:3]
        vox_ml = float(np.prod(zoom)) / 1000.0
        T = seg.shape[3]
        pred = np.stack([np.asarray(nib.load(
            f"{d}/{ARM}/recon_breath/vol_t{t:02d}.nii.gz").dataobj).astype(np.float32)
            for t in range(T)], axis=-1)                                     # (X,Y,Z,T)
    except Exception as e:
        print(f"skip {subj}: {e}", file=sys.stderr); continue

    lv_curve = (seg == LV).sum(axis=(0, 1, 2))
    tED, tES = int(np.argmax(lv_curve)), int(np.argmin(lv_curve))
    lvED, lvES = seg[..., tED] == LV, seg[..., tES] == LV
    myoES = seg[..., tES] == MYO
    if lvES.sum() < 20 or myoES.sum() < 20:
        print(f"skip {subj}: tiny ES seg", file=sys.stderr); continue

    gtN = norm01(gt); prN = norm01(pred)
    gES, pES = gtN[..., tES], prN[..., tES]
    gED, pED = gtN[..., tED], prN[..., tED]

    # intensity refs from GT ES: blood = eroded LV core, myo = myocardium
    core = ndi.binary_erosion(lvES, iterations=2)
    if core.sum() < 10: core = lvES
    blood_g, myo_g = np.median(gES[core]), np.median(gES[myoES])
    blood_p, myo_p = np.median(pES[core]), np.median(pES[myoES])

    def a_norm(img, m, blood, myo):
        return float((img[m].mean() - myo) / max(blood - myo, 1e-3))

    # A) annulus
    annulus = lvED & ~ndi.binary_dilation(lvES, iterations=1)
    if annulus.sum() < 20:
        aG = aP = np.nan
    else:
        aG = a_norm(gES, annulus, blood_g, myo_g)
        aP = a_norm(pES, annulus, blood_p, myo_p)

    # B) boundary sharpness (in-plane gradients only; z is anisotropic)
    def sharp(img, m):
        shell = ndi.binary_dilation(m, iterations=1) & ~ndi.binary_erosion(m, iterations=1)
        gx, gy = np.gradient(img, axis=(0, 1))
        return float(np.hypot(gx, gy)[shell].mean())
    sES = sharp(pES, lvES) / max(sharp(gES, lvES), 1e-6)
    sED = sharp(pED, lvED) / max(sharp(gED, lvED), 1e-6)

    # C) threshold pseudo-ESV inside dilated LV-ED region
    roi = ndi.binary_dilation(lvED, iterations=2)
    ths = np.linspace(0.35, 0.65, 7)
    def pseudo(img, blood, myo):
        lev = myo + ths * (blood - myo)
        return np.array([np.count_nonzero(img[roi] >= L) for L in lev]) * vox_ml
    pvG, pvP = pseudo(gES, blood_g, myo_g), pseudo(pES, blood_p, myo_p)
    mid = 3  # th=0.5
    # sensitivity: relative change of pseudo-vol across the sweep, per unit threshold
    sensG = float((pvG[0] - pvG[-1]) / max(pvG[mid], 1e-3))
    sensP = float((pvP[0] - pvP[-1]) / max(pvP[mid], 1e-3))

    # D) effective contraction fraction from matched pseudo-volume curves
    def curve_pseudo(vol4, blood, myo):
        lev = myo + 0.5 * (blood - myo)
        return np.array([np.count_nonzero(vol4[..., t][roi] >= lev)
                         for t in range(T)]) * vox_ml
    cG, cP = curve_pseudo(gtN, blood_g, myo_g), curve_pseudo(prN, blood_p, myo_p)
    edG, esG = cG.max(), cG.min()
    edP, esP = cP.max(), cP.min()
    # where pred's minimum lands on GT's contraction span (1 = full GT contraction)
    cfrac = float((edG - esP) / max(edG - esG, 1e-3))
    cfrac_gt = 1.0

    rows.append(dict(subj=subj, coh=coh, group="worst20" if subj in worst20 else "rest",
        err=err[subj], ef_gt=float(r["ef"]),
        esv_gt_seg=ef.get(subj, {}).get("esv_gt"), esv_pred_seg=ef.get(subj, {}).get("esv_breath"),
        tED=tED, tES=tES,
        annulus_gt=aG, annulus_pred=aP,
        sharp_ratio_ED=sED, sharp_ratio_ES=sES,
        pseudoESV_gt=float(pvG[mid]), pseudoESV_pred=float(pvP[mid]),
        th_sens_gt=sensG, th_sens_pred=sensP,
        contract_frac_pred=cfrac))

with open("temp/dice_ef_sweep/es_diag_noreg.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

import pandas as pd
df = pd.DataFrame(rows)
print(f"n={len(df)}  (worst20={len(df[df.group=='worst20'])})")
def rep(col):
    a, b = df[df.group == "worst20"][col].dropna(), df[df.group == "rest"][col].dropna()
    u = stats.mannwhitneyu(a, b) if len(a) and len(b) else None
    print(f"{col:22s} all={df[col].median():7.3f}  worst20={a.median():7.3f}  "
          f"rest={b.median():7.3f}  MWU p={u.pvalue:.2g}" if u else col)
for c in ["annulus_gt", "annulus_pred", "sharp_ratio_ED", "sharp_ratio_ES",
          "th_sens_gt", "th_sens_pred", "contract_frac_pred"]:
    rep(c)
d = df.dropna(subset=["annulus_gt", "annulus_pred"])
print("\npaired annulus pred-vs-gt: Wilcoxon p=%.2g, median Δ=%+.3f" % (
    stats.wilcoxon(d.annulus_pred, d.annulus_gt).pvalue,
    (d.annulus_pred - d.annulus_gt).median()))
print("paired sharpness ES-vs-ED ratio: Wilcoxon p=%.2g, median Δ=%+.3f" % (
    stats.wilcoxon(df.sharp_ratio_ES, df.sharp_ratio_ED).pvalue,
    (df.sharp_ratio_ES - df.sharp_ratio_ED).median()))
print("pseudoESV pred-gt (ml): median %+.1f  (seg-based ESV bias for reference)" % (
    (df.pseudoESV_pred - df.pseudoESV_gt).median()))
for c in ["annulus_pred", "sharp_ratio_ES", "th_sens_pred", "contract_frac_pred"]:
    r_, p_ = stats.spearmanr(df[c], df.err, nan_policy="omit")
    print(f"spearman({c:20s}, EF err): r={r_:+.2f} p={p_:.2g}")
