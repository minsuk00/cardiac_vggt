"""Does the TRAINED model regress to the mean (under-contract)?

Reads Task114 segmentations of the model's predicted volumes (V_canon) and the GT
canonical volumes, both swept over target_t=0..11 per subject. Computes:

  PRIMARY (splat-robust): per-subject predicted ES phase (argmin LV) vs GT ES phase.
    If the model regressed to the population mean, predicted ES would collapse to the
    cohort mode regardless of the subject; if it tracks the subject, they correlate.
    Splat blur shrinks absolute LV ~uniformly across phases, so it preserves WHICH
    phase is the minimum -> this test is robust to the rendering confound.

  SECONDARY (amplitude, splat-confounded): predicted EF vs GT EF curve. Reported with
    the caveat that the splat renderer itself blurs/under-segments V_canon (docs/10),
    so an EF gap mixes model error + rendering, bounded above by the model-free 2% ceiling.

Canonical voxel = 1.4*1.4*12.0 mm = 0.02352 mL (constant). EF/ES are spacing-invariant
(EF is a ratio, ES an argmin) so this only scales the absolute mL printout.
"""
import argparse, glob, json, os, re
import numpy as np
import nibabel as nib

VOX_ML = 1.4 * 1.4 * 12.0 / 1000.0
T = 12
LV = 1


def lv_curve(seg_dir, subj, kind):
    c = np.full(T, np.nan)
    for t in range(T):
        f = os.path.join(seg_dir, f"{subj}_t{t:02d}_{kind}.nii.gz")
        if os.path.exists(f):
            c[t] = (np.asarray(nib.load(f).dataobj) == LV).sum() * VOX_ML
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    segs = glob.glob(os.path.join(args.seg_dir, "*.nii.gz"))
    subjects = sorted({re.match(r"^(.*)_t\d{2}_(pred|gt)$", os.path.basename(s)[:-7]).group(1)
                       for s in segs if re.match(r"^(.*)_t\d{2}_(pred|gt)$", os.path.basename(s)[:-7])})

    rows = []
    for subj in subjects:
        pred = lv_curve(args.seg_dir, subj, "pred")
        gt = lv_curve(args.seg_dir, subj, "gt")
        if np.isnan(pred).any() or np.isnan(gt).any() or (gt <= 0).any() or (pred <= 0).any():
            continue
        rows.append(dict(
            subj=subj, pred=pred.tolist(), gt=gt.tolist(),
            pred_es=int(pred.argmin()), gt_es=int(gt.argmin()),
            pred_ed=int(pred.argmax()), gt_ed=int(gt.argmax()),
            pred_ef=float((pred.max() - pred.min()) / pred.max() * 100),
            gt_ef=float((gt.max() - gt.min()) / gt.max() * 100),
            curve_corr=float(np.corrcoef(pred, gt)[0, 1]),
        ))

    N = len(rows)
    pred_es = np.array([r["pred_es"] for r in rows])
    gt_es = np.array([r["gt_es"] for r in rows])
    pred_ef = np.array([r["pred_ef"] for r in rows])
    gt_ef = np.array([r["gt_ef"] for r in rows])

    es_corr = float(np.corrcoef(pred_es, gt_es)[0, 1]) if N > 2 else float("nan")
    es_within1 = float((np.abs(pred_es - gt_es) <= 1).mean() * 100)
    # what full regression-to-mean predicts: pred_es ~ constant (cohort mode), uncorrelated with gt_es
    gt_es_mode = int(np.bincount(gt_es, minlength=T).argmax())

    summary = dict(
        n=N,
        # PRIMARY: timing
        es_corr_pred_vs_gt=es_corr,
        es_within1_pct=es_within1,
        pred_es_mean=float(pred_es.mean()), pred_es_std=float(pred_es.std()),
        gt_es_mean=float(gt_es.mean()), gt_es_std=float(gt_es.std()),
        gt_es_mode=gt_es_mode,
        # if regressing to mean, pred_es would cluster at the mode; measure its spread
        pred_es_spread_vs_gt=float(pred_es.std() / max(gt_es.std(), 1e-6)),
        # SECONDARY: amplitude
        pred_ef_mean=float(pred_ef.mean()), pred_ef_std=float(pred_ef.std()),
        gt_ef_mean=float(gt_ef.mean()), gt_ef_std=float(gt_ef.std()),
        ef_gap_mean=float((gt_ef - pred_ef).mean()),
        curve_corr_mean=float(np.mean([r["curve_corr"] for r in rows])),
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out_json, "w"), indent=2)

    print(f"N={N} val subjects (model pred vs GT, both segmented, swept over target_t)\n")
    print("=== PRIMARY: ES-phase timing (splat-robust regression test) ===")
    print(f"  corr(pred ES phase, GT ES phase) = {es_corr:+.3f}")
    print(f"  pred ES within +/-1 of GT ES     = {es_within1:.0f}%")
    print(f"  pred ES: mean {pred_es.mean():.2f} std {pred_es.std():.2f}  |  "
          f"GT ES: mean {gt_es.mean():.2f} std {gt_es.std():.2f}  | GT mode t{gt_es_mode}")
    print(f"  pred-ES spread / GT-ES spread = {summary['pred_es_spread_vs_gt']:.2f}  "
          f"(≈1 → tracks subjects; ≈0 → collapsed to mode = regression to mean)")
    print("\n=== SECONDARY: contraction amplitude (EF; splat-confounded) ===")
    print(f"  pred EF {pred_ef.mean():.1f}±{pred_ef.std():.1f}%   GT EF {gt_ef.mean():.1f}±{gt_ef.std():.1f}%   "
          f"gap {summary['ef_gap_mean']:+.1f} pts")
    print(f"  per-subject curve corr(pred,gt) = {summary['curve_corr_mean']:.3f}")
    print("\nper-subject (subj: gt_es->pred_es | gt_ef->pred_ef):")
    for r in rows[:30]:
        print(f"  {r['subj']:12s} ES t{r['gt_es']}->t{r['pred_es']}  EF {r['gt_ef']:.0f}->{r['pred_ef']:.0f}%")


if __name__ == "__main__":
    main()
