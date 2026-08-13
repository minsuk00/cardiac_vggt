"""E0 scorer — one common CorSeg readout for every arm/checkpoint (docs/66 campaign).

Consumes a dump dir produced by tools/e0_dump_phase_sweep.py (t00/..t11/ of val_volumes-format
NIfTIs, splat order (D,H,W) = (Z,Y,X)), segments every pred+GT volume with CorSeg, and reports:

  1. EF vs GT-EF (cardiac_phase.csv ED/ES phases): slope / Spearman / MAE + bootstrap CIs
     — same definition as the in-training `ef_eval` metric.
  2. Curve EF: EF from each subject's own LV(t) extremes (max/min over the 12 phases).
  3. Phase-transfer coefficient (per subject): OLS slope + Pearson of LV_pred(t) on LV_gt(t).
     Because the dump driver holds non-reference slots fixed across t, this directly measures
     whether slot 0 (the target-phase reference) controls the reconstruction. A conditioning
     failure shows a flat LV_pred(t) -> coefficient ~0; a working model tracks LV_gt(t) -> ~1.
  4. Amplitude ratio: (max-min LV_pred) / (max-min LV_gt), cohort median.

GT segs are cached via --gt-seg-dir (the GT volumes are identical across arms on the same split).

Usage:
  micromamba run -n svr python tools/e0_score_volumes.py \
      --dump result/e0_dumps/<name> --out result/e0_dumps/<name>_score.json \
      --gt-seg-dir result/e0_dumps/_gt_segs_cmrx24val [--dz 12.0]
"""
import argparse
import glob
import json
import os
import re
import sys

import nibabel as nib
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)                              # `vggt.*` (checkpoint staging in ef_eval)
sys.path.insert(0, os.path.join(_REPO, "training"))
from ef_eval import LV_LABEL, _ef_stats, _lv_ml, _spearman, load_gt_ef, run_corseg  # noqa: E402

IN_PLANE_MM = 1.4
_FNAME = re.compile(r"subj\d+_t(\d+)_(.+)_(pred|gt)\.nii\.gz$")


def discover(dump_dir):
    """{(subject, t, kind): path} from every t??/ subdir."""
    found = {}
    for path in glob.glob(os.path.join(dump_dir, "t??", "*.nii.gz")):
        m = _FNAME.search(os.path.basename(path))
        if m:
            found[(m.group(2), int(m.group(1)), m.group(3))] = path
    if not found:
        raise SystemExit(f"no val_volumes dumps under {dump_dir}/t??/")
    return found


def stage_for_corseg(found, kind, stage_dir, dz_flag):
    """Rewrite each (D,H,W) dump as CorSeg input: (X,Y,Z) array, true (1.4,1.4,dz) affine.

    CorSeg reads spacing from the header (corseg_infer.segment_nifti), and old-code dumps
    carry an identity affine — so the affine is ALWAYS rewritten here, from the dump header
    when it looks canonical (in-plane 1.4) and from --dz otherwise.
    """
    os.makedirs(stage_dir, exist_ok=True)
    for (subject, t, k), path in sorted(found.items()):
        if k != kind:
            continue
        dst = os.path.join(stage_dir, f"{subject}_t{t:02d}_0000.nii.gz")
        if os.path.exists(dst):
            continue
        im = nib.load(path)
        zooms = [float(z) for z in im.header.get_zooms()[:3]]
        if abs(zooms[1] - IN_PLANE_MM) < 0.01:      # current-format dump: diag(dz, 1.4, 1.4)
            dz = zooms[0]
        elif dz_flag is not None:                    # old-format identity affine
            dz = float(dz_flag)
        else:
            raise SystemExit(f"{path}: identity affine and no --dz given")
        arr = np.transpose(np.asarray(im.dataobj, dtype=np.float32), (2, 1, 0))  # (D,H,W)->(X,Y,Z)
        nib.save(nib.Nifti1Image(arr, np.diag([IN_PLANE_MM, IN_PLANE_MM, dz, 1.0])), dst)


def segment_dir(stage_dir, seg_dir):
    have = {os.path.basename(p) for p in glob.glob(os.path.join(seg_dir, "*.nii.gz"))}
    need = {os.path.basename(p)[: -len("_0000.nii.gz")] + ".nii.gz"
            for p in glob.glob(os.path.join(stage_dir, "*_0000.nii.gz"))}
    if not need.issubset(have):
        run_corseg(stage_dir, seg_dir)


def lv_curves(seg_dir, keys):
    """{subject: {t: LV_ml}} from the segs for the requested (subject, t) keys."""
    lv = {}
    for subject, t in keys:
        p = os.path.join(seg_dir, f"{subject}_t{t:02d}.nii.gz")
        if os.path.exists(p):
            lv.setdefault(subject, {})[t] = _lv_ml(p, LV_LABEL["corseg"])
    return lv


def csv_id(raw_subject, csv_keys):
    """Map a dump's flattened seq_name subject (e.g.
    'CMRxRecon2024_Cine_combined_CMRx24_Test_P012_sax') to its cardiac_phase.csv id
    ('CMRx24_Test_P012') by longest-substring match. Returns None when the subject has
    no EF GT (fine — the phase-transfer metrics don't need the CSV)."""
    hits = [k for k in csv_keys if k in raw_subject]
    return max(hits, key=len) if hits else None


def bootstrap_ci(gts, preds, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    gts, preds = np.asarray(gts, float), np.asarray(preds, float)
    slopes, spears = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(gts), len(gts))
        g, p = gts[idx], preds[idx]
        if np.std(g) < 1.0:
            continue
        slopes.append(float(np.polyfit(g, p, 1)[0]))
        spears.append(_spearman(g, p))
    pct = lambda v: [float(np.nanpercentile(v, q)) for q in (2.5, 97.5)] if v else None
    return {"slope_ci95": pct(slopes), "spearman_ci95": pct(spears)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None, help="JSON output (default <dump>_score.json)")
    ap.add_argument("--csv", default=os.path.join(_REPO, "scratch/data/whs/cardiac_phase.csv"))
    ap.add_argument("--gt-seg-dir", default=None,
                    help="shared GT seg cache (GT identical across arms on one split)")
    ap.add_argument("--dz", type=float, default=None,
                    help="z pitch (mm) for old-format dumps with identity affines")
    args = ap.parse_args()

    dump = os.path.abspath(args.dump)
    out_json = args.out or dump.rstrip("/") + "_score.json"
    found = discover(dump)
    subjects = sorted({s for s, _, _ in found})
    phases = sorted({t for _, t, _ in found})
    print(f"[e0] {len(subjects)} subjects x {len(phases)} phases from {dump}")

    stage_p = os.path.join(dump, "_stage_pred")
    seg_p = os.path.join(dump, "_seg_pred")
    stage_for_corseg(found, "pred", stage_p, args.dz)
    segment_dir(stage_p, seg_p)

    seg_g = args.gt_seg_dir or os.path.join(dump, "_seg_gt")
    stage_g = seg_g + "_stage"
    stage_for_corseg(found, "gt", stage_g, args.dz)
    segment_dir(stage_g, seg_g)

    keys = sorted({(s, t) for s, t, _ in found})
    lv_pred = lv_curves(seg_p, keys)
    lv_gt = lv_curves(seg_g, keys)

    gt_csv = load_gt_ef(args.csv)
    rows, per_subject = [], {}
    for s in subjects:
        cp, cg = lv_pred.get(s, {}), lv_gt.get(s, {})
        ts = sorted(set(cp) & set(cg))
        if len(ts) < 3:
            continue
        p = np.array([cp[t] for t in ts]); g = np.array([cg[t] for t in ts])
        d = {
            "lv_pred": {int(t): round(cp[t], 2) for t in ts},
            "lv_gt": {int(t): round(cg[t], 2) for t in ts},
            "transfer_slope": float(np.polyfit(g, p, 1)[0]) if np.std(g) > 1e-6 else float("nan"),
            "transfer_pearson": (float(np.corrcoef(g, p)[0, 1])
                                 if np.std(g) > 1e-6 and np.std(p) > 1e-6 else float("nan")),
            "amp_ratio": float((p.max() - p.min()) / max(g.max() - g.min(), 1e-6)),
            "ef_curve_pred": float((p.max() - p.min()) / max(p.max(), 1e-6) * 100.0),
            "ef_curve_gt": float((g.max() - g.min()) / max(g.max(), 1e-6) * 100.0),
        }
        cid = csv_id(s, gt_csv.keys())
        if cid is not None and gt_csv[cid][3] == "ok":
            ed, es, ef_gt, _ = gt_csv[cid]
            if ed in cp and es in cp and cp[ed] > 0:
                d["ef_csv_pred"] = float((cp[ed] - cp[es]) / cp[ed] * 100.0)
                d["ef_csv_gt"] = float(ef_gt)
                rows.append((s, float(ef_gt), d["ef_csv_pred"]))
        per_subject[s] = d

    gts = [g for _, g, _ in rows]; preds = [p for _, _, p in rows]
    ef_stats = _ef_stats(gts, preds)
    result = {
        "dump": dump,
        "n_subjects": len(per_subject),
        "n_ef": len(rows),
        "ef_csv": ef_stats,
        "ef_csv_bootstrap": bootstrap_ci(gts, preds) if ef_stats else None,
        "ef_curve": _ef_stats([per_subject[s]["ef_curve_gt"] for s in per_subject],
                              [per_subject[s]["ef_curve_pred"] for s in per_subject]),
        "phase_transfer": {
            "slope_median": float(np.nanmedian([d["transfer_slope"] for d in per_subject.values()])),
            "slope_iqr": [float(np.nanpercentile([d["transfer_slope"] for d in per_subject.values()], q))
                          for q in (25, 75)],
            "pearson_median": float(np.nanmedian([d["transfer_pearson"] for d in per_subject.values()])),
            "amp_ratio_median": float(np.nanmedian([d["amp_ratio"] for d in per_subject.values()])),
        },
        "per_subject": per_subject,
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=1)

    pt = result["phase_transfer"]
    print(f"[e0] EF(csv):   " + (f"slope {ef_stats['slope']:+.3f}  spearman {ef_stats['spearman']:+.3f}  "
                                 f"MAE {ef_stats['mae_pct']:.2f}pp  n={ef_stats['n']}" if ef_stats else "n/a"))
    if result["ef_csv_bootstrap"]:
        print(f"[e0]            slope CI95 {result['ef_csv_bootstrap']['slope_ci95']}  "
              f"spearman CI95 {result['ef_csv_bootstrap']['spearman_ci95']}")
    print(f"[e0] transfer:  slope median {pt['slope_median']:+.3f}  IQR {pt['slope_iqr']}  "
          f"pearson {pt['pearson_median']:+.3f}  amp_ratio {pt['amp_ratio_median']:.3f}")
    print(f"[e0] -> {out_json}")


if __name__ == "__main__":
    main()
