#!/usr/bin/env python
"""WHEN does Fetal CMR 4D fail? A conditional failure analysis on already-scored data.

No method is modified and nothing is re-run: the baseline keeps its author-faithful input contract
(all T frames of every slice). This asks a different question -- given that fixed method, which
PROPERTIES OF A SUBJECT make it fail, and are any of them properties where VGGT does not fail.

Conditions, all measured per subject from data already on disk:

  OUT OF PHASE (its self-gating is wrong)
    ed_err_abs      mean |ED-anchor error| in frames, vs the bundle's known roll (gate.json's
                    `ed_err_frames_diag`). This is the per-slice constant error -- the one the
                    paper's cross-slice sync is supposed to remove and our SAX adaptation cannot.
    circ_std_deg    circular sd of those errors ACROSS slices. Slices anchored to inconsistent
                    phases are mutually mis-registered, which no output-phase relabeling fixes.
    within_1_frame  fraction of slices anchored to within one frame.

  UNGATEABLE (its self-gating has nothing to work with)
    anchored_frac   fraction of slices where an LV was segmentable at all. Base/apex slices have
                    none (docs/35 par.6); they keep offset 0 and are left to robust statistics.
    lv_coverage     mean fraction of frames in which a slice shows any LV.

  STATIC HEART (there is little motion to reconstruct)
    gt_motion_rms   mean pairwise RMS intensity difference between the 12 GT volumes inside the
                    heart ROI. An IMAGE-domain measure on purpose: the obvious volume-domain one,
                    (max-min)/max of the LV curve, is ALGEBRAICALLY IDENTICAL to EF (measured:
                    r = 1.000 against gt_ef), so it tests nothing that gt_ef does not.
    gt_ef           the GT ejection fraction itself.

  GEOMETRY
    D, dz_mm        slice count and pitch.

Outcomes: per-subject EF error and mean Dice for fetal_cmr_4d and a VGGT arm, and the PAIRED
DIFFERENCE (VGGT error - fetal error; > 0 means fetal wins). The last is the one that matters --
a condition that hurts both methods equally is not a finding.

Usage:
    PYTHONPATH=training:. python tools/fetal_failure_modes.py [--vggt <arm>] [--json OUT.json]
    PYTHONPATH=training:. python tools/fetal_failure_modes.py --cohorts cmrx2024_af   # rhythm arms
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

T = 12
FETAL = "fetal_cmr_4d"
DICE_KEYS = [("LV", "ED"), ("LV", "ES"), ("MYO", "ED"), ("MYO", "ES"), ("RV", "ED"), ("RV", "ES")]


def ang(x):
    return (np.asarray(x, float) + np.pi) % (2 * np.pi) - np.pi


def gate_conditions(cohort, subj, arm=FETAL):
    """-> dict of gating-quality conditions from the arm's own gate.json, or None."""
    p = paths.subject_dir(cohort, subj) / arm / "gate" / "gate.json"
    if not p.is_file():
        return None
    j = json.load(open(p))
    per = j.get("per_slice") or []
    errs = [s["ed_err_frames_diag"] for s in per if "ed_err_frames_diag" in s]
    D = int(j.get("D") or len(per) or 1)
    out = {"D": D,
           "anchored_frac": (j.get("n_anchored") or 0) / max(D, 1),
           "lv_coverage": float(np.mean([s.get("lv_coverage", 0.0) for s in per])) if per else np.nan}
    if errs:
        a = np.array(errs, float) * 2 * np.pi / T
        R = abs(np.mean(np.exp(1j * a)))
        out["ed_err_abs"] = float(np.mean(np.abs(ang(a))) * T / (2 * np.pi))
        out["circ_std_deg"] = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-9)))))
        out["within_1_frame"] = float(np.mean(np.abs(errs) <= 1))
    else:
        out.update(ed_err_abs=np.nan, circ_std_deg=np.nan, within_1_frame=np.nan)
    return out


def gt_motion(cohort, subj):
    """-> how much the GT actually MOVES, in the image domain, inside the heart ROI.

    Mean pairwise RMS over the 12 GT volumes, restricted to `mask_heart_pad10` (the scorer's ROI),
    so it measures visible motion rather than LV cavity volume. Deliberately not the LV-curve
    excursion: that equals EF exactly and would be a duplicate regressor.
    """
    d = paths.subject_dir(cohort, subj)
    mp = d / "mask_heart_pad10.nii.gz"
    if not mp.is_file():
        return None
    try:
        V = np.stack([np.asarray(nib.load(str(d / "gt" / f"gt_t{f:02d}.nii.gz")).dataobj,
                                 dtype=np.float32) for f in range(T)])
    except FileNotFoundError:
        return None
    m = np.asarray(nib.load(str(mp)).dataobj) > 0.5
    if m.sum() == 0:
        return None
    v = V[:, m]
    iu = np.triu_indices(T, 1)
    return {"gt_motion_rms": float(np.sqrt(((v[iu[0]] - v[iu[1]]) ** 2).mean(1)).mean())}


def ef_rows(path):
    if not os.path.isfile(path):
        return {}
    d = json.load(open(path))
    out = {}
    for r in d.get("per_subject", []):
        if r.get("ef_breath") is None or r.get("ef_gt") is None:
            continue
        dice = [r.get(f"dice_breath_{a}_{b}") for a, b in DICE_KEYS]
        dice = [x for x in dice if x is not None]
        out[(r.get("cohort"), r.get("subject"))] = {
            "ef_err": abs(r["ef_breath"] - r["ef_gt"]), "ef_gt": r["ef_gt"],
            "ef_pred": r["ef_breath"], "dice": float(np.mean(dice)) if dice else np.nan}
    return out


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 4 or x.std() == 0 or y.std() == 0:
        return np.nan, np.nan, len(x)
    r = float(np.corrcoef(x, y)[0, 1])
    try:
        from scipy.stats import pearsonr
        return r, float(pearsonr(x, y)[1]), len(x)
    except Exception:
        return r, np.nan, len(x)


def tercile_table(rows, cond, outs, label):
    v = np.array([r.get(cond, np.nan) for r in rows], float)
    m = np.isfinite(v)
    if m.sum() < 9:
        return
    q = np.nanpercentile(v[m], [33.3, 66.7])
    bands = [("low ", v <= q[0]), ("mid ", (v > q[0]) & (v <= q[1])), ("HIGH", v > q[1])]
    print(f"\n  {label}  ({cond})")
    print(f"    {'band':5s} {'n':>3s} {'range':>15s} " + " ".join(f"{o:>16s}" for o in outs))
    for name, sel in bands:
        sel = sel & m
        if sel.sum() == 0:
            continue
        vals = []
        for o in outs:
            a = np.array([r.get(o, np.nan) for r in rows], float)[sel]
            vals.append(np.nanmean(a))
        print(f"    {name:5s} {sel.sum():3d} [{v[sel].min():6.2f},{v[sel].max():6.2f}] "
              + " ".join(f"{x:16.2f}" for x in vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vggt", default="vggt_final518_diff1000_ep300")
    ap.add_argument("--cohorts", nargs="+", default=None,
                    help="default: every non-rhythm cohort in the test _ef files (the live campaign)")
    ap.add_argument("--ef-dir", default=None,
                    help="dir holding <arm>.json EF files (default evaluation/metric_results/test/_ef)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    efdir = args.ef_dir or os.path.join(ROOT, "evaluation", "metric_results", "test", "_ef")
    fet = ef_rows(os.path.join(efdir, f"{FETAL}.json"))
    vgg = ef_rows(os.path.join(efdir, f"{args.vggt}.json"))
    keys = sorted(set(fet) & set(vgg))
    if args.cohorts:
        keys = [k for k in keys if k[0] in args.cohorts]
    else:
        keys = [k for k in keys if not k[0].startswith("cmrx2024_")]
    print(f"fetal_cmr_4d vs {args.vggt}: {len(keys)} subjects with both scored")

    rows = []
    for c, s in keys:
        g = gate_conditions(c, s)
        if g is None:
            continue
        r = {"cohort": c, "subject": s, **g}
        mo = gt_motion(c, s)
        if mo:
            r.update(mo)
        man = paths.manifest(c, s)
        if os.path.isfile(man):
            m = json.load(open(man))
            r["dz_mm"] = m.get("dz_mm")
        r["fetal_ef_err"] = fet[(c, s)]["ef_err"]
        r["vggt_ef_err"] = vgg[(c, s)]["ef_err"]
        r["gt_ef"] = fet[(c, s)]["ef_gt"]
        r["fetal_dice"] = fet[(c, s)]["dice"]
        r["vggt_dice"] = vgg[(c, s)]["dice"]
        r["ef_margin"] = r["vggt_ef_err"] - r["fetal_ef_err"]      # >0 = fetal wins
        r["dice_margin"] = r["vggt_dice"] - r["fetal_dice"]        # >0 = VGGT wins
        rows.append(r)
    print(f"with a readable gate.json: {len(rows)}\n")
    if not rows:
        sys.exit("no rows -- check --cohorts / --ef-dir")

    CONDS = ["ed_err_abs", "circ_std_deg", "within_1_frame", "anchored_frac", "lv_coverage",
             "gt_motion_rms", "gt_ef", "D", "dz_mm"]
    OUTS = ["fetal_ef_err", "vggt_ef_err", "ef_margin", "fetal_dice", "vggt_dice"]

    print("=== correlations: condition vs outcome  (r, and p) ===")
    print(f"  {'condition':16s} " + " ".join(f"{o:>22s}" for o in OUTS))
    for c in CONDS:
        x = [r.get(c, np.nan) for r in rows]
        line = f"  {c:16s} "
        for o in OUTS:
            r_, p_, n_ = pearson(x, [r.get(o, np.nan) for r in rows])
            star = "*" if (np.isfinite(p_) and p_ < 0.05) else " "
            line += f"{r_:+8.3f} (p={p_:7.3g}){star}" if np.isfinite(r_) else f"{'--':>22s}"
        print(line)

    print("\n\n=== tercile tables: is there a regime where FETAL loses? "
          "(ef_margin > 0 = fetal wins) ===")
    for c in CONDS:
        tercile_table(rows, c, ["fetal_ef_err", "vggt_ef_err", "ef_margin",
                                "fetal_dice", "vggt_dice"], c)

    print("\n\n=== the worst subjects for fetal (top 12 by EF error) ===")
    w = sorted(rows, key=lambda r: -r["fetal_ef_err"])[:12]
    print(f"  {'subject':22s} {'fetalEF':>8s} {'vggtEF':>7s} {'gtEF':>6s} {'excur':>6s} "
          f"{'edErr':>6s} {'circSd':>7s} {'anch':>5s}")
    for r in w:
        print(f"  {r['subject'][:22]:22s} {r['fetal_ef_err']:8.1f} {r['vggt_ef_err']:7.1f} "
              f"{r['gt_ef']:6.1f} {r.get('gt_motion_rms', np.nan):6.3f} "
              f"{r.get('ed_err_abs', np.nan):6.2f} {r.get('circ_std_deg', np.nan):7.1f} "
              f"{r.get('anchored_frac', np.nan):5.2f}")

    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
