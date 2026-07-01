#!/usr/bin/env python
"""Segmentation-based metrics for the in-distribution CMRxRecon eval: ejection fraction (EF)
and Dice, from M&Ms nnU-Net (Task114) segmentations of the per-phase volumes dumped by
`eval/run_cmrxrecon.py --dump-volumes` and segmented by `eval/seg_cmrxrecon.sh`.

WHY EF here: the reference-slot design (docs/24 → docs/25) exists specifically to fix the
FLAT-EF failure — the old target_t model regressed every patient's EF to the cohort mean
(slope≈0). This reproduces the docs/24 pred-EF-vs-true-EF scatter on real CMRxRecon val with
the actual trained model, per respiratory mode (clean vs breathing-corrupted input). "True" EF
is nnU-Net on the GT canonical phases — pseudo-truth (CMRxRecon Cine ships no manual seg), but
self-consistent: the SAME segmenter scores pred and GT, exactly as docs/17 + docs/24.

EF per subject: LV blood-pool (label 1) volume curve over the 12 phases →
    EF = (max − min) / max · 100,  ED = argmax,  ES = argmin   (spacing-invariant).
Dice: Dice(seg(pred_vol), seg(gt_vol)) per structure at the GT's ED and ES phases — does the
reconstruction reproduce the same anatomy the GT volume does, at the two clinically-read phases.

Case naming (from run_cmrxrecon dumps, nnU-Net strips the `_0000`):
    pred:  subj{N}_{clean,breathing}_pred_t{tt}
    gt:    subj{N}_gt_t{tt}                          (mode-independent; dumped once)

Pure numpy/nibabel/scipy/matplotlib — no torch, no nnU-Net import (runs in svr or nnunet).

Usage:
  python eval/seg_metrics_cmrxrecon.py --seg_dir <seg_out_dir> [--out_json ...] [--out_png ...]
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import nibabel as nib

LABELS = {1: "LV", 2: "MYO", 3: "RV"}
LV = 1
VOX_ML = 1.4 * 1.4 * 12.0 / 1000.0     # canonical voxel volume (mL); docs/27 spacing
T = 12

PRED_RE = re.compile(r"^(subj\d+)_(clean|breathing)_pred_t(\d{2})$")
GT_RE = re.compile(r"^(subj\d+)_gt_t(\d{2})$")


def _load(seg_dir, case):
    f = os.path.join(seg_dir, case + ".nii.gz")
    return np.asarray(nib.load(f).dataobj).astype(np.int16) if os.path.exists(f) else None


def _lv_ml(seg):
    return float((seg == LV).sum() * VOX_ML)


def _dice(a, b, lbl):
    A, B = a == lbl, b == lbl
    s = int(A.sum()) + int(B.sum())
    return float("nan") if s == 0 else 2.0 * float((A & B).sum()) / s


def _ef(curve):
    """EF% + ED/ES phase from an LV-volume curve; None if the curve is unusable."""
    if np.isnan(curve).any() or (curve <= 0).any():
        return None
    hi, lo = float(curve.max()), float(curve.min())
    return dict(ef=(hi - lo) / hi * 100.0, ed=int(curve.argmax()), es=int(curve.argmin()))


def _fit(g, p):
    """slope/intercept/pearson/spearman of pred vs true, guarded for tiny N."""
    from scipy.stats import pearsonr, spearmanr
    g, p = np.asarray(g, float), np.asarray(p, float)
    if len(g) < 3:
        return dict(n=len(g), slope=None, intercept=None, pearson=None, spearman=None)
    slope, intc = np.polyfit(g, p, 1)
    return dict(n=len(g), slope=float(slope), intercept=float(intc),
                pearson=float(pearsonr(g, p)[0]), spearman=float(spearmanr(g, p)[0]))


def save_ef_examples(rows, seg_dir, vol_dir, path, n=6, mode="clean"):
    """Demonstration figure: for n subjects spanning the true-EF range, show the
    mid-ventricular slice at ED vs ES for GT and pred (LV contour in red) — you can SEE the
    cavity contract ED→ES, which is what EF = (max−min)/max measures. Needs the input volumes
    (`vol_dir` = the run_cmrxrecon --dump-volumes dir) plus the segs (`seg_dir`)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mrows = sorted([r for r in rows if r["mode"] == mode], key=lambda r: r["gt_ef"])
    if not mrows:
        return
    sel = ([mrows[i] for i in np.linspace(0, len(mrows) - 1, n).round().astype(int)]
           if len(mrows) > n else mrows)   # span the EF range (low→high) instead of the first n

    def _vol(tag, ph):
        f = os.path.join(vol_dir, f"{tag}_t{ph:02d}_0000.nii.gz")
        return np.asarray(nib.load(f).dataobj) if os.path.exists(f) else None

    def _seg(case):
        return _load(seg_dir, case)

    cols = [("GT ED", "gt", "ed"), ("GT ES", "gt", "es"),
            (f"pred ED", mode, "ed"), (f"pred ES", mode, "es")]
    R = len(sel)
    fig, axes = plt.subplots(R, 4, figsize=(9, 2.4 * R), squeeze=False)
    for ri, r in enumerate(sel):
        s, ed, es = r["subj"], r["gt_ed"], r["gt_es"]
        gseg_ed = _seg(f"{s}_gt_t{ed:02d}")
        if gseg_ed is None:
            continue
        zc = int((gseg_ed == LV).sum(axis=(0, 1)).argmax())   # mid-ventricular plane (max LV area)
        for ci, (hdr, kind, which) in enumerate(cols):
            ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
            ph = ed if which == "ed" else es
            tag = f"{s}_gt" if kind == "gt" else f"{s}_{mode}_pred"
            vol = _vol(tag, ph); seg = _seg(f"{tag}_t{ph:02d}")
            if vol is not None:
                ax.imshow(vol[:, :, zc].T, cmap="gray")
            if seg is not None and (seg[:, :, zc] == LV).any():
                ax.contour((seg[:, :, zc] == LV).T, colors="r", linewidths=0.8)
            if ri == 0:
                ax.set_title(hdr, fontsize=10)
        axes[ri][0].set_ylabel(f"subj{s}\ntrue {r['gt_ef']:.0f}%\npred {r['pred_ef']:.0f}%",
                               fontsize=8, rotation=0, labelpad=32, va="center")
    fig.suptitle(f"EF demo ({mode}, z=mid-ventricle) — LV cavity (red) contracts ED→ES", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True, help="nnU-Net segmentation output dir")
    ap.add_argument("--vol_dir", default=None,
                    help="run_cmrxrecon --dump-volumes dir; enables the ED-vs-ES EF demo figure")
    ap.add_argument("--n_examples", type=int, default=6, help="subjects in the EF demo (span the EF range)")
    ap.add_argument("--example_mode", default="clean", choices=("clean", "breathing"),
                    help="which pred mode to show in the EF demo figure")
    ap.add_argument("--out_json", default=None, help="default: <seg_dir>/../seg_metrics.json")
    ap.add_argument("--out_png", default=None, help="default: result/cmrxrecon_ef_correlation.png")
    args = ap.parse_args()

    cases = sorted(os.path.basename(f)[:-7] for f in glob.glob(os.path.join(args.seg_dir, "*.nii.gz")))
    subjects = sorted({m.group(1) for c in cases for m in [GT_RE.match(c)] if m})
    modes = ("clean", "breathing")
    if not subjects:
        raise SystemExit(f"no gt cases (subj*_gt_t*) found in {args.seg_dir}")

    # ── LV-volume curves (pass 1: cheap voxel counts) ────────────────────────
    gt_curve = {s: np.full(T, np.nan) for s in subjects}
    pred_curve = {(s, md): np.full(T, np.nan) for s in subjects for md in modes}
    for c in cases:
        m = GT_RE.match(c)
        if m:
            s, t = m.group(1), int(m.group(2))
            if s in gt_curve and t < T:
                gt_curve[s][t] = _lv_ml(_load(args.seg_dir, c))
            continue
        m = PRED_RE.match(c)
        if m:
            s, md, t = m.group(1), m.group(2), int(m.group(3))
            if (s, md) in pred_curve and t < T:
                pred_curve[(s, md)][t] = _lv_ml(_load(args.seg_dir, c))

    # ── Per-subject EF + Dice (Dice at the GT's ED/ES phases) ────────────────
    # A subject/mode is excluded from EF when its LV curve has a 0-LV phase (failed
    # segmentation) — same convention as tools/cmrxrecon_phase_analysis/
    # analyze_model_contraction.py (docs/24). Track WHICH ones so N never shrinks silently.
    rows = []
    excluded_gt = []          # subjects dropped (whole subject) for an unusable GT LV curve
    excluded_pred = []        # (subj, mode) dropped for an unusable pred LV curve
    for s in subjects:
        gt_ef = _ef(gt_curve[s])
        if gt_ef is None:
            excluded_gt.append(s)
            continue
        ed, es = gt_ef["ed"], gt_ef["es"]
        gt_seg = {ph: _load(args.seg_dir, f"{s}_gt_t{ph:02d}") for ph in (ed, es)}
        for md in modes:
            pe = _ef(pred_curve[(s, md)])
            if pe is None:
                excluded_pred.append(f"{s}/{md}")
                continue
            dice = {}
            for ph, ph_name in ((ed, "ED"), (es, "ES")):
                pseg = _load(args.seg_dir, f"{s}_{md}_pred_t{ph:02d}")
                if pseg is None or gt_seg[ph] is None or pseg.shape != gt_seg[ph].shape:
                    continue
                for lbl, name in LABELS.items():
                    dice[f"{ph_name}_{name}"] = _dice(pseg, gt_seg[ph], lbl)
            rows.append(dict(subj=s, mode=md, gt_ef=gt_ef["ef"], pred_ef=pe["ef"],
                             gt_ed=ed, gt_es=es, pred_ed=pe["ed"], pred_es=pe["es"],
                             gt_lv=gt_curve[s].tolist(), pred_lv=pred_curve[(s, md)].tolist(),
                             dice=dice))

    # ── Aggregate per mode ───────────────────────────────────────────────────
    per_mode = {}
    for md in modes:
        mrows = [r for r in rows if r["mode"] == md]
        if not mrows:
            continue
        fit = _fit([r["gt_ef"] for r in mrows], [r["pred_ef"] for r in mrows])
        dkeys = sorted({k for r in mrows for k in r["dice"]})
        dice_mean = {}
        for k in dkeys:
            vals = [r["dice"][k] for r in mrows if k in r["dice"] and r["dice"][k] == r["dice"][k]]
            if vals:
                dice_mean[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)), n=len(vals))
        per_mode[md] = dict(
            n=len(mrows), ef_fit=fit, dice_mean=dice_mean,
            pred_ef_mean=float(np.mean([r["pred_ef"] for r in mrows])),
            gt_ef_mean=float(np.mean([r["gt_ef"] for r in mrows])),
            es_within1_pct=float(np.mean([min(d, T - d) <= 1 for d in
                                          (abs(r["pred_es"] - r["gt_es"]) for r in mrows)]) * 100),
        )

    out = dict(seg_dir=os.path.abspath(args.seg_dir), n_subjects=len(subjects),
               excluded_gt_subjects=excluded_gt, excluded_pred_modes=excluded_pred,
               per_mode=per_mode, rows=rows)
    out_json = args.out_json or os.path.join(os.path.dirname(args.seg_dir.rstrip("/")), "seg_metrics.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    # ── Print + scatter ──────────────────────────────────────────────────────
    print(f"\n{len(subjects)} subjects discovered; EF uses {len(subjects) - len(excluded_gt)} "
          f"with a usable GT LV curve.")
    if excluded_gt:
        print(f"  EXCLUDED {len(excluded_gt)} subject(s) (0-LV phase in GT curve, all modes): "
              f"{', '.join(excluded_gt)}")
    if excluded_pred:
        print(f"  EXCLUDED {len(excluded_pred)} (subj/mode) pair(s) (0-LV phase in pred curve): "
              f"{', '.join(excluded_pred)}")
    for md, pm in per_mode.items():
        fit = pm["ef_fit"]
        sl = "n/a" if fit["slope"] is None else f"{fit['slope']:+.2f}"
        pr = "n/a" if fit["pearson"] is None else f"{fit['pearson']:+.2f}"
        print(f"\n=== [{md}] N={pm['n']}  EF ===")
        print(f"  pred-EF vs true-EF: slope={sl}  pearson={pr}  "
              f"(pred {pm['pred_ef_mean']:.1f}%  true {pm['gt_ef_mean']:.1f}%)  "
              f"ES within±1: {pm['es_within1_pct']:.0f}%")
        for k, d in pm["dice_mean"].items():
            print(f"  Dice {k:8s} = {d['mean']:.3f}±{d['std']:.3f} (n={d['n']})")

    out_png = args.out_png or os.path.join("result", "cmrxrecon_ef_correlation.png")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        fig, ax = plt.subplots(figsize=(5.6, 5.6))
        colors = {"clean": "#2ca02c", "breathing": "#d62728"}
        xs = np.linspace(0, 100, 50)
        for md in modes:
            mrows = [r for r in rows if r["mode"] == md]
            if not mrows:
                continue
            g = np.array([r["gt_ef"] for r in mrows]); p = np.array([r["pred_ef"] for r in mrows])
            fit = per_mode[md]["ef_fit"]
            lbl = f"{md} (n={len(mrows)}"
            lbl += f", slope={fit['slope']:+.2f})" if fit["slope"] is not None else ")"
            ax.scatter(g, p, c=colors[md], s=40, alpha=0.8, edgecolor="white", linewidth=0.5, label=lbl, zorder=3)
            if fit["slope"] is not None:
                ax.plot(xs, fit["slope"] * xs + fit["intercept"], c=colors[md], lw=2.0, zorder=2)
        ax.plot(xs, xs, "--", color="0.55", lw=1, label="identity (slope 1)")
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("true EF (%)  [nnU-Net on GT phases]")
        ax.set_ylabel("predicted EF (%)  [nnU-Net on V_canon]")
        ax.set_title(f"CMRxRecon val EF — reference-slot model (N={len(subjects)})", fontsize=11)
        ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
        fig.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"\nwrote {out_png}")
    except Exception as e:
        print(f"(scatter skipped: {e})")

    # ED-vs-ES demonstration figure (needs the input volumes from --dump-volumes).
    if args.vol_dir:
        ex_png = os.path.join(os.path.dirname(out_png) or ".", "cmrxrecon_ef_examples.png")
        try:
            save_ef_examples(rows, args.seg_dir, args.vol_dir, ex_png,
                             n=args.n_examples, mode=args.example_mode)
        except Exception as e:
            print(f"(EF examples skipped: {e})")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
