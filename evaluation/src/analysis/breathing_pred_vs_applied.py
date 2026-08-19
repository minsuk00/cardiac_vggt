"""breathing_pred_vs_applied.py — cohort through-plane breathing diagnostic.

For a (dataset, arm), reads each subject's `resp_diag.json` (breath arm) — predicted through-plane
Δz vs the applied respiratory displacement per input slot, produced by `run_vggt.resp_diag` — and
reports how well the model recovers breathing:

  - per-subject-averaged  slope / corr / EPE(mm)   (mean ± std over subjects; run_vggt's own values)
  - pooled                slope / corr / EPE(mm)   (over ALL slots of ALL subjects, same formulas:
                                                    slope=polyfit(applied,pred,1)[0], corr=pearson,
                                                    EPE=mean|pred-applied|) — the honest cohort number
  - a pooled scatter PNG (applied vs predicted Δz, y=x + fit line).

Pure disk read (no GPU/model). Writes JSON + PNG under comparison_figures/<ds>/ (GPFS, regenerable).

Run: python evaluation/src/analysis/breathing_pred_vs_applied.py --dataset cmrx2024 --arm vggt_augaggr224hw2_ep300
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402


def fit(applied, pred):
    """(slope, corr, epe) with the SAME definitions as run_vggt.resp_diag."""
    applied, pred = np.asarray(applied, float), np.asarray(pred, float)
    epe = float(np.mean(np.abs(pred - applied))) if pred.size else float("nan")
    slope = corr = float("nan")
    if pred.size >= 2 and applied.std() > 1e-6:
        slope = float(np.polyfit(applied, pred, 1)[0])
        corr = float(np.corrcoef(applied, pred)[0, 1])
    return slope, corr, epe


def mean_std(rows, key):
    xs = [r[key] for r in rows if r[key] is not None and not np.isnan(r[key])]  # drop NaN too (a
    # degenerate subject stores NaN slope/corr — not None — and plain np.mean would poison the cohort)
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs)} if xs \
        else {"mean": None, "std": None, "n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default=None, help="JSON path (PNG alongside); default comparison_figures/<ds>/<arm>_breathing.json")
    args = ap.parse_args()
    ds, arm = args.dataset, args.arm

    rows, pooled_app, pooled_pred = [], [], []
    # Same split rail as run_vggt/aggregate: a stray train/test bundle in out/ must not join
    # the citable breathing-slope cohort (paths.filter_by_split contract).
    split = os.environ.get("SPLIT", "val")
    keep, dropped = paths.filter_by_split(ds, paths.subjects(ds), split)
    for subj, why in dropped:
        print(f"  !! skipping {subj}: {why}", file=sys.stderr)
    for subj in keep:
        rp = paths.resp_diag(ds, subj, arm)
        if not rp.is_file():
            continue
        d = json.loads(rp.read_text()).get("breath")
        if not d or not d.get("pred_dz_mm"):          # skip clean-only / empty
            continue
        rows.append({"subject": subj, "slope": d.get("slope"), "corr": d.get("corr"),
                     "epe": d.get("epe_dz_mm"), "n_slots": d.get("n_slots")})
        pooled_app += d["applied_dz_mm"]
        pooled_pred += d["pred_dz_mm"]
    if not rows:
        sys.exit(f"no breath resp_diag found for {ds}/{arm}")

    p_slope, p_corr, p_epe = fit(pooled_app, pooled_pred)
    summary = {
        "dataset": ds, "arm": arm, "n_subjects": len(rows), "n_slots_pooled": len(pooled_app),
        "per_subject_mean": {k: mean_std(rows, k) for k in ("slope", "corr", "epe")},
        "pooled": {"slope": p_slope, "corr": p_corr, "epe_mm": p_epe},
        "per_subject": rows,
    }
    # Cohort-level per-arm summary -> the FIGURES tree (GPFS): comparison_figures/<ds>/<arm>_breathing.{json,png}.
    out = Path(args.out) if args.out else paths.cohort_fig_dir(ds) / f"{arm}_breathing.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    # pooled scatter
    app, pred = np.asarray(pooled_app), np.asarray(pooled_pred)
    lim = float(max(1.0, np.abs(np.concatenate([app, pred])).max())) * 1.05
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(app, pred, s=6, alpha=0.35)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y=x (perfect)")
    if np.isfinite(p_slope):
        b = pred.mean() - p_slope * app.mean()
        xs = np.array([-lim, lim])
        ax.plot(xs, p_slope * xs + b, "r-", lw=1.5, label=f"fit slope={p_slope:.2f}")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("applied Δz (mm)"); ax.set_ylabel("predicted Δz (mm)")
    ax.set_title(f"{ds} / {arm}\npooled slope {p_slope:.2f}  corr {p_corr:.2f}  "
                 f"EPE {p_epe:.2f} mm   (n_subj={len(rows)})", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    png = out.with_suffix(".png")
    fig.savefig(png, dpi=110); plt.close(fig)

    print(f"-> {out}\n-> {png}")
    print(f"   pooled : slope {p_slope:.3f}  corr {p_corr:.3f}  EPE {p_epe:.3f} mm"
          f"  (n_subj={len(rows)}, n_slots={len(pooled_app)})")
    psm = summary["per_subject_mean"]
    def f3(v):                                        # a degenerate cohort can leave a mean None
        return "n/a" if v is None else f"{v:.3f}"
    print(f"   per-subj mean: slope {f3(psm['slope']['mean'])}  corr {f3(psm['corr']['mean'])}  "
          f"EPE {f3(psm['epe']['mean'])} mm")


if __name__ == "__main__":
    main()
