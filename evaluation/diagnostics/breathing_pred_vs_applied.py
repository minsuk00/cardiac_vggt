"""breathing_pred_vs_applied.py — cohort through-plane breathing diagnostic.

For a (dataset, arm), reads each subject's `resp_diag.json` (breath arm) — predicted through-plane
Δz vs the applied respiratory displacement per input slot, produced by `run_vggt.resp_diag` — and
reports how well the model recovers breathing:

  - per-subject-averaged  slope / corr / EPE(mm)   (mean ± std over subjects; run_vggt's own values)
  - pooled                slope / corr / EPE(mm)   (over ALL slots of ALL subjects, same formulas:
                                                    slope=polyfit(applied,pred,1)[0], corr=pearson,
                                                    EPE=mean|pred-applied|) — the honest cohort number
  - a pooled scatter PNG (applied vs predicted Δz, y=x + fit line).

Pure disk read (no GPU/model). Writes JSON + PNG under diagnostics/out/<ds>/ (regenerable).

Run: python evaluation/diagnostics/breathing_pred_vs_applied.py --dataset cmrxrecon --arm vggt_20260719_1f_gather05_ep99
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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
    xs = [r[key] for r in rows if r[key] is not None]
    return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "n": len(xs)} if xs \
        else {"mean": None, "std": None, "n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", default=None, help="JSON path (PNG alongside); default diagnostics/out/<ds>/<arm>_breathing.json")
    args = ap.parse_args()
    ds, arm = args.dataset, args.arm

    rows, pooled_app, pooled_pred = [], [], []
    for subj in paths.subjects(ds):
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
    out = Path(args.out) if args.out else paths.EVAL_ROOT / "diagnostics" / "out" / ds / f"{arm}_breathing.json"
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
