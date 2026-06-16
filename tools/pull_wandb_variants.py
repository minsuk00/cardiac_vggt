"""Pull training-time history for the 5 respiratory-variant runs from wandb (online).

Saves per-run metric trajectories (train loss/psnr, val mean psnr full/bbox/motion) +
config + summary to result/variants_eval/wandb_<runid>.json. Column names with run-specific
baseline suffixes (e.g. val_psnr_bbox/mean_n200_base30.4) are matched by prefix.

NOTE: val curves are on DIFFERENT tasks across the resp/no-resp boundary (resp runs val on
breathing-corrupted inputs, no-resp on clean) — usable for convergence WITHIN a family only.
"""
import json
import os
import sys

import numpy as np
import wandb

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "result", "variants_eval")

RUNS = [
    dict(var=1, runid="5n5yc4nj", name="resp_zt",         family="resp"),
    dict(var=2, runid="t59w6nqy", name="resp_no_t",       family="resp"),
    dict(var=3, runid="xjm8r890", name="resp_aug_no_t",   family="resp"),
    dict(var=4, runid="lwgifjma", name="noresp_no_t",     family="noresp"),
    dict(var=5, runid="n1rqb0yd", name="noresp_aug_no_t", family="noresp"),
]

# Prefixes to keep (suffix varies per run). "epoch"/"trainer/*" capture the x-axis.
KEEP_PREFIXES = [
    "Loss/train_loss_objective", "Loss/train_metric_psnr_3d_full", "Loss/train_metric_psnr_3d_bbox",
    "Train_Loss/loss_objective", "Train_Loss/metric_psnr_3d_full", "Train_Loss/metric_psnr_3d_bbox",
    "Val_Loss/", "val_psnr_full/mean", "val_psnr_bbox/mean", "val_motion/mean",
    "train/resp_disp_mm", "val/resp_disp_mm", "epoch", "Trainer/epoch", "trainer/epoch",
]


def keep(col):
    return any(col == p or col.startswith(p) for p in KEEP_PREFIXES)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    api = wandb.Api(timeout=30)
    for r in RUNS:
        try:
            run = api.run(f"vggt-mri/{r['runid']}")
        except Exception as e:
            print(f"var{r['var']} {r['runid']}: FAILED {e}")
            continue
        hist = run.history(samples=20000, pandas=True)
        cols = [c for c in hist.columns if keep(c)]
        series = {}
        for c in cols:
            s = hist[["_step", c]].dropna()
            series[c] = dict(step=s["_step"].tolist(),
                             value=[None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)
                                    for v in s[c].tolist()])
        summary = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                   for k, v in dict(run.summary).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str))}
        payload = dict(var=r["var"], name=r["name"], family=r["family"], runid=r["runid"],
                       state=run.state, last_step=run.lastHistoryStep,
                       columns=cols, series=series, summary=summary)
        with open(os.path.join(OUT_DIR, f"wandb_var{r['var']}_{r['runid']}.json"), "w") as f:
            json.dump(payload, f, indent=2)
        print(f"var{r['var']} {r['runid']} ({r['name']}): state={run.state} laststep={run.lastHistoryStep} "
              f"kept {len(cols)} cols, {len(hist)} rows")


if __name__ == "__main__":
    main()
