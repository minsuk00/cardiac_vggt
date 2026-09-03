#!/usr/bin/env python
"""plot_burst_ef_scatter.py — pooled GT-vs-pred EF scatter, one panel per burst-2x2 arm.

Reads evaluation/metric_results/_ef/<arm>.json (ef_dice.py score output; `per_subject` rows with
ef_gt / ef_breath), pools all cohorts, and draws pred EF vs GT EF with identity, fitted slope,
Spearman, median bias and MAE. Arms without an _ef file are skipped.

  PYTHONPATH=training:. python tools/plot_burst_ef_scatter.py --out temp/burst_eval/ef_scatter.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EF_DIR = ROOT / "evaluation" / "metric_results" / "_ef"
ARMS = [
    ("vggt_noreg224_ep300", "noreg224 @1 frame (baseline)"),
    ("vggt_burst5noreg224_k1_ep300", "burst5 @1 frame"),
    ("vggt_noreg224_k5_ep300", "noreg224 @5 frames"),
    ("vggt_burst5noreg224_ep300", "burst5 @5 frames"),
]
COHORT_COLORS = {"cmrx2023": "#1f77b4", "cmrx2024": "#ff7f0e", "cmrx2025": "#2ca02c",
                 "acdc": "#d62728", "mnms": "#9467bd", "miitt": "#8c564b", "ocmr": "#e377c2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "temp/burst_eval/ef_scatter.png"))
    ap.add_argument("--arm", default="breath", choices=["breath", "clean"])
    a = ap.parse_args()

    arms = [(k, lab) for k, lab in ARMS if (EF_DIR / f"{k}.json").is_file()]
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.8), squeeze=False)
    for ax, (arm, label) in zip(axes[0], arms):
        rows = json.load(open(EF_DIR / f"{arm}.json"))["per_subject"]
        rows = [r for r in rows if r.get(f"ef_{a.arm}") is not None and r.get("ef_gt") is not None]
        g = np.array([r["ef_gt"] for r in rows]); p = np.array([r[f"ef_{a.arm}"] for r in rows])
        for c, col in COHORT_COLORS.items():
            m = np.array([r["cohort"] == c for r in rows])
            if m.any():
                ax.scatter(g[m], p[m], s=20, c=col, alpha=0.8, edgecolor="none", label=f"{c} (n={m.sum()})")
        sl, b = np.polyfit(g, p, 1)
        rho = spearmanr(g, p).correlation
        xs = np.array([g.min(), g.max()])
        ax.plot(xs, sl * xs + b, c="k", lw=1.4)
        ax.plot([0, 100], [0, 100], "--", c="0.6", lw=1, zorder=0)
        ax.set_xlim(0, 90); ax.set_ylim(0, 90); ax.set_aspect("equal")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("GT EF (%)"); ax.set_ylabel("pred EF (%)")
        ax.annotate(f"n={len(g)}  slope {sl:.2f}  ρ {rho:.2f}\nbias {np.median(p - g):+.1f}  "
                    f"MAE {np.mean(np.abs(p - g)):.1f}", xy=(0.03, 0.97), xycoords="axes fraction",
                    va="top", fontsize=8.5, bbox=dict(fc="white", ec="0.8", alpha=0.9))
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"nnU-Net Task114 EF, {a.arm} arm, val split pooled over 7 sources", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
