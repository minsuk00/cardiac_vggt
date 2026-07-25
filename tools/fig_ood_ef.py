"""OOD EF recovery for the ep100 hub (gather05), 3 gated cohorts — figure for docs/46 + _html/46.

Reads scratch/eval/_ef_ood/gather05/ef_ood_gather05.json (nnU-Net Task114 segmentations of the
per-phase recon + GT volumes; EF = (LV_max - LV_min)/LV_max, a RATIO so the 12 mm pitch caveat of
docs/39 cancels).

Form: pred-vs-GT scatter, one small multiple per cohort, with the identity line and a per-arm fit.
Axes are SHARED across panels on purpose — the restriction-of-range story (ACDC's GT EF spans
11.7-74.2 while MIITT/OCMR sit in a ~20-point band) is only visible if the panels share a scale.
Colours: house teal (fig_breathing.png) + orange, CVD-validated dE 12.8; marker shape carries the
arm as a secondary encoding so identity is never colour-alone.
"""
import json
import os

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EF = "/home/minsukc/vggt/scratch/eval/_ef_ood/gather05/ef_ood_gather05.json"
OUT = "/home/minsukc/vggt/result/1frame_ep100"
COH = [("miitt", "MIITT (n=13)"), ("ocmr", "OCMR (n=8)"), ("acdc", "ACDC — pathology (n=40)")]
ARMS = [("clean", "clean input", "#2a9d8f", "o"), ("breath", "breathing-corrupted", "#e07a1f", "^")]


def main():
    d = json.load(open(EF))
    ps, agg = d["per_subject"], d["aggregate"]
    os.makedirs(OUT, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.9), sharex=True, sharey=True)
    lo, hi = 0, 85
    for ax, (c, lbl) in zip(axes, COH):
        r = [x for x in ps if x["cohort"] == c]
        gt = np.array([x["ef_gt"] for x in r])
        ax.plot([lo, hi], [lo, hi], ls="--", lw=1.2, color="#888", zorder=1)
        txt = []
        for arm, albl, col, mk in ARMS:
            pr = np.array([x[f"ef_{arm}"] for x in r])
            ax.scatter(gt, pr, s=34, marker=mk, facecolor=col, edgecolor="white", lw=0.7,
                       alpha=0.9, zorder=3, label=albl if c == "miitt" else None)
            sl = agg[c][f"{arm}_ef_slope"]; sp = agg[c][f"{arm}_ef_spearman"]
            b = np.polyfit(gt, pr, 1)
            xs = np.array([gt.min(), gt.max()])
            ax.plot(xs, np.polyval(b, xs), color=col, lw=2, zorder=2)
            txt.append(f"{albl}:  slope {sl:.2f}   $\\rho$ {sp:+.2f}   bias {pr.mean()-gt.mean():+.1f}")
        ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", ha="left",
                fontsize=7.6, color="#333",
                bbox=dict(fc="white", ec="#ddd", lw=0.6, boxstyle="round,pad=0.35"))
        ax.text(0.97, 0.05, f"GT EF sd = {gt.std():.1f}\nrange {gt.min():.0f}–{gt.max():.0f}",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=7.4, color="#666")
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("ground-truth EF (%)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    axes[0].set_ylabel("predicted EF (%)", fontsize=9)
    # figure-level legend: an in-axes one covers panel 1's GT-EF-sd annotation
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.4, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.suptitle("OOD ejection fraction — ep100 hub (gather05), nnU-Net Task114.  Dashed = identity.  "
                 "EF is under-predicted by 20–29 points on every cohort;\nper-patient ranking holds only "
                 "on ACDC, the one cohort with a wide GT EF spread (restriction of range).", fontsize=9.5)
    fig.tight_layout(rect=[0, 0.13, 1, 0.91])      # leave room for the xlabel + figure legend
    fig.savefig(f"{OUT}/fig_ood_ef.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT}/fig_ood_ef.png")

    # console table for the doc
    print(f"\n{'cohort':8s} {'n':>3s} {'arm':7s} {'slope':>6s} {'rho':>6s} {'MAE%':>6s} "
          f"{'LV_ED':>6s} {'MYO_ED':>7s} {'RV_ED':>6s}")
    for c, _ in COH:
        a = agg[c]
        for arm in ("clean", "breath"):
            print(f"{c:8s} {a['n']:3d} {arm:7s} {a[f'{arm}_ef_slope']:6.3f} "
                  f"{a[f'{arm}_ef_spearman']:+6.3f} {a[f'{arm}_ef_mae_pct']:6.1f} "
                  f"{a[f'{arm}_dice_LV_ED']:6.3f} {a[f'{arm}_dice_MYO_ED']:7.3f} "
                  f"{a[f'{arm}_dice_RV_ED']:6.3f}")


if __name__ == "__main__":
    main()
