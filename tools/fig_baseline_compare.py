"""VGGT vs classical SVR baselines (SVRTK, NeSVoR) — PSNR/SSIM/NCC, clean vs breath, over the heart ROI.
Baselines exist only on cmrxrecon + miitt. Pure disk read (metrics.json). -> result/1frame_ep100/."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/minsukc/vggt/result/1frame_ep100"; os.makedirs(OUT, exist_ok=True)
METHODS = [("SVRTK", "svrtk3d"), ("NeSVoR", "nesvor"),
           ("VGGT hub", "vggt_20260719_1f_gather05_ep99"), ("VGGT aug", "vggt_20260719_1f_aug_moderate_ep99")]
COHORTS = ["cmrxrecon", "miitt"]
METRICS = [("PSNR (dB)", "psnr"), ("SSIM", "ssim"), ("NCC", "ncc")]


def cohort_mean(ds, arm, key):
    fs = glob.glob(f"/home/minsukc/vggt/scratch/eval/{ds}/out/*/{arm}/metrics.json")
    return float(np.nanmean([json.load(open(f)).get(key, np.nan) for f in fs]))


fig, axes = plt.subplots(len(COHORTS), len(METRICS), figsize=(13, 7))
x = np.arange(len(METHODS)); w = 0.38
for ri, ds in enumerate(COHORTS):
    for ci, (mlabel, m) in enumerate(METRICS):
        ax = axes[ri, ci]
        clean = [cohort_mean(ds, arm, f"clean_{m}_mean") for _, arm in METHODS]
        breath = [cohort_mean(ds, arm, f"breath_{m}_mean") for _, arm in METHODS]
        ax.bar(x - w/2, clean, w, label="clean input", color="#b8c4d0")
        ax.bar(x + w/2, breath, w, label="breathing input", color="#c0392b")
        for xi, (c, b) in enumerate(zip(clean, breath)):
            ax.text(xi - w/2, c, f"{c:.2f}" if m == "psnr" else f"{c:.3f}", ha="center", va="bottom", fontsize=6.5)
            ax.text(xi + w/2, b, f"{b:.2f}" if m == "psnr" else f"{b:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels([lbl for lbl, _ in METHODS], fontsize=8, rotation=12)
        ax.set_title(f"{ds} — {mlabel}", fontsize=10)
        if m != "psnr":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        if ri == 0 and ci == 0:
            ax.legend(fontsize=8, loc="lower left")
fig.suptitle("VGGT vs classical SVR baselines (heart ROI) — SVR is strong on CLEAN input but collapses under BREATHING; VGGT is breathing-robust",
             fontsize=11, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{OUT}/fig_baseline_compare.png", dpi=135); plt.close(fig)
print(f"wrote {OUT}/fig_baseline_compare.png")
