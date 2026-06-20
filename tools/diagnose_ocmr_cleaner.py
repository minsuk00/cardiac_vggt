"""Settle WHY OCMR looks cleaner than in-distribution val, using the PROJECT-PRIMARY
metric (motion PSNR, dynamic voxels) and controlled tests for the two remaining
hypotheses after prove-it refuted the coverage-mechanism claim:

  H-breathing  : val visuals feed breathing-corrupted inputs the model under-corrects.
                 TEST = motion PSNR, val breathing ON vs OFF (same subjects). val-only
                 (OCMR has no V_gt). This is the SOLID, primary-metric result.
  H-resolution : OCMR is lower-res (2.16mm) upsampled to 1.4mm -> intrinsically smoother
                 -> looks "cleaner". TEST = Laplacian-variance (sharpness) of the INPUT
                 slices and of V_canon, val vs OCMR.
  H-presentation: OCMR shown mid-z, no V_gt/diff; val shown all-z + diff. TEST = matched
                 mid-z V_canon render (val-ON/OFF/OCMR), identical style, look directly.

Run: PYTHONPATH=training:. micromamba run -n svr python tools/diagnose_ocmr_cleaner.py
"""
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.diagnose_ood_clean_paradox import (
    load_model, build_val_dataset, val_forward, ocmr_forward, CKPT,
)

OUT = os.path.join(_ROOT, "result", "ocmr_cleaner")
os.makedirs(OUT, exist_ok=True)
N_VAL = 5
OCMR = ["us_0084_1_5T", "us_0173_pt_1_5T", "us_0197_pt_1_5T", "us_0183_pt_1_5T"]
DEV = torch.device("cuda")


def sharpness(imgs, mask_thr=0.1):
    """Mean Laplacian-magnitude over anatomy pixels (>thr) — higher = sharper/noisier,
    lower = smoother. imgs: (S, H, W) in [0,1]. 4-neighbour Laplacian, no deps."""
    vals = []
    for im in imgs:
        lap = (4 * im - np.roll(im, 1, 0) - np.roll(im, -1, 0)
               - np.roll(im, 1, 1) - np.roll(im, -1, 1))
        m = im > mask_thr
        if m.sum() > 50:
            vals.append(float(np.abs(lap[m]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


def vcanon_sharpness(V):
    """Laplacian-magnitude over the mid populated z-planes of a V_canon volume."""
    pmax = V.reshape(V.shape[0], -1).max(1)
    pop = np.where(pmax > 0.3 * pmax.max())[0]
    return sharpness(V[pop])


def main():
    model = load_model(CKPT, DEV)
    mri_ds, rcfg = build_val_dataset()

    # ── H-breathing: motion PSNR (primary metric), val ON vs OFF ──
    print("\n=== H-breathing: motion PSNR (PRIMARY), val ON vs OFF — same subjects ===")
    mot_on, mot_off, full_on, full_off = [], [], [], []
    sharp_val_in, sharp_val_vcanon = [], []
    val_demo = {}
    for i in range(N_VAL):
        on = val_forward(model, mri_ds, rcfg, i, breathing=True)
        off = val_forward(model, mri_ds, rcfg, i, breathing=False)
        mot_on.append(on["psnr_motion"]); mot_off.append(off["psnr_motion"])
        full_on.append(on["psnr_full"]); full_off.append(off["psnr_full"])
        sharp_val_in.append(sharpness(off["intensity"]))
        sharp_val_vcanon.append(vcanon_sharpness(off["V_canon"]))
        print(f"  val{i} (t{on['t_target']}, motion_frac={on['motion_frac']:.3f}): "
              f"motionPSNR ON={on['psnr_motion']:.2f} OFF={off['psnr_motion']:.2f} "
              f"(Δ={off['psnr_motion']-on['psnr_motion']:+.2f}) | fullPSNR ON={on['psnr_full']:.2f} OFF={off['psnr_full']:.2f}")
        if i == 0:
            val_demo = {"on": on, "off": off}

    # ── H-resolution: input + V_canon sharpness, val vs OCMR ──
    print("\n=== H-resolution: sharpness (Laplacian mag) val vs OCMR ===")
    sharp_oc_in, sharp_oc_vcanon = [], []
    oc_demo = None
    for name in OCMR:
        o = ocmr_forward(model, os.path.join(_ROOT, "scratch/data/ocmr/recon", name))
        si = sharpness(o["intensity"]); sv = vcanon_sharpness(o["V_canon"])
        sharp_oc_in.append(si); sharp_oc_vcanon.append(sv)
        print(f"  OCMR {name}: input_sharp={si:.4f}  Vcanon_sharp={sv:.4f}")
        if oc_demo is None:
            oc_demo = o

    # ── matched render (presentation) ──
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (lbl, o) in zip(axes, [("val breathing ON", val_demo["on"]),
                                   ("val breathing OFF", val_demo["off"]),
                                   ("OCMR (OOD)", oc_demo)]):
        D = o["V_canon"].shape[0]; mid = D // 2
        ax.imshow(o["V_canon"][mid], cmap="gray", vmin=0, vmax=float(o["V_canon"].max()) or 1e-3)
        ax.set_title(lbl, fontsize=9); ax.axis("off")
    fig.suptitle("Matched mid-z V_canon (identical render, no diff panel)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "matched_render.png"), dpi=110); plt.close(fig)

    summary = {
        "motion_psnr_val_ON_mean": round(float(np.nanmean(mot_on)), 2),
        "motion_psnr_val_OFF_mean": round(float(np.nanmean(mot_off)), 2),
        "motion_psnr_breathing_drop": round(float(np.nanmean(mot_off) - np.nanmean(mot_on)), 2),
        "full_psnr_val_ON_mean": round(float(np.nanmean(full_on)), 2),
        "full_psnr_val_OFF_mean": round(float(np.nanmean(full_off)), 2),
        "input_sharpness_val_mean": round(float(np.nanmean(sharp_val_in)), 4),
        "input_sharpness_ocmr_mean": round(float(np.nanmean(sharp_oc_in)), 4),
        "vcanon_sharpness_val_OFF_mean": round(float(np.nanmean(sharp_val_vcanon)), 4),
        "vcanon_sharpness_ocmr_mean": round(float(np.nanmean(sharp_oc_vcanon)), 4),
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:34s} {v}")
    print("\nInterpretation keys:")
    print("  - motion_psnr_breathing_drop > 0  => breathing degrades the PRIMARY metric (val side).")
    print("  - input_sharpness_ocmr  <  input_sharpness_val => OCMR inputs are smoother (resolution).")
    print("DONE")


if __name__ == "__main__":
    main()
