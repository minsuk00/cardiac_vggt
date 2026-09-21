#!/usr/bin/env python
"""Side-by-side predicted DVF of two VGGT arms on the same subject (from each arm's ed_dvf.npz).

ed_dvf.npz holds the point head's residual displacement at the ED pass: delta (S, 518, 518, 3) in
normalized units, one map per input slot, with slot_z / slot_t (= the frame that slot was fed).
Both arms saw the identical input (same bundle, same slot draw), so any difference is the model.

Per subject one PNG; rows = slots spanning the whole z range (never just the mid slice), columns:
  input slice | arm A in-plane quiver | arm B in-plane quiver | arm A dz | arm B dz | |A - B| (mm)
In-plane arrows are in mm (scaled x ARROW for visibility); dz is the through-plane shift in mm, and
the row title carries the TRUE applied breathing dz for that slot's frame.

Usage:
  PYTHONPATH=training:. python tools/render_dvf_compare.py --arm af24 \
      --subjects cmrx2024:CMRx24_Test_P017 mnms:MNMs_K7L2Y6
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import nibabel as nib                                              # noqa: E402
import numpy as np                                                 # noqa: E402
from scipy.ndimage import zoom                                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
from build_af_bundle import DT, T_BREATH, breathing_model                 # noqa: E402
from data.respiratory import lujan_displacement                          # noqa: E402

EVAL = os.path.join(ROOT, "scratch", "eval")
MM = np.array([0.5 * 255 * 1.4, 0.5 * 255 * 1.4, 90.0])          # run_vggt.MM_PER_NORM (x, y, z)
ARROW, STEP, NROWS = 4.0, 22, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subjects", nargs="+", required=True, help="source:subject")
    ap.add_argument("--a", default="vggt_final518_base_ep300")
    ap.add_argument("--b", default="vggt_final518_diff1000_ep300")
    args = ap.parse_args()
    outdir = os.path.join(ROOT, "figs", "rhythm24", f"{args.arm}_dvf_compare")
    os.makedirs(outdir, exist_ok=True)
    sa, sb = (x.replace("vggt_final518_", "").replace("_ep300", "") for x in (args.a, args.b))

    for item in args.subjects:
        src, subj = item.split(":")
        sd = os.path.join(EVAL, f"{src}_{args.arm}", "out", subj)
        man = json.load(open(os.path.join(sd, "manifest.json")))
        A, B = (np.load(os.path.join(sd, m, "ed_dvf.npz")) for m in (args.a, args.b))
        assert np.array_equal(A["slot_z"], B["slot_z"]) and np.array_equal(A["slot_t"], B["slot_t"]), "slot draws differ"
        u, amp, r0, n, _ = breathing_model(man)
        S = len(A["slot_z"])
        order = np.argsort(A["slot_z"])
        rows = [int(order[i]) for i in np.linspace(0, S - 1, min(NROWS, S)).round().astype(int)]

        fig, axs = plt.subplots(len(rows), 6, figsize=(17, 2.9 * len(rows)), dpi=120, squeeze=False)
        dA, dB = (np.asarray(X["delta"], np.float32) * MM for X in (A, B))          # (S,518,518,3) mm
        dzmax = max(2.0, float(np.percentile(np.abs(np.concatenate([dA[rows, ..., 2].ravel(), dB[rows, ..., 2].ravel()])), 99)))
        dfmax = max(1.0, float(np.percentile(np.linalg.norm(dA[rows] - dB[rows], axis=-1), 99)))
        yy, xx = np.mgrid[STEP // 2:518:STEP, STEP // 2:518:STEP]
        for ri, s in enumerate(rows):
            z, j = int(round(float(A["slot_z"][s]))), int(A["slot_t"][s])
            img = np.asarray(nib.load(os.path.join(sd, "breath", f"stack_t{j:02d}.nii.gz")).dataobj, np.float32)[:, :, z].T
            img = zoom(img, 518 / img.shape[0], order=1)
            fov = img > 0.05 * img.max()
            appl = float((u * amp * float(lujan_displacement(float((r0[z] + j * DT / T_BREATH) % 1.0), 1.0, n=n)))[0])
            vmax = float(np.percentile(img[fov], 99.5)) if fov.any() else 1.0
            tag = "  [slot 0 = reference]" if s == 0 else ""
            axs[ri, 0].imshow(img, cmap="gray", vmax=vmax)
            axs[ri, 0].set_title(f"input  z={z}  frame={j}{tag}\ntrue breathing dz = {appl:+.2f} mm", fontsize=8.5)
            for ci, (d, nm) in ((1, (dA, sa)), (2, (dB, sb))):
                ax = axs[ri, ci]
                ax.imshow(img, cmap="gray", vmax=vmax)
                m = fov[yy, xx]
                ax.quiver(xx[m], yy[m], d[s, yy, xx, 0][m] / 1.4 * 518 / 256 * ARROW, d[s, yy, xx, 1][m] / 1.4 * 518 / 256 * ARROW,
                          np.hypot(d[s, yy, xx, 0], d[s, yy, xx, 1])[m], cmap="autumn_r", angles="xy", scale_units="xy",
                          scale=1, width=0.004, clim=(0, 6))
                ip = np.hypot(d[s, ..., 0], d[s, ..., 1])[fov]
                ax.set_title(f"{nm}: in-plane (arrows x{ARROW:.0f})\nmean |d| = {ip.mean():.2f} mm", fontsize=8.5)
            for ci, (d, nm) in ((3, (dA, sa)), (4, (dB, sb))):
                im = axs[ri, ci].imshow(np.where(fov, d[s, ..., 2], np.nan), cmap="coolwarm", vmin=-dzmax, vmax=dzmax)
                axs[ri, ci].set_title(f"{nm}: through-plane dz\nmean = {d[s, ..., 2][fov].mean():+.2f} mm", fontsize=8.5)
            plt.colorbar(im, ax=axs[ri, 4], fraction=0.046, pad=0.02)
            df = np.linalg.norm(dA[s] - dB[s], axis=-1)
            im2 = axs[ri, 5].imshow(np.where(fov, df, np.nan), cmap="magma", vmin=0, vmax=dfmax)
            axs[ri, 5].set_title(f"|{sa} - {sb}| (3D)\nmean = {df[fov].mean():.2f} mm", fontsize=8.5)
            plt.colorbar(im2, ax=axs[ri, 5], fraction=0.046, pad=0.02)
            for ax in axs[ri]:
                ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{src}_{args.arm} / {subj}: predicted DVF at the ED pass, {sa} vs {sb}  (same input slots)",
                     fontsize=11, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        out = os.path.join(outdir, f"{subj}__{sa}_vs_{sb}.png")
        fig.savefig(out, facecolor="white"); plt.close(fig)
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
