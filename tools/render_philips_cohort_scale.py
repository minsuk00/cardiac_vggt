"""All Philips subjects: shipped spacing vs the ReadOutOversample-implied spacing.

Deliberately uses NO automatic segmentation -- the blob detector used in earlier checks was shown
to fail on a known-good Siemens control, so this figure carries only things that cannot be got
wrong: a fixed millimetre window, a 100 mm scale bar, and a dashed 340 mm line (a typical adult
chest left-right width) as an external anatomical ruler. Judge by eye whether the torso is
plausible against that line.

  left  column: as shipped, pixel = FOVx / ReconMatrix_X   = 299/256 = 1.168 mm
  right column: proposed,   pixel = FOVx / (nx/ReadOutOversample) = 299/152 = 1.967 mm
  last row    : CMRx24 Siemens, a known-good subject, for calibration of the eye

Both columns are the SAME voxels; FOVx = FOVy and the readout/phase base matrices are both ~152,
so the correction is a pure ISOTROPIC rescale -- the x/y ratio does not change (aniso stays 1.000).

Usage: python tools/render_philips_cohort_scale.py
Writes result/cmrx2025_recon_check/philips_cohort_scale.png
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_uih_fov_check import ROOT, midslice  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/result/cmrx2025_recon_check/philips_cohort_scale.png"
REF = f"{REPO}/scratch/data/CMRxRecon2024/Cine_combined/CMRx24_Train_P001/sax/4d_recon.nii.gz"
HALF = 280.0        # mm half-window, shared by every panel
CHEST_MM = 340.0    # typical adult chest left-right width, the external ruler
NSUB = 5


def draw(ax, sl, px, py, title):
    nx, ny = sl.shape
    ext = [-nx * px / 2, nx * px / 2, -ny * py / 2, ny * py / 2]
    ax.imshow(sl.T[::-1], cmap="gray", extent=ext, aspect="equal",
              vmin=0, vmax=np.percentile(sl, 99.5))
    ax.plot([-CHEST_MM / 2, CHEST_MM / 2], [0, 0], "--", color="red", lw=1.4, alpha=0.85)
    ax.text(0, 12, f"{CHEST_MM:.0f} mm typical adult chest", color="red", fontsize=7,
            ha="center")
    ax.plot([-HALF + 20, -HALF + 120], [-HALF + 28] * 2, "-", color="yellow", lw=3.5)
    ax.text(-HALF + 20, -HALF + 42, "100 mm", color="yellow", fontsize=8)
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="every Philips subject at the PROPOSED spacing only (grid layout)")
    args = ap.parse_args()

    rows = json.load(open(f"{ROOT}/recon_report.json"))
    allph = [r for r in rows if "Philips" in r["scanner"]]
    fix = 299.0 / 152.0      # x: crop preserves the acquired pixel
    fiy = 299.0 / 256.0      # y: zero-fill preserves the FOV -> pixel = FOVy/ry (already shipped)

    if args.all:
        refs = [("CMRx24 Siemens REF", REF)]
        n = len(allph) + len(refs)
        ncol = 4
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.9 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, r in zip(axes, allph):
            sl, _, _ = midslice(f"{ROOT}/Cine_combined/{r['cid']}/sax/4d_recon.nii.gz")
            sid = "_".join(r["cid"].split("_")[1:2] + r["cid"].split("_")[-1:])
            draw(ax, sl, fix, fiy, f"{sid}  C: {fix:.3f} x {fiy:.3f} mm  (nz={r['shape_in'][1]})")
        for ax, (tag, path) in zip(axes[len(allph):], refs):
            sl, px, py = midslice(path)
            draw(ax, sl, px, py, f"{tag}  {px:.3f} x {py:.3f} mm")
        for ax in axes[n:]:
            ax.axis("off")
        title = (f"ALL {len(allph)} Philips subjects under hypothesis C: x = 299/152 = 1.967 mm, "
                 "y = 299/256 = 1.168 mm (aspect 1.684), "
                 "plus a known-good CMRx24 Siemens reference.\n"
                 "Same +-280 mm window, same 100 mm bar, same red 340 mm adult-chest ruler in every "
                 "panel. No segmentation used.")
        out = OUT.replace(".png", "_all.png")
    else:
        ph = allph[:NSUB]
        fig, axes2 = plt.subplots(NSUB + 1, 2, figsize=(9.5, 4.6 * (NSUB + 1)))
        for i, r in enumerate(ph):
            sl, px, py = midslice(f"{ROOT}/Cine_combined/{r['cid']}/sax/4d_recon.nii.gz")
            sid = r["cid"].split("_")[-1]
            draw(axes2[i, 0], sl, px, py,
                 f"{sid}  AS SHIPPED  {px:.3f} mm   (ny={r['shape_in'][3]}, nx=304, base=152)")
            draw(axes2[i, 1], sl, fix, fix, f"{sid}  PROPOSED  {fix:.3f} mm")
        slr, pxr, pyr = midslice(REF)
        draw(axes2[NSUB, 0], slr, pxr, pyr, f"CMRx24 Siemens REFERENCE  {pxr:.3f} x {pyr:.3f} mm")
        axes2[NSUB, 1].axis("off")
        axes2[NSUB, 1].text(0.5, 0.5,
                            "reference is known-good:\nnx = 2*rx exactly, so\nReconMatrix IS the "
                            "acquired\nbase matrix and FOV/ReconMatrix\nis provably the pixel size",
                            ha="center", va="center", fontsize=10)
        title = ("Philips cohort at true physical scale -- shipped vs ReadOutOversample-implied "
                 "spacing.\nEvery panel: same +-280 mm window, same 100 mm bar, same red 340 mm "
                 "adult-chest ruler.\nCorrection is a pure ISOTROPIC rescale (x/y ratio unchanged); "
                 "no segmentation is used.")
        out = OUT

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
