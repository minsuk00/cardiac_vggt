"""Philips spacing: three candidate hypotheses, judged on LV roundness.

Headers cannot distinguish these -- they differ in what you believe FOVy refers to after the ky
zero-fill. Anatomy can: a true mid-ventricular short-axis LV cavity is round.

  A  as shipped   1.168 x 1.168   FOVx/rx, FOVy/ry
  B  isotropic    1.967 x 1.967   both axes referred to the ~152 acquired base matrix
  C  UIH-style    1.967 x 1.168   x = acquired readout resolution (crop preserves pixel size),
                                  y = finer because ky was zero-filled (fill preserves FOV)
                                  <- this is exactly the treatment UIH volumes already carry

C is the internally consistent one if you follow the operations literally: cropping does not
change pixel size, zero-filling does. B additionally assumes FOVy refers to the acquired line
count rather than the filled grid, which is what was previously asserted without justification.

Usage: python tools/render_philips_three_hypotheses.py
Writes result/cmrx2025_recon_check/philips_three_hypotheses.png
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
OUT = f"{REPO}/result/cmrx2025_recon_check/philips_three_hypotheses.png"
REF = f"{REPO}/scratch/data/CMRxRecon2024/Cine_combined/CMRx24_Train_P001/sax/4d_recon.nii.gz"
HYP = [("A  as shipped", 1.168, 1.168), ("B  isotropic", 1.967, 1.967), ("C  UIH-style", 1.967, 1.168)]
HALF = 75.0   # mm half-window on the heart
NSUB = 4


def heart_centre(sl):
    """Brightest smoothed location in the central third -- a crude but spacing-INDEPENDENT
    anchor (it works in pixel space, so it cannot favour any hypothesis)."""
    from scipy import ndimage
    nx, ny = sl.shape
    i0, j0 = nx // 3, ny // 3
    sub = ndimage.gaussian_filter(sl[i0:2 * nx // 3, j0:2 * ny // 3], 6)
    k = np.unravel_index(np.argmax(sub), sub.shape)
    return i0 + k[0], j0 + k[1]


def draw(ax, sl, px, py, ci, cj, title):
    nx, ny = sl.shape
    ext = [-nx * px / 2, nx * px / 2, -ny * py / 2, ny * py / 2]
    cx, cy = (ci - nx / 2) * px, (cj - ny / 2) * py
    ax.imshow(sl.T[::-1], cmap="gray", extent=ext, aspect="equal",
              vmin=0, vmax=np.percentile(sl, 99.5))
    # 45 mm reference circle: a typical mid-LV cavity. Round LV -> matches the circle's shape.
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(cx + 22.5 * np.cos(th), cy + 22.5 * np.sin(th), "-", color="lime", lw=1.2, alpha=0.8)
    ax.plot([cx - HALF + 10, cx - HALF + 60], [cy - HALF + 10] * 2, "-", color="yellow", lw=3)
    ax.text(cx - HALF + 10, cy - HALF + 17, "50 mm", color="yellow", fontsize=8)
    ax.set_xlim(cx - HALF, cx + HALF)
    ax.set_ylim(cy - HALF, cy + HALF)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main():
    rows = json.load(open(f"{ROOT}/recon_report.json"))
    ph = [r for r in rows if "Philips" in r["scanner"]][:NSUB]
    fig, axes = plt.subplots(NSUB + 1, 3, figsize=(13, 4.4 * (NSUB + 1)))
    for i, r in enumerate(ph):
        sl, _, _ = midslice(f"{ROOT}/Cine_combined/{r['cid']}/sax/4d_recon.nii.gz")
        ci, cj = heart_centre(sl)
        sid = r["cid"].split("_")[-1]
        for c, (tag, px, py) in enumerate(HYP):
            draw(axes[i, c], sl, px, py, ci, cj, f"{sid}   {tag}   {px:.3f} x {py:.3f} mm")
    slr, pxr, pyr = midslice(REF)
    ci, cj = heart_centre(slr)
    draw(axes[NSUB, 0], slr, pxr, pyr, ci, cj,
         f"CMRx24 Siemens REFERENCE  {pxr:.3f} x {pyr:.3f} mm")
    for c in (1, 2):
        axes[NSUB, c].axis("off")
    axes[NSUB, 1].text(0.5, 0.5,
                       "green circle = 45 mm, a typical\nmid-ventricular LV cavity.\n\n"
                       "Whichever column makes the LV\nmatch the circle's ROUNDNESS is the\n"
                       "correct pixel ASPECT RATIO;\nmatching its SIZE fixes the scale.",
                       ha="center", va="center", fontsize=11)
    fig.suptitle(
        "Philips: three spacing hypotheses on the same voxels. Headers cannot separate them; "
        "anatomy can.\nA = as shipped, B = isotropic rescale, C = the rule UIH volumes already "
        "carry (crop preserves pixel size, zero-fill shrinks it).", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=115, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
