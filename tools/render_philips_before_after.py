"""Philips in-plane spacing: as-shipped vs the ReadOutOversample-implied fix, against a reference.

One question, one figure. All panels share millimetre axes and a common scale bar, so size is
directly comparable by eye:

  col 1  Philips AS SHIPPED   pixel = FOVx / ReconMatrix_X = 299/256 = 1.168 mm
  col 2  Philips PROPOSED     pixel = FOVx / (nx/ReadOutOversample) = 299/152 = 1.967 mm
  col 3  CMRx24 Siemens       known-good reference (nx == 2*rx exactly, so ReconMatrix IS the
                              acquired base matrix and FOV/ReconMatrix is provably the pixel size)

Row 1 = whole slice on a +-260 mm window with a 100 mm bar.
Row 2 = 140 mm heart window centred on the detected blood pool, with a 50 mm bar.

Nothing here is written to disk anywhere; col 2 is a rendering hypothesis, not a change.

Usage: python tools/render_philips_before_after.py
Writes result/cmrx2025_recon_check/philips_before_after.png
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_uih_fov_check import ROOT, caliper, find_bloodpool, midslice  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{REPO}/result/cmrx2025_recon_check/philips_before_after.png"
REF = f"{REPO}/scratch/data/CMRxRecon2024/Cine_combined/CMRx24_Train_P001/sax/4d_recon.nii.gz"
FULL_HALF, ZOOM_HALF = 260.0, 70.0


def lv_centre(sl, px, py):
    """Centroid (mm, relative to image centre) of the blood pool found in the central 130 mm."""
    nx, ny = sl.shape
    hx, hy = int(65 / px), int(65 / py)
    i0, j0 = nx // 2 - hx, ny // 2 - hy
    sub = sl[i0:i0 + 2 * hx, j0:j0 + 2 * hy]
    m = find_bloodpool(sub)
    if m is None:
        return 0.0, 0.0, None
    ii, jj = np.nonzero(m)
    cx = (i0 + ii.mean() - nx / 2) * px
    cy = (j0 + jj.mean() - ny / 2) * py
    return cx, cy, (m, px, py)


def draw(ax, sl, px, py, half, centre, bar, barlabel, title):
    nx, ny = sl.shape
    ext = [-nx * px / 2, nx * px / 2, -ny * py / 2, ny * py / 2]
    ax.imshow(sl.T[::-1], cmap="gray", extent=ext, aspect="equal",
              vmin=0, vmax=np.percentile(sl, 99.5))
    cx, cy = centre
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    x0, y0 = cx - half + 0.08 * half, cy - half + 0.12 * half
    ax.plot([x0, x0 + bar], [y0, y0], "-", color="yellow", lw=3.5)
    ax.text(x0, y0 + 0.05 * half, barlabel, color="yellow", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    rows = json.load(open(f"{ROOT}/recon_report.json"))
    ph = next(r for r in rows if "Philips" in r["scanner"])
    pth = f"{ROOT}/Cine_combined/{ph['cid']}/sax/4d_recon.nii.gz"

    sl_p, px_p, py_p = midslice(pth)            # as shipped: 1.168 mm
    px_fix = 299.0 / 152.0                      # proposed:  1.967 mm
    sl_r, px_r, py_r = midslice(REF)

    cols = [
        (sl_p, px_p, py_p, "Philips IngeniaCX  AS SHIPPED\n299/256 = 1.168 mm"),
        (sl_p, px_fix, px_fix, "Philips IngeniaCX  PROPOSED\n299/152 = 1.967 mm"),
        (sl_r, px_r, py_r, f"CMRx24 Siemens  REFERENCE\n344/256 = {px_r:.3f} mm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 10))
    for c, (sl, px, py, title) in enumerate(cols):
        cx, cy, blob = lv_centre(sl, px, py)
        lv = ""
        if blob is not None:
            w, h, d = caliper(blob[0], px, py)
            lv = f"   LV d_eq {d:.0f} mm"
        draw(axes[0, c], sl, px, py, FULL_HALF, (0, 0), 100, "100 mm", title)
        draw(axes[1, c], sl, px, py, ZOOM_HALF, (cx, cy), 50, "50 mm",
             f"heart, 140 mm window{lv}")
    fig.suptitle(
        "Philips in-plane spacing: same voxels, two spacing hypotheses, one reference.\n"
        "Top row shares a 520 mm window and a 100 mm bar; bottom row a 140 mm window and a 50 mm bar.\n"
        "If the shipped spacing were right, the Philips heart and torso would match the reference "
        "at left. They do not.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")
    print("subject:", ph["cid"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
