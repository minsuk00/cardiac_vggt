"""Cross-vendor PHYSICAL-SCALE check of the stamped in-plane spacing (docs/54 follow-up).

Every panel is drawn on the SAME millimetre axes with the SAME 100 mm scale bar, so bodies can be
compared directly. If the stamped spacing is right, every adult thorax should come out at a
similar physical size regardless of vendor -- torso width is an external anatomical ruler that
owes nothing to any header convention.

Included:
  * one subject per 2025 scanner model (11),
  * the same Philips subject re-rendered at the pixel size implied by ReadOutOversample=2
    (pixel = FOVx / (nx/2) = 299/152 = 1.967 mm) -- the proposed fix, NOT what is on disk,
  * one CMRx24 and one CMRx23 subject as known-good anchors (Siemens, nx == 2*rx exactly, so
    ReconMatrix == the acquired base matrix and FOV/ReconMatrix is provably the pixel size).

The measured body box is printed per panel; the body mask is drawn so the measurement is
auditable rather than taken on trust.

Usage: python tools/render_physical_scale_montage.py
Writes result/cmrx2025_recon_check/physical_scale_montage.png
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = f"{REPO}/scratch/data"
ROOT = f"{DATA}/CMRxRecon2025"
OUT = f"{REPO}/result/cmrx2025_recon_check/physical_scale_montage.png"
HALF = 260.0  # mm half-window for the common axes


def midslice(path):
    img = nib.load(path)
    zx, zy = img.header.get_zooms()[:2]
    a = np.asarray(img.dataobj[..., 0], dtype=np.float32)
    return a[:, :, a.shape[2] // 2], float(zx), float(zy)


def body_box(sl, px, py):
    """Largest connected bright region = the body. Returns (mask, width_mm, height_mm)."""
    thr = 0.10 * np.percentile(sl, 99.0)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(sl > thr, np.ones((5, 5))))
    lab, n = ndimage.label(m)
    if n == 0:
        return None, 0.0, 0.0
    k = 1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1))))
    m = lab == k
    ii, jj = np.nonzero(m)
    return m, (ii.max() - ii.min() + 1) * px, (jj.max() - jj.min() + 1) * py


def panel(ax, sl, px, py, title, warn=False):
    m, w, h = body_box(sl, px, py)
    nx, ny = sl.shape
    ext = [-nx * px / 2, nx * px / 2, -ny * py / 2, ny * py / 2]
    ax.imshow(sl.T[::-1], cmap="gray", extent=ext, aspect="equal",
              vmin=0, vmax=np.percentile(sl, 99.5))
    if m is not None:
        ax.contour(np.linspace(ext[0], ext[1], nx), np.linspace(ext[2], ext[3], ny),
                   m.T[::-1].astype(float), levels=[0.5], colors="deepskyblue", linewidths=0.9)
    ax.plot([-HALF + 25, -HALF + 125], [-HALF + 30] * 2, "-", color="yellow", lw=3)
    ax.text(-HALF + 25, -HALF + 45, "100 mm", color="yellow", fontsize=7)
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(-HALF, HALF)
    ax.set_xticks([])
    ax.set_yticks([])
    col = "red" if warn else "black"
    ax.set_title(f"{title}\n{px:.3f} x {py:.3f} mm   body {w:.0f} x {h:.0f} mm",
                 fontsize=8, color=col)
    return w, h


def main():
    rows = json.load(open(f"{ROOT}/recon_report.json"))
    models = sorted({r["scanner"] for r in rows})
    picks = []
    for mdl in models:
        for r in sorted([x for x in rows if x["scanner"] == mdl], key=lambda x: x["cid"]):
            p = f"{ROOT}/Cine_combined/{r['cid']}/sax/4d_recon.nii.gz"
            if os.path.exists(p):
                picks.append((f"CMRx25 {mdl}", p, None, "Philips" in mdl))
                break
    # Philips re-rendered at the ReadOutOversample-implied pixel size (proposed fix)
    ph = [p for p in picks if "Philips" in p[0]][0]
    picks.append(("CMRx25 Philips_IngeniaCX\nPROPOSED 299/152", ph[1], 299.0 / 152.0, False))
    for tag, pat in [("CMRx24 Siemens (anchor)", f"{DATA}/CMRxRecon2024/Cine_combined/CMRx24_Train_P001/sax/4d_recon.nii.gz"),
                     ("CMRx23 Siemens (anchor)", f"{DATA}/CMRxRecon2023/Cine_combined/CMRx23_Train_P001/sax/4d_recon.nii.gz")]:
        g = glob.glob(pat)
        if g:
            picks.append((tag, g[0], None, False))

    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 4.3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    print(f"{'panel':46s} {'px':>7} {'py':>7} {'body_w':>8} {'body_h':>8}")
    for ax, (tag, path, force_px, warn) in zip(axes, picks):
        sl, px, py = midslice(path)
        if force_px is not None:
            px = py = force_px
        w, h = panel(ax, sl, px, py, tag, warn)
        print(f"{tag[:46]:46s} {px:7.3f} {py:7.3f} {w:8.0f} {h:8.0f}")
    for ax in axes[len(picks):]:
        ax.axis("off")
    fig.suptitle(
        "In-plane spacing sanity check -- every panel on the SAME millimetre axes, same 100 mm bar.\n"
        "Blue = detected body. An adult thorax is ~300-400 mm wide; a panel that looks small IS small.\n"
        "Red title = Philips AS SHIPPED (suspected 1.68x under-scaled); the PROPOSED panel is the "
        "ReadOutOversample-implied fix, not what is on disk.", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches="tight")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
