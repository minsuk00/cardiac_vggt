"""Visual + quantitative check of the UIH FOVx convention (docs/54 follow-up).

`tools/reconstruct_cmrx2025.py` assumes UIH's `FOVx` describes the FULL ACQUIRED readout grid
(nx samples) while Siemens/Philips report the POST-crop FOV (rx pixels). It therefore stamps

    pixel_x = FOVx / nx        instead of the 2024 formula   pixel_x = FOVx / rx

i.e. a factor rx/nx ~ 0.75 smaller, but ONLY for UIH -- for Siemens/Philips the two formulas are
algebraically identical whenever nx >= rx. That rule was INFERRED from "cine in-plane voxels
should be near-isotropic", never checked against vendor documentation.

The VOXELS are byte-identical under both hypotheses; only the stamped spacing differs. So the
sole observable is the physical aspect ratio. Two adjudicators, neither of which assumes the
conclusion:

  1. VISUAL  -- a mid-ventricular SAX LV cavity is round. Render both hypotheses at true
     physical aspect and look.
  2. CALIPER -- an adult LV cavity is ~40-55 mm across at mid-ventricle. The detected blood-pool
     blob is measured in MILLIMETRES under each hypothesis. Only the correct spacing can give
     both a round blob AND a physiological diameter. The detected blob is outlined in the figure
     so the measurement is auditable rather than taken on trust.

CONTROLS (this script must be able to fail):
  * Siemens rows: px_naive == px by construction, so the two panels are pixel-identical. If they
    ever differ, this script is lying. (An earlier version got this wrong by applying the UIH
    rescale to every vendor; the control is what caught it.)
  * UIH 1.5T rows (umr670/680) sit at nx/rx ~ 0.99, so the correction is a near no-op for them --
    a within-vendor control showing the effect tracks nx/rx, not the vendor label.

Usage: python tools/render_uih_fov_check.py
Writes result/cmrx2025_recon_check/uih_fov_convention_check.png
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = f"{REPO}/scratch/data/CMRxRecon2025"
OUT = f"{REPO}/result/cmrx2025_recon_check/uih_fov_convention_check.png"

# nx/rx per model measured from recon_report.json:
#   umr780/790/880 ~1.33 (correction bites) | umr670/680 ~0.99 (no-op) | Siemens ~2.0 (identical)
WANT = [
    ("UIH_30T_umr780", "UIH"),
    ("UIH_30T_umr790", "UIH"),
    ("UIH_30T_umr880", "UIH"),
    ("UIH_15T_umr670", "UIH-noop"),
    ("Siemens_30T_Vida", "CONTROL"),
    ("Siemens_15T_Aera", "CONTROL"),
]
ZOOM_MM = 130.0


def pick():
    rows = json.load(open(f"{ROOT}/recon_report.json"))
    out = []
    for scanner, kind in WANT:
        cands = sorted([r for r in rows if r["scanner"] == scanner],
                       key=lambda r: -r["nx_over_rx"])
        for r in cands:
            p = f"{ROOT}/Cine_combined/{r['cid']}/sax/4d_recon.nii.gz"
            if os.path.exists(p):
                out.append((r, kind, p))
                break
    return out


def midslice(path):
    img = nib.load(path)
    zx, zy = img.header.get_zooms()[:2]
    arr = np.asarray(img.dataobj[..., 0], dtype=np.float32)  # (X, Y, Z), frame 0 = ED
    return arr[:, :, arr.shape[2] // 2], float(zx), float(zy)


def zoom_box(sl, px, py):
    """Central ZOOM_MM box, returned as the (X, Y) sub-array plus its index origin."""
    nx, ny = sl.shape
    hx, hy = int(ZOOM_MM / 2 / px), int(ZOOM_MM / 2 / py)
    i0, j0 = max(0, nx // 2 - hx), max(0, ny // 2 - hy)
    return sl[i0:i0 + 2 * hx, j0:j0 + 2 * hy], i0, j0


def find_bloodpool(z):
    """Brightest compact blob in the central zoom = LV cavity (bSSFP blood is bright).

    Returns a boolean mask in (X, Y). Selection is on SHAPE-FREE criteria only (brightness,
    area, distance from centre) so it cannot prefer either spacing hypothesis.
    """
    thr = np.percentile(z, 90)
    lab, n = ndimage.label(z > thr)
    if n == 0:
        return None
    cx, cy = np.array(z.shape) / 2
    best, best_score = None, -1e9
    for k in range(1, n + 1):
        m = lab == k
        a = int(m.sum())
        if a < 60 or a > 0.25 * z.size:
            continue
        yy, xx = np.nonzero(m)
        d = np.hypot(yy.mean() - cx, xx.mean() - cy)
        score = a - 8.0 * d           # big and central; no shape term
        if score > best_score:
            best, best_score = m, score
    return best


def caliper(mask, px, py):
    """Physical extent of the blob: full width/height in mm + equivalent diameter."""
    ii, jj = np.nonzero(mask)
    w = (ii.max() - ii.min() + 1) * px
    h = (jj.max() - jj.min() + 1) * py
    d = 2.0 * np.sqrt(mask.sum() * px * py / np.pi)
    return w, h, d


def show(ax, d, ext, title, mask=None):
    ax.imshow(d, cmap="gray", extent=ext, aspect="equal", vmin=0, vmax=np.percentile(d, 99.5))
    if mask is not None:
        ax.contour(np.linspace(ext[0], ext[1], mask.shape[1]),
                   np.linspace(ext[2], ext[3], mask.shape[0]),
                   mask.astype(float), levels=[0.5], colors="lime", linewidths=1.1)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    subs = pick()
    fig, axes = plt.subplots(len(subs), 4, figsize=(15.5, 3.5 * len(subs)))
    summary = []
    for row, (r, kind, path) in enumerate(subs):
        sl, px, py = midslice(path)
        ratio = r["nx_over_rx"]
        # The 2024 formula differs from what was stamped ONLY for UIH (see module docstring).
        px_naive = px * max(ratio, 1.0) if r["scanner"].startswith("UIH") else px

        nx, ny = sl.shape
        z, _, _ = zoom_box(sl, px, py)
        mask = find_bloodpool(z)

        for col, (pxx, tag) in enumerate([(px, "CURRENT  FOVx/nx"), (px_naive, "NAIVE 2024  FOVx/rx")]):
            show(axes[row, col], sl.T[::-1], [0, nx * pxx, 0, ny * py],
                 f"{r['cid'][:40]}\n{r['scanner']}  nx/rx={ratio:.2f}\n{tag}   {pxx:.3f} x {py:.3f} mm")
            zt = z.T[::-1]
            mt = mask.T[::-1] if mask is not None else None
            ext = [0, z.shape[0] * pxx, 0, z.shape[1] * py]
            if mask is not None:
                w, h, dia = caliper(mask, pxx, py)
                lab = f"LV blob  {w:.0f} x {h:.0f} mm   d_eq {dia:.0f} mm   w/h {w/h:.2f}"
                summary.append((r["cid"], r["scanner"], ratio, tag.split()[0], w, h, dia))
            else:
                lab = "no blob found"
            show(axes[row, col + 2], zt, ext, f"heart zoom - {tag.split()[0]}\n{lab}", mt)
        if kind == "CONTROL":
            axes[row, 1].set_title(axes[row, 1].get_title() + "\n(IDENTICAL by construction)", fontsize=8)
    fig.suptitle(
        "UIH FOVx convention: stamped rule (FOVx/nx) vs the 2024 formula (FOVx/rx)\n"
        "identical voxels, different stamped spacing -> only the physical aspect changes. "
        "Green = auto-detected LV blood pool; an adult mid-LV cavity is ~40-55 mm.\n"
        "Siemens rows are controls (must be identical); UIH 1.5T rows have nx/rx~1 so the "
        "correction is a no-op there.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=125, bbox_inches="tight")

    print(f"{'subject':42s} {'scanner':22s} {'nx/rx':>6} {'hyp':>8} {'w_mm':>7} {'h_mm':>7} {'d_eq':>6} {'w/h':>5}")
    for cid, sc, ra, hyp, w, h, d in summary:
        print(f"{cid[:42]:42s} {sc:22s} {ra:6.2f} {hyp:>8} {w:7.1f} {h:7.1f} {d:6.1f} {w/h:5.2f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
