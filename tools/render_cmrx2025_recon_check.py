"""Visual gate for the CMRxRecon2025 recon, before any batch run.

Two of the three 2025-specific fixes fail SILENTLY if they are wrong -- a misplaced k-space fill
puts the heart off-centre, and a wrong vendor FOV rule stretches it -- so headers and exit codes
prove nothing here. Everything is drawn at TRUE PHYSICAL ASPECT (`aspect = pixel_y/pixel_x` from
the NIfTI affine), because a spacing bug is invisible in a square-pixel render and obvious in
this one: the LV must read as a circle, not an ellipse.

Outputs (project dir, not /tmp -- these are meant to be looked at):
    result/cmrx2025_recon_check/<cid>.png   per subject: every z at ED + mid-z across the cycle
    result/cmrx2025_recon_check/_contact_sheet.png   all subjects, mid-z, same physical scale

Usage:
    python tools/render_cmrx2025_recon_check.py --root /tmp/cmrx2025_validate
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "result", "cmrx2025_recon_check")


def load(cid_dir):
    """-> (vol (T,Z,Y,X), (pix_y, pix_x, pitch_z))."""
    p = os.path.join(cid_dir, "sax", "4d_recon.nii.gz")
    img = nib.load(p)
    a = np.asarray(img.dataobj, dtype=np.float32)      # sitk wrote (X,Y,Z,T)
    zoom = img.header.get_zooms()
    a = np.transpose(a, (3, 2, 1, 0))                  # -> (T,Z,Y,X)
    return a, (float(zoom[1]), float(zoom[0]), float(zoom[2]))


def norm(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def panel(cid, vol, sp, path):
    T, Z, Y, X = vol.shape
    py, px, pz = sp
    aspect = py / px                                    # rows are y, cols are x
    ncol = max(Z, T)
    fig, axes = plt.subplots(2, ncol, figsize=(1.5 * ncol, 3.6))
    axes = np.atleast_2d(axes)
    for j in range(ncol):
        for i in range(2):
            axes[i, j].axis("off")
    for z in range(Z):
        axes[0, z].imshow(norm(vol[0, z]), cmap="gray", aspect=aspect)
        axes[0, z].set_title(f"z{z}", fontsize=6)
    zmid = Z // 2
    for t in range(T):
        axes[1, t].imshow(norm(vol[t, zmid]), cmap="gray", aspect=aspect)
        axes[1, t].set_title(f"t{t}", fontsize=6)
    fig.suptitle(f"{cid}   {Y}x{X}x{Z}, {py:.2f} x {px:.2f} x {pz:.1f} mm   "
                 f"(top: all z at ED — bottom: mid-z over the cycle)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, dpi=110)
    plt.close(fig)


def body_width_mm(vol, py, px):
    """(width, height) of the body silhouette in mm, from a robust absolute threshold.

    Averaged over all slices and frames so a dim single slice cannot drive it; the largest
    connected component only, so noise specks and wrap do not inflate the bounding box.
    """
    from scipy import ndimage
    m0 = vol.mean(axis=(0, 1))
    m = ndimage.binary_opening(m0 > 0.15 * np.percentile(m0, 99.5), np.ones((3, 3)))
    m = ndimage.binary_closing(m, np.ones((9, 9)))
    lab, n = ndimage.label(m)
    if n:
        m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    ys, xs = np.where(m)
    if len(xs) == 0:
        return float("nan"), float("nan")
    return (xs.max() - xs.min() + 1) * px, (ys.max() - ys.min() + 1) * py


def contact_sheet(entries, path):
    """Mid-z at ED for every subject, with a 100 mm scale bar so physical size is checkable.

    The scale bar is the point of this figure: spacing errors are invisible in a square-pixel
    render, but with a bar you can see directly whether a thorax is ~350 mm or absurdly 700 mm.
    """
    n = len(entries)
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (cid, vol, sp, rep) in zip(axes, entries):
        py, px, pz = sp
        Z, Y, X = vol.shape[1], vol.shape[2], vol.shape[3]
        bx, by = body_width_mm(vol, py, px)
        # draw in MILLIMETRES so every panel shares one physical scale
        ax.imshow(norm(vol[0, Z // 2]), cmap="gray",
                  extent=[0, X * px, Y * py, 0], aspect="equal")
        ax.plot([10, 110], [Y * py - 14, Y * py - 14], "-", color="yellow", lw=3)
        ax.text(10, Y * py - 20, "100 mm", color="yellow", fontsize=8, va="bottom")
        fill = rep.get("fill_y", "?")
        ax.set_title(f"{rep.get('scanner','?')}  [{rep.get('matver','?')}]\n"
                     f"{py:.2f} x {px:.2f} x {pz:.0f} mm   ky fill {fill:+}\n"
                     f"body {bx:.0f} x {by:.0f} mm", fontsize=8)
        ax.axis("off")
    fig.suptitle("CMRxRecon2025 recon check — mid-ventricular slice at ED, drawn in millimetres\n"
                 "PASS = LV reads as a circle AND the body is ~300-400 mm wide against the 100 mm bar",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/cmrx2025_validate")
    ap.add_argument("--report", default="/tmp/cmrx2025_validate_report.json")
    args = ap.parse_args()

    reps = {}
    if os.path.exists(args.report):
        reps = {r["cid"]: r for r in json.load(open(args.report))}
    os.makedirs(OUT, exist_ok=True)

    entries = []
    for d in sorted(glob.glob(os.path.join(args.root, "CMRx25_*"))):
        cid = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "sax", "4d_recon.nii.gz")):
            continue
        vol, sp = load(d)
        panel(cid, vol, sp, os.path.join(OUT, f"{cid}.png"))
        entries.append((cid, vol, sp, reps.get(cid, {})))
        print(f"  {cid:52} {vol.shape}  {sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.1f} mm", flush=True)

    if entries:
        contact_sheet(entries, os.path.join(OUT, "_contact_sheet.png"))
    print(f"\n{len(entries)} subjects -> {OUT}")


if __name__ == "__main__":
    main()
