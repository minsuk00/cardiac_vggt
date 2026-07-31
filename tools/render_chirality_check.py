"""Visual chirality check: render CMRx subjects flagged R>0 vs R<0 by probe_chirality.py, with an
affine-derived R/A direction arrow overlaid, so a human can see whether RV really sits on opposite
sides of LV between the two groups (a true left-right mirror) or whether the groups just look
similar (meaning the R-sign measurement was catching something else, e.g. an in-plane rotation,
not a real flip -- rotations preserve handedness, mirrors don't).

Usage: python render_chirality_check.py [--csv result/chirality_check/chirality.csv] [--n 4]
"""
import argparse, os, sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/minsukc/vggt"


def best_z(seg4d):
    """z-plane with the most combined LV+RV signal (summed across all T for robustness)."""
    lv_rv = (seg4d == 1) | (seg4d == 3)
    area = lv_rv.sum(axis=(0, 1, 3))
    return int(np.argmax(area))


def r_a_screen_dirs(affine):
    """-> ((dx,dy) for +R, (dx,dy) for +A) in on-screen coords, for imshow(arr[:,:,z], origin='upper')
    where array axis 0 (X) is the row/vertical-DOWN axis and array axis 1 (Y) is the column/
    horizontal-RIGHT axis."""
    L = np.asarray(affine, dtype=np.float64)[:3, :3]
    M = L[:2, :2]                                   # world (R,A) <- array (X,Y), in-plane block
    Minv = np.linalg.inv(M)

    def to_screen(world_dir):
        ax_, ay_ = Minv @ np.asarray(world_dir, dtype=np.float64)     # delta in (array-X, array-Y)
        dx, dy = ay_, ax_                                              # screen-x=array-Y, screen-y=array-X
        n = np.hypot(dx, dy)
        return (dx / n, dy / n) if n > 0 else (0.0, 0.0)

    return to_screen((1, 0)), to_screen((0, 1))


def render_one(ax, out_dir, title):
    img_f = os.path.join(out_dir, "3d_recon", "sax_frame_00.nii.gz")
    seg_f = os.path.join(out_dir, "heart_seg.nii.gz")
    im = nib.load(img_f)
    seg_im = nib.load(seg_f)
    frame = np.asarray(im.dataobj)                  # (X,Y,Z)
    seg4d = np.asarray(seg_im.dataobj).astype(np.uint8)   # (X,Y,Z,T)
    z = best_z(seg4d)
    sl = frame[:, :, z]
    seg_sl = seg4d[:, :, z, 0]

    # Raw native images, not yet resampled to the canonical isotropic grid -- in-plane spacing
    # varies per subject and isn't always square, so imshow's default aspect='equal' (1 array-index
    # step == 1 array-index step on screen) visibly squashes/stretches anisotropic subjects.
    # aspect = physical-mm-per-row-step / physical-mm-per-column-step, per matplotlib's "ratio of
    # y-unit to x-unit" convention for Axes.set_aspect.
    sx, sy, _ = nib.affines.voxel_sizes(seg_im.affine)      # (row/X spacing, col/Y spacing) mm
    ax.imshow(sl, cmap="gray", origin="upper", aspect=sx / sy)
    ax.contour(seg_sl == 1, colors="red", linewidths=1.3, levels=[0.5])
    ax.contour(seg_sl == 3, colors="deepskyblue", linewidths=1.3, levels=[0.5])

    (rdx, rdy), (adx, ady) = r_a_screen_dirs(seg_im.affine)
    cx, cy = sl.shape[1] / 2.0, sl.shape[0] / 2.0
    Larrow = min(sl.shape) * 0.38
    ax.annotate("R", xy=(cx + rdx * Larrow, cy + rdy * Larrow), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="yellow", lw=2),
                color="yellow", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.annotate("A", xy=(cx + adx * Larrow, cy + ady * Larrow), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="lime", lw=2),
                color="lime", fontsize=13, fontweight="bold", ha="center", va="center")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main():
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "result/chirality_check/chirality.csv"))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(ROOT, "result/chirality_check/render_check.png"))
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv)) if r["dataset"] == "cmrx"]
    pos = sorted([r for r in rows if float(r["R"]) > 0], key=lambda r: -float(r["R"]))[:args.n]
    neg = sorted([r for r in rows if float(r["R"]) < 0], key=lambda r: float(r["R"]))[:args.n]

    fig, axes = plt.subplots(2, args.n, figsize=(3.2 * args.n, 7.6),
                              gridspec_kw={"hspace": 0.5, "wspace": 0.15})
    for col, r in enumerate(pos):
        out_dir = os.path.join(ROOT, "scratch/data", r["source"], "Cine_combined", r["subject"], "sax")
        render_one(axes[0, col], out_dir, f"R>0 {r['subject']}\nR={float(r['R']):.1f}")
    for col, r in enumerate(neg):
        out_dir = os.path.join(ROOT, "scratch/data", r["source"], "Cine_combined", r["subject"], "sax")
        render_one(axes[1, col], out_dir, f"R<0 {r['subject']}\nR={float(r['R']):.1f}")

    axes[0, 0].set_ylabel("R>0 (normal?)", fontsize=11)
    axes[1, 0].set_ylabel("R<0 (flipped?)", fontsize=11)
    fig.suptitle("Chirality check: LV=red, RV=blue. Yellow/green arrows = this subject's own "
                 "affine-derived R/A direction.\nIf R<0 row's RV sits on the OPPOSITE side of LV "
                 "relative to the R arrow vs the R>0 row -> real mirror. If arrows just point a "
                 "different way but RV-vs-arrow relationship matches -> rotation, not a flip.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
