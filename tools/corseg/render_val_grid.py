"""CorSeg output on our CMRxRecon canonical volumes — multi-subject, all z-planes, ED vs ES.

There is no human GT on this cohort, so nnU-Net is shown as a *reference*, not truth; the numbers
in the row labels are agreement Dice between the two models, not accuracy.

Usage:
  micromamba run -n svr python tools/corseg/render_val_grid.py \
      --subjects Val_P048 Val_P054 Val_P055 --corseg_root <dir> --t 0 \
      --out result/corseg/val_grid_t00.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import nibabel as nib              # noqa: E402
import numpy as np                 # noqa: E402

COLORS = {1: "#4488FF", 2: "#FF4444", 3: "#44CC44"}
NAMES = {1: "LV cavity", 2: "myocardium", 3: "RV"}
EVAL_ROOT = "scratch/eval/cmrxrecon/out"


def corseg_to_common(c):
    o = np.zeros_like(c)
    o[c == 2] = 1
    o[c == 1] = 2
    o[c == 3] = 3
    return o


def overlay(ax, img2d, seg2d, vmax):
    ax.imshow(img2d, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    if seg2d is not None:
        rgba = np.zeros(seg2d.shape + (4,), np.float32)
        for lab, col in COLORS.items():
            sel = (seg2d == lab)
            if sel.any():
                rgba[sel, :3] = matplotlib.colors.to_rgb(col)
                rgba[sel, 3] = 0.45
        ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def dice(a, b):
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--corseg_root", required=True,
                    help="dir containing <subject>_paper/gt_tXX.nii.gz")
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=26)
    args = ap.parse_args()
    t = args.t

    blocks = []
    for s in args.subjects:
        img = np.asarray(nib.load(f"{EVAL_ROOT}/{s}/gt/gt_t{t:02d}.nii.gz").dataobj, np.float32)
        cs = corseg_to_common(np.asarray(
            nib.load(f"{args.corseg_root}/{s}_paper/gt_t{t:02d}.nii.gz").dataobj).astype(np.uint8))
        nn = np.asarray(nib.load(f"{EVAL_ROOT}/{s}/heart_seg.nii.gz").dataobj)[..., t].astype(np.uint8)
        d = {n: dice(cs == l, nn == l) for l, n in NAMES.items()}
        blocks.append((s, img, cs, nn, d))

    Z = blocks[0][1].shape[2]
    fig, axes = plt.subplots(2 * len(blocks), Z, figsize=(1.42 * Z, 1.55 * 2 * len(blocks)),
                             squeeze=False)
    r = 0
    for s, img, cs, nn, d in blocks:
        union = (cs > 0) | (nn > 0)
        if union.any():
            xs, ys, _ = np.where(union)
            m = args.margin
            x0, x1 = max(0, xs.min() - m), min(img.shape[0], xs.max() + 1 + m)
            y0, y1 = max(0, ys.min() - m), min(img.shape[1], ys.max() + 1 + m)
        else:
            x0, x1, y0, y1 = 0, img.shape[0], 0, img.shape[1]
        vmax = float(np.percentile(img[img > 0], 99.5)) if (img > 0).any() else 1.0
        agree = "  ".join(f"{k.split()[0]} {v:.2f}" for k, v in d.items())
        for title, seg in [("CorSeg", cs), ("nnU-Net (ref)", nn)]:
            for z in range(Z):
                overlay(axes[r][z], img[x0:x1, y0:y1, z].T, seg[x0:x1, y0:y1, z].T, vmax)
                if r == 0:
                    axes[r][z].set_title(f"z={z}", fontsize=8)
                if z == 0:
                    lab = f"{s}\n{title}" + (f"\nagreement {agree}" if title == "CorSeg" else "")
                    axes[r][z].set_ylabel(lab, fontsize=6.6, rotation=0, ha="right", va="center",
                                          labelpad=4)
            r += 1
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.45, label=NAMES[l])
               for l, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"CorSeg on CMRxRecon canonical GT volumes — phase t={t}, every z-plane "
                 f"(no human GT here; nnU-Net shown as reference, not truth)", fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
