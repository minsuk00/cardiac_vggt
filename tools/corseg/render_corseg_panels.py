"""Render an all-z-slice comparison panel: image | CorSeg(paper) | CorSeg(gui) | nnU-Net Task114.

Spans EVERY canonical z-plane (the mid plane alone is the easiest/least informative one), so
base/apex failures are visible. Contours are drawn on the greyscale image; labels are remapped to
one common convention (LV cavity / myocardium / RV) so the two models are directly comparable.

Usage:
  micromamba run -n svr python tools/corseg/render_corseg_panels.py \
      --subject_dir scratch/eval/cmrxrecon/out/Val_P048 \
      --corseg_paper <dir> --corseg_gui <dir> --t 0 --out result/corseg/panel_Val_P048_t00.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                     # noqa: E402
import numpy as np                        # noqa: E402

# common convention used for PLOTTING: 1=LV cavity, 2=myocardium, 3=RV
COLORS = {1: "#4488FF", 2: "#FF4444", 3: "#44CC44"}
NAMES = {1: "LV cavity", 2: "myocardium", 3: "RV"}


def corseg_to_common(c):
    """CorSeg (1=myo, 2=LV cav, 3=RV) -> common (1=LV cav, 2=myo, 3=RV)."""
    o = np.zeros_like(c)
    o[c == 2] = 1
    o[c == 1] = 2
    o[c == 3] = 3
    return o


def t114_to_common(c):
    """Task114 is already (1=LV cav, 2=myo, 3=RV)."""
    return c.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject_dir", required=True)
    ap.add_argument("--corseg_paper", required=True)
    ap.add_argument("--corseg_gui", default=None)
    ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zoom", action="store_true",
                    help="crop the DISPLAY window to the union-of-segmentations bbox (+margin)")
    ap.add_argument("--zoom_margin", type=int, default=24)
    args = ap.parse_args()

    t = args.t
    img = np.asarray(nib.load(os.path.join(args.subject_dir, "gt", f"gt_t{t:02d}.nii.gz")).dataobj,
                     dtype=np.float32)                                   # (X,Y,Z)
    nn = np.asarray(nib.load(os.path.join(args.subject_dir, "heart_seg.nii.gz")).dataobj)
    nn = t114_to_common(nn[..., t].astype(np.uint8))

    rows = [("image", None)]
    cs_p = np.asarray(nib.load(os.path.join(args.corseg_paper, f"gt_t{t:02d}.nii.gz")).dataobj)
    rows.append(("CorSeg (paper prep: 1.25mm + crop224)", corseg_to_common(cs_p.astype(np.uint8))))
    if args.corseg_gui:
        cs_g = np.asarray(nib.load(os.path.join(args.corseg_gui, f"gt_t{t:02d}.nii.gz")).dataobj)
        rows.append(("CorSeg (shipped GUI prep: naive resize)", corseg_to_common(cs_g.astype(np.uint8))))
    rows.append(("nnU-Net Task114 (M&Ms) - current", nn))

    Z = img.shape[2]
    # optional zoom: common display window = bbox of the union of all segmentations, + margin
    xs0, xs1, ys0, ys1 = 0, img.shape[0], 0, img.shape[1]
    if args.zoom:
        union = np.zeros(img.shape[:3], bool)
        for _, s in rows:
            if s is not None:
                union |= (s > 0)
        if union.any():
            xs, ys, _ = np.where(union)
            m = args.zoom_margin
            xs0, xs1 = max(0, xs.min() - m), min(img.shape[0], xs.max() + 1 + m)
            ys0, ys1 = max(0, ys.min() - m), min(img.shape[1], ys.max() + 1 + m)

    fig, axes = plt.subplots(len(rows), Z, figsize=(1.45 * Z, 1.55 * len(rows)))
    axes = np.atleast_2d(axes)
    vmax = float(np.percentile(img[img > 0], 99.5)) if (img > 0).any() else 1.0

    for r, (title, seg) in enumerate(rows):
        for z in range(Z):
            ax = axes[r, z]
            ax.imshow(img[xs0:xs1, ys0:ys1, z].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
            if seg is not None:
                s = seg[xs0:xs1, ys0:ys1, z].T
                for lab, col in COLORS.items():
                    m = (s == lab)
                    if m.any():
                        ax.contour(m.astype(float), levels=[0.5], colors=[col], linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={z}", fontsize=8)
            if z == 0:
                ax.set_ylabel(title, fontsize=7.5, rotation=0, ha="right", va="center", labelpad=4)
    handles = [plt.Line2D([0], [0], color=c, lw=2, label=NAMES[l]) for l, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{os.path.basename(args.subject_dir)}  phase t={t}  -- all {Z} canonical z-planes",
                 fontsize=11)
    fig.tight_layout(rect=[0.0, 0.035, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
