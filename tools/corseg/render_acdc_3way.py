"""All-z-slice 3-way panels on ACDC: human GT vs CorSeg vs nnU-Net Task114.

The only view here with real truth in it, so it is the one worth looking at closely. Contours are
remapped to one convention (1=LV cavity, 2=myocardium, 3=RV) and drawn on the greyscale frame.

Usage:
  micromamba run -n svr python tools/corseg/render_acdc_3way.py \
      --cases patient117_ED patient103_ES --out result/corseg/acdc_3way.png
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
CONV = {"acdc": {1: 3, 2: 2, 3: 1}, "t114": {1: 1, 2: 2, 3: 3}, "corseg": {1: 2, 2: 1, 3: 3}}


def to_common(arr, conv):
    """Remap a label map into common 1=LV cav, 2=myo, 3=RV."""
    m = CONV[conv]
    out = np.zeros_like(arr)
    for common, native in m.items():
        out[arr == native] = common
    return out


def dice(a, b):
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--img_dir", default="scratch/data/nnunet_mnms/acdc/inputs")
    ap.add_argument("--gt_dir", default="scratch/data/nnunet_mnms/acdc/gt")
    ap.add_argument("--corseg_dir", default="/tmp/corseg_acdc/corseg_full_paper")
    ap.add_argument("--nnunet_dir", default="scratch/data/nnunet_mnms/acdc/seg_2d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin", type=int, default=28)
    args = ap.parse_args()

    blocks = []
    for case in args.cases:
        img = np.asarray(nib.load(f"{args.img_dir}/{case}_0000.nii.gz").dataobj, dtype=np.float32)
        gt = to_common(np.asarray(nib.load(f"{args.gt_dir}/{case}.nii.gz").dataobj), "acdc")
        cs = to_common(np.asarray(nib.load(f"{args.corseg_dir}/{case}.nii.gz").dataobj), "corseg")
        nn = to_common(np.asarray(nib.load(f"{args.nnunet_dir}/{case}.nii.gz").dataobj), "t114")
        dc = {n: dice(cs == l, gt == l) for l, n in NAMES.items()}
        dn = {n: dice(nn == l, gt == l) for l, n in NAMES.items()}
        rows = [("human GT", gt, ""),
                (f"CorSeg", cs, "  ".join(f"{k.split()[0]} {v:.2f}" for k, v in dc.items())),
                (f"nnU-Net 2d", nn, "  ".join(f"{k.split()[0]} {v:.2f}" for k, v in dn.items()))]
        blocks.append((case, img, rows))

    Z = max(b[1].shape[2] for b in blocks)
    nrow = sum(len(b[2]) for b in blocks)
    fig, axes = plt.subplots(nrow, Z, figsize=(1.5 * Z, 1.62 * nrow), squeeze=False)
    r = 0
    for case, img, rows in blocks:
        union = np.zeros(img.shape, bool)
        for _, s, _ in rows:
            union |= (s > 0)
        xs, ys, _ = np.where(union)
        m = args.margin
        x0, x1 = max(0, xs.min() - m), min(img.shape[0], xs.max() + 1 + m)
        y0, y1 = max(0, ys.min() - m), min(img.shape[1], ys.max() + 1 + m)
        vmax = float(np.percentile(img[img > 0], 99.5))
        for title, seg, dtxt in rows:
            for z in range(Z):
                ax = axes[r][z]
                if z < img.shape[2]:
                    ax.imshow(img[x0:x1, y0:y1, z].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
                    s = seg[x0:x1, y0:y1, z].T
                    # filled translucent overlay: a contour for the LV cavity is invisible because
                    # it coincides exactly with the myocardium's inner boundary
                    rgba = np.zeros(s.shape + (4,), np.float32)
                    for lab, col in COLORS.items():
                        sel = (s == lab)
                        if sel.any():
                            rgba[sel, :3] = matplotlib.colors.to_rgb(col)
                            rgba[sel, 3] = 0.45
                    ax.imshow(rgba, origin="lower", interpolation="nearest")
                else:
                    ax.set_facecolor("black")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"z={z}", fontsize=8)
                if z == 0:
                    lab = f"{case}\n{title}" if title == "human GT" else title
                    ax.set_ylabel(lab + (f"\nDice {dtxt}" if dtxt else ""), fontsize=6.8,
                                  rotation=0, ha="right", va="center", labelpad=4)
            r += 1
    handles = [plt.Rectangle((0,0),1,1, fc=c, alpha=0.45, label=NAMES[l]) for l, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("ACDC — human GT vs CorSeg vs nnU-Net Task114, every z-plane", fontsize=12)
    fig.tight_layout(rect=[0, 0.022, 1, 0.975])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
