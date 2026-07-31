"""CorSeg vs nnU-Net on the REAL heart-ROI volumes our SVR baselines emit.

These are what `ef_dice.py dump` actually feeds the segmenter for the SVR arms:
`<subject>/{svrtk3d,nesvor}/recon_clean/vol_tXX.nii.gz` — small ~1.4 mm ISOTROPIC heart-centred
grids (e.g. 72x87x86), not the canonical (256,256,12) @ (1.4,1.4,12) cube that the GT and VGGT arms
use. Slice axis is axis 2 (verified by adjacent-slice correlation). No human GT exists here, so this
is qualitative: the question is whether each model produces plausible cardiac anatomy at all.

Usage:
  micromamba run -n svr python tools/corseg/render_svr_roi.py \
      --cases Test_P012__svrtk3d Test_P012__nesvor --nslices 12 \
      --out result/corseg/svr_roi_3way.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import nibabel as nib              # noqa: E402
import numpy as np                 # noqa: E402

ROOT = "/home/minsukc/vggt/scratch/data/corseg/bench/svr_roi"
COLORS = {1: "#4488FF", 2: "#FF4444", 3: "#44CC44"}
NAMES = {1: "LV cavity", 2: "myocardium", 3: "RV"}


def corseg_to_common(c):
    o = np.zeros_like(c)
    o[c == 2] = 1
    o[c == 1] = 2
    o[c == 3] = 3
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--nslices", type=int, default=12)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    blocks = []
    for case in args.cases:
        img = np.asarray(nib.load(f"{ROOT}/in/{case}_0000.nii.gz").dataobj, np.float32)
        cs = corseg_to_common(np.asarray(nib.load(f"{ROOT}/corseg/{case}.nii.gz").dataobj).astype(np.uint8))
        nnf = f"{ROOT}/nnunet/{case}.nii.gz"
        nn = np.asarray(nib.load(nnf).dataobj).astype(np.uint8) if os.path.exists(nnf) else None
        blocks.append((case, img, cs, nn))

    Z = args.nslices
    nrow = sum(1 + 1 + (1 if b[3] is not None else 0) for b in blocks)
    fig, axes = plt.subplots(nrow, Z, figsize=(1.45 * Z, 1.6 * nrow), squeeze=False)
    r = 0
    for case, img, cs, nn in blocks:
        zs = np.linspace(0, img.shape[2] - 1, Z).round().astype(int)
        vmax = float(np.percentile(img[img != 0], 99.5)) if (img != 0).any() else 1.0
        vmin = float(np.percentile(img[img != 0], 1.0)) if (img != 0).any() else 0.0
        rows = [("image", None), ("CorSeg", cs)]
        if nn is not None:
            rows.append(("nnU-Net 2d", nn))
        for title, seg in rows:
            nvox = int((seg > 0).sum()) if seg is not None else 0
            for j, z in enumerate(zs):
                ax = axes[r][j]
                ax.imshow(img[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax, origin="lower")
                if seg is not None:
                    s = seg[:, :, z].T
                    rgba = np.zeros(s.shape + (4,), np.float32)
                    for lab, col in COLORS.items():
                        sel = (s == lab)
                        if sel.any():
                            rgba[sel, :3] = matplotlib.colors.to_rgb(col)
                            rgba[sel, 3] = 0.45
                    ax.imshow(rgba, origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"z={z}", fontsize=8)
                if j == 0:
                    lab = (f"{case}\n{img.shape}\n{title}" if title == "image"
                           else f"{title}\n{nvox:,} vox" + ("  BLANK" if nvox == 0 else ""))
                    ax.set_ylabel(lab, fontsize=6.6, rotation=0, ha="right", va="center", labelpad=4)
            r += 1
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.45, label=NAMES[l]) for l, c in COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle("CorSeg vs nnU-Net on the REAL heart-ROI volumes our SVR baselines emit "
                 "(~1.4 mm isotropic; no human GT — qualitative)", fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=135, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
