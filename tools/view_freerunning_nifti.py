"""Quick visual montage of the free-running 5D demo NIfTIs (Zenodo 15033956).

Renders, for the converted resp*_4d.nii.gz:
  - slices stepping through two spatial axes (anatomy / where the heart sits)
  - the cardiac cycle at a fixed mid-slice (beating heart)
  - the 5 respiratory phases at a fixed mid-slice (breathing)
Output: result/freerunning_demo/montage.png
"""
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_NIFTI = "/home/minsukc/vggt/scratch/data/FRF/nifti"
DEFAULT_OUT = "/home/minsukc/vggt/result/freerunning_demo"


def show(ax, im, vmax, title=None):
    ax.imshow(im.T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nifti-dir", default=DEFAULT_NIFTI)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--name", default="montage.png")
    ap.add_argument("--tag", default="Free-running 5D demo (Zenodo 15033956)")
    args = ap.parse_args()
    NIFTI, OUT = args.nifti_dir, args.out

    os.makedirs(OUT, exist_ok=True)
    img = nib.load(os.path.join(NIFTI, "resp0_4d.nii.gz"))
    vol = np.asarray(img.dataobj)  # (200,186,234,20)
    X, Y, Z, T = vol.shape
    vmax = float(np.quantile(vol[..., 0], 0.99))

    vmax *= 0.7  # brighten (UTE is dim)
    # heart sits near the middle of axis0; this (Y,Z) view shows the chambers
    xh = int(X * 0.58)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(4, 8, hspace=0.18, wspace=0.05)

    # Row 1: step through axis 2 (the 234/350mm = SI-ish axis), cardiac phase 0
    zs = np.linspace(Z * 0.18, Z * 0.82, 8).astype(int)
    for j, z in enumerate(zs):
        show(fig.add_subplot(gs[0, j]), vol[:, :, z, 0], vmax, f"ax2={z}")

    # Row 2: step through axis 0 (200/300mm axis), cardiac phase 0  -> heart views
    xs = np.linspace(X * 0.18, X * 0.82, 8).astype(int)
    for j, x in enumerate(xs):
        show(fig.add_subplot(gs[1, j]), vol[x, :, :, 0], vmax, f"ax0={x}")

    # Row 3: cardiac cycle at the heart slice (axis0=xh), 8 of 20 phases
    ts = np.linspace(0, T - 1, 8).astype(int)
    for j, t in enumerate(ts):
        show(fig.add_subplot(gs[2, j]), vol[xh, :, :, t], vmax, f"card t={t}")

    # Row 4: 5 respiratory phases at the same heart slice, cardiac phase 0
    for r in range(5):
        rv = nib.load(os.path.join(NIFTI, f"resp{r}_4d.nii.gz")).dataobj[xh, :, :, 0]
        show(fig.add_subplot(gs[3, r]), np.asarray(rv), vmax, f"resp {r}")
    for r in range(5, 8):
        fig.add_subplot(gs[3, r]).axis("off")

    fig.suptitle(
        f"{args.tag} — 200x186x234 iso 1.5mm, 20 cardiac x 5 resp\n"
        "row1: slices along axis2 | row2: slices along axis0 (heart) | row3: cardiac cycle (heart slice) | row4: respiratory phases (heart slice)",
        fontsize=10,
    )
    path = os.path.join(OUT, args.name)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    main()
