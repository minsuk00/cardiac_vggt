"""Rebuild the 5-panel beating-heart comparison GIF at FULL 256-level grayscale.

Reproduces the (lost, ephemeral) `allpanels.py` builder — same mid-z, full uncropped
slice, animated over the cardiac cycle: GT gated | Real-time input | SVRTK 3D+t (robust
ON) | SVRTK 4D-joint | NiftyMIC 3D+t. The original file was saved with a 16-color palette
(`quantize(colors=16)`), which posterized every panel — including the GT. This version
saves grayscale `L` frames with NO color reduction (PIL keeps 256 gray levels).

Run: micromamba run -n svr python baselines/fetal_cmr_4d/build_allpanels.py --subject Volunteer1
"""
import argparse, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def panels_for(subject):
    return [
        ("GT gated",               f"scratch/data/MIITT/nifti/{subject}/gated/sax/4d_recon.nii.gz"),
        ("Real-time (input)",      f"scratch/data/MIITT/nifti/{subject}/realtime/sax/4d_recon.nii.gz"),
        ("SVRTK 3D+t (robust ON)", f"scratch/fetal_cmr_4d/recon/{subject}/perphase_cine_K7rob/cine.nii.gz"),
        ("SVRTK 4D-joint",         f"scratch/fetal_cmr_4d/recon/{subject}/selfgate_cine/cine.nii.gz"),
        ("NiftyMIC 3D+t",          f"scratch/niftymic/recon/{subject}_binned/cine.nii.gz"),
    ]


def load_midz(path):
    """Return (mid-z, all-phase) slices as (H, W, T), + a global (1,99)-percentile vmax."""
    a = nib.load(os.path.join(REPO, path)).get_fdata().astype(np.float32)   # (X, Y, Z, T)
    z = a.shape[2] // 2
    sl = a[:, :, z, :]                                                      # (X, Y, T)
    fg = sl[sl > 0]
    lo, hi = np.percentile(fg, 1), np.percentile(fg, 99)
    return sl, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="Volunteer1")
    ap.add_argument("--nframes", type=int, default=30, help="output GIF frame count (master clock)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--duration", type=int, default=100, help="ms/frame")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    names_paths = panels_for(args.subject)
    slabs = [(nm, *load_midz(p)) for nm, p in names_paths]   # (name, (X,Y,T), lo, hi)
    NF = args.nframes

    frames = []
    for f in range(NF):
        fig, axes = plt.subplots(1, len(slabs), figsize=(3.4 * len(slabs), 3.6), squeeze=False)
        for c, (nm, sl, lo, hi) in enumerate(slabs):
            T = sl.shape[2]
            t = int(round(f / NF * T)) % T                   # proportional phase per volume
            img = np.clip((sl[:, :, t] - lo) / (hi - lo + 1e-6), 0, 1)
            ax = axes[0][c]
            ax.imshow(img.T, cmap="gray", vmin=0, vmax=1)    # .T -> upright (row=Y, col=X)
            ax.set_title(nm, fontsize=10); ax.axis("off")
        fig.suptitle(f"{args.subject} — same mid-z, full slice (uncropped)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=args.dpi); plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("L"))          # 256-level grayscale, NO quantize

    out = args.out or os.path.join(REPO, "scratch", "fetal_cmr_4d", "recon",
                                   f"allpanels_{args.subject}.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=args.duration, loop=0)
    # verify color depth
    im = Image.open(out); im.seek(im.n_frames // 2)
    ncol = len(np.unique(np.array(im.convert("L"))))
    print(f"wrote {out}  ({im.n_frames} frames, {im.size[0]}x{im.size[1]}, {ncol} gray levels)")


if __name__ == "__main__":
    main()
