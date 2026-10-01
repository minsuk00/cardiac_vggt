"""Curation examples: retained (aligned) vs discarded (breath-hold misaligned) gated SAX stacks.

Columns = subjects; rows = two long-axis views reformatted from the SAX stack through the LV+myocardium
centroid (heart_seg labels 1,2): along x (fixed y) and along y (fixed x). Nearest-neighbour in z at true
physical aspect, so a slice acquired at a different breath-hold position shows as a step. Frame 0. Each
panel is W_MM wide by the full stack height; column widths follow that aspect so panels tile with no gaps.

Usage: micromamba run -n svr python tools/paper_figures/render_curation_examples.py --out curation_examples.pdf
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from render_runtime_accuracy import TEXTW, paper_rc

DATA = "/home/minsukc/vggt/scratch/data"
SUBJECTS = [  # (rel, group)
    ("MNMs_sax/MNMs_A8I1U6", "Retained"),
    ("MNMs_sax/MNMs_C5M4S2", "Retained"),
    ("ACDC_sax/ACDC_patient085", "Discarded"),
    ("CMRxRecon2024/Cine_combined/CMRx24_Test_P039", "Discarded"),
]
W_MM = 110   # in-plane physical width of every panel


def load(rel):
    f = sorted(glob.glob(f"{DATA}/{rel}/sax/3d_recon/sax_frame_*.nii.gz"))[0]
    img = nib.load(f)
    vol = np.asarray(img.dataobj, dtype=np.float32)            # (X, Y, Z), LPS, apex at z0
    seg = np.asarray(nib.load(f"{DATA}/{rel}/sax/heart_seg.nii.gz").dataobj)[..., 0]
    dx, dy, dz = (float(v) for v in img.header.get_zooms()[:3])
    heart = np.isin(seg, (1, 2))
    hi = np.percentile(vol[heart], 99.5)                         # window on the heart, not the whole FOV
    return np.clip(vol / hi, 0, 1), heart, (dx, dy, dz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    paper_rc()
    data = [load(rel) for rel, _ in SUBJECTS]
    ratios = [W_MM / (v.shape[2] * z[2]) for v, _, z in data]   # full stack height, W_MM wide, true aspect
    gap = 0.03                                                   # inches between panels
    ph = (TEXTW - gap * (len(ratios) - 1)) / sum(ratios)         # panel height (inches)
    th, bh = 0.20, 0.16                                          # group-header strip, (a)-(d) label strip
    H = 2 * ph + gap + th + bh
    fig, axes = plt.subplots(2, len(SUBJECTS), figsize=(TEXTW, H),
                             gridspec_kw=dict(width_ratios=ratios))
    fig.subplots_adjust(left=0, right=1, bottom=bh / H, top=1 - th / H,
                        wspace=gap / (TEXTW / len(ratios)), hspace=gap / ph)
    for j, (vol, heart, (dx, dy, dz)) in enumerate(data):
        X, Y, Z = vol.shape
        ix, iy, _ = np.nonzero(heart)
        cx, cy = int(round(ix.mean())), int(round(iy.mean()))
        for i, (sl, N, d, c) in enumerate([(vol[:, cy, :], X, dx, cx), (vol[cx, :, :], Y, dy, cy)]):
            ax = axes[i, j]
            ax.imshow(sl.T, cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest",
                      extent=(0, N * d, 0, Z * dz), aspect="auto")  # base up, apex down
            ax.set_xlim(c * d - W_MM / 2, c * d + W_MM / 2)
            ax.set_ylim(0, Z * dz)
            ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        b = axes[1, j].get_position()
        fig.text((b.x0 + b.x1) / 2, b.y0 - 0.02 / H, f"({'abcd'[j]})", ha="center", va="top", fontsize=8)
    # one header per group, with a bar spanning its columns
    for g in dict.fromkeys(grp for _, grp in SUBJECTS):
        cols = [j for j, (_, grp) in enumerate(SUBJECTS) if grp == g]
        x0, x1 = axes[0, cols[0]].get_position().x0, axes[0, cols[-1]].get_position().x1
        yb = 1 - th / H + 0.03 / H
        inset = 0.04 / TEXTW                                     # keep the two bars visibly apart
        fig.add_artist(plt.Line2D([x0 + inset, x1 - inset], [yb, yb], color="black", lw=0.8, transform=fig.transFigure))
        fig.text((x0 + x1) / 2, yb + 0.02 / H, g, ha="center", va="bottom", fontsize=8)
    fig.savefig(args.out, dpi=300, pad_inches=0.01)
    fig.savefig(os.path.splitext(args.out)[0] + ".png", dpi=200, pad_inches=0.01)


if __name__ == "__main__":
    main()
