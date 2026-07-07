#!/usr/bin/env python3
"""Visual QC: fetal_cmr_4d recon (from MIITT real-time) vs the gated breath-hold GT.

Both are 13-slice SAX cines of the same volunteer but SEPARATE acquisitions with
PLACEHOLDER affines, so they are NOT voxel-aligned -- this is a *visual coherence*
check (did self-gating + SVR produce a sane beating heart?), not a pixel metric.
Quantitative PSNR/SSIM needs a rigid registration between the two scans first
(TODO, separate step).

For a chosen slice it renders two rows of per-phase frames (recon over its cardiac
phases, GT over its 30 phases) so you can eyeball structure + contraction.

Usage:
  micromamba run -n svr python baselines/fetal_cmr_4d/qc_compare.py Volunteer1 \
      [--recon scratch/fetal_cmr_4d/recon/Volunteer1/direct_cine/cine.nii.gz]
"""
import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_NIFTI = "/home/minsukc/MIITT/nifti/{vol}/gated/sax/4d_recon.nii.gz"
RECON_DEFAULT = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}/direct_cine/cine.nii.gz"
OUT = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}/qc_recon_vs_gt.png"

N_SHOW = 8  # phases to display per row


def load4d(path):
    a = nib.load(path).get_fdata().astype(np.float32)
    if a.ndim == 3:
        a = a[..., None]
    return a  # (X,Y,Z,P)


def pick_phases(P, n):
    return np.linspace(0, P - 1, min(n, P)).round().astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("--recon", default=None)
    ap.add_argument("--slice", type=int, default=None, help="slice index (default: max-motion slice)")
    args = ap.parse_args()

    recon_path = args.recon or RECON_DEFAULT.format(vol=args.vol)
    gt = load4d(GT_NIFTI.format(vol=args.vol))            # (Xg,Yg,13,30)
    rc = load4d(recon_path)                                # (Xr,Yr,Zr,Pr)
    print(f"recon {rc.shape}  gt {gt.shape}")

    # choose a slice with strong contraction in the GT (temporal variance peak in z)
    if args.slice is None:
        zvar = gt.std(axis=3).sum(axis=(0, 1))
        zg = int(np.argmax(zvar))
    else:
        zg = args.slice
    # map to recon z by relative depth (both cover the same 13-slice stack)
    zr = int(round(zg / (gt.shape[2] - 1) * (rc.shape[2] - 1)))
    print(f"GT slice {zg}, recon slice {zr}")

    gp = pick_phases(gt.shape[3], N_SHOW)
    rp = pick_phases(rc.shape[3], N_SHOW)

    def norm(im):
        lo, hi = np.percentile(im, [1, 99])
        return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)

    n = N_SHOW
    fig, ax = plt.subplots(2, n, figsize=(2.0 * n, 4.4))
    for j, p in enumerate(rp):
        ax[0, j].imshow(norm(rc[:, :, zr, p]).T, cmap="gray")
        ax[0, j].set_title(f"ph{p}", fontsize=8); ax[0, j].axis("off")
    for j, p in enumerate(gp):
        ax[1, j].imshow(norm(gt[:, :, zg, p]).T, cmap="gray")
        ax[1, j].set_title(f"ph{p}", fontsize=8); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("fetal_cmr_4d\n(RT self-gated)", fontsize=9)
    ax[1, 0].set_ylabel("gated GT", fontsize=9)
    fig.suptitle(f"{args.vol}: self-gated RT recon vs gated breath-hold GT "
                 f"(recon z{zr} / gt z{zg}) — NOT registered", fontsize=11)
    out = OUT.format(vol=args.vol)
    plt.tight_layout(); plt.savefig(out, dpi=72, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
