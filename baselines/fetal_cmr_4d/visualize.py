#!/usr/bin/env python3
"""Visualize a fetal_cmr_4d recon (from MIITT real-time) as a beating-heart cine.

Qualitative only. The recon (from the real-time acquisition) and the gated GT are
SEPARATE scans with different FOV and heart position, so there is NO voxel
correspondence -- PSNR/SSIM are meaningless here. Quantitative comparison is done
functionally (ejection fraction via nnU-Net segmentation), separately.

Produces under the recon dir:
  vis_sax_montage.png   recon mid-ventricular SAX slice across all cardiac phases
  vis_sax.gif           the same slice, animated (beating heart)
  vis_multislice.png    all SAX slices at ED vs ES (the 3D volume, two phases)
  vis_vs_gt.png         recon vs gated GT, mid-slice, several phases (QUALITATIVE,
                        different FOV/position -- shown only to compare beating)

Usage:
  micromamba run -n svr python baselines/fetal_cmr_4d/visualize.py Volunteer1 \
      [--recon .../fast_cine/cine.nii.gz]
"""
import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GT_NIFTI = "/home/minsukc/MIITT/nifti/{vol}/gated/sax/4d_recon.nii.gz"
RECON_DEFAULT = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}/fast_cine/cine.nii.gz"
OUT_DIR = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}"


def norm(im, p=(1, 99)):
    lo, hi = np.percentile(im, p)
    return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)


def mid_ventricular_z(v):
    """slice with the strongest cardiac motion = max temporal variance."""
    return int(np.argmax(v.std(axis=3).sum(axis=(0, 1))))


def es_phase(v, z, heart):
    """crude ES = phase with smallest bright-blood area in the heart ROI."""
    areas = [(v[:, :, z, p][heart] > np.percentile(v[:, :, z, p][heart], 70)).sum()
             for p in range(v.shape[3])]
    return int(np.argmin(areas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("--recon", default=None)
    args = ap.parse_args()
    recon_path = args.recon or RECON_DEFAULT.format(vol=args.vol)
    outdir = OUT_DIR.format(vol=args.vol)

    rc = nib.load(recon_path).get_fdata().astype(np.float32)
    rc = np.clip(rc, 0, None)
    if rc.ndim == 3:
        rc = rc[..., None]
    X, Y, Z, P = rc.shape
    z = mid_ventricular_z(rc)
    print(f"recon {rc.shape}, mid-ventricular slice z={z}")

    # --- 1. SAX montage: mid slice, all phases ---
    fig, ax = plt.subplots(1, P, figsize=(1.1 * P, 1.5))
    for p in range(P):
        ax[p].imshow(norm(rc[:, :, z, p]).T, cmap="gray"); ax[p].axis("off")
        ax[p].set_title(str(p), fontsize=6)
    fig.suptitle(f"{args.vol}: recon mid-ventricular SAX slice z{z}, cardiac phases 0..{P-1}", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{outdir}/vis_sax_montage.png", dpi=80, bbox_inches="tight"); plt.close()

    # --- 2. beating GIF ---
    try:
        import imageio
        frames = []
        for p in range(P):
            im = (norm(rc[:, :, z, p]).T * 255).astype(np.uint8)
            frames.append(im)
        imageio.mimsave(f"{outdir}/vis_sax.gif", frames, duration=0.08, loop=0)
        print("saved vis_sax.gif")
    except Exception as e:
        print("gif skipped:", e)

    # --- 3. all slices at ED vs ES ---
    heart2d = rc[:, :, z].std(axis=2) > np.percentile(rc[:, :, z].std(axis=2), 90)
    ed, es = 0, es_phase(rc, z, heart2d)
    print(f"ED phase 0, ES phase {es}")
    zs = [zz for zz in range(Z) if rc[:, :, zz].std() > 1e-4]
    zs = zs[::max(1, len(zs) // 12)][:12]
    fig, ax = plt.subplots(2, len(zs), figsize=(1.3 * len(zs), 3))
    for j, zz in enumerate(zs):
        ax[0, j].imshow(norm(rc[:, :, zz, ed]).T, cmap="gray"); ax[0, j].axis("off"); ax[0, j].set_title(f"z{zz}", fontsize=7)
        ax[1, j].imshow(norm(rc[:, :, zz, es]).T, cmap="gray"); ax[1, j].axis("off")
    ax[0, 0].set_ylabel(f"ED (ph{ed})", fontsize=9); ax[1, 0].set_ylabel(f"ES (ph{es})", fontsize=9)
    fig.suptitle(f"{args.vol}: recon SAX stack at ED vs ES", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{outdir}/vis_multislice.png", dpi=80, bbox_inches="tight"); plt.close()

    # --- 4. qualitative recon vs GT (different FOV/position) ---
    gt = nib.load(GT_NIFTI.format(vol=args.vol)).get_fdata().astype(np.float32)
    zg = mid_ventricular_z(gt)
    cols_r = np.linspace(0, P - 1, 8).round().astype(int)
    cols_g = np.linspace(0, gt.shape[3] - 1, 8).round().astype(int)
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.4))
    for j in range(8):
        ax[0, j].imshow(norm(rc[:, :, z, cols_r[j]]).T, cmap="gray"); ax[0, j].axis("off"); ax[0, j].set_title(f"ph{cols_r[j]}", fontsize=8)
        ax[1, j].imshow(norm(gt[:, :, zg, cols_g[j]]).T, cmap="gray"); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("fetal_cmr_4d\n(RT self-gated)", fontsize=9)
    ax[1, 0].set_ylabel("gated GT", fontsize=9)
    fig.suptitle(f"{args.vol}: recon (z{z}) vs gated GT (z{zg}) — QUALITATIVE, different FOV/position", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{outdir}/vis_vs_gt.png", dpi=72, bbox_inches="tight"); plt.close()

    print(f"saved visualizations to {outdir}/vis_*.png,gif")


if __name__ == "__main__":
    main()
