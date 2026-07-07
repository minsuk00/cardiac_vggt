#!/usr/bin/env python3
"""Score a fetal_cmr_4d recon (from MIITT real-time) against the gated GT cine.

Unlike the NeSVoR/NiftyMIC baselines (where GT *was* the input, same grid), here
the recon comes from the REAL-TIME acquisition and the GT is a SEPARATE gated
breath-hold scan. So they are not voxel-aligned and need bringing into register
before any intensity metric means anything:

  1. spatial: resample the recon onto the gated-GT grid (world-coordinate based).
  2. rigid-ish: estimate a 3D translation (dominant residual = respiratory/table
     shift between the two scans) on the ED phase over the heart region, apply to
     all phases. (Full rigid/affine is a later refinement.)
  3. temporal: recon has 25 cardiac phases, GT has 30 -- both ED-anchored (phase 0
     = R-trigger), so resample the recon's phase axis 25->30 by circular linear
     interpolation.
  4. intensity: fit recon ~= k*gt + c on the heart region and invert (SVR recons
     don't preserve absolute scale; same convention as niftymic/nesvor score.py).

Then per-phase PSNR + SSIM over the heart region, plus a mid-slice beating montage.

Usage:
  micromamba run -n svr python baselines/fetal_cmr_4d/score.py Volunteer1 \
      [--recon .../fast_cine/cine.nii.gz]
"""
import argparse
import os
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
from scipy import ndimage

GT_NIFTI = "/home/minsukc/MIITT/nifti/{vol}/gated/sax/4d_recon.nii.gz"
RECON_DEFAULT = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}/fast_cine/cine.nii.gz"
OUT_DIR = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon/{vol}"


def _ssim(a, b, mask):
    """Global SSIM over masked voxels (single window; simple + alignment-honest)."""
    a, b = a[mask], b[mask]
    if a.size < 2:
        return float("nan")
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    L = max(a.max(), b.max()) - min(a.min(), b.min())
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def _psnr(a, b, mask):
    mse = (((a - b) ** 2)[mask]).mean()
    peak = b[mask].max()
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-10)))


def temporal_resample(vol, n_out):
    """(X,Y,Z,P) -> (X,Y,Z,n_out) circular linear interp along the phase axis."""
    P = vol.shape[3]
    if P == n_out:
        return vol
    src = np.arange(P)
    dst = np.linspace(0, P, n_out, endpoint=False)
    out = np.empty(vol.shape[:3] + (n_out,), np.float32)
    ext = np.concatenate([vol, vol[..., :1]], axis=3)  # wrap for circular
    for i, d in enumerate(dst):
        lo = int(np.floor(d)) % P
        w = d - np.floor(d)
        out[..., i] = (1 - w) * ext[..., lo] + w * ext[..., lo + 1]
    return out


def estimate_shift(recon_ed, gt_ed, heart):
    """Integer-voxel 3D translation aligning recon_ed to gt_ed over the heart box."""
    from scipy.signal import fftconvolve
    m = heart
    a = (recon_ed - recon_ed[m].mean()) * m
    b = (gt_ed - gt_ed[m].mean()) * m
    corr = fftconvolve(b, a[::-1, ::-1, ::-1], mode="same")
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    center = tuple(s // 2 for s in corr.shape)
    return tuple(int(p - c) for p, c in zip(peak, center))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vol")
    ap.add_argument("--recon", default=None)
    args = ap.parse_args()

    recon_path = args.recon or RECON_DEFAULT.format(vol=args.vol)
    gt_img = nib.load(GT_NIFTI.format(vol=args.vol))
    gt = gt_img.get_fdata().astype(np.float32)              # (Xg,Yg,Zg,30)
    recon_img = nib.load(recon_path)
    rc = recon_img.get_fdata().astype(np.float32)
    if rc.ndim == 3:
        rc = rc[..., None]
    Pr = rc.shape[3]
    print(f"recon {rc.shape} @ {tuple(round(z,2) for z in recon_img.header.get_zooms()[:3])}  "
          f"gt {gt.shape} @ {tuple(round(z,2) for z in gt_img.header.get_zooms()[:3])}")

    # 1. spatial resample recon -> GT grid, per phase
    gt3 = nib.Nifti1Image(gt[..., 0], gt_img.affine)
    rc_on_gt = np.empty(gt.shape[:3] + (Pr,), np.float32)
    for p in range(Pr):
        ph = nib.Nifti1Image(rc[..., p], recon_img.affine)
        rc_on_gt[..., p] = np.asarray(
            nibproc.resample_from_to(ph, gt3, order=1).dataobj, dtype=np.float32)

    # 3. temporal 25 -> 30
    rc_t = temporal_resample(rc_on_gt, gt.shape[3])

    # heart region from GT temporal variance (where it beats)
    tvar = gt.std(axis=3)
    heart = ndimage.binary_dilation(tvar > np.percentile(tvar[tvar > 0], 88),
                                    iterations=2)
    if heart.sum() < 50:
        heart = tvar > np.percentile(tvar, 95)

    # 2. translation align on ED (phase 0)
    sh = estimate_shift(rc_t[..., 0], gt[..., 0], heart)
    rc_al = np.roll(rc_t, shift=sh, axis=(0, 1, 2))
    print(f"estimated ED alignment shift (vox): {sh}")

    # 4. intensity calibrate on heart region (all phases pooled)
    hb = np.broadcast_to(heart[..., None], rc_al.shape)
    g, p = gt[hb], rc_al[hb]
    k, c = np.linalg.lstsq(np.stack([g, np.ones_like(g)], 1), p, rcond=None)[0]
    corr = float(np.corrcoef(g, p)[0, 1])
    rc_cal = (rc_al - c) / k if abs(k) > 1e-8 else rc_al
    print(f"intensity fit: recon = {k:.3g}*gt + {c:.3g}  (corr={corr:.3f})")

    # per-phase metrics over the heart region
    psnrs, ssims = [], []
    for ph in range(gt.shape[3]):
        psnrs.append(_psnr(rc_cal[..., ph], gt[..., ph], heart))
        ssims.append(_ssim(rc_cal[..., ph], gt[..., ph], heart))
    print(f"[{args.vol}] per-phase PSNR  mean {np.mean(psnrs):.2f} dB  "
          f"(min {np.min(psnrs):.2f}, max {np.max(psnrs):.2f})")
    print(f"[{args.vol}] per-phase SSIM  mean {np.mean(ssims):.3f}  "
          f"(min {np.min(ssims):.3f}, max {np.max(ssims):.3f})")
    print(f"[{args.vol}] heart-region intensity corr {corr:.3f}")

    _montage(args.vol, rc_cal, gt, heart)
    return dict(psnr=float(np.mean(psnrs)), ssim=float(np.mean(ssims)), corr=corr)


def _montage(vol, rc, gt, heart):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = int(np.argmax(heart.sum(axis=(0, 1))))
    P = gt.shape[3]
    cols = np.linspace(0, P - 1, 8).round().astype(int)

    def norm(im):
        lo, hi = np.percentile(im, [1, 99]); return np.clip((im - lo) / max(hi - lo, 1e-6), 0, 1)

    fig, ax = plt.subplots(2, len(cols), figsize=(2 * len(cols), 4.4))
    for j, p in enumerate(cols):
        ax[0, j].imshow(norm(rc[:, :, z, p]).T, cmap="gray"); ax[0, j].set_title(f"ph{p}", fontsize=8); ax[0, j].axis("off")
        ax[1, j].imshow(norm(gt[:, :, z, p]).T, cmap="gray"); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("fetal_cmr_4d\n(RT self-gated)", fontsize=9)
    ax[1, 0].set_ylabel("gated GT", fontsize=9)
    fig.suptitle(f"{vol}: aligned+calibrated recon vs gated GT (z{z})", fontsize=11)
    out = os.path.join(OUT_DIR.format(vol=vol), "score_montage.png")
    plt.tight_layout(); plt.savefig(out, dpi=72, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
