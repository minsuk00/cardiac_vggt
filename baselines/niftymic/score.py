"""Score a NiftyMIC reconstruction against our canonical V_gt.

NiftyMIC outputs an isotropic volume at its own resolution/grid. We resample it
onto the exact canonical (12, 256, 256) @ (1.4, 1.4, 12.0) mm grid the export
stack came from (== V_gt's grid), then compute the same MAE/PSNR/SSIM the
in-pipeline baselines use (training/loss.py, baselines/eval_all_baselines.py).

Usage: micromamba run -n svr python baselines/niftymic/score.py <tag> [<tag> ...]
  e.g. python baselines/niftymic/score.py CMRx24_Train_P053_t0 CMRx24_Val_P055_t0
"""
import os
import sys

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np

DATA_DIR = "/home/minsukc/vggt/scratch/niftymic/data"
RECON_DIR = "/home/minsukc/vggt/scratch/niftymic/recon"
OUT_DIR = "/home/minsukc/vggt/result/niftymic"


def _metrics(pred, gt):
    valid = (gt > 1e-3)
    denom = max(valid.sum(), 1)
    mae_full = np.abs(pred - gt).mean()
    mse_full = ((pred - gt) ** 2).mean()
    mae_anat = np.abs((pred - gt) * valid).sum() / denom
    mse_anat = (((pred - gt) ** 2) * valid).sum() / denom
    psnr_full = 10 * np.log10(1.0 / max(mse_full, 1e-10))
    psnr_anat = 10 * np.log10(1.0 / max(mse_anat, 1e-10))
    return dict(mae_full=float(mae_full), psnr_full=float(psnr_full),
                mae_anat=float(mae_anat), psnr_anat=float(psnr_anat))


def _calibrate_intensity(pred, gt):
    """Fit pred ~= k*gt + c on the anatomy region and invert it.

    NiftyMIC's Tikhonov SRR solve doesn't preserve absolute input intensity
    scale (bias-field/regularization drift) — standard in the SVR literature,
    which is why cross-method PSNR/SSIM comparisons calibrate intensity first.
    Structure, not absolute scale, is what the reconstruction is judged on.
    """
    anat = gt > 1e-3
    g, p = gt[anat], pred[anat]
    k, c = np.linalg.lstsq(np.stack([g, np.ones_like(g)], axis=1), p, rcond=None)[0]
    corr = float(np.corrcoef(g, p)[0, 1])
    pred_cal = (pred - c) / k if abs(k) > 1e-8 else pred
    return pred_cal, dict(k=float(k), c=float(c), corr=corr)


def score_one(tag):
    stack_path = os.path.join(DATA_DIR, f"{tag}_stack.nii.gz")
    recon_path = os.path.join(RECON_DIR, tag, "recon.nii.gz")
    if not os.path.exists(recon_path):
        print(f"SKIP {tag}: no reconstruction at {recon_path}")
        return None

    # V_gt == the exported stack itself (see export_stack.py docstring): the
    # clean per-phase canonical volume IS our ground truth (no finer GT exists).
    gt_img = nib.load(stack_path)
    gt_xyz = np.asarray(gt_img.dataobj, dtype=np.float32)

    recon_img = nib.load(recon_path)
    # Resample NiftyMIC's (higher-res, its own grid) output onto V_gt's exact
    # canonical grid/affine for a voxel-aligned comparison.
    recon_on_gt = nibproc.resample_from_to(recon_img, gt_img, order=1)
    pred_raw = np.asarray(recon_on_gt.dataobj, dtype=np.float32)

    pred_cal, fit = _calibrate_intensity(pred_raw, gt_xyz)

    print(f"[{tag}] gt range=({gt_xyz.min():.3f},{gt_xyz.max():.3f}) "
          f"pred_raw range=({pred_raw.min():.3f},{pred_raw.max():.3f}) "
          f"recon native shape={recon_img.shape}")
    print(f"     intensity fit: pred = {fit['k']:.3f}*gt + {fit['c']:.3f}  "
          f"(corr={fit['corr']:.3f})")

    m_raw = _metrics(pred_raw, gt_xyz)
    m_cal = _metrics(pred_cal, gt_xyz)
    print(f"     NiftyMIC RAW         MAE_full={m_raw['mae_full']:.4f}  PSNR_full={m_raw['psnr_full']:6.2f} dB  "
          f"MAE_anat={m_raw['mae_anat']:.4f}  PSNR_anat={m_raw['psnr_anat']:6.2f} dB")
    print(f"     NiftyMIC CALIBRATED  MAE_full={m_cal['mae_full']:.4f}  PSNR_full={m_cal['psnr_full']:6.2f} dB  "
          f"MAE_anat={m_cal['mae_anat']:.4f}  PSNR_anat={m_cal['psnr_anat']:6.2f} dB")
    m_cal["corr"] = fit["corr"]
    return m_cal


def main():
    tags = sys.argv[1:]
    if not tags:
        print("usage: score.py <tag> [<tag> ...]")
        sys.exit(1)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    for tag in tags:
        m = score_one(tag)
        if m is not None:
            results[tag] = m
    if results:
        mean_psnr = np.mean([m["psnr_full"] for m in results.values()])
        mean_psnr_anat = np.mean([m["psnr_anat"] for m in results.values()])
        mean_corr = np.mean([m["corr"] for m in results.values()])
        print(f"\n(all numbers below are intensity-CALIBRATED)")
        print(f"mean PSNR_full={mean_psnr:.2f} dB  mean PSNR_anat={mean_psnr_anat:.2f} dB  "
              f"mean corr={mean_corr:.3f}  over {len(results)} subject(s)")


if __name__ == "__main__":
    main()
