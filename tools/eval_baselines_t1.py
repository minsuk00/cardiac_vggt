"""Neighboring-phase baseline: PSNR / SSIM / MAE between phase-0 (ED) and phase-1
on the canonical (12, 256, 256) grid, for all 30 val subjects.

Combined with model and identity-Δ numbers from eval_val_multi.py for a unified
table. Run after eval_val_multi.py.
"""
import os
import sys
import glob
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GRID_SHAPE = (12, 256, 256)
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"


def val_subjects():
    out = []
    current = None
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].lower()
            elif current == "val":
                p = os.path.join(DATA_ROOT, line, "sax")
                if os.path.isdir(p):
                    out.append(p)
    return out


def _resample_phase(subject_path, t_idx, v_min, v_max):
    """Same resample logic as MRIDataset (post-fix-3, mode='nearest')."""
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    img_obj = nib.load(nii_files[t_idx])
    vol = img_obj.get_fdata()
    spacing = np.array(img_obj.header.get_zooms()[:3], dtype=np.float32)
    W, H, Z = vol.shape

    D_t, H_t, W_t = GRID_SHAPE
    half_t = np.array([W*spacing[0]/2, H*spacing[1]/2, Z*spacing[2]/2], np.float32)
    center_t = np.array([(W-1)/2*spacing[0], (H-1)/2*spacing[1], (Z-1)/2*spacing[2]], np.float32)
    dd, hh, ww = np.meshgrid(np.linspace(-1, 1, D_t, dtype=np.float32),
                              np.linspace(-1, 1, H_t, dtype=np.float32),
                              np.linspace(-1, 1, W_t, dtype=np.float32), indexing="ij")
    x_vox = (center_t[0] + half_t[0]*ww) / spacing[0]
    y_vox = (center_t[1] + half_t[1]*hh) / spacing[1]
    z_vox = (center_t[2] + half_t[2]*dd) / spacing[2]
    vox = np.stack([x_vox.ravel(), y_vox.ravel(), z_vox.ravel()])
    V = map_coordinates(vol, vox, order=1, mode="nearest").reshape(D_t, H_t, W_t)
    return np.clip((V - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32)


def ssim3d(a, b):
    try:
        from fused_ssim import fused_ssim3d
        ta = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float().cuda().contiguous()
        tb = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).float().cuda().contiguous()
        return float(fused_ssim3d(ta, tb, train=False).item())
    except Exception:
        return float("nan")


def main():
    subjects = val_subjects()
    print(f"{len(subjects)} val subjects")
    rows = []
    for i, sp in enumerate(subjects):
        name = os.path.basename(os.path.dirname(sp))
        nii_files = sorted(glob.glob(os.path.join(sp, "3d_recon", "sax_frame_*.nii.gz")))
        T = len(nii_files)
        vol0 = nib.load(nii_files[0]).get_fdata()
        v_min, v_max = np.percentile(vol0, 1), np.percentile(vol0, 99.5)
        V0 = _resample_phase(sp, 0, v_min, v_max)
        V1 = _resample_phase(sp, 1, v_min, v_max)
        mae = np.abs(V1 - V0).mean()
        mse = ((V1 - V0) ** 2).mean()
        psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
        s = ssim3d(V1, V0)
        rows.append((name, T, psnr, mae, s))
        print(f"  [{i:2d}] {name:<14}  T={T}  PSNR(t=1 vs t=0) = {psnr:.2f} dB  MAE = {mae:.4f}  SSIM = {s:.4f}")
    psnr_arr = np.array([r[2] for r in rows])
    ssim_arr = np.array([r[4] for r in rows])
    print(f"\n  mean PSNR (t=1 vs t=0) = {psnr_arr.mean():.2f} ± {psnr_arr.std():.2f} dB")
    print(f"  mean SSIM                = {ssim_arr.mean():.4f} ± {ssim_arr.std():.4f}")


if __name__ == "__main__":
    main()
