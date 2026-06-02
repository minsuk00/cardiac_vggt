"""Baseline: PSNR / SSIM between ED (phase 0) and each other cardiac phase, computed
on the canonical (12, 256, 256) grid using the same resample as MRIDataset.

This is the "no-motion-model" floor — if the input slice was just at the right (t, z),
how close to phase 0 would it naturally be? Anything our model beats this by is
genuine motion reconstruction.

Compares against the model's V_canon-vs-V_gt PSNR from the sequential test (train ≈
33.5 dB, val ≈ 31.1 dB after Option A fix).
"""
import os
import sys
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GRID_SHAPE = (12, 256, 256)
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"


def _read_split_section(split):
    subjects = []
    current = None
    with open(SPLIT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].lower()
            elif current == split.lower():
                p = os.path.join(DATA_ROOT, line, "sax")
                if os.path.isdir(p):
                    subjects.append(p)
    return subjects


def _resample_phase_to_canonical(subject_path, t_idx, v_min, v_max):
    """Reproduce MRIDataset.gt_target_volume logic but for arbitrary t."""
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    img_obj = nib.load(nii_files[t_idx])
    vol = img_obj.get_fdata()
    spacing = np.array(img_obj.header.get_zooms()[:3], dtype=np.float32)
    W, H, Z = vol.shape

    D_t, H_t, W_t = GRID_SHAPE
    half_t = np.array([W * spacing[0] / 2, H * spacing[1] / 2, Z * spacing[2] / 2], np.float32)
    center_t = np.array([(W-1)/2*spacing[0], (H-1)/2*spacing[1], (Z-1)/2*spacing[2]], np.float32)
    d_canon = np.linspace(-1, 1, D_t, dtype=np.float32)
    h_canon = np.linspace(-1, 1, H_t, dtype=np.float32)
    w_canon = np.linspace(-1, 1, W_t, dtype=np.float32)
    dd, hh, ww = np.meshgrid(d_canon, h_canon, w_canon, indexing="ij")
    x_vox = (center_t[0] + half_t[0] * ww) / spacing[0]
    y_vox = (center_t[1] + half_t[1] * hh) / spacing[1]
    z_vox = (center_t[2] + half_t[2] * dd) / spacing[2]
    vox_coords = np.stack([x_vox.ravel(), y_vox.ravel(), z_vox.ravel()])
    V = map_coordinates(vol, vox_coords, order=1, mode="nearest").reshape(D_t, H_t, W_t)
    return np.clip((V - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32)


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def ssim_3d(a, b):
    try:
        import torch
        from fused_ssim import fused_ssim3d
        ta = torch.from_numpy(a).unsqueeze(0).unsqueeze(0).float().cuda().contiguous()
        tb = torch.from_numpy(b).unsqueeze(0).unsqueeze(0).float().cuda().contiguous()
        return float(fused_ssim3d(ta, tb, train=False).item())
    except Exception as e:
        return float("nan")


def report_subject(subject_path):
    name = os.path.basename(os.path.dirname(subject_path))
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    T = len(nii_files)
    vol0 = nib.load(nii_files[0]).get_fdata()
    v_min = np.percentile(vol0, 1)
    v_max = np.percentile(vol0, 99.5)

    V0 = _resample_phase_to_canonical(subject_path, 0, v_min, v_max)
    print(f"\n=== {name}  T={T}  Z={vol0.shape[2]} ===")
    print(f"  phase  vs phase 0:  PSNR(dB)   MAE     SSIM")
    psnr_vals = []
    for t in range(1, T):
        Vt = _resample_phase_to_canonical(subject_path, t, v_min, v_max)
        p = psnr(Vt, V0)
        m = np.abs(Vt - V0).mean()
        s = ssim_3d(Vt, V0)
        psnr_vals.append(p)
        print(f"  t = {t:2d}                {p:6.2f}    {m:.4f}  {s:.4f}")
    arr = np.array(psnr_vals)
    print(f"  ---")
    print(f"  min  PSNR (most motion, ≈ES): {arr.min():.2f} dB at t={1 + arr.argmin()}")
    print(f"  mean PSNR across t=1..T-1   : {arr.mean():.2f} dB")
    print(f"  max  PSNR (least motion)    : {arr.max():.2f} dB at t={1 + arr.argmax()}")


if __name__ == "__main__":
    train_subjects = _read_split_section("train")
    val_subjects = _read_split_section("val")
    print(f"Found {len(train_subjects)} train, {len(val_subjects)} val subjects")
    # Same subjects the sequential test used (index 0)
    report_subject(train_subjects[0])
    report_subject(val_subjects[0])

    # Sanity: a couple more val subjects to see between-subject variance
    print("\n--- additional val subjects for variance ---")
    for s in val_subjects[1:4]:
        report_subject(s)
