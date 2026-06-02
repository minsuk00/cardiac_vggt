"""Restrict PSNR comparison to the elastix body-mask region for a fair
apples-to-apples test against elastix (which zeros DVF outside that mask).

For each val subject, computes PSNR / SSIM / MAE three ways for
model / elastix-Δ / identity-Δ:
  - full           : over all 12·256·256 canonical voxels
  - inside body    : over voxels where the canonical-resampled body mask is 1
  - outside body   : over voxels where the mask is 0 (the periphery)

The "inside body" column is the fair registration-quality comparison.
"""
import os
import sys
import random
import glob
import numpy as np
import torch
import nibabel as nib
from scipy.ndimage import map_coordinates
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss

CKPT = "/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
GRID_SHAPE = (12, 256, 256)
N_SUBJECTS = 30


class IdentityRandom:
    def __init__(self, *a, **kw): pass
    def shuffle(self, x): pass
    def random(self): return 0.0


def _build_batch(seq_idx):
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518,
                    dvf_dirname="dvf_elastix")
    subject_path = ds.subjects[seq_idx]
    orig_class, orig_shuffle = random.Random, random.shuffle
    random.Random = IdentityRandom
    random.shuffle = lambda x: None
    try:
        data = ds.get_data(seq_index=seq_idx, img_per_seq=12)
    finally:
        random.Random = orig_class
        random.shuffle = orig_shuffle
    def st(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {
        "images": imgs, "scanner_coords": st("scanner_coords"),
        "world_points": st("world_points"),  # scanner_coords + Δ_elastix
        "z_indices": st("z_indices"), "t_indices": st("t_indices"),
        "target_t_indices": st("target_t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
    }
    if "gt_target_volume" in data:
        batch["gt_target_volume"] = torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)
    if "t_target" in data:
        batch["t_target"] = torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)
    return subject_path, batch


def _resample_body_mask(subject_path):
    """Load mask_frame_00.nii.gz and resample to canonical grid (12, 256, 256).
    Returns boolean array, True inside body.
    """
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    img_obj = nib.load(nii_files[0])
    spacing = np.array(img_obj.header.get_zooms()[:3], dtype=np.float32)
    W, H, Z = img_obj.shape

    mask_path = os.path.join(subject_path, "dvf_elastix", "mask_frame_00.nii.gz")
    mask_vol = nib.load(mask_path).get_fdata().astype(np.float32)
    # mask_vol shape matches the cine volume (W, H, Z)

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
    M = map_coordinates(mask_vol, vox, order=0, mode="nearest").reshape(D_t, H_t, W_t)
    return M > 0.5  # binary


def _psnr(V_canon, V_gt, mask=None):
    """PSNR (in dB). If mask is provided (bool array), restrict to mask voxels."""
    diff2 = (V_canon - V_gt) ** 2
    if mask is None:
        mse = diff2.mean()
    else:
        n = mask.sum()
        if n == 0:
            return float("nan"), float("nan")
        mse = (diff2 * mask).sum() / n
    abs_diff = np.abs(V_canon - V_gt)
    if mask is None:
        mae = abs_diff.mean()
    else:
        mae = (abs_diff * mask).sum() / max(int(mask.sum()), 1)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))
    return float(psnr), float(mae)


def main():
    device = "cuda"
    print("Building model...")
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=True, use_target_t_pose_embedding=True,
                 train_on_residual_dvf=True).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print("Loaded checkpoint.\n")

    rows = []
    for i in range(N_SUBJECTS):
        subject_path, batch = _build_batch(i)
        name = os.path.basename(os.path.dirname(subject_path))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)

        # All three position fields → V_canon via the same loss helper
        scanner = batch["scanner_coords"].float()
        wp_model = preds["world_points"].float()
        wp_elastix = batch["world_points"].float()

        out_m = compute_volume_intensity_loss({"world_points": wp_model},   batch, grid_shape=GRID_SHAPE)
        out_e = compute_volume_intensity_loss({"world_points": wp_elastix}, batch, grid_shape=GRID_SHAPE)
        out_i = compute_volume_intensity_loss({"world_points": scanner},    batch, grid_shape=GRID_SHAPE)

        V_m = out_m["V_canon"][0].float().cpu().numpy()
        V_e = out_e["V_canon"][0].float().cpu().numpy()
        V_i = out_i["V_canon"][0].float().cpu().numpy()
        V_gt = out_m["V_gt"][0].float().cpu().numpy()

        body_mask = _resample_body_mask(subject_path)
        n_in = body_mask.sum()
        n_out = (~body_mask).sum()
        body_frac = float(n_in) / body_mask.size

        p_m_full,  _ = _psnr(V_m, V_gt)
        p_e_full,  _ = _psnr(V_e, V_gt)
        p_i_full,  _ = _psnr(V_i, V_gt)
        p_m_in,    _ = _psnr(V_m, V_gt, body_mask)
        p_e_in,    _ = _psnr(V_e, V_gt, body_mask)
        p_i_in,    _ = _psnr(V_i, V_gt, body_mask)
        p_m_out,   _ = _psnr(V_m, V_gt, ~body_mask)
        p_e_out,   _ = _psnr(V_e, V_gt, ~body_mask)
        p_i_out,   _ = _psnr(V_i, V_gt, ~body_mask)

        rows.append(dict(name=name, body_frac=body_frac,
                         m_full=p_m_full, e_full=p_e_full, i_full=p_i_full,
                         m_in=p_m_in,   e_in=p_e_in,   i_in=p_i_in,
                         m_out=p_m_out, e_out=p_e_out, i_out=p_i_out))
        print(f"[{i:2d}] {name:<14}  body_frac={body_frac:.3f}  "
              f"full: m={p_m_full:5.2f} e={p_e_full:5.2f} i={p_i_full:5.2f}  "
              f"in: m={p_m_in:5.2f} e={p_e_in:5.2f} i={p_i_in:5.2f}  "
              f"out: m={p_m_out:5.2f} e={p_e_out:5.2f} i={p_i_out:5.2f}")

    # --- Aggregate ---
    def arr(k): return np.array([r[k] for r in rows])
    print("\n=== summary (mean ± std, PSNR dB, n=30) ===")
    print(f"  {'region':<22}  {'model':>13}  {'elastix':>13}  {'identity':>13}  {'Δ(m-e)':>7}")
    for label, m, e, i in [
        ("full volume",      "m_full", "e_full", "i_full"),
        ("inside body mask", "m_in",   "e_in",   "i_in"),
        ("outside body",     "m_out",  "e_out",  "i_out"),
    ]:
        am, ae, ai = arr(m), arr(e), arr(i)
        print(f"  {label:<22}  {am.mean():6.2f}±{am.std():4.2f}  "
              f"{ae.mean():6.2f}±{ae.std():4.2f}  "
              f"{ai.mean():6.2f}±{ai.std():4.2f}  "
              f"{(am-ae).mean():+6.2f}")

    body_arr = arr("body_frac")
    print(f"\n  body mask covers {body_arr.mean()*100:.1f}% ± {body_arr.std()*100:.1f}% of canonical voxels")


if __name__ == "__main__":
    main()
