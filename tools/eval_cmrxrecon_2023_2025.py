#!/usr/bin/env python
"""Evaluate VGGT-MRI moderate augmentation model on CMRxRecon 2023 & 2025 subjects (Breathing condition only).

Outputs 3D/4D NIfTIs, animated GIFs, and metrics strictly under result/cmrxrecon_2023_2025/.
No existing repository files are edited.
"""

import glob
import json
import os
import sys
import time
import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

VGGT_ROOT = "/home/minsukc/vggt"
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, os.path.join(VGGT_ROOT, "training"))

from inference.inference import load_rtfb_model_reference
from inference.adapters.base import GRID_SHAPE, INPUT_IMG_SIZE, D_CANON
from data.preprocess import (
    TARGET_SPACING, TARGET_SHAPE, NUM_PHASES,
    ScaleIntensityByT0PercentilesD, AddOnesMaskD,
)
from data.respiratory import RespiratoryConfig, sample_resp_disp
from monai.transforms import (
    Compose, EnsureChannelFirstd, Orientationd, Spacingd, ResizeWithPadOrCropd
)
from monai.data.meta_tensor import MetaTensor
from vggt.utils.splat import splat_predictions

CKPT_PATH = os.path.join(VGGT_ROOT, "scratch/logs/216003592_mri_volume_diffusion_oneframe_aug_moderate_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt")
RESULTS_ROOT = os.path.join(VGGT_ROOT, "result/cmrxrecon_2023_2025")

TEST_SUBJECTS = {
    "cmrxrecon2023": {
        "root": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/data/CMRxRecon2023_recon_v1_espirit_imagedomain",
        "subjects": ["CMRx23_Train_P001", "CMRx23_Train_P002", "CMRx23_Test_P001"]
    },
    "cmrxrecon2025": {
        "root": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/data/CMRxRecon2025_recon_v1_espirit_imagedomain",
        "subjects": [
            "CMRx25_R1test_Center001_Siemens_30T_Vida_P003",
            "CMRx25_R1test_Center006_Siemens_30T_Prisma_P021",
            "CMRx25_R1test_Center004_Siemens_15T_Aera_P002",
            "CMRx25_R1test_Center012_Philips_30T_IngeniaCX_P001",
            "CMRx25_R1test_Center001_UIH_30T_umr780_P003",
            "CMRx25_R1test_Center003_UIH_15T_umr670_P001"
        ]
    }
}


def load_subject_canonical(subject_dir):
    sax_nii = os.path.join(subject_dir, "sax/4d_recon.nii.gz")
    if not os.path.exists(sax_nii):
        niis = glob.glob(os.path.join(subject_dir, "**/*.nii.gz"), recursive=True)
        if not niis:
            raise FileNotFoundError(f"Missing NIfTI: {sax_nii}")
        sax_nii = niis[0]
    
    img_4d = nib.load(sax_nii)
    data_4d = np.asarray(img_4d.dataobj, dtype=np.float32)
    affine = torch.as_tensor(img_4d.affine, dtype=torch.float64)
    
    keys = [f"phase_{t:02d}" for t in range(NUM_PHASES)]
    data_dict = {}
    for t in range(NUM_PHASES):
        p_data = data_4d[..., t] if data_4d.ndim == 4 else data_4d
        data_dict[f"phase_{t:02d}"] = MetaTensor(p_data, meta={"affine": affine, "spatial_shape": p_data.shape})
        
    spatial_keys = keys + ["content_mask"]
    spacing_modes = ["bilinear"] * NUM_PHASES + ["nearest"]
    
    transform = Compose([
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        AddOnesMaskD(ref_key="phase_00", output_key="content_mask"),
        Orientationd(keys=spatial_keys, axcodes="LPS"),
        Spacingd(keys=spatial_keys, pixdim=TARGET_SPACING, mode=spacing_modes),
        ResizeWithPadOrCropd(keys=spatial_keys, spatial_size=TARGET_SHAPE, mode="constant", value=0),
        ScaleIntensityByT0PercentilesD(keys=keys, ref_key="phase_00"),
    ])
    
    res = transform(data_dict)
    
    phases = []
    for t in range(NUM_PHASES):
        p = res[f"phase_{t:02d}"].numpy()[0]  # (256, 256, 12)
        phases.append(np.transpose(p, (2, 1, 0)))  # (12, 256, 256)
    phases = np.stack(phases, axis=0)  # (12, 12, 256, 256)
    
    mask = res["content_mask"].numpy()[0]  # (256, 256, 12)
    content_mask = np.transpose(mask, (2, 1, 0)) > 0.5  # (12, 256, 256)
    
    return phases, content_mask


def reslice_volume_per_slice(V, disp_mm, spacing=(12.0, 1.4, 1.4)):
    D, H, W = V.shape
    device = V.device
    inp = V.float().view(1, 1, D, H, W)
    
    zs = torch.arange(D, device=device, dtype=torch.float32)
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    
    z_base = (zs / (D - 1) * 2.0 - 1.0).view(D, 1, 1).expand(D, H, W)
    y_base = (ys / (H - 1) * 2.0 - 1.0).view(1, H, 1).expand(D, H, W)
    x_base = (xs / (W - 1) * 2.0 - 1.0).view(1, 1, W).expand(D, H, W)
    
    disp = torch.as_tensor(disp_mm, dtype=torch.float32, device=device)  # (D, 3)
    dz = (disp[:, 0] / max(1e-6, spacing[0] * (D - 1) / 2.0)).view(D, 1, 1).expand(D, H, W)
    dy = (disp[:, 1] / max(1e-6, spacing[1] * (H - 1) / 2.0)).view(D, 1, 1).expand(D, H, W)
    dx = (disp[:, 2] / max(1e-6, spacing[2] * (W - 1) / 2.0)).view(D, 1, 1).expand(D, H, W)
    
    grid = torch.stack([x_base + dx, y_base + dy, z_base + dz], dim=-1).unsqueeze(0)
    out = F.grid_sample(inp, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return out.view(D, H, W)


def run_model_reconstruct(model, phases, content_mask, disp_mm, device="cuda"):
    T, D, H, W = phases.shape
    planes = [int(z) for z in np.where(content_mask.any(axis=(1, 2)))[0]]
    if len(planes) == 0:
        planes = list(range(D))
    
    phases_input = []
    for t in range(T):
        vol_t = torch.from_numpy(phases[t]).float()
        vol_corrupted = reslice_volume_per_slice(vol_t, disp_mm)
        phases_input.append(vol_corrupted.numpy())
    phases_input = np.stack(phases_input, axis=0)
    
    z_mid = (min(planes) + max(planes) + 1) // 2
    ref_k = int(np.argmin([abs(p - z_mid) for p in planes]))
    
    slots = [(ref_k, 0)] + [(k, (k * 3) % T) for k in range(len(planes)) if k != ref_k]
    
    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    
    coords, zidx = [], []
    for k, phase_idx in slots:
        z_plane = planes[k]
        zv = z_plane / max(1, D - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, zv)], -1))
        zidx.append([zv])
        
    coords_tensor = torch.from_numpy(np.stack(coords)).float()[None].to(device)
    zidx_tensor = torch.tensor(zidx, dtype=torch.float32)[None].to(device)
    
    pred_vols = []
    for t in range(T):
        imgs = []
        for s_idx, (k, phase_idx) in enumerate(slots):
            actual_t = t if s_idx == 0 else phase_idx
            z_plane = planes[k]
            img_slice = phases_input[actual_t, z_plane]
            up = F.interpolate(torch.from_numpy(img_slice)[None, None].float(), size=(hw, hw),
                               mode="bilinear", align_corners=True)[0, 0].numpy()
            imgs.append(np.repeat(up[None], 3, axis=0))
            
        imgs_tensor = torch.from_numpy(np.stack(imgs)).float()[None].to(device)
        
        batch = {
            "images": imgs_tensor,
            "scanner_coords": coords_tensor,
            "z_indices": zidx_tensor,
        }
        
        dev_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.no_grad():
            with torch.amp.autocast(dev_type, enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)
            wp = preds["world_points"].float()
            V_canon, _ = splat_predictions({"world_points": wp}, batch, (D, H, W))
            V_out = preds.get("V_refined", V_canon)[0].float().cpu().numpy()
            pred_vols.append(V_out)
            
    return np.stack(pred_vols), phases_input


def compute_metrics(v_gt, v_pred, mask):
    T = v_gt.shape[0]
    psnrs, ssims, l1s = [], [], []
    for t in range(T):
        gt = v_gt[t]
        pred = v_pred[t]
        l1 = np.abs(gt[mask] - pred[mask]).mean()
        mz, my, mx = np.where(mask)
        z0, z1 = mz.min(), mz.max() + 1
        y0, y1 = my.min(), my.max() + 1
        x0, x1 = mx.min(), mx.max() + 1
        gt_crop = gt[z0:z1, y0:y1, x0:x1]
        pred_crop = pred[z0:z1, y0:y1, x0:x1]
        psnrs.append(psnr_fn(gt_crop, pred_crop, data_range=1.0))
        ssims.append(ssim_fn(gt_crop, pred_crop, data_range=1.0))
        l1s.append(l1)
        
    return {
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "l1": float(np.mean(l1s))
    }


def render_gif(out_path, rows, planes, T, vmax=1.0, fps=3):
    nrow, ncol = len(rows), len(planes)
    H = nrow * 1.15 + 0.8
    top = 1.0 - 0.68 / H
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.15, H))
        axes = np.atleast_2d(axes)
        for ri, (label, cine) in enumerate(rows):
            for ci, z in enumerate(planes):
                ax = axes[ri, ci]
                ax.imshow(cine[t, z, :, :].T, cmap="gray", vmin=0, vmax=vmax,
                          origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"z{z}", fontsize=6.5)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=8)
        fig.suptitle(f"Cardiac Cycle Phase t={t:02d}", fontsize=9, y=0.985, va="top")
        fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=0.01,
                            wspace=0.03, hspace=0.06)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(out_path, frames, duration=1.0 / fps, loop=0)


def save_artifacts(out_dir, v_gt, v_pred):
    recon_dir = os.path.join(out_dir, "recon_breath")
    os.makedirs(recon_dir, exist_ok=True)
    affine = np.diag([1.4, 1.4, 12.0, 1.0])
    
    # Save per-phase 3D NIfTIs
    for t in range(NUM_PHASES):
        vol_xyz = np.ascontiguousarray(v_pred[t].transpose(2, 1, 0))
        nib.save(nib.Nifti1Image(vol_xyz, affine), os.path.join(recon_dir, f"vol_t{t:02d}.nii.gz"))
        
    # Save 4D cine NIfTIs (X, Y, Z, T)
    gt_4d = np.ascontiguousarray(np.moveaxis(v_gt, 0, -1).transpose(2, 1, 0, 3))
    pred_4d = np.ascontiguousarray(np.moveaxis(v_pred, 0, -1).transpose(2, 1, 0, 3))
    
    nib.save(nib.Nifti1Image(gt_4d, affine), os.path.join(out_dir, "cine_gt.nii.gz"))
    nib.save(nib.Nifti1Image(pred_4d, affine), os.path.join(out_dir, "cine_breath.nii.gz"))
    
    # Render animated GIF
    planes = list(range(D_CANON))
    gif_path = os.path.join(out_dir, "gif_breath.gif")
    render_gif(gif_path, [("GT", v_gt), ("VGGT (Breath)", v_pred)], planes, NUM_PHASES, vmax=1.0, fps=3)


def main():
    print("=" * 70)
    print("Evaluating VGGT 1-Frame Moderate Aug Model (Breathing Condition Only)")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"Results Output Dir: {RESULTS_ROOT}")
    print("=" * 70)
    
    os.makedirs(RESULTS_ROOT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Selected inference device: {device}")
    
    print("\nLoading model checkpoint into GPU...")
    model = load_rtfb_model_reference(CKPT_PATH, refiner=False, device=device)
    
    rcfg = RespiratoryConfig(enable=True, amplitude_mm=5.0, amplitude_jitter=0.2, group_by_burst=True)
    
    all_summary = {}
    
    for ds_name, ds_info in TEST_SUBJECTS.items():
        print(f"\n--- Processing dataset: {ds_name} ---")
        ds_root = ds_info["root"]
        subjects = ds_info["subjects"]
        
        ds_results = {}
        for idx, subj in enumerate(subjects):
            subj_dir = os.path.join(ds_root, subj)
            print(f"\nSubject: {subj}")
            
            try:
                t0 = time.time()
                phases_gt, content_mask = load_subject_canonical(subj_dir)
                print(f"  Loaded canonical phases shape: {phases_gt.shape}, content voxels: {content_mask.sum()}")
                
                # Breathing-corrupted run
                seq_idx = torch.tensor([[idx + 42]], dtype=torch.int64)
                disp_mm, _ = sample_resp_disp(1, D_CANON, rcfg, device="cpu", train=False, seq_index=seq_idx)
                disp_mm = disp_mm[0]  # (12, 3)
                
                v_breath_pred, _ = run_model_reconstruct(model, phases_gt, content_mask, disp_mm=disp_mm, device=device)
                m_breath = compute_metrics(phases_gt, v_breath_pred, content_mask)
                print(f"  [Breath] PSNR: {m_breath['psnr']:.2f} dB | SSIM: {m_breath['ssim']:.4f} | L1: {m_breath['l1']:.4f}")
                
                # Save 3D/4D NIfTIs and Animated GIF under result/cmrxrecon_2023_2025/<ds_name>/<subj>/
                out_subj_dir = os.path.join(RESULTS_ROOT, ds_name, subj)
                save_artifacts(out_subj_dir, phases_gt, v_breath_pred)
                
                dt = time.time() - t0
                subj_res = {
                    "breath": m_breath,
                    "elapsed_sec": round(dt, 2)
                }
                ds_results[subj] = subj_res
                
                with open(os.path.join(out_subj_dir, "metrics.json"), "w") as f:
                    json.dump(subj_res, f, indent=2)
                    
                torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"  ERROR processing {subj}: {e}")
                import traceback
                traceback.print_exc()
                
        all_summary[ds_name] = ds_results
        
    summary_path = os.path.join(RESULTS_ROOT, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)
        
    print("\n" + "=" * 70)
    print(f"Evaluation Complete! Summary saved to {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
