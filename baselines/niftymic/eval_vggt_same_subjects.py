"""Score VGGT-MRI (current reference-conditioning checkpoint) on the exact same
2 subjects + target phase used for the NiftyMIC baseline run (docs/29), with the
exact same PSNR_anat metric, for a genuinely fair comparison.

Checkpoint: the live in-progress retrain confirming target-phase reference-slice
conditioning (CLAUDE.md "PRIMARY PIPELINE"), NOT a finished/converged model —
epoch ~182/500 as of this run. Report accordingly.
"""
import os
import sys

import nibabel as nib
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data.datasets.mri_dataset import MRIDataset  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402
from loss import compute_volume_intensity_loss  # noqa: E402

CKPT = "/home/minsukc/vggt/scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
SUBJECT_INDICES = [0, 1]  # same as export_stack.py: Train_P053, Val_P055
TARGET_PHASE = 0
SAVE_DIR = "/home/minsukc/vggt/scratch/niftymic/vggt_volumes"
SPACING_XYZ = (1.4, 1.4, 12.0)  # same canonical affine as export_stack.py, for viewer compatibility


def _save_nifti(vol_dhw, path):
    """splat (D=Z,H=Y,W=X) -> nibabel (X,Y,Z), same convention as export_stack.py."""
    vol_xyz = np.asarray(vol_dhw, dtype=np.float32).transpose(2, 1, 0)
    affine = np.diag([SPACING_XYZ[0], SPACING_XYZ[1], SPACING_XYZ[2], 1.0])
    nib.save(nib.Nifti1Image(vol_xyz, affine), path)


def _metrics(pred, gt):
    valid = (gt > 1e-3).float()
    denom = valid.sum().clamp(min=1.0)
    mae_full = (pred - gt).abs().mean()
    mse_full = ((pred - gt) ** 2).mean()
    mae_anat = ((pred - gt).abs() * valid).sum() / denom
    mse_anat = (((pred - gt) ** 2) * valid).sum() / denom
    psnr_full = 10 * torch.log10(1.0 / mse_full.clamp(min=1e-10))
    psnr_anat = 10 * torch.log10(1.0 / mse_anat.clamp(min=1e-10))
    return dict(mae_full=mae_full.item(), psnr_full=psnr_full.item(),
                mae_anat=mae_anat.item(), psnr_anat=psnr_anat.item())


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Building VGGT-1B (reference-slice conditioning: z-only, no t, camera_token anchor)...")
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=False,
        use_reference_token=True,
        train_on_residual_dvf=True,
    ).to(device)
    print(f"Loading checkpoint: {CKPT}")
    # map_location="cpu": the live training job (checkpoint_last.pt is still being
    # written by it) already holds ~37/46 GB of this A40. Loading straight to CUDA
    # deserializes the WHOLE checkpoint dict (incl. optimizer/scaler state for 637M
    # trainable params — ~2x model size for Adam) onto the GPU before we ever touch
    # ck["model"], which OOM'd. Load to CPU RAM instead, keep only the model weights.
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    epoch = ck.get("epoch", "?")
    print(f"  checkpoint epoch: {epoch}  (live in-progress run — NOT converged)")
    model_sd = ck["model"]
    del ck
    model.load_state_dict(model_sd, strict=False)
    del model_sd
    torch.cuda.empty_cache()
    model.eval()

    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    # num_slices=12, NOT 20: checkpoint 217721337 started training 2026-06-24, six days
    # before the multi-frame commit (9aeb760, 2026-06-30) landed. At the commit it actually
    # trained from (845e11f, carrying 1856a82's reference-slot config), sampling was
    # S = min(T_total, bbox_z_size, num_slices=12), z WITHOUT replacement — one frame per
    # z-plane, no repeats. Using num_slices=20 here silently evaluated the model outside
    # its trained input distribution.
    ds = MRIDataset(
        common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
        mode="dynamic", mri_mode="axial", num_slices=12, target_size=518,
        t_target_fixed=TARGET_PHASE, reference_slot=True,
    )

    results = {}
    for idx in SUBJECT_INDICES:
        subject_path = ds.subjects[idx % len(ds.subjects)]
        name = os.path.basename(os.path.dirname(subject_path))
        data = ds.get_data(seq_index=idx, img_per_seq=12)

        def stack(k, dt=np.float32):
            return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)

        imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
        batch = {
            "images": imgs.to(device),
            "scanner_coords": stack("scanner_coords").to(device),
            "z_indices": stack("z_indices").to(device),
            "t_indices": stack("t_indices").to(device),
            "target_t_indices": stack("target_t_indices").to(device),
            "gt_target_volume": torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(device),
        }

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)

        out = compute_volume_intensity_loss(preds, batch, grid_shape=(12, 256, 256), tv_weight=0.1)
        V_canon = out["V_canon"][0].float()
        V_gt = out["V_gt"][0].float()
        m = _metrics(V_canon, V_gt)
        results[f"{name}_t{TARGET_PHASE}"] = m
        print(f"[{idx}] {name}  t={TARGET_PHASE}  "
              f"PSNR_full={m['psnr_full']:6.2f} dB  PSNR_anat={m['psnr_anat']:6.2f} dB  "
              f"MAE_anat={m['mae_anat']:.4f}")

        np.save(os.path.join(SAVE_DIR, f"{name}_t{TARGET_PHASE}_Vcanon.npy"), V_canon.cpu().numpy())
        np.save(os.path.join(SAVE_DIR, f"{name}_t{TARGET_PHASE}_Vgt.npy"), V_gt.cpu().numpy())
        # slot 0 = the reference input image (the t=0 anchor slice actually fed to the model)
        ref_img = batch["images"][0, 0].permute(1, 2, 0).cpu().numpy()  # (518,518,3)
        np.save(os.path.join(SAVE_DIR, f"{name}_t{TARGET_PHASE}_refimg.npy"), ref_img)

        # NIfTI, same canonical affine as baselines/niftymic/export_stack.py — open directly
        # in ITK-SNAP / 3D Slicer / FSLeyes, no Python required.
        _save_nifti(V_canon.cpu().numpy(), os.path.join(SAVE_DIR, f"{name}_t{TARGET_PHASE}_Vcanon.nii.gz"))
        _save_nifti(V_gt.cpu().numpy(), os.path.join(SAVE_DIR, f"{name}_t{TARGET_PHASE}_Vgt.nii.gz"))
        print(f"     saved NIfTI: {SAVE_DIR}/{name}_t{TARGET_PHASE}_{{Vcanon,Vgt}}.nii.gz")

    if results:
        mean_full = np.mean([m["psnr_full"] for m in results.values()])
        mean_anat = np.mean([m["psnr_anat"] for m in results.values()])
        print(f"\nmean PSNR_full={mean_full:.2f} dB  mean PSNR_anat={mean_anat:.2f} dB  "
              f"over {len(results)} subject(s)  (checkpoint epoch {epoch}, in-progress run)")


if __name__ == "__main__":
    main()
