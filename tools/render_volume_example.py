"""Render one nprlshyj inference example with the requested viz changes:
  - Predicted world_points NOT masked.
  - V_canon - V_gt panel with colorbar + narrow range.
  - Coverage panel with colorbar.

Run: micromamba run -n svr python tools/render_volume_example.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss


CKPT = "/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
OUT_PNG = "/home/minsukc/vggt/result/nprlshyj_example_volume.png"
OUT_PNG_GRAD = "/home/minsukc/vggt/result/nprlshyj_example_gradients.png"
SEQ_IDX = 0
ERR_VRANGE = 0.05


def build_batch():
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(
        common_conf, DATA_ROOT,
        split="val", split_file=SPLIT_FILE,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
    )
    print(f"Loaded {len(ds.subjects)} val subjects. Using subject: {ds.subjects[SEQ_IDX]}")
    globals()["ds_subject_path"] = ds.subjects[SEQ_IDX]
    data = ds.get_data(seq_index=SEQ_IDX, img_per_seq=12)

    def stack(key, dtype=np.float32):
        return torch.from_numpy(np.stack(data[key]).astype(dtype)).unsqueeze(0)

    imgs = stack("images")                              # (1, S, H, W, 3)
    imgs = imgs.permute(0, 1, 4, 2, 3).contiguous() / 255.0  # → (1, S, 3, H, W)
    batch = {
        "images": imgs,
        "scanner_coords": stack("scanner_coords"),
        "world_points": stack("world_points"),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "z_indices": stack("z_indices"),
        "t_indices": stack("t_indices"),
        "target_t_indices": stack("target_t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
    }
    if "gt_target_volume" in data:
        batch["gt_target_volume"] = torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)
    if "t_target" in data:
        batch["t_target"] = torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)
    return batch


def main():
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Building batch...")
    batch = build_batch()
    batch = {k: v.to(device) for k, v in batch.items()}
    S = batch["images"].shape[1]
    print(f"Batch images: {tuple(batch['images'].shape)}  (B, S, 3, H, W)")
    print(f"GT target volume: {tuple(batch['gt_target_volume'].shape)}  (t_target={int(batch['t_target'].item())})")

    print("Building VGGT-1B model...")
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=True, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
    ).to(device)

    print(f"Loading checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    print("Forward pass...")
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    preds["world_points"] = preds["world_points"].float()

    print("Computing volume loss...")
    out = compute_volume_intensity_loss(preds, batch, grid_shape=(12, 256, 256), tv_weight=0.1)
    V_canon = out["V_canon"][0].float().cpu().numpy()
    V_gt = out["V_gt"][0].float().cpu().numpy()
    coverage = out["coverage"][0].float().cpu().numpy()
    print(f"  loss_volume = {out['loss_volume'].item():.4f}")
    print(f"  PSNR = {out['metric_psnr_3d'].item():.2f}  SSIM = {out.get('metric_ssim_3d', torch.tensor(float('nan'))).item():.4f}")
    print(f"  V_canon range [{V_canon.min():.3f}, {V_canon.max():.3f}]")
    print(f"  V_gt range    [{V_gt.min():.3f}, {V_gt.max():.3f}]")
    print(f"  diff |max|={np.abs(V_canon - V_gt).max():.3f}  mean|abs|={np.abs(V_canon - V_gt).mean():.4f}")

    # ── Figure 1: Volume panel ────────────────────────────────────────────
    # Layout: axial MIPs (square) on the left, plus a mid-z slice; coronal/sagittal
    # are anisotropic (D=12 × W=256), so render them with aspect='auto' on shorter rows.
    diff = V_canon - V_gt
    v_vmax = float(max(V_canon.max(), V_gt.max(), 1e-3))
    cov_vmax = float(max(coverage.max(), 1e-3))
    D, Hv, Wv = V_canon.shape
    mid_d = D // 2

    fig = plt.figure(figsize=(15, 14), dpi=90)
    # 4 rows × 5 cols: axial-MIP | mid-z slice | coronal-MIP | sagittal-MIP | colorbar
    gs = gridspec.GridSpec(
        4, 5,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 0.04],
        height_ratios=[1, 1, 1, 1],
        wspace=0.07, hspace=0.20,
    )

    def show(ax, img, cmap, vmin, vmax, title=None, aspect="equal"):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)
        ax.set_xticks([]); ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=9)
        return im

    def row(r, get_view, cmap, vmin, vmax, ylabel, titles=None):
        v_ax, v_mid, v_cor, v_sag = get_view
        ax0 = fig.add_subplot(gs[r, 0]); show(ax0, v_ax,  cmap, vmin, vmax, titles[0] if titles else None)
        ax0.set_ylabel(ylabel, fontsize=11)
        ax1 = fig.add_subplot(gs[r, 1]); show(ax1, v_mid, cmap, vmin, vmax, titles[1] if titles else None)
        ax2 = fig.add_subplot(gs[r, 2]); im = show(ax2, v_cor, cmap, vmin, vmax, titles[2] if titles else None, aspect="auto")
        ax3 = fig.add_subplot(gs[r, 3]); show(ax3, v_sag, cmap, vmin, vmax, titles[3] if titles else None, aspect="auto")
        cax = fig.add_subplot(gs[r, 4]); plt.colorbar(im, cax=cax)

    titles = [f"axial MIP (∨ over D={D})",
              f"axial slice z={mid_d}",
              f"coronal MIP (∨ over H, shape D×W={D}×{Wv})",
              f"sagittal MIP (∨ over W, shape D×H={D}×{Hv})"]

    row(0,
        [V_canon.max(0), V_canon[mid_d], V_canon.max(1), V_canon.max(2)],
        "gray", 0, v_vmax, "V_canon (pred)", titles)
    row(1,
        [V_gt.max(0), V_gt[mid_d], V_gt.max(1), V_gt.max(2)],
        "gray", 0, v_vmax, "V_gt", titles=None)
    row(2,
        [diff.mean(0), diff[mid_d], diff.mean(1), diff.mean(2)],
        "RdBu_r", -ERR_VRANGE, ERR_VRANGE, f"V_canon - V_gt\n(clipped ±{ERR_VRANGE})", titles=None)
    row(3,
        [coverage.max(0), coverage[mid_d], coverage.max(1), coverage.max(2)],
        "viridis", 0, cov_vmax, "coverage", titles=None)

    fig.suptitle(f"nprlshyj — subject {os.path.basename(os.path.dirname(ds_subject_path))}  "
                 f"PSNR={out['metric_psnr_3d'].item():.2f} dB  "
                 f"SSIM={out['metric_ssim_3d'].item():.4f}  "
                 f"loss_volume={out['loss_volume'].item():.4f}  S={S}",
                 fontsize=11, y=0.995)
    plt.savefig(OUT_PNG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_PNG}")

    # ── Figure 2: gradient (world_points) viz — NO MASK on pred ──────────────
    gt_wp = batch["world_points"][0].cpu().numpy()    # (S, H, W, 3) in [-1, 1]
    pred_wp = preds["world_points"][0].cpu().numpy()  # (S, H, W, 3)
    def to_rgb(wp):
        rgb = (wp + 1.0) / 2.0
        return np.clip(rgb, 0, 1)

    fig2, axes = plt.subplots(2, S, figsize=(1.4 * S, 3.2), dpi=110)
    for s in range(S):
        axes[0, s].imshow(to_rgb(gt_wp[s])); axes[0, s].axis("off")
        axes[0, s].set_title(f"z={batch['slice_indices'][0, s].item()}, t={batch['timesteps'][0, s].item()}",
                             fontsize=8)
        axes[1, s].imshow(to_rgb(pred_wp[s])); axes[1, s].axis("off")
    axes[0, 0].set_title("GT (top) / Pred (bot, UN-masked)\n" + axes[0, 0].get_title(), fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PNG_GRAD, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved: {OUT_PNG_GRAD}")


if __name__ == "__main__":
    main()
