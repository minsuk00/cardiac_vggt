"""Test inference with SEQUENTIAL (t, z) picks instead of random.

Slot k uses (t=k, z=k) — one slice per heart phase, z increasing in lockstep.
Slot 0 stays at t=0 (required for GT phase-0 volume load).

Runs on one train subject + one val subject, renders:
  row 1: S input slices (at the chosen (t, z))
  row 2: V_gt[z]  for z = 0..D-1 (canonical axial slices)
  row 3: V_canon[z]
  row 4: V_canon[z] - V_gt[z]  (colorbar, narrow range)

V_gt and V_canon share shape (12, 256, 256) — directly comparable per-z.
Input slices are at the native (padded to 518²) input resolution — shown for
context, not pixel-for-pixel comparison.
"""
import os
import sys
import random
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
OUT_DIR = "/home/minsukc/vggt/result"
ERR_VRANGE = 0.10  # narrow signed-error range in [0,1] intensity units


class IdentityRandom:
    """Replacement for random.Random: shuffle is a no-op so lists stay sorted."""
    def __init__(self, *a, **kw): pass
    def shuffle(self, x): pass
    def random(self): return 0.0


def build_sequential_batch(split: str, seq_index: int = 0):
    """Load one subject and force sequential (t, z) picks via monkey-patched random."""
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(
        common_conf, DATA_ROOT,
        split=split, split_file=SPLIT_FILE,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
    )
    subject_path = ds.subjects[seq_index]
    print(f"  [{split}] {len(ds.subjects)} subjects; using: {os.path.basename(os.path.dirname(subject_path))}")

    # Patch the two RNG entry points the dataset uses so shuffles become no-ops.
    orig_random_class = random.Random
    orig_shuffle = random.shuffle
    random.Random = IdentityRandom
    random.shuffle = lambda x: None
    try:
        data = ds.get_data(seq_index=seq_index, img_per_seq=12)
    finally:
        random.Random = orig_random_class
        random.shuffle = orig_shuffle

    t_picks = [int(t) for t in data["timesteps"]]
    z_picks = [int(z) for z in data["slice_indices"]]
    print(f"     S={len(t_picks)}  t_seq={t_picks}  z_seq={z_picks}")

    def stack(key, dtype=np.float32):
        return torch.from_numpy(np.stack(data[key]).astype(dtype)).unsqueeze(0)

    imgs = stack("images")                                    # (1, S, H, W, 3)
    imgs = imgs.permute(0, 1, 4, 2, 3).contiguous() / 255.0   # → (1, S, 3, H, W)
    batch = {
        "images": imgs,
        "scanner_coords": stack("scanner_coords"),
        "world_points": stack("world_points"),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
        "z_indices": stack("z_indices"),
        "t_indices": stack("t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
    }
    if "gt_target_volume" in data:
        batch["gt_target_volume"] = torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)
    if "t_target" in data:
        batch["t_target"] = torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)

    # Inspect one native NIfTI to get native (W, H, Z) for aspect-correct viz.
    import glob, nibabel as nib
    nii0 = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))[0]
    native_shape = nib.load(nii0).header.get_data_shape()[:3]  # (W, H, Z)
    return subject_path, batch, native_shape


def run_model(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    preds["world_points"] = preds["world_points"].float()
    out = compute_volume_intensity_loss(preds, batch, grid_shape=(12, 256, 256), tv_weight=0.1)

    # Compute BOTH PSNR formulas side by side so we can compare apples to apples
    # with the training-time numbers (which used the anatomy-masked version).
    with torch.no_grad():
        V, Vg = out["V_canon"], out["V_gt"]
        valid = (Vg > 1e-3).float()
        denom = valid.sum().clamp(min=1.0)
        mse_anat = ((V - Vg) ** 2 * valid).sum() / denom
        mse_full = ((V - Vg) ** 2).mean()
        out["metric_psnr_anatomy"] = 10 * torch.log10(1.0 / mse_anat.clamp(min=1e-10))
        out["metric_psnr_full"]    = 10 * torch.log10(1.0 / mse_full.clamp(min=1e-10))
        out["metric_mae_anatomy"]  = ((V - Vg).abs() * valid).sum() / denom
        out["metric_mae_full"]     = (V - Vg).abs().mean()
        out["anatomy_frac"]        = valid.mean()
    return batch, preds, out


def render_dvf_figure(split, subject_path, batch, preds, png_path):
    """Per-slice DVF: input intensity + Δx, Δy, Δz (normalized [-1,1] units).
    Inputs are transposed so W (vertical native) matches V_gt's W-horizontal layout.
    """
    imgs = batch["images"][0].float().cpu().numpy().mean(axis=1)              # (S, W, H)
    imgs = imgs.transpose(0, 2, 1)                                            # (S, H, W) — match V_gt orientation
    dvf = preds.get("dvfs")
    if dvf is None:
        dvf = preds["world_points"] - batch["scanner_coords"]
    dvf = dvf[0].float().cpu().numpy().transpose(0, 2, 1, 3)                  # (S, H, W, 3)
    t_picks = batch["timesteps"][0].cpu().numpy()
    z_picks = batch["slice_indices"][0].cpu().numpy()
    S = imgs.shape[0]

    p50 = float(np.percentile(np.abs(dvf), 50))
    p95 = float(np.percentile(np.abs(dvf), 95))
    p99 = float(np.percentile(np.abs(dvf), 99))
    print(f"     |Δ| percentiles  p50={p50:.4f}  p95={p95:.4f}  p99={p99:.4f}")
    dvf_vrange = 0.05  # fixed signed range; ~5mm in a ~100mm half-extent ≈ 0.05 in normalized units
    fig = plt.figure(figsize=(2.4 * S + 1.6, 11), dpi=180)
    gs = gridspec.GridSpec(4, S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.18)

    row_specs = [
        ("input intensity", imgs,        "gray",   0,            1.0,         True),
        ("Δx (norm)",       dvf[..., 0], "RdBu_r", -dvf_vrange,  dvf_vrange,  False),
        ("Δy (norm)",       dvf[..., 1], "RdBu_r", -dvf_vrange,  dvf_vrange,  False),
        ("Δz (norm)",       dvf[..., 2], "RdBu_r", -dvf_vrange,  dvf_vrange,  False),
    ]
    for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(row_specs):
        last_im = None
        for s in range(S):
            ax = fig.add_subplot(gs[r, s])
            last_im = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if is_top:
                ax.set_title(f"t={int(t_picks[s])}, z={int(z_picks[s])}", fontsize=9)
            if s == 0:
                ax.set_ylabel(lbl, fontsize=10)
        cax = fig.add_subplot(gs[r, S]); plt.colorbar(last_im, cax=cax)

    fig.suptitle(f"{split.upper()} — {os.path.basename(os.path.dirname(subject_path))}  "
                 f"per-slice DVF (residual, normalized [-1,1])  "
                 f"fixed range ±{dvf_vrange:.2f}  |  |Δ| p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}",
                 fontsize=12, y=0.995)
    plt.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {png_path}")


def _resize_native(img2d, native_shape):
    """Resize a (256,256) canonical slice back to native (H_native, W_native) for aspect-correct viz."""
    import cv2
    W_n, H_n, _ = native_shape
    return cv2.resize(img2d.astype(np.float32), (W_n, H_n), interpolation=cv2.INTER_LINEAR)


def _load_native_input_slices(subject_path, t_picks, z_picks):
    """Load raw native vol[:,:,z] slices for the (t,z) picks, normalized with t=0
    percentiles (same convention as MRIDataset after fix 3). Returns (S, H_n, W_n).
    Used so the displayed input row has exactly the same shape/aspect as V_gt[d],
    without going through the 518² padded resize that compresses anatomy.
    """
    import nibabel as nib
    import glob
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    vol0 = nib.load(nii_files[0]).get_fdata()
    v_min = np.percentile(vol0, 1)
    v_max = np.percentile(vol0, 99.5)
    cache = {0: vol0}
    out = []
    for t, z in zip(t_picks, z_picks):
        if int(t) not in cache:
            cache[int(t)] = nib.load(nii_files[int(t)]).get_fdata()
        slc = cache[int(t)][:, :, int(z)]  # (W_n, H_n)
        slc_norm = np.clip((slc - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32)
        out.append(slc_norm.T)  # transpose to (H_n, W_n) to match V_gt orientation
    return np.stack(out)


def render_figure(split, subject_path, batch, out, png_path, native_shape):
    V_canon = out["V_canon"][0].float().cpu().numpy()       # (D, 256, 256)
    V_gt    = out["V_gt"][0].float().cpu().numpy()
    D, H, W = V_canon.shape
    W_n, H_n, _ = native_shape
    V_canon_n = np.stack([_resize_native(V_canon[d], native_shape) for d in range(D)])  # (D, H_n, W_n)
    V_gt_n    = np.stack([_resize_native(V_gt[d],    native_shape) for d in range(D)])

    print(f"     [diag] per-d V_gt    mean/max:  " +
          "  ".join([f"d{d}:{V_gt[d].mean():.3f}/{V_gt[d].max():.3f}" for d in range(D)]))
    print(f"     [diag] per-d V_canon mean/max:  " +
          "  ".join([f"d{d}:{V_canon[d].mean():.3f}/{V_canon[d].max():.3f}" for d in range(D)]))

    # Load input slices from the raw NIfTI directly, normalized with t=0 percentiles
    # (same as MRIDataset post-fix-3). Avoids going through the padded 518² resize,
    # so anatomy aspect ratio is preserved exactly.
    t_picks = batch["timesteps"][0].cpu().numpy()
    z_picks = batch["slice_indices"][0].cpu().numpy()
    imgs_gray = _load_native_input_slices(subject_path, t_picks, z_picks)  # (S, H_n, W_n)
    S = imgs_gray.shape[0]
    diff = V_canon_n - V_gt_n

    print(f"     [diag] input slice at (t=0,z=0) intensity mean/max:  "
          f"{imgs_gray[0].mean():.3f}/{imgs_gray[0].max():.3f}")

    v_vmax = float(max(V_canon_n.max(), V_gt_n.max(), 1e-3))
    n_cols = max(S, D)
    fig = plt.figure(figsize=(2.0 * n_cols + 1.6, 8.5), dpi=180)
    gs = gridspec.GridSpec(4, n_cols + 1,
                           width_ratios=[1.0] * n_cols + [0.05],
                           wspace=0.04, hspace=0.18)

    # Row 0: input slices
    for s in range(S):
        ax = fig.add_subplot(gs[0, s])
        ax.imshow(imgs_gray[s], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={int(t_picks[s])}, z={int(z_picks[s])}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        if s == 0:
            ax.set_ylabel("input slice", fontsize=9)
    # trailing blanks (input row may have fewer than D cols)
    for s in range(S, n_cols):
        fig.add_subplot(gs[0, s]).axis("off")
    fig.add_subplot(gs[0, n_cols]).axis("off")  # blank colorbar slot

    # Helper to render a row of D axial slices with one shared colorbar on the right
    def vol_row(r, vol, cmap, vmin, vmax, ylabel):
        last_im = None
        for d in range(D):
            ax = fig.add_subplot(gs[r, d])
            last_im = ax.imshow(vol[d], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 1:
                ax.set_title(f"z={d}", fontsize=7)
            if d == 0:
                ax.set_ylabel(ylabel, fontsize=9)
        for d in range(D, n_cols):
            fig.add_subplot(gs[r, d]).axis("off")
        cax = fig.add_subplot(gs[r, n_cols])
        plt.colorbar(last_im, cax=cax)

    vol_row(1, V_gt_n,    "gray",   0,           v_vmax,     f"V_gt\n({H_n}×{W_n})")
    vol_row(2, V_canon_n, "gray",   0,           v_vmax,     f"V_canon\n({H_n}×{W_n})")
    vol_row(3, diff,      "RdBu_r", -ERR_VRANGE, ERR_VRANGE, f"V_canon − V_gt\n(±{ERR_VRANGE})")

    fig.suptitle(
        f"{split.upper()} — {os.path.basename(os.path.dirname(subject_path))}  "
        f"S={S}  D={D}   "
        f"MAE_full={out['metric_mae_full'].item():.4f}  "
        f"PSNR_full={out['metric_psnr_full'].item():.2f} dB  |  "
        f"MAE_anat={out['metric_mae_anatomy'].item():.4f}  "
        f"PSNR_anat={out['metric_psnr_anatomy'].item():.2f} dB  "
        f"SSIM={out['metric_ssim_3d'].item():.4f}",
        fontsize=9, y=0.995,
    )
    plt.savefig(png_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {png_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Building VGGT-1B model...")
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=True,
        train_on_residual_dvf=True,
    ).to(device)
    print(f"Loading checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    for split, seq_idx in [("train", 0), ("val", 0)]:
        print(f"\n=== {split} ===")
        subject_path, batch, native_shape = build_sequential_batch(split, seq_idx)
        print(f"     native (W,H,Z) = {native_shape}")
        batch_dev, preds, out = run_model(model, batch, device)
        print(f"     anatomy_frac = {out['anatomy_frac'].item():.3f}  "
              f"MAE_full = {out['metric_mae_full'].item():.4f}  "
              f"MAE_anat = {out['metric_mae_anatomy'].item():.4f}")
        print(f"     PSNR_full = {out['metric_psnr_full'].item():.2f} dB  "
              f"PSNR_anat = {out['metric_psnr_anatomy'].item():.2f} dB  "
              f"SSIM(full) = {out['metric_ssim_3d'].item():.4f}")
        out_png = os.path.join(OUT_DIR, f"nprlshyj_sequential_{split}.png")
        render_figure(split, subject_path, batch_dev, out, out_png, native_shape)
        dvf_png = os.path.join(OUT_DIR, f"nprlshyj_sequential_{split}_dvf.png")
        render_dvf_figure(split, subject_path, batch_dev, preds, dvf_png)


if __name__ == "__main__":
    main()
