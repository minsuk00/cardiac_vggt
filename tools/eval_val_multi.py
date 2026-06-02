"""Run the nprlshyj checkpoint on N val subjects with sequential (t=k, z=k)
sampling. Report MAE / PSNR / SSIM per subject and a summary line at the end.
Also renders the 4-row volume figure for each subject for visual inspection.
"""
import os
import sys
import random
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec
from omegaconf import OmegaConf
import nibabel as nib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.mri_dataset import MRIDataset
from vggt.models.vggt import VGGT
from loss import compute_volume_intensity_loss

CKPT = "/home/minsukc/vggt/scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
OUT_DIR = "/home/minsukc/vggt/result/val_multi"
N_SUBJECTS = 30            # val[0..N-1]; full val set
ERR_VRANGE = 0.10


class IdentityRandom:
    def __init__(self, *a, **kw): pass
    def shuffle(self, x): pass
    def random(self): return 0.0


def _build_sequential_batch(seq_idx):
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)
    subject_path = ds.subjects[seq_idx]
    orig_class, orig_shuffle = random.Random, random.shuffle
    random.Random = IdentityRandom
    random.shuffle = lambda x: None
    try:
        data = ds.get_data(seq_index=seq_idx, img_per_seq=12)
    finally:
        random.Random = orig_class
        random.shuffle = orig_shuffle

    def stack(k, dt=np.float32): return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0)
    imgs = stack("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    batch = {
        "images": imgs,
        "scanner_coords": stack("scanner_coords"),
        "world_points": stack("world_points"),
        "z_indices": stack("z_indices"),
        "t_indices": stack("t_indices"),
        "target_t_indices": stack("target_t_indices"),
        "timesteps": torch.from_numpy(np.stack(data["timesteps"]).astype(np.int64)).unsqueeze(0),
        "slice_indices": torch.from_numpy(np.stack(data["slice_indices"]).astype(np.int64)).unsqueeze(0),
        "point_masks": torch.from_numpy(np.stack(data["point_masks"])).unsqueeze(0),
    }
    if "gt_target_volume" in data:
        batch["gt_target_volume"] = torch.from_numpy(data["gt_target_volume"].astype(np.float32)).unsqueeze(0)
    if "t_target" in data:
        batch["t_target"] = torch.from_numpy(data["t_target"].astype(np.int64)).unsqueeze(0)
    nii0 = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))[0]
    native_shape = nib.load(nii0).header.get_data_shape()[:3]
    return subject_path, batch, native_shape


def _resize_native(img2d, native_shape):
    import cv2
    W_n, H_n, _ = native_shape
    return cv2.resize(img2d.astype(np.float32), (W_n, H_n), interpolation=cv2.INTER_LINEAR)


def _load_native_input_slices(subject_path, t_picks, z_picks):
    nii_files = sorted(glob.glob(os.path.join(subject_path, "3d_recon", "sax_frame_*.nii.gz")))
    vol0 = nib.load(nii_files[0]).get_fdata()
    v_min, v_max = np.percentile(vol0, 1), np.percentile(vol0, 99.5)
    cache = {0: vol0}
    out = []
    for t, z in zip(t_picks, z_picks):
        if int(t) not in cache:
            cache[int(t)] = nib.load(nii_files[int(t)]).get_fdata()
        slc = cache[int(t)][:, :, int(z)]
        out.append(np.clip((slc - v_min) / (v_max - v_min + 1e-8), 0, 1).astype(np.float32).T)
    return np.stack(out)


def _metrics(V_canon, V_gt):
    valid = (V_gt > 1e-3).float()
    denom = valid.sum().clamp(min=1.0)
    mae_full = (V_canon - V_gt).abs().mean()
    mae_anat = ((V_canon - V_gt).abs() * valid).sum() / denom
    mse_full = ((V_canon - V_gt) ** 2).mean()
    mse_anat = (((V_canon - V_gt) ** 2) * valid).sum() / denom
    psnr_full = 10 * torch.log10(1.0 / mse_full.clamp(min=1e-10))
    psnr_anat = 10 * torch.log10(1.0 / mse_anat.clamp(min=1e-10))
    try:
        from fused_ssim import fused_ssim3d
        # fused_ssim3d expects (B, 1, D, H, W); V_canon here is (D, H, W).
        a = V_canon.unsqueeze(0).unsqueeze(0).float().contiguous()
        b = V_gt.unsqueeze(0).unsqueeze(0).float().contiguous()
        ssim = fused_ssim3d(a, b, train=False)
    except Exception as e:
        print(f"     [warn] SSIM failed: {e}")
        ssim = torch.tensor(float("nan"))
    return dict(mae_full=mae_full.item(), mae_anat=mae_anat.item(),
                psnr_full=psnr_full.item(), psnr_anat=psnr_anat.item(),
                ssim=float(ssim.item()))


def _render(subject_path, batch, V_canon, V_gt, native_shape, metrics, out_png):
    W_n, H_n, _ = native_shape
    D = V_canon.shape[0]
    V_canon_n = np.stack([_resize_native(V_canon[d], native_shape) for d in range(D)])
    V_gt_n    = np.stack([_resize_native(V_gt[d],    native_shape) for d in range(D)])
    t_picks = batch["timesteps"][0].cpu().numpy()
    z_picks = batch["slice_indices"][0].cpu().numpy()
    imgs_gray = _load_native_input_slices(subject_path, t_picks, z_picks)
    S = imgs_gray.shape[0]
    diff = V_canon_n - V_gt_n
    v_vmax = float(max(V_canon_n.max(), V_gt_n.max(), 1e-3))
    n_cols = max(S, D)
    fig = plt.figure(figsize=(2.0 * n_cols + 1.6, 8.5), dpi=160)
    gs = gridspec.GridSpec(4, n_cols + 1, width_ratios=[1.0]*n_cols + [0.05], wspace=0.04, hspace=0.18)
    # Input row
    for s in range(S):
        ax = fig.add_subplot(gs[0, s])
        ax.imshow(imgs_gray[s], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={int(t_picks[s])}, z={int(z_picks[s])}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if s == 0:
            ax.set_ylabel("input", fontsize=10)
    for s in range(S, n_cols): fig.add_subplot(gs[0, s]).axis("off")
    fig.add_subplot(gs[0, n_cols]).axis("off")

    def vol_row(r, vol, cmap, vmin, vmax, ylabel):
        im = None
        for d in range(D):
            ax = fig.add_subplot(gs[r, d])
            im = ax.imshow(vol[d], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 1: ax.set_title(f"z={d}", fontsize=8)
            if d == 0: ax.set_ylabel(ylabel, fontsize=10)
        for d in range(D, n_cols): fig.add_subplot(gs[r, d]).axis("off")
        cax = fig.add_subplot(gs[r, n_cols]); plt.colorbar(im, cax=cax)

    vol_row(1, V_gt_n,    "gray",   0,            v_vmax,    f"V_gt\n({H_n}×{W_n})")
    vol_row(2, V_canon_n, "gray",   0,            v_vmax,    f"V_canon\n({H_n}×{W_n})")
    vol_row(3, diff,      "RdBu_r", -ERR_VRANGE,  ERR_VRANGE,f"V_canon − V_gt\n(±{ERR_VRANGE})")

    fig.suptitle(
        f"VAL — {os.path.basename(os.path.dirname(subject_path))}  S={S}  "
        f"native (W,H,Z)={native_shape}  |  MAE_full={metrics['mae_full']:.4f}  "
        f"PSNR_full={metrics['psnr_full']:.2f} dB  PSNR_anat={metrics['psnr_anat']:.2f} dB  "
        f"SSIM={metrics['ssim']:.4f}",
        fontsize=11, y=0.995)
    plt.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Building VGGT-1B model...")
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=True, use_target_t_pose_embedding=True,
        train_on_residual_dvf=True,
    ).to(device)
    print(f"Loading checkpoint: {CKPT}")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    rows = []
    for i in range(N_SUBJECTS):
        subject_path, batch, native_shape = _build_sequential_batch(i)
        name = os.path.basename(os.path.dirname(subject_path))
        S = batch["images"].shape[1]
        print(f"\n[{i}] {name}  native={native_shape}  S={S}")
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch_dev["images"], batch=batch_dev)
        preds["world_points"] = preds["world_points"].float()

        # Model prediction
        out_model = compute_volume_intensity_loss(preds, batch_dev,
                                                  grid_shape=(12, 256, 256), tv_weight=0.1)
        m = _metrics(out_model["V_canon"][0].float(), out_model["V_gt"][0].float())

        # Identity-Δ baseline: world_points = scanner_coords (no motion correction at all).
        # Same splat, same V_gt — answers "how well do raw mixed-phase inputs reconstruct ED?"
        ident_preds = {"world_points": batch_dev["scanner_coords"].float()}
        out_ident = compute_volume_intensity_loss(ident_preds, batch_dev,
                                                  grid_shape=(12, 256, 256), tv_weight=0.1)
        mi = _metrics(out_ident["V_canon"][0].float(), out_ident["V_gt"][0].float())

        rows.append((name, native_shape, S, m, mi))
        print(f"     model:    MAE={m['mae_full']:.4f}  PSNR_full={m['psnr_full']:.2f} dB  "
              f"PSNR_anat={m['psnr_anat']:.2f} dB  SSIM={m['ssim']:.4f}")
        print(f"     identity: MAE={mi['mae_full']:.4f}  PSNR_full={mi['psnr_full']:.2f} dB  "
              f"PSNR_anat={mi['psnr_anat']:.2f} dB  SSIM={mi['ssim']:.4f}")
        print(f"     Δ(model - identity):  ΔPSNR_full={m['psnr_full']-mi['psnr_full']:+.2f} dB  "
              f"ΔSSIM={m['ssim']-mi['ssim']:+.4f}")
        out_png = os.path.join(OUT_DIR, f"val_{i:02d}_{name}.png")
        _render(subject_path, batch_dev, out_model["V_canon"][0].float().cpu().numpy(),
                out_model["V_gt"][0].float().cpu().numpy(), native_shape, m, out_png)
        print(f"     saved: {out_png}")

    print("\n=== summary: model vs identity-Δ baseline ===")
    print(f"  {'idx':<3} {'subject':<14} {'native':<14} {'S':>3}  "
          f"{'PSNR_full':>9} {'PSNR_id':>8} {'ΔPSNR':>7}  "
          f"{'SSIM':>7} {'SSIM_id':>7} {'ΔSSIM':>7}")
    for i, (name, native, S, m, mi) in enumerate(rows):
        print(f"  [{i}] {name:<14} {str(tuple(native)):<14} {S:>3}  "
              f"{m['psnr_full']:>9.2f} {mi['psnr_full']:>8.2f} {m['psnr_full']-mi['psnr_full']:>+7.2f}  "
              f"{m['ssim']:>7.4f} {mi['ssim']:>7.4f} {m['ssim']-mi['ssim']:>+7.4f}")
    psnr_m = np.array([m['psnr_full']   for _,_,_,m,_ in rows])
    psnr_i = np.array([mi['psnr_full']  for _,_,_,_,mi in rows])
    ssim_m = np.array([m['ssim']        for _,_,_,m,_ in rows])
    ssim_i = np.array([mi['ssim']       for _,_,_,_,mi in rows])
    print(f"  ---")
    print(f"  model    mean PSNR_full = {psnr_m.mean():.2f} ± {psnr_m.std():.2f}  "
          f"SSIM = {ssim_m.mean():.4f} ± {ssim_m.std():.4f}")
    print(f"  identity mean PSNR_full = {psnr_i.mean():.2f} ± {psnr_i.std():.2f}  "
          f"SSIM = {ssim_i.mean():.4f} ± {ssim_i.std():.4f}")
    print(f"  Δ(model − identity)    = {(psnr_m-psnr_i).mean():+.2f} dB PSNR, "
          f"{(ssim_m-ssim_i).mean():+.4f} SSIM")


if __name__ == "__main__":
    main()
