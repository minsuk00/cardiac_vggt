"""All val baselines + model in one sweep.

For each of the 30 val subjects with sequential (t=k, z=k) picks:
  - model       : world_points = scanner_coords + Δ_model
  - identity    : world_points = scanner_coords
  - elastix-Δ   : world_points = scanner_coords + Δ_elastix  (from dvf_elastix/)
  - carmen-Δ    : world_points = scanner_coords + Δ_carmen   (from dvf_carmen/)

Same splat / V_gt / loss for all four — drop-in different position fields.
Renders one 6-row figure per subject (input / V_gt / V_model / V_elastix /
V_carmen / V_identity).
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
OUT_DIR = "/home/minsukc/vggt/result/val_all_baselines"
N_SUBJECTS = 30
ERR_VRANGE = 0.10


class IdentityRandom:
    def __init__(self, *a, **kw): pass
    def shuffle(self, x): pass
    def random(self): return 0.0


def _build_batch(seq_idx, dvf_dirname):
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518,
                    dvf_dirname=dvf_dirname)
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
        "world_points": stack("world_points"),    # = scanner_coords + Δ_<dvf_dirname>
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
        a = V_canon.unsqueeze(0).unsqueeze(0).float().contiguous()
        b = V_gt.unsqueeze(0).unsqueeze(0).float().contiguous()
        ssim = fused_ssim3d(a, b, train=False)
    except Exception:
        ssim = torch.tensor(float("nan"))
    return dict(mae_full=mae_full.item(), psnr_full=psnr_full.item(),
                psnr_anat=psnr_anat.item(), ssim=float(ssim.item()))


def _run_method(method_world_points, batch, scanner_coords, grid_shape=(12, 256, 256)):
    """Splat with the given world_points field, compute metrics & V_canon."""
    if method_world_points is scanner_coords:
        preds = {"world_points": scanner_coords}
    else:
        preds = {"world_points": method_world_points}
    out = compute_volume_intensity_loss(preds, batch, grid_shape=grid_shape, tv_weight=0.1)
    V_canon = out["V_canon"][0].float()
    V_gt    = out["V_gt"][0].float()
    return _metrics(V_canon, V_gt), V_canon.cpu().numpy(), V_gt.cpu().numpy()


def _render(subject_path, batch, native_shape, V_gt_n, panels, summary, png):
    W_n, H_n, _ = native_shape
    D, *_ = V_gt_n.shape
    t_picks = batch["timesteps"][0].cpu().numpy()
    z_picks = batch["slice_indices"][0].cpu().numpy()
    imgs_gray = _load_native_input_slices(subject_path, t_picks, z_picks)
    S = imgs_gray.shape[0]
    n_cols = max(S, D)
    v_vmax = float(max(V_gt_n.max(), max(p["V_n"].max() for p in panels.values()), 1e-3))

    n_rows = 2 + 2 * len(panels)  # input + V_gt + (V_method + diff) per method
    fig = plt.figure(figsize=(2.0 * n_cols + 1.6, 2.0 * n_rows + 1.0), dpi=140)
    gs = gridspec.GridSpec(n_rows, n_cols + 1, width_ratios=[1.0]*n_cols + [0.05],
                           wspace=0.04, hspace=0.20)

    # Row 0: input slices
    for s in range(S):
        ax = fig.add_subplot(gs[0, s])
        ax.imshow(imgs_gray[s], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={int(t_picks[s])}, z={int(z_picks[s])}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if s == 0: ax.set_ylabel("input", fontsize=10)
    for s in range(S, n_cols): fig.add_subplot(gs[0, s]).axis("off")
    fig.add_subplot(gs[0, n_cols]).axis("off")

    def vol_row(r, vol, cmap, vmin, vmax, ylabel, show_z_title=False):
        im = None
        for d in range(D):
            ax = fig.add_subplot(gs[r, d])
            im = ax.imshow(vol[d], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if show_z_title: ax.set_title(f"z={d}", fontsize=8)
            if d == 0: ax.set_ylabel(ylabel, fontsize=10)
        for d in range(D, n_cols): fig.add_subplot(gs[r, d]).axis("off")
        cax = fig.add_subplot(gs[r, n_cols]); plt.colorbar(im, cax=cax)

    # Row 1: V_gt
    vol_row(1, V_gt_n, "gray", 0, v_vmax, f"V_gt\n({H_n}×{W_n})", show_z_title=True)

    # Rows 2..: per-method V_canon + signed diff
    r = 2
    for method_name, p in panels.items():
        psnr = p["psnr"]
        vol_row(r,   p["V_n"],          "gray",   0,           v_vmax,
                f"V_canon: {method_name}\n{psnr:.2f} dB")
        vol_row(r+1, p["V_n"] - V_gt_n, "RdBu_r", -ERR_VRANGE, ERR_VRANGE,
                f"diff: {method_name}\n(±{ERR_VRANGE})")
        r += 2

    fig.suptitle(summary, fontsize=11, y=0.995)
    plt.savefig(png, bbox_inches="tight", facecolor="white")
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
        # Build the elastix batch first (default dvf_dirname); use it for model & identity.
        subject_path, batch_e, native_shape = _build_batch(i, "dvf_elastix")
        # Also build the carmen batch (same seed = same picks) for Δ_carmen.
        _, batch_c, _ = _build_batch(i, "dvf_carmen")
        name = os.path.basename(os.path.dirname(subject_path))
        S = batch_e["images"].shape[1]
        # Sanity: picks match across the two batches
        assert torch.equal(batch_e["timesteps"], batch_c["timesteps"]), "seq picks diverged"
        assert torch.equal(batch_e["slice_indices"], batch_c["slice_indices"]), "z picks diverged"

        batch_e = {k: v.to(device) for k, v in batch_e.items()}
        batch_c = {k: v.to(device) for k, v in batch_c.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch_e["images"], batch=batch_e)
        wp_model = preds["world_points"][0].float()
        scanner = batch_e["scanner_coords"][0].float()
        wp_elastix = batch_e["world_points"][0].float()
        wp_carmen  = batch_c["world_points"][0].float()

        # Run all four methods through the same loss path
        m_model, V_m, V_gt = _run_method(wp_model.unsqueeze(0),    batch_e, scanner.unsqueeze(0))
        m_id,    V_i, _    = _run_method(scanner.unsqueeze(0),     batch_e, scanner.unsqueeze(0))
        m_ela,   V_e, _    = _run_method(wp_elastix.unsqueeze(0),  batch_e, scanner.unsqueeze(0))
        m_car,   V_c, _    = _run_method(wp_carmen.unsqueeze(0),   batch_c, scanner.unsqueeze(0))

        rows.append((name, native_shape, S, m_model, m_id, m_ela, m_car))
        print(f"[{i:2d}] {name:<14}  native={native_shape}  S={S}")
        print(f"     model    PSNR_full={m_model['psnr_full']:6.2f} dB  SSIM={m_model['ssim']:.4f}")
        print(f"     elastix  PSNR_full={m_ela  ['psnr_full']:6.2f} dB  SSIM={m_ela  ['ssim']:.4f}")
        print(f"     carmen   PSNR_full={m_car  ['psnr_full']:6.2f} dB  SSIM={m_car  ['ssim']:.4f}")
        print(f"     identity PSNR_full={m_id   ['psnr_full']:6.2f} dB  SSIM={m_id   ['ssim']:.4f}")

        # Resize V_gt and per-method V_canon to native shape for the figure
        D = V_gt.shape[0]
        V_gt_n = np.stack([_resize_native(V_gt[d], native_shape) for d in range(D)])
        panels = {}
        for nm, V, m in [("model", V_m, m_model), ("elastix", V_e, m_ela),
                         ("carmen", V_c, m_car), ("identity", V_i, m_id)]:
            panels[nm] = {"V_n": np.stack([_resize_native(V[d], native_shape) for d in range(D)]),
                          "psnr": m["psnr_full"]}
        summary = (f"VAL — {name}  native={native_shape}  S={S}    "
                   f"PSNR_full:  model={m_model['psnr_full']:.2f}  "
                   f"elastix={m_ela['psnr_full']:.2f}  "
                   f"carmen={m_car['psnr_full']:.2f}  "
                   f"identity={m_id['psnr_full']:.2f}")
        out_png = os.path.join(OUT_DIR, f"val_{i:02d}_{name}.png")
        _render(subject_path, batch_e, native_shape, V_gt_n, panels, summary, out_png)
        print(f"     saved: {out_png}")

    print("\n=== summary table (PSNR_full / SSIM) ===")
    print(f"  {'idx':>3} {'subject':<14} {'native':<14} {'S':>3}  "
          f"{'model':>14}  {'elastix':>14}  {'carmen':>14}  {'identity':>14}")
    for i, (name, native, S, mm, mi, me, mc) in enumerate(rows):
        print(f"  [{i:2d}] {name:<14} {str(tuple(native)):<14} {S:>3}  "
              f"{mm['psnr_full']:6.2f}/{mm['ssim']:.4f}  "
              f"{me['psnr_full']:6.2f}/{me['ssim']:.4f}  "
              f"{mc['psnr_full']:6.2f}/{mc['ssim']:.4f}  "
              f"{mi['psnr_full']:6.2f}/{mi['ssim']:.4f}")
    psnr_m = np.array([r[3]['psnr_full'] for r in rows])
    psnr_e = np.array([r[5]['psnr_full'] for r in rows])
    psnr_c = np.array([r[6]['psnr_full'] for r in rows])
    psnr_i = np.array([r[4]['psnr_full'] for r in rows])
    ssim_m = np.array([r[3]['ssim'] for r in rows])
    ssim_e = np.array([r[5]['ssim'] for r in rows])
    ssim_c = np.array([r[6]['ssim'] for r in rows])
    ssim_i = np.array([r[4]['ssim'] for r in rows])
    print("  ---")
    print(f"  model    mean PSNR={psnr_m.mean():.2f}±{psnr_m.std():.2f}  SSIM={ssim_m.mean():.4f}±{ssim_m.std():.4f}")
    print(f"  elastix  mean PSNR={psnr_e.mean():.2f}±{psnr_e.std():.2f}  SSIM={ssim_e.mean():.4f}±{ssim_e.std():.4f}")
    print(f"  carmen   mean PSNR={psnr_c.mean():.2f}±{psnr_c.std():.2f}  SSIM={ssim_c.mean():.4f}±{ssim_c.std():.4f}")
    print(f"  identity mean PSNR={psnr_i.mean():.2f}±{psnr_i.std():.2f}  SSIM={ssim_i.mean():.4f}±{ssim_i.std():.4f}")
    print(f"  Δ(model − elastix) = {(psnr_m-psnr_e).mean():+.2f} dB PSNR, {(ssim_m-ssim_e).mean():+.4f} SSIM")
    print(f"  Δ(model − carmen)  = {(psnr_m-psnr_c).mean():+.2f} dB PSNR, {(ssim_m-ssim_c).mean():+.4f} SSIM")
    print(f"  Δ(model − identity)= {(psnr_m-psnr_i).mean():+.2f} dB PSNR, {(ssim_m-ssim_i).mean():+.4f} SSIM")


if __name__ == "__main__":
    main()
