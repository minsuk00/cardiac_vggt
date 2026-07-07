"""Cardiac-cycle filmstrip + beating-heart GIF for the 3 active reference-slot models
(reference / bspline / diffusion) on CMRxRecon **val** subjects, rendered at **5 z-slices**
(including the middle) instead of just the mid-z plane that the in-trainer filmstrip shows.

For each (model, val subject, respiration on/off) it rebuilds the scattered reference-slot input
batch, sweeps the queried target phase t = 0..T-1 (slot 0 = the target-phase reference slice,
docs/25), and reconstructs the full V_canon cube at each phase. It then renders:
  * a filmstrip PNG: 5 z-rows, each a (V_gt / V_canon) pair × 12 phase columns
  * a beating-heart GIF: per-phase frame = [V_gt 5-z montage] over [V_canon 5-z montage]

This is faithful to the trainer's `_log_cardiac_cycle_filmstrip` reference-slot path
(training/trainer.py) — same batch construction, same per-phase slot-0 re-extraction, same
deterministic val breathing — only the captured z-planes and the output layout differ.

Run: micromamba run -n svr python tools/render_cardiac_filmstrip_multislice.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec as _gs
from PIL import Image
from hydra import compose, initialize
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

# Hydra custom resolvers (registered in launch.py; needed for standalone compose()).
OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
OmegaConf.register_new_resolver(
    "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)

from hydra.utils import instantiate
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss

DEV = torch.device("cuda")
LOGS = os.path.join(_ROOT, "scratch", "logs")
MODELS = [
    ("reference", "mri_volume",
     f"{LOGS}/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
    ("bspline", "mri_volume_bspline",
     f"{LOGS}/217719798_mri_volume_bspline_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
    ("diffusion", "mri_volume_diffusion",
     f"{LOGS}/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
]
SUBJECTS = [0, 7]          # a couple of deterministic val subjects
N_ZSLICES = 5             # filmstrip z-rows (includes the middle)
OUT = os.path.join(_ROOT, "result", "cardiac_filmstrip_multislice")


def compose_cfg(config_name):
    with initialize(version_base=None, config_path=os.path.join("..", "training", "config")):
        cfg = compose(config_name=config_name)
    return cfg


def build_model(cfg, ckpt_path):
    model = instantiate(cfg.model, _recursive_=False).to(DEV).eval()
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    # strict=false because the 941M dict carries disabled heads; flag any *trainable* gaps.
    real_miss = [m for m in miss if "patch_embed" not in m]
    print(f"    loaded {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))} "
          f"(epoch {ck.get('epoch', '?')}): missing={len(miss)} unexpected={len(unexp)}",
          flush=True)
    if real_miss:
        print(f"      WARN non-patch_embed missing keys: {real_miss[:5]}", flush=True)
    return model


def build_mri_dataset(cfg):
    val = cfg.data.val
    ds_cfg = val.dataset.dataset_configs[0]
    # Mirror ComposedDataset: the MRIDataset needs the shared common_conf injected.
    return instantiate(ds_cfg, common_conf=val.common_config, _recursive_=False)


def reconstruct_cycle(model, mri_ds, subj_idx, do_resp, reference_slot, resp_cfg):
    """Sweep target phase 0..T-1 → (canon_vols, gt_vols, bbox). Mirrors trainer reference path."""
    T_total = mri_ds.gt_grid_shape[0]
    grid_shape = tuple(mri_ds.gt_grid_shape)
    num_slices = mri_ds.num_slices

    data = mri_ds.get_data(seq_index=subj_idx, img_per_seq=num_slices)

    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(DEV)

    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    S = imgs.shape[1]
    batch = {
        "images": imgs,
        "scanner_coords": st("scanner_coords"),
        "z_indices": st("z_indices"),
        "t_indices": st("t_indices"),
    }
    phases_bundle = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(DEV)
    batch["phases"] = phases_bundle.unsqueeze(0)
    batch["timesteps"] = st("timesteps", np.int64)
    batch["slice_indices"] = st("slice_indices", np.int64)
    batch["seq_index"] = torch.tensor([[subj_idx]], dtype=torch.int64, device=DEV)

    bb = np.asarray(data["anatomy_bbox"]).astype(np.int64)
    hw = batch["images"].shape[-1]

    # Non-reference (legacy) path applies breathing once up front; reference path re-applies
    # per phase inside the loop (slot 0 changes). All 3 active models are reference_slot.
    if not reference_slot and do_resp:
        batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=resp_cfg, train=False)

    ref_zmid = (int(bb[0]) + int(bb[1])) // 2
    canon, gt = [], []
    for t in range(T_total):
        t_norm = (t / max(1, T_total)) * 2.0 - 1.0
        batch["target_t_indices"] = torch.full((1, S, 1), t_norm, dtype=torch.float32, device=DEV)
        if reference_slot:
            batch["timesteps"][:, 0] = t  # slot 0 observes the queried phase
            if do_resp:
                batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=resp_cfg, train=False)
            else:
                ref_up = F.interpolate(
                    phases_bundle[t, ref_zmid][None, None].float(), size=(hw, hw),
                    mode="bilinear", align_corners=True)
                batch["images"][:, 0] = ref_up.repeat(1, 3, 1, 1)
        batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            out = compute_volume_intensity_loss(
                {"world_points": preds["world_points"].float()},
                batch, grid_shape=grid_shape, tv_weight=0.0,
            )
        canon.append(out["V_canon"][0].float().cpu().numpy())  # (D, H, W)
        gt.append(out["V_gt"][0].float().cpu().numpy())
    return canon, gt, bb


def pick_zslices(bb, n=N_ZSLICES):
    z0, z1 = int(bb[0]), int(bb[1])
    if z1 - z0 < 2:  # degenerate bbox → fall back to full-depth spread
        z0, z1 = 0, 12
    zs = np.unique(np.linspace(z0, z1 - 1, n).round().astype(int))
    return zs.tolist()


def render_filmstrip(canon, gt, zs, title, path):
    T = len(gt)
    vmax = float(max(max(f.max() for f in canon), max(f.max() for f in gt), 1e-3))
    nrows = 2 * len(zs)  # per z: gt row + canon row
    fig = plt.figure(figsize=(1.15 * T + 0.5, 1.15 * nrows + 0.4), dpi=90)
    grid = _gs.GridSpec(nrows, T + 1, width_ratios=[1.0] * T + [0.04],
                        wspace=0.04, hspace=0.06)
    im = None
    for zi, z in enumerate(zs):
        for t in range(T):
            ax = fig.add_subplot(grid[2 * zi, t])
            ax.imshow(gt[t][z], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if zi == 0:
                ax.set_title(f"t{t}", fontsize=7)
            if t == 0:
                ax.set_ylabel(f"z{z}\nGT", fontsize=6)
            ax2 = fig.add_subplot(grid[2 * zi + 1, t])
            im = ax2.imshow(canon[t][z], cmap="gray", vmin=0, vmax=vmax)
            ax2.set_xticks([]); ax2.set_yticks([])
            if t == 0:
                ax2.set_ylabel("pred", fontsize=6)
    cax = fig.add_subplot(grid[:, T])
    plt.colorbar(im, cax=cax)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def render_gif(canon, gt, zs, path):
    T = len(gt)
    vmax = float(max(max(f.max() for f in canon), max(f.max() for f in gt), 1e-3))
    frames = []
    for t in range(T):
        gt_row = np.concatenate([gt[t][z] for z in zs], axis=1)      # 5 z side by side
        cn_row = np.concatenate([canon[t][z] for z in zs], axis=1)
        montage = np.concatenate([gt_row, cn_row], axis=0)            # GT over pred
        g = np.clip(montage / vmax, 0, 1)
        frames.append(Image.fromarray((g * 255).astype(np.uint8)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=140, loop=0)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, config_name, ckpt in MODELS:
        print(f"\n=== {name} ({config_name}) ===", flush=True)
        if not os.path.exists(ckpt):
            print(f"  skip: ckpt not found {ckpt}", flush=True)
            continue
        cfg = compose_cfg(config_name)
        reference_slot = bool(cfg.get("reference_slot", False))
        resp_cfg = RespiratoryConfig.from_cfg(
            cfg.data.augmentation.get("respiratory", None))
        model = build_model(cfg, ckpt)
        mri_ds = build_mri_dataset(cfg)
        mdir = os.path.join(OUT, name)
        os.makedirs(mdir, exist_ok=True)
        for subj in SUBJECTS:
            for do_resp, tag in [(False, "noresp"), (True, "resp")]:
                canon, gt, bb = reconstruct_cycle(
                    model, mri_ds, subj, do_resp, reference_slot, resp_cfg)
                zs = pick_zslices(bb)
                title = (f"{name} — val subj {subj} — {'with' if do_resp else 'no'} respiration "
                         f"(z={zs}, GT top / pred bottom per z)")
                base = os.path.join(mdir, f"subj{subj:02d}_{tag}")
                render_filmstrip(canon, gt, zs, title, base + "_filmstrip.png")
                render_gif(canon, gt, zs, base + "_beating.gif")
        del model
        torch.cuda.empty_cache()
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
