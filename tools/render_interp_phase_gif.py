"""Smoke test: continuous-phase interpolation gif.

The all-phases aggregator-finetune run (wandb fc8d065g) feeds the reconstruction
target phase through a *continuous* Fourier embedding (`target_t_embedder`), so at
inference we can query ANY real t in [0, T), not just the 12 trained integers. This
script builds the scattered input ONCE for one val subject and sweeps a dense set of
fractional target phases, rendering the resulting V_canon mid-z slice as a gif — to
eyeball whether the in-between (untrained) phases interpolate smoothly.

No training-code changes; this only reads the trained checkpoint + dataset.

    micromamba run -n svr python tools/render_interp_phase_gif.py
"""
import os
import time
import numpy as np
import torch
import imageio.v2 as imageio
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vggt.utils.splat import splat_to_volume

CKPT = "scratch/logs/219576158_mri_volume_allphases_aggft/ckpts/checkpoint_last.pt"
OUT_DIR = "_html/assets"
SUBJ_IDX = 0
N_INTERP = 100          # dense fractional phases
FPS_INTERP = 20         # 100 frames / 20 fps = 5 s loop
FPS_DISCRETE = 4
MID_Z = None            # default D//2

# Hydra resolvers (mirror launch.py) so compose() works standalone.
for name, fn in [
    ("rev_ts", lambda: "0"),
    ("basename", lambda p: os.path.basename(p)),
    ("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}"),
]:
    try:
        OmegaConf.register_new_resolver(name, fn)
    except ValueError:
        pass  # already registered


def splat_v_canon(world_points, images, grid_shape):
    """Replicate compute_volume_intensity_loss's splat (no GT needed)."""
    B, S, H, W, _ = world_points.shape
    intensity = images.float().mean(dim=2)
    if intensity.max() > 2.0:
        intensity = intensity / 255.0
    pos_flat = world_points.reshape(B, S * H * W, 3)
    int_flat = intensity.reshape(B, S * H * W)
    splat_weight = (int_flat > 1e-3).to(int_flat.dtype)
    V_canon, _ = splat_to_volume(pos_flat, int_flat, grid_shape, weight=splat_weight)
    return V_canon


def main():
    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)

    # DynamicTorchDataset queries the (single-process) distributed group.
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    with initialize(version_base=None, config_path="../training/config"):
        cfg = compose(config_name="mri_volume")

    # ── Val dataset → inner MRIDataset (walk the wrapper chain like the trainer). ──
    val_ds = instantiate(cfg.data.val, _recursive_=False)
    mri_ds = val_ds.dataset.base_dataset.datasets[0]
    T_total = mri_ds.gt_grid_shape[0]
    grid_shape = tuple(mri_ds.gt_grid_shape)
    num_slices = mri_ds.num_slices
    print(f"[data] T_total={T_total} grid={grid_shape} num_slices={num_slices} "
          f"n_val_subj={len(mri_ds.subjects)}")

    # ── Model. ──
    print("[model] building VGGT + loading checkpoint (cold load ~1 min)…")
    model = instantiate(cfg.model, _recursive_=False).to(device).eval()
    ckpt = torch.load(CKPT, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    step = ckpt.get("steps", ckpt.get("epoch", "?"))
    print(f"[model] loaded step={step} | missing={len(missing)} unexpected={len(unexpected)}")

    # ── Build the scattered input ONCE for the subject. ──
    data = mri_ds.get_data(seq_index=SUBJ_IDX, img_per_seq=num_slices)

    def st(k, dt=np.float32):
        return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(device)

    imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
    S = imgs.shape[1]
    batch = {
        "images": imgs,
        "scanner_coords": st("scanner_coords"),
        "z_indices": st("z_indices"),
        "t_indices": st("t_indices"),
    }
    phases_bundle = torch.from_numpy(
        np.asarray(data["phases"]).astype(np.float32)).to(device)  # (T, D, H, W)
    mid_d = MID_Z if MID_Z is not None else phases_bundle.shape[1] // 2
    amp = torch.bfloat16

    def run_phase(t_float):
        """t_float in [0, T). Returns V_canon mid-z slice (numpy)."""
        t_norm = (t_float / max(1, T_total)) * 2.0 - 1.0
        batch["target_t_indices"] = torch.full(
            (1, S, 1), float(t_norm), dtype=torch.float32, device=device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=amp):
            preds = model(batch["images"], batch=batch)
            V_canon = splat_v_canon(preds["world_points"].float(), batch["images"], grid_shape)
        return V_canon[0, mid_d].float().cpu().numpy()

    # ── (1) Discrete 12 phases: V_gt | V_canon reference. ──
    t0 = time.time()
    discrete_canon, gt_frames = [], []
    for t in range(T_total):
        discrete_canon.append(run_phase(t))
        gt_frames.append(phases_bundle[t, mid_d].float().cpu().numpy())
    print(f"[render] 12 discrete phases done in {time.time()-t0:.1f}s")

    # ── (2) Dense 100 fractional phases: V_canon only (the interpolation). ──
    t0 = time.time()
    interp_phases = np.linspace(0.0, T_total, N_INTERP, endpoint=False)
    interp_canon = [run_phase(float(t)) for t in interp_phases]
    print(f"[render] {N_INTERP} interpolated phases done in {time.time()-t0:.1f}s")

    # ── Shared intensity scale. ──
    vmax = float(max(max(f.max() for f in discrete_canon),
                     max(f.max() for f in gt_frames),
                     max(f.max() for f in interp_canon), 1e-3))

    def to_u8(frame):
        return np.clip(frame / vmax * 255.0, 0, 255).astype(np.uint8)

    # ── GIF A: discrete reference (V_gt left | V_canon right), 12 frames. ──
    discrete_gif = []
    for t in range(T_total):
        side = np.concatenate([to_u8(gt_frames[t]), to_u8(discrete_canon[t])], axis=1)
        discrete_gif.append(np.repeat(side[:, :, None], 3, axis=2))
    path_discrete = os.path.join(OUT_DIR, "interp_discrete12_aggft.gif")
    imageio.mimsave(path_discrete, discrete_gif, fps=FPS_DISCRETE, loop=0)

    # ── GIF B: 100-phase interpolated V_canon. ──
    interp_gif = [np.repeat(to_u8(f)[:, :, None], 3, axis=2) for f in interp_canon]
    path_interp = os.path.join(OUT_DIR, "interp100_aggft.gif")
    imageio.mimsave(path_interp, interp_gif, fps=FPS_INTERP, loop=0)

    print(f"[done] wrote:\n  {path_discrete} ({len(discrete_gif)} frames)\n"
          f"  {path_interp} ({len(interp_gif)} frames)")


if __name__ == "__main__":
    main()
