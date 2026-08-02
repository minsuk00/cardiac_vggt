"""ED-phase (t=0) input-slice vs predicted-volume comparison for the 3 active reference-slot
models (reference / bspline / diffusion), respiration ON — to inspect whether the per-pixel DPT
head's 14-px patch / splat-lattice artifact in V_canon is reduced by the B-spline warp head
(smooth-by-construction) or the L2-diffusion smoothness head.

For each model it queries the ED target phase (t_target=0) on a couple of val subjects with val
breathing applied to the inputs, then writes:
  * per model  : subj{ii}_ED_input_vs_pred.png — the S scattered INPUT slices fed in (top),
                 the predicted V_canon at all 12 canonical z-planes (mid), GT z-planes (bottom)
  * cross-head : subj{ii}_compare_heads_ED.png — pred V_canon z-planes stacked one row per head
                 + a GT row, so the splat/patch artifact compares directly across heads

Run: micromamba run -n svr python tools/render_ed_input_vs_pred_artifact.py
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
from hydra import compose, initialize
from omegaconf import OmegaConf

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

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
# Config names repointed 2026-08-01 (docs/62 §5.5) — see render_cardiac_filmstrip_multislice.py.
MODELS = [
    ("reference", "default",
     f"{LOGS}/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
    ("diffusion", "default",
     f"{LOGS}/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
    ("bspline", "exp_bspline",
     f"{LOGS}/217719798_mri_volume_bspline_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"),
]
SUBJECTS = [0, 7]
T_TARGET = 0               # ED
OUT = os.path.join(_ROOT, "result", "ed_input_vs_pred_artifact")


def compose_cfg(config_name):
    with initialize(version_base=None, config_path=os.path.join("..", "training", "config")):
        return compose(config_name=config_name)


def build_model(cfg, ckpt_path):
    model = instantiate(cfg.model, _recursive_=False).to(DEV).eval()
    ck = torch.load(ckpt_path, map_location=DEV, weights_only=False)
    miss, unexp = model.load_state_dict(ck["model"], strict=False)
    print(f"    loaded {os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))}: "
          f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    return model


def build_mri_dataset(cfg):
    val = cfg.data.val
    ds_cfg = val.dataset.dataset_configs[0]
    return instantiate(ds_cfg, common_conf=val.common_config, _recursive_=False)


def reconstruct_ed(model, mri_ds, subj_idx, reference_slot, resp_cfg):
    """Single ED-phase forward, resp ON. Returns (input_slices, slot_tz, V_canon, V_gt)."""
    grid_shape = tuple(mri_ds.gt_grid_shape)
    T_total = mri_ds.gt_grid_shape[0]
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

    t_norm = (T_TARGET / max(1, T_total)) * 2.0 - 1.0
    batch["target_t_indices"] = torch.full((1, S, 1), t_norm, dtype=torch.float32, device=DEV)
    if reference_slot:
        batch["timesteps"][:, 0] = T_TARGET
    # Respiration ON: deterministic val breathing overwrites the input slices.
    batch = gpu_augment_batch(batch, None, DEV, respiratory_cfg=resp_cfg, train=False)
    batch["gt_target_volume"] = phases_bundle[T_TARGET].unsqueeze(0)

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss(
            {"world_points": preds["world_points"].float()},
            batch, grid_shape=grid_shape, tv_weight=0.0,
        )
    input_slices = batch["images"][0, :, 0].float().cpu().numpy()  # (S, 518, 518) ch0
    slot_t = batch["timesteps"][0].cpu().numpy()
    slot_z = batch["slice_indices"][0].cpu().numpy()
    V_canon = out["V_canon"][0].float().cpu().numpy()
    V_gt = out["V_gt"][0].float().cpu().numpy()
    return input_slices, list(zip(slot_t.tolist(), slot_z.tolist())), V_canon, V_gt


def _grid_of(ax_parent, fig, sub_gs, imgs, vmax, labels=None, row_label=None):
    n = len(imgs)
    for i in range(n):
        ax = fig.add_subplot(sub_gs[i])
        ax.imshow(imgs[i], cmap="gray", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if labels is not None:
            ax.set_title(labels[i], fontsize=6)
        if i == 0 and row_label is not None:
            ax.set_ylabel(row_label, fontsize=8)


def render_per_model(name, inputs, slot_tz, V_canon, V_gt, path):
    D = V_canon.shape[0]
    S = inputs.shape[0]
    vmax_vol = float(max(V_canon.max(), V_gt.max(), 1e-3))
    fig = plt.figure(figsize=(1.0 * max(S, D) + 1.0, 9.0), dpi=110)
    outer = _gs.GridSpec(3, 1, height_ratios=[1, 1, 1], hspace=0.22)
    g_in = _gs.GridSpecFromSubplotSpec(1, S, subplot_spec=outer[0], wspace=0.05)
    g_pr = _gs.GridSpecFromSubplotSpec(1, D, subplot_spec=outer[1], wspace=0.05)
    g_gt = _gs.GridSpecFromSubplotSpec(1, D, subplot_spec=outer[2], wspace=0.05)
    in_labels = [f"t{t},z{z}" for (t, z) in slot_tz]
    _grid_of(None, fig, g_in, [inputs[i] for i in range(S)], 1.0,
             labels=in_labels, row_label="input slices")
    _grid_of(None, fig, g_pr, [V_canon[d] for d in range(D)], vmax_vol,
             labels=[f"z{d}" for d in range(D)], row_label="pred V_canon")
    _grid_of(None, fig, g_gt, [V_gt[d] for d in range(D)], vmax_vol,
             labels=[f"z{d}" for d in range(D)], row_label="GT")
    fig.suptitle(f"{name} — ED (t=0), respiration ON — input slices | pred volume | GT",
                 fontsize=11)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def render_compare_heads(per_head_canon, V_gt, subj, path):
    """Rows = pred V_canon per head + GT; cols = 12 canonical z-planes."""
    D = V_gt.shape[0]
    names = list(per_head_canon.keys())
    vmax = float(max(max(v.max() for v in per_head_canon.values()), V_gt.max(), 1e-3))
    nrows = len(names) + 1
    fig = plt.figure(figsize=(1.05 * D + 0.6, 1.05 * nrows + 0.4), dpi=110)
    grid = _gs.GridSpec(nrows, D + 1, width_ratios=[1.0] * D + [0.04],
                        wspace=0.04, hspace=0.08)
    im = None
    for r, nm in enumerate(names):
        for d in range(D):
            ax = fig.add_subplot(grid[r, d])
            im = ax.imshow(per_head_canon[nm][d], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z{d}", fontsize=7)
            if d == 0:
                ax.set_ylabel(nm, fontsize=8)
    for d in range(D):
        ax = fig.add_subplot(grid[nrows - 1, d])
        ax.imshow(V_gt[d], cmap="gray", vmin=0, vmax=vmax)
        ax.set_xticks([]); ax.set_yticks([])
        if d == 0:
            ax.set_ylabel("GT", fontsize=8)
    cax = fig.add_subplot(grid[:, D]); plt.colorbar(im, cax=cax)
    fig.suptitle(f"val subj {subj} — ED (t=0), respiration ON — pred V_canon per head vs GT "
                 f"(patch/splat artifact check)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def render_zoom(per_head_canon, V_gt, subj, z, path, crop=110):
    """High-res nearest-neighbor crop of ONE z-plane per head + GT — reveals the DPT 14-px
    patch / splat lattice (if present) that bspline/diffusion are meant to remove."""
    names = list(per_head_canon.keys())
    H, W = V_gt[z].shape
    cy, cx = H // 2, W // 2
    y0, y1 = max(0, cy - crop), min(H, cy + crop)
    x0, x1 = max(0, cx - crop), min(W, cx + crop)
    vmax = float(max(max(v[z].max() for v in per_head_canon.values()), V_gt[z].max(), 1e-3))
    panels = [(nm, per_head_canon[nm][z]) for nm in names] + [("GT", V_gt[z])]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), dpi=140)
    for ax, (nm, img) in zip(axes, panels):
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(nm, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"val subj {subj} — ED (t=0), respiration ON — z{z} center crop "
                 f"(nearest-neighbor; patch/splat lattice check)", fontsize=11)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    # subj -> {head_name: V_canon}, and a shared GT per subj
    compare = {s: {} for s in SUBJECTS}
    gt_by_subj = {}
    for name, config_name, ckpt in MODELS:
        print(f"\n=== {name} ({config_name}) ===", flush=True)
        if not os.path.exists(ckpt):
            print(f"  skip: ckpt not found {ckpt}", flush=True)
            continue
        cfg = compose_cfg(config_name)
        reference_slot = bool(cfg.get("reference_slot", False))
        resp_cfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.get("respiratory", None))
        model = build_model(cfg, ckpt)
        mri_ds = build_mri_dataset(cfg)
        mdir = os.path.join(OUT, name)
        os.makedirs(mdir, exist_ok=True)
        for subj in SUBJECTS:
            inputs, slot_tz, V_canon, V_gt = reconstruct_ed(
                model, mri_ds, subj, reference_slot, resp_cfg)
            render_per_model(name, inputs, slot_tz, V_canon, V_gt,
                             os.path.join(mdir, f"subj{subj:02d}_ED_input_vs_pred.png"))
            compare[subj][name] = V_canon
            gt_by_subj[subj] = V_gt
        del model
        torch.cuda.empty_cache()
    for subj in SUBJECTS:
        if not compare[subj]:
            continue
        V_gt = gt_by_subj[subj]
        render_compare_heads(compare[subj], V_gt, subj,
                             os.path.join(OUT, f"subj{subj:02d}_compare_heads_ED.png"))
        # Heart-center plane = max GT slice energy → most structure for the lattice check.
        z_star = int(np.argmax([V_gt[d].sum() for d in range(V_gt.shape[0])]))
        render_zoom(compare[subj], V_gt, subj, z_star,
                    os.path.join(OUT, f"subj{subj:02d}_zoom_z{z_star}_ED.png"))
        # Cache volumes so future zoom/analysis needs no GPU re-run.
        np.savez_compressed(
            os.path.join(OUT, f"subj{subj:02d}_volumes.npz"),
            gt=V_gt, **{nm: v for nm, v in compare[subj].items()})
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
