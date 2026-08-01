#!/usr/bin/env python
"""In-distribution CMRxRecon val inference — the standalone counterpart to inference/run_rtfb.py.

Loads the reference-slot z-only model (docs/25) and builds one real CMRxRecon val subject's
DEPLOYMENT-REALISTIC multi-frame batch (docs/28), mirroring inference/run_rtfb.py exactly but
sourced from the canonical cache. Training samples companions randomly because it is capped by
the S-slot GPU budget (`MRIDataset.get_data`); inference is NOT budget-limited, so — as in
run_rtfb — we deterministically feed what a genuinely short real-time acquisition would record:
the mid-ventricular reference plane contributes ALL its cardiac phases (swept as the query),
and every other in-bbox plane contributes its first `--frames-per-slice` consecutive phases
(short-burst sim). Unlike the OOD real-time datasets, CMRxRecon HAS ground truth, so this
reports PSNR per phase. Runs WITH and WITHOUT the simulated respiratory corruption (docs/01,
docs/05) on the input slices — target/GT stay at the unshifted reference either way, so this is
the clean-vs-breathing-corrupted comparison.

Usage:
  micromamba run -n svr python inference/run_cmrxrecon.py --subjects 0 7 --ckpt PATH
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from inference.inference import load_rtfb_model_reference
from inference.adapters.base import DEFAULT_CKPT_REFERENCE, INPUT_IMG_SIZE, MM_PER_NORM

CANON_SPACING = (1.4, 1.4, 12.0)  # x,y,z mm — true CMRx pitch (preprocess.TARGET_SPACING; docs/27)


def build_mri_dataset():
    """Instantiate the mri_volume.yaml val dataset standalone (no live Trainer needed) —
    same construction trainer.py uses internally."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29564")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    OmegaConf.register_new_resolver("rev_ts", lambda: "0", replace=True)
    OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p), replace=True)
    OmegaConf.register_new_resolver("phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}", replace=True)
    with initialize_config_dir(version_base=None, config_dir=os.path.join(_ROOT, "training", "config")):
        cfg = compose(config_name="default")
    from data.respiratory import RespiratoryConfig
    rcfg = RespiratoryConfig.from_cfg(cfg.data.augmentation.respiratory)
    val = instantiate(cfg.data.val, _recursive_=False)
    mri_ds = val.dataset.base_dataset.datasets[0]
    return mri_ds, rcfg


def _build_multiframe_batch(phases_bundle, bbox, frames_per_slice, seq_index, device):
    """Deployment-realistic multi-frame + reference-slot batch built straight from the canonical
    cache (docs/28), the in-distribution twin of inference.adapters.base._build_batch_multiframe_core.
    No random sampling — that's a TRAINING-only augmentation forced by the S-slot GPU budget; at
    inference we deterministically use what a short real-time acquisition would record.

    Slot 0 = swept reference placeholder (overwritten per phase). The reference plane (z_mid)
    contributes ALL T phases as companions; every OTHER in-bbox plane contributes a short
    consecutive burst of `frames_per_slice` phases starting at a RANDOM phase (cyclic, seeded by
    seq_index so clean/breathing match and runs reproduce). Random-start (not always phase 0)
    matters because a real short acquisition of a slice lands at an arbitrary point in the cycle
    — if every plane started at phase 0, ED (phase 0) would be cleanly observed everywhere and
    the ED reconstruction would look artificially easy. Companions are held constant across the
    sweep. -> (batch, z_mid).
    """
    T, D, H, W = phases_bundle.shape
    z0, z1 = int(bbox[0]), int(bbox[1])
    z_mid = (z0 + z1) // 2
    in_bbox_z = list(range(z0, z1)) or [z_mid]
    rng = np.random.default_rng(seq_index)   # reproducible per subject; identical for clean/breathing

    # (z_plane, phase) per slot.
    slots = [(z_mid, 0)]                                   # slot 0: overwritten each sweep step
    slots += [(z_mid, t) for t in range(T)]               # full reference-plane cine as companions
    n = min(frames_per_slice, T)
    for z in in_bbox_z:
        if z == z_mid:
            continue
        s0 = int(rng.integers(T))                         # random burst START (cyclic), NOT phase 0
        slots += [(z, (s0 + k) % T) for k in range(n)]
    S = len(slots)

    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (hw - 1) * 2.0 - 1.0).astype(np.float32)

    slot_t = torch.tensor([t for _, t in slots], dtype=torch.long)
    slot_z = torch.tensor([z for z, _ in slots], dtype=torch.long)
    canon = phases_bundle[slot_t, slot_z].unsqueeze(1)    # (S,1,256,256) in [0,1]
    up = F.interpolate(canon, size=(hw, hw), mode="bilinear", align_corners=True).squeeze(1)  # (S,518,518)
    images = up.unsqueeze(1).repeat(1, 3, 1, 1)           # (S,3,518,518)

    coords, z_idx = [], []
    for z, _t in slots:
        z_val = z / max(1, D - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        z_idx.append([z_val])

    batch = {
        "images": images.unsqueeze(0).to(device).float(),                                # (1,S,3,518,518)
        "scanner_coords": torch.from_numpy(np.stack(coords)).unsqueeze(0).to(device),     # (1,S,518,518,3)
        "z_indices": torch.tensor(z_idx, dtype=torch.float32).unsqueeze(0).to(device),    # (1,S,1)
        "timesteps": slot_t.view(1, S).to(device),                                        # (1,S) int64
        "slice_indices": slot_z.float().view(1, S).to(device),                            # (1,S)
        "phases": phases_bundle.unsqueeze(0),                                             # (1,T,D,H,W)
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
        # (1,6) geometric content bbox (z0,z1,y0,y1,x0,x1) — enables bbox PSNR in
        # compute_volume_intensity_loss (mirrors what MRIDataset.get_data emits).
        "anatomy_bbox": torch.as_tensor(bbox[:6], dtype=torch.int64).view(1, 6).to(device),
    }
    return batch, z_mid


_METRIC_KEYS = {
    "full": "metric_psnr_3d_full",
    "bbox": "metric_psnr_3d_bbox",
    "motion": "metric_psnr_3d_motion",
    "ssim": "metric_ssim_3d_full",
}

ED_PHASE = 0   # CMRxRecon is ED-anchored (docs/17): phase 0 = end-diastole
# Δ mm scale = MM_PER_NORM (the splat's align-corners factor 0.5*(N-1)*spacing = (178.5,178.5,66.0)),
# NOT the physical cube half-extent — a normalized displacement of 1 spans 0.5*(N-1) voxels, not 0.5*N.


@torch.no_grad()
def reconstruct_from_bundle(model, phases_bundle, bbox, rcfg, seq_index, breathing, device,
                            grid_shape, frames_per_slice=5):
    """Core of `reconstruct_cycle`, parameterized by an already-built canonical phase bundle +
    geometric bbox (so an OOD gated adapter can drive the SAME protocol as CMRxRecon — see
    inference/run_gated_ood.py). Sweep the reference slot (slot 0) over the bundle's T phases (docs/25)
    using the deployment-realistic multi-frame batch, with/without the simulated respiratory
    corruption on the INPUT slices — target/GT stay at the unshifted reference either way (the
    model is meant to correct breathing, not see it in the GT). Companions (slots 1..S-1) are
    built once and reused across all phases.

    Metrics come from the SAME `compute_volume_intensity_loss` the trainer uses, so full/bbox/
    motion PSNR + SSIM are defined identically to training (motion = voxels with
    max_t-min_t > MOTION_MASK_TAU; loss.py). Full per-phase pred/GT cubes (splat-order D,H,W) are
    always returned (for the multi-slice GIF, ED montage, and --dump-volumes), plus `ed_pack` —
    the ED-phase forward internals (input images, predicted Δ, per-slot z/phase, applied
    breathing displacement) for the ED input / DVF panels. Reference slot = 0 (the swept query).

    phases_bundle: (T,D,H,W) torch on `device`. grid_shape: (D,H,W) splat grid.
    -> dict(metrics={full,bbox,motion,ssim:[per-phase]}, pred_vols (T,D,H,W), gt_vols (T,D,H,W),
            bbox, z_mid, ed_pack).
    """
    from data.gpu_aug import gpu_augment_batch
    from loss import compute_volume_intensity_loss
    T = phases_bundle.shape[0]   # number of cardiac phases (the sweep length)
    bbox = np.asarray(bbox).astype(np.int64)
    batch, z_mid = _build_multiframe_batch(phases_bundle, bbox, frames_per_slice, seq_index, device)
    hw = batch["images"].shape[-1]

    metrics = {k: [] for k in _METRIC_KEYS}
    pred_vols, gt_vols, ed_pack = [], [], None
    for t in range(T):
        batch["timesteps"][:, 0] = t   # slot 0 observes the queried target phase t
        if breathing:
            batch = gpu_augment_batch(batch, None, device, respiratory_cfg=rcfg, train=False)
        else:
            ref_up = F.interpolate(phases_bundle[t, z_mid][None, None].float(), size=(hw, hw),
                                   mode="bilinear", align_corners=True)  # (1,1,518,518) in [0,1]
            batch["images"][:, 0] = ref_up.repeat(1, 3, 1, 1)
        batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)   # (1,D,H,W) = V_gt at phase t
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            out = compute_volume_intensity_loss(
                {"world_points": preds["world_points"].float()},
                batch, grid_shape=grid_shape, tv_weight=0.0)
        for k, mk in _METRIC_KEYS.items():
            metrics[k].append(float(out[mk]) if mk in out else float("nan"))
        pred_vols.append(out["V_canon"][0].float().cpu().numpy())
        gt_vols.append(out["V_gt"][0].float().cpu().numpy())
        if t == ED_PHASE:
            delta = (preds["world_points"][0].float() - batch["scanner_coords"][0].float()).cpu().numpy()
            resp = batch.get("resp_disp_mm")
            ed_pack = dict(
                images=batch["images"][0].mean(1).cpu().numpy(),         # (S,hw,hw) the input model saw
                delta=delta,                                             # (S,hw,hw,3) normalized Δ
                slice_z=batch["slice_indices"][0].cpu().numpy(),         # (S,) z-plane per slot
                timesteps=batch["timesteps"][0].cpu().numpy(),           # (S,) phase per slot
                resp_disp_mm=(resp[0].cpu().numpy() if resp is not None else None),  # (S,3) or None
            )
    return dict(metrics=metrics, pred_vols=np.stack(pred_vols), gt_vols=np.stack(gt_vols),
                bbox=bbox, z_mid=z_mid, ed_pack=ed_pack)


def reconstruct_cycle(model, mri_ds, rcfg, seq_index, breathing, device, frames_per_slice=5):
    """In-distribution CMRxRecon wrapper around `reconstruct_from_bundle`: fetch the canonical
    phase bundle + geometric bbox from the val dataset (`MRIDataset.get_data` — its own slot
    sampling is discarded, inference builds deterministic multi-frame slots), then run the shared
    core. Bit-identical to the pre-refactor path."""
    grid_shape = tuple(mri_ds.gt_grid_shape)   # (D,H,W) for the splat
    data = mri_ds.get_data(seq_index=seq_index, img_per_seq=mri_ds.num_slices)
    phases_bundle = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(device)  # (T,D,H,W)
    bbox = np.asarray(data["anatomy_bbox"]).astype(np.int64)
    return reconstruct_from_bundle(model, phases_bundle, bbox, rcfg, seq_index, breathing, device,
                                   grid_shape, frames_per_slice=frames_per_slice)


# ── Visualization helpers ────────────────────────────────────────────────────
def _planes_across_bbox(bbox, D, n=5):
    """n z-planes evenly spanning the anatomy bbox (clipped to the volume)."""
    z0, z1 = int(bbox[0]), int(bbox[1])
    if z1 <= z0:
        z0, z1 = 0, D
    return np.unique(np.clip(np.linspace(z0, z1 - 1, n).round().astype(int), 0, D - 1))


def _fig_to_pil(fig):
    """Render a matplotlib figure to a PIL RGB image (version-proof via a PNG buffer)."""
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _ed_slots(ed_pack, bbox, z_mid):
    """One representative input slot per in-bbox plane; the reference plane maps to slot 0 (the
    swept query, at ED). Non-reference planes are observed at their OWN random burst phase (not
    ED), so we take that plane's first companion regardless of phase — the panels label each with
    its true phase so the non-ED inputs are honest. -> list of (z_plane, slot_index, is_reference)."""
    zc = np.round(ed_pack["slice_z"]).astype(int)
    z0, z1 = int(bbox[0]), int(bbox[1])
    planes = list(range(z0, z1)) or [z_mid]
    out = []
    for z in planes:
        if z == z_mid:
            out.append((z, 0, True))                       # slot 0 = swept reference query (at ED)
            continue
        cand = np.where(zc == z)[0]                        # this plane's input (its own burst phase)
        if len(cand):
            out.append((z, int(cand[0]), False))
    return out


def save_multislice_gif(pred_vols, gt_vols, bbox, path, n_slices=5):
    """2 rows (GT top / pred bottom) × n z-planes spanning the anatomy bbox, animated over the
    cardiac cycle. Only the mid plane is given (as the reference frame); the others are inferred,
    so this is the honest cycle view — unlike a single mid-slice, which the model gets for free."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, D = pred_vols.shape[0], pred_vols.shape[1]
    planes = _planes_across_bbox(bbox, D, n_slices)
    vmax = float(max(gt_vols[:, planes].max(), pred_vols[:, planes].max(), 1e-3))
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(2, len(planes), figsize=(1.6 * len(planes), 3.6), squeeze=False)
        for r, (vol, name) in enumerate(((gt_vols, "GT"), (pred_vols, "pred"))):
            for c, z in enumerate(planes):
                ax = axes[r][c]
                ax.imshow(vol[t, z], cmap="gray", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"z={z}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(name, fontsize=11)
        fig.suptitle(f"phase t={t}", fontsize=9)
        fig.tight_layout()
        frames.append(_fig_to_pil(fig)); plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def save_ed_montage(pred_vol, gt_vol, path):
    """All D z-planes at ED: pred / GT / |diff| rows (the full reconstructed ED volume vs truth)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = pred_vol.shape[0]
    vmax = float(max(pred_vol.max(), gt_vol.max(), 1e-3))
    diff = np.abs(pred_vol - gt_vol); dmax = float(max(diff.max(), 1e-3))
    rows = [("pred", pred_vol, vmax, "gray"), ("GT", gt_vol, vmax, "gray"),
            ("|diff|", diff, dmax, "magma")]
    fig, axes = plt.subplots(3, D, figsize=(1.4 * D, 4.4), squeeze=False)
    for r, (name, vol, vm, cmap) in enumerate(rows):
        for z in range(D):
            ax = axes[r][z]
            ax.imshow(vol[z], cmap=cmap, vmin=0, vmax=vm)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z={z}", fontsize=7)
            if z == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle(f"{os.path.basename(path)}  ED (t={ED_PHASE})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_ed_input_png(ed_pack, bbox, z_mid, path):
    """Input slice fed at ED for each in-bbox plane; the REFERENCE slot (slot 0) is boxed/red.
    Under breathing these are the respiratory-shifted inputs the model must correct."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slots = _ed_slots(ed_pack, bbox, z_mid)
    imgs = ed_pack["images"]; ts = np.round(ed_pack["timesteps"]).astype(int)
    n = len(slots); cols = min(n, 6); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_frame_on(False)
    for i, (z, s, is_ref) in enumerate(slots):
        ax = axes[i // cols][i % cols]
        ax.imshow(imgs[s], cmap="gray")
        ax.set_title(f"z={z} t={ts[s]}" + ("  [REF]" if is_ref else ""), fontsize=8,
                     color=("red" if is_ref else "black"))
        if is_ref:
            ax.set_frame_on(True)
            for sp in ax.spines.values():
                sp.set_color("red"); sp.set_linewidth(3)
    fig.suptitle(f"{os.path.basename(path)}  inputs (red = reference at ED; others at their own "
                 f"burst phase t)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_ed_dvf_png(ed_pack, bbox, z_mid, path, breathing):
    """Predicted Δ (x/y/z, mm) at ED for one slot per in-bbox plane (input row + 3 Δ rows).
    Under breathing the title reports the APPLIED breathing displacement amplitude — what the
    model's Δ should be undoing — so predicted correction vs applied shift is directly comparable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slots = _ed_slots(ed_pack, bbox, z_mid)
    imgs = ed_pack["images"]; delta = ed_pack["delta"]; ts = np.round(ed_pack["timesteps"]).astype(int)
    labels = ["Δx (mm)", "Δy (mm)", "Δz (mm)"]
    n = len(slots)
    fig, axes = plt.subplots(4, n, figsize=(1.7 * n, 7.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for i, (z, s, is_ref) in enumerate(slots):
        up = imgs[s]
        axes[0][i].imshow(up, cmap="gray")
        axes[0][i].set_title(f"z={z} t={ts[s]}" + ("*" if is_ref else ""), fontsize=8,
                             color=("red" if is_ref else "black"))
        for c in range(3):
            dm = delta[s, ..., c] * MM_PER_NORM[c]
            # Show the ENTIRE Δ field (no intensity mask — that hid low-signal anatomy too).
            # vlim from the 99th pct over the whole slice so a few outliers don't wash it out;
            # a very light input overlay gives anatomical context without occluding the field.
            vlim = max(float(np.percentile(np.abs(dm), 99)), 1e-3)
            im = axes[c + 1][i].imshow(dm, cmap="bwr", vmin=-vlim, vmax=vlim)
            axes[c + 1][i].imshow(up, cmap="gray", alpha=0.15)   # faint anatomy overlay
            if i == 0:
                axes[c + 1][i].set_ylabel(labels[c], fontsize=9)
            if i == n - 1:
                fig.colorbar(im, ax=axes[c + 1][i], fraction=0.046, pad=0.02)
    axes[0][0].set_ylabel("input", fontsize=9)
    title = f"{os.path.basename(path)}  predicted Δ at ED (* = reference slot)"
    resp = ed_pack.get("resp_disp_mm")
    if breathing and resp is not None:
        amp = np.abs(resp).mean(0)   # mean |disp| over slots; canonical (D=z/SI, H=y/AP, W=x) mm
        title += (f"\napplied breathing |disp| mean: z(SI)={amp[0]:.1f}  "
                  f"y(AP)={amp[1]:.1f}  x={amp[2]:.1f} mm")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_nnunet_nii(vol_dhw, path):
    """Full volume (D=Z,H=Y,W=X) splat order -> nnU-Net (X,Y,Z) NIfTI with canonical spacing.
    Matches tools/nnunet_mnms_eval/prep_inputs.py so the seg stage consumes it identically."""
    import nibabel as nib
    arr = np.transpose(np.asarray(vol_dhw, np.float32), (2, 1, 0))     # (D,H,W)->(X,Y,Z)
    nib.save(nib.Nifti1Image(arr, np.diag([*CANON_SPACING, 1.0])), path)


def _nanmean(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(xs)) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_REFERENCE)
    ap.add_argument("--frames-per-slice", type=int, default=5,
                    help="first N consecutive phases per non-reference in-bbox plane (short-burst sim)")
    ap.add_argument("--refiner", action="store_true", help="model has a coverage refiner head")
    ap.add_argument("--subjects", nargs="*", type=int, default=[0, 7], help="val seq_index list")
    ap.add_argument("--out", default="result/cmrxrecon_eval")
    ap.add_argument("--dump-volumes", default=None,
                    help="dir to write per-phase pred/GT NIfTIs for the seg-metric stage "
                         "(EF/Dice via inference/seg_metrics_cmrxrecon.py). GT is written once per subject.")
    ap.add_argument("--metrics-json", default=None,
                    help="path to write the per-subject/per-mode PSNR+SSIM summary (default: <out>/metrics.json)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.dump_volumes:
        os.makedirs(args.dump_volumes, exist_ok=True)
    device = torch.device("cuda")
    model = load_rtfb_model_reference(args.ckpt, refiner=args.refiner, device=device)
    mri_ds, rcfg = build_mri_dataset()

    summary = []
    for seq_index in args.subjects:
        for breathing, tag in [(False, "clean"), (True, "breathing")]:
            res = reconstruct_cycle(
                model, mri_ds, rcfg, seq_index, breathing, device,
                frames_per_slice=args.frames_per_slice)
            base = os.path.join(args.out, f"subj{seq_index}_{tag}")
            # Multi-slice beating-heart GIF (2×5, GT/pred × spanning planes) + ED-phase panels.
            save_multislice_gif(res["pred_vols"], res["gt_vols"], res["bbox"], base + "_cycle.gif")
            save_ed_montage(res["pred_vols"][ED_PHASE], res["gt_vols"][ED_PHASE], base + "_ED.png")
            save_ed_input_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_input.png")
            save_ed_dvf_png(res["ed_pack"], res["bbox"], res["z_mid"], base + "_ED_dvf.png", breathing)
            m = res["metrics"]
            means = {k: _nanmean(v) for k, v in m.items()}
            summary.append(dict(subject=int(seq_index), mode=tag, per_phase=m, mean=means))
            print(f"[subj{seq_index} {tag}] "
                  f"motion={means['motion']:.2f}  bbox={means['bbox']:.2f}  "
                  f"full={means['full']:.2f}dB  ssim={means['ssim']:.3f}  "
                  f"(motion/phase={['%.1f' % p for p in m['motion']]}) "
                  f"-> {base}_cycle.gif", flush=True)

            if args.dump_volumes:
                for t in range(res["pred_vols"].shape[0]):
                    save_nnunet_nii(res["pred_vols"][t], os.path.join(
                        args.dump_volumes, f"subj{seq_index}_{tag}_pred_t{t:02d}_0000.nii.gz"))
                if not breathing:  # GT is mode-independent (target stays unshifted) -> dump once
                    for t in range(res["gt_vols"].shape[0]):
                        save_nnunet_nii(res["gt_vols"][t], os.path.join(
                            args.dump_volumes, f"subj{seq_index}_gt_t{t:02d}_0000.nii.gz"))

    mpath = args.metrics_json or os.path.join(args.out, "metrics.json")
    with open(mpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote metrics -> {mpath}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
