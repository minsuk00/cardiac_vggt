#!/usr/bin/env python
"""Decisive mechanism test for WHY fps=5 > fps=1. Compares three input constructions per subject
on the reference-slot model 4wokxzov, measuring motion PSNR:

  fps1            : 1 frame per non-ref plane (baseline)
  fps5_distinct   : 5 frames per non-ref plane at 5 DIFFERENT phases (the normal multi-frame setup)
  fps5_dup        : 5 frames per non-ref plane all at the SAME phase (duplicated)

Logic:
  * CLEAN: fps5_dup feeds 5 IDENTICAL slices -> no new cardiac-phase info AND no averaging benefit
    (identical content). If fps5_dup ~= fps1 while fps5_distinct > fps1, the clean gain is
    NEW-PHASE COVERAGE, not averaging.
  * BREATHING: respiratory shift is sampled per-slot iid, so 5 duplicated (z,t) slices still get 5
    DIFFERENT breathing displacements. If fps5_dup ~= fps5_distinct >> fps1, the breathing gain is
    NOISE AVERAGING of the respiratory corruption (independent of cardiac phase).

  micromamba run -n svr python tools/exp_mechanism.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))

from eval.inference import load_rtfb_model_reference
from eval.run_cmrxrecon import build_mri_dataset
from eval.adapters.base import INPUT_IMG_SIZE

SNAPSHOT = "/tmp/vggt_4wokxzov_snapshot.pt"


def build_batch(phases_bundle, bbox, fps, seq_index, device, dup):
    """Same slots as run_cmrxrecon._build_multiframe_batch, but `dup=True` makes each non-ref
    plane's burst all the SAME phase instead of `fps` distinct consecutive phases."""
    T, D, H, W = phases_bundle.shape
    z0, z1 = int(bbox[0]), int(bbox[1]); z_mid = (z0 + z1) // 2
    in_bbox_z = list(range(z0, z1)) or [z_mid]
    rng = np.random.default_rng(seq_index)
    slots = [(z_mid, 0)] + [(z_mid, t) for t in range(T)]
    n = min(fps, T)
    for z in in_bbox_z:
        if z == z_mid:
            continue
        s0 = int(rng.integers(T))
        if dup:
            slots += [(z, s0) for _ in range(n)]                 # SAME phase ×n
        else:
            slots += [(z, (s0 + k) % T) for k in range(n)]       # n distinct phases
    S = len(slots)
    hw = INPUT_IMG_SIZE
    py, px = np.meshgrid(np.arange(hw), np.arange(hw), indexing="ij")
    x_norm = (px / (hw - 1) * 2 - 1).astype(np.float32); y_norm = (py / (hw - 1) * 2 - 1).astype(np.float32)
    slot_t = torch.tensor([t for _, t in slots]); slot_z = torch.tensor([z for z, _ in slots])
    canon = phases_bundle[slot_t, slot_z].unsqueeze(1)
    up = F.interpolate(canon, size=(hw, hw), mode="bilinear", align_corners=True).squeeze(1)
    images = up.unsqueeze(1).repeat(1, 3, 1, 1)
    coords, z_idx = [], []
    for z, _t in slots:
        zv = z / max(1, D - 1) * 2 - 1
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, zv)], -1)); z_idx.append([zv])
    return {
        "images": images.unsqueeze(0).to(device).float(),
        "scanner_coords": torch.from_numpy(np.stack(coords)).unsqueeze(0).to(device),
        "z_indices": torch.tensor(z_idx, dtype=torch.float32).unsqueeze(0).to(device),
        "timesteps": slot_t.view(1, S).to(device),
        "slice_indices": slot_z.float().view(1, S).to(device),
        "phases": phases_bundle.unsqueeze(0),
        "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=device),
        "anatomy_bbox": torch.as_tensor(bbox[:6], dtype=torch.int64).view(1, 6).to(device),
    }, z_mid


@torch.no_grad()
def motion_psnr(model, mri_ds, rcfg, seq_index, mode, fps, dup, device):
    from data.gpu_aug import gpu_augment_batch
    from loss import compute_volume_intensity_loss
    grid = tuple(mri_ds.gt_grid_shape)
    data = mri_ds.get_data(seq_index=seq_index, img_per_seq=mri_ds.num_slices)
    ph = torch.from_numpy(np.asarray(data["phases"]).astype(np.float32)).to(device)
    T = ph.shape[0]; bbox = np.asarray(data["anatomy_bbox"]).astype(np.int64)
    batch, z_mid = build_batch(ph, bbox, fps, seq_index, device, dup)
    hw = batch["images"].shape[-1]; breathing = mode == "breathing"
    vals = []
    for t in range(T):
        batch["timesteps"][:, 0] = t
        if breathing:
            batch = gpu_augment_batch(batch, None, device, respiratory_cfg=rcfg, train=False)
        else:
            ref = F.interpolate(ph[t, z_mid][None, None].float(), size=(hw, hw),
                                mode="bilinear", align_corners=True)
            batch["images"][:, 0] = ref.repeat(1, 3, 1, 1)
        batch["gt_target_volume"] = ph[t].unsqueeze(0)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
            out = compute_volume_intensity_loss({"world_points": preds["world_points"].float()},
                                                batch, grid_shape=grid, tv_weight=0.0)
        vals.append(float(out["metric_psnr_3d_motion"]))
    torch.cuda.empty_cache()
    return float(np.mean(vals))


def main():
    device = torch.device("cuda")
    model = load_rtfb_model_reference(SNAPSHOT, refiner=False, device=device)
    mri_ds, rcfg = build_mri_dataset()
    print(f"[breathing sim] group_by_burst={rcfg.group_by_burst} "
          f"(True = realistic shared-breath-per-slice; False = legacy per-frame iid)", flush=True)
    print("subj mode       fps1   fps5_dup  fps5_distinct   |  gain_distinct  gain_dup  dup/distinct%")
    cases = [(3, "clean"), (7, "breathing"), (7, "clean"), (3, "breathing")]  # decisive two first
    for seq_index, mode in cases:
        f1 = motion_psnr(model, mri_ds, rcfg, seq_index, mode, 1, False, device)
        fd = motion_psnr(model, mri_ds, rcfg, seq_index, mode, 5, True, device)
        fx = motion_psnr(model, mri_ds, rcfg, seq_index, mode, 5, False, device)
        gx, gd = fx - f1, fd - f1
        frac = 100 * gd / gx if abs(gx) > 1e-6 else float("nan")
        print(f"{seq_index:>3} {mode:9} {f1:6.2f}  {fd:7.2f}   {fx:10.2f}     |  "
              f"{gx:+.2f}         {gd:+.2f}      {frac:5.0f}%", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
