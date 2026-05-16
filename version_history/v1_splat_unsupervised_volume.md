# v1 — Unsupervised splat-based slice-to-volume

**Date:** 2026-05-08
**Config:** `training/config/mri_volume.yaml`
**Status:** Implemented, unit-tested, not yet run on real data.

## What this version does

Replaces the previous supervised DVF-regression pipeline with a **fully unsupervised intensity-based** slice-to-volume reconstruction. The model architecture is unchanged; only the loss and the interpretation of the point head's output change.

- Point head output (`world_points`, shape `(B, S, H, W, 3)`) is reinterpreted as a **per-pixel predicted position in canonical phase-0 space**, normalized to `[-1, 1]`.
- No GT DVF used in the loss. Carmen/Elastix DVFs are still on disk and still loaded by the dataset (just ignored by the new loss).
- **GT canonical volume is loaded from disk** — the dataset loads the full phase-0 NIfTI (`sax_frame_00.nii.gz`) and resamples it once to the canonical grid `(D=12, H=256, W=256)` using the subject's own `center_mm`/`scale_factor`. The predicted V_canon is splatted to the SAME grid. Voxel-aligned, direct comparison. No splat-fake-GT.
- Cardiac mask is OPTIONAL (config `use_cardiac_mask`, default `false`). For intensity-based loss the agreement-or-disagreement structure of the splat already prevents trivial collapse.
- Target volume resolution: `(D=12, H=256, W=256)` — matches typical CMRxRecon2024 z-coverage. **No through-plane upsampling** (Option 1 from planning).

## Pipeline (end to end)

```
images (B, S, 3, H, W)  +  scanner_coords (B, S, H, W, 3)
        │
        ▼
   Aggregator                                     (z/t embeddings on, residual-DVF path on)
        │
        ▼
   Point head (DPTHead, output_dim=4)              ← unchanged
        │
        ▼
   world_points = scanner_coords + head_dvf        ← residual-DVF formulation
   (B, S, H, W, 3), in [-1, 1]  =  pos_pred
        │
        │  for each pixel: (pos_pred, I_input)
        ▼
   splat_to_volume(pos_pred, I_input, weight=cardiac_mask)
        │       trilinear scatter, gated by cardiac mask
        ▼
   V_canon  (B, D, H, W)  — accumulated intensity / accumulated weight
   coverage (B, D, H, W)  — per-voxel accumulated weight
        │
        ▼
   sample_volume(V_canon, pos_pred) → I_recon  (B, S, H, W)
        │
        ▼
   L = L_intensity + tv_weight · L_pos_tv
        │
        │  L_intensity = |I_input - I_recon|, masked by cardiac mask
        │  L_pos_tv    = TV(pos_pred) along (H, W), masked
        ▼
   backward through grid_sample (sample) and weighted scatter_add_ (splat)
   gradient reaches pos_pred → DPTHead → Aggregator (only Aggregator is frozen by config)
```

## Why it works (in one paragraph)

Pixels from different cardiac phases that show the **same anatomy** must map to the **same canonical voxel** for the resample loss to be small. If the model predicts inconsistent canonical positions, multiple disagreeing intensities collide at the same voxel; the average misses each contributor, and the resample loss is high. Smoothness regularization on `pos_pred` prevents the trivial "scatter every pixel to a unique voxel" failure mode. No GT positions or DVFs are ever compared against — the only supervision is *agreement of intensities at shared predicted positions*.

## Files changed / created

| File | Status | Purpose |
|---|---|---|
| `vggt/utils/splat.py` | **NEW** | `splat_to_volume()` differentiable trilinear scatter; `sample_volume()` thin wrapper over 3D `F.grid_sample` |
| `tests/test_splat.py` | **NEW** | 8 unit tests: forward correctness, roundtrip, agreement averaging, OOB handling, weight gating, gradient flow, batching, constant-field |
| `training/loss.py` | MODIFIED | New `compute_volume_intensity_loss()`; gated existing `point` loss on `weight > 0`; new `volume` slot in `MultitaskLoss.__init__` |
| `training/config/mri_volume.yaml` | **NEW** | Inherits from `mri_finetune.yaml`. Sets `point.weight=0`, enables `volume` loss with `grid_shape=(12, 256, 256)`, `tv_weight=0.1` |
| `training/trainer.py` | MODIFIED | Added V_canon / coverage MIP + I_input/I_recon comparison panel in `_log_tb_visuals` |

**Also modified for the disk-based GT volume + per-axis normalization:**
- `training/data/datasets/mri_dataset.py` —
  - **Switched from cubic to per-axis normalization** for `scanner_coords` and `world_points`. Each axis's `[-1, 1]` now spans that axis's own physical extent: `half_extent = [W·sx/2, H·sy/2, Z·sz/2]`. Canonical voxels approximately match native acquisition resolution (~8mm Z, ~1.34mm X/Y).
  - Loads full phase-0 NIfTI, resamples once to `gt_grid_shape=(12, 256, 256)` via `map_coordinates` using the same per-axis `half_extent`. Emits as `gt_phase0_volume`.
  - `scale_factors` (legacy scalar) now emits `mean(half_extent)` for backward compat with the residual-DVF viz path; unused in the new volume loss.
  - New ctor arg `gt_grid_shape`.
- `training/data/composed_dataset.py` — passes `gt_phase0_volume` through to the batch.

**Untouched** (so the rest of the working tree's WIP doesn't get entangled with v1):
- `vggt/models/vggt.py`
- `vggt/models/aggregator.py`

## What is logged to WandB

### Scalar metrics (every `log_freq` steps, both train and val)

**Per-slice self-consistency (I_input vs I_recon, model's own input reconstruction):**

| Key | Meaning | Range |
|---|---|---|
| `loss_objective` | Total loss | ≥ 0 |
| `loss_intensity` | L1 of `I_input - I_recon` averaged over all pixels (cardiac-masked iff `use_cardiac_mask=true`) | ≥ 0 |
| `loss_pos_tv` | Spatial smoothness of `pos_pred`, weighted by `tv_weight` | ≥ 0 |
| `metric_mae_slice` | Same as `loss_intensity` (alias) | ≥ 0 |
| `metric_mse_slice` | MSE of `(I_input - I_recon)` | ≥ 0 |
| `metric_psnr_slice` | `10·log10(1/MSE_slice)` | dB, higher better |

**3D metrics (V_canon vs V_gt where V_gt = full phase-0 NIfTI resampled by the dataset to the canonical grid):**

| Key | Meaning | Range |
|---|---|---|
| `metric_mae_3d` | `\|V_canon − V_gt\|` averaged over voxels with phase-0 coverage | ≥ 0 |
| `metric_mse_3d` | `(V_canon − V_gt)²` averaged over voxels with phase-0 coverage | ≥ 0 |
| `metric_psnr_3d` | `10·log10(1/MSE_3d)` — the actual volume-reconstruction quality vs GT | dB, higher better |
| `metric_ssim_3d` | 3D SSIM via `fused_ssim3d` on `V_canon * eval_mask` vs `V_gt * eval_mask` | ∈ [-1, 1], higher better. CUDA-only — falls back silently on CPU |
| `metric_gt_coverage_frac` | Fraction of `V_canon` voxels where phase-0 GT exists (limits the eval region) | ∈ [0, 1] |

**Diagnostics:**

| Key | Meaning |
|---|---|
| `metric_mean_disp_norm` | Mean `\|pos_pred − scanner_coords\|_1` — how much the model is "moving" pixels in canonical space (≈ 0 means model is identity, growing means motion correction is happening) |
| `metric_coverage_frac` | Fraction of V_canon voxels with non-zero accumulated weight |
| `metric_coverage_mean` | Mean accumulated weight per voxel — proxy for redundancy at covered voxels |

### Visualizations (every `log_visual_frequency` steps)

1. **`{Train|Val}_Visuals_images`** — input slice grid (carried over from before).
2. **`{Train|Val}_Visuals_Volume`** — 4×3 MIP panel when phase-0 reference is available, else 2×3:
   - Row 0: V_canon (pred) MIPs in three orthogonal projections (axial = max over z, coronal = max over y, sagittal = max over x).
   - Row 1: V_gt (phase-0 splat at native scanner_coords) MIPs, same scaling as V_canon — *direct visual comparison*.
   - Row 2: V_canon − V_gt **mean** projection along each axis (signed, RdBu colormap) — *where pred disagrees with gt*.
   - Row 3: coverage MIPs (viridis colormap). *Where data actually exists*.

3. **`{Train|Val}_Visuals_VolumeSlices`** — 3×D z-strip showing every z-plane:
   - Row 0: V_canon at each z (z=0..D-1, side by side).
   - Row 1: V_gt at each z.
   - Row 2: signed error at each z.

4. **`{Train|Val}_Visuals_Slices`** — 3×S panel showing every input slice:
   - Row 0: I_input for each of the S input slices.
   - Row 1: I_recon for each.
   - Row 2: signed error for each.
   - Caption labels which slot is t=0, t=1, etc.
3. **`{Train|Val}_Visuals_DVF`** — NOT logged in this config because `point.weight=0` disables the path that produces `pred_dvfs`.
4. **3D point cloud panels** — still log when `pred_world_points` is present (they show pos_pred directly as a point cloud, colored by input intensity).

### What is NOT logged

- **Volume saved to disk**: NO. `V_canon` is recomputed each step and only the matplotlib MIP is sent to WandB. To inspect a volume offline, add a callback that dumps `V_canon[0].cpu().numpy()` or writes a NIfTI. ~20 lines, can add when needed.
- **3D SSIM**: NO. Would need a 3D SSIM implementation (e.g. `fused_ssim3d` like MRI2CT, or torchmetrics 3D). Easy follow-up; skipped for first version.
- **Per-slice 2D SSIM**: NO. Skipped — masked MSE/PSNR cover similar ground for the cardiac patch.

## Why 3D PSNR works here

The phase-0 frames are themselves the canonical reference. The dataset loads the full phase-0 NIfTI from disk, resamples it once to the canonical (12, 256, 256) grid using `scipy.ndimage.map_coordinates` and the subject's own `center_mm`/`scale_factor`, and emits it as `gt_phase0_volume`. This is the actual GT canonical volume — dense across all 12 z-planes, sourced from disk, NOT a splat of one slice from the input batch.

The model's `V_canon`, built from ALL phases via `pos_pred` and splatted to the same (12, 256, 256) canonical grid, should agree with this loaded GT. We compute MAE/MSE/PSNR/SSIM over the region where the GT has anatomy (intensity > 0 from the resampled subject volume).

## Volume-shape contract

| Volume | Where it comes from | Shape | Coordinate frame |
|---|---|---|---|
| `V_canon` (pred) | `splat_to_volume(pos_pred, intensity)` in loss | `(B, 12, 256, 256)` | canonical normalized `[-1, 1]` |
| `V_gt` (target) | full phase-0 NIfTI, resampled at dataset load time | `(B, 12, 256, 256)` | canonical normalized `[-1, 1]`, **same subject-specific `center_mm`/`scale_factor`** |

Both live in the same coordinate frame at the same resolution → element-wise comparable.

**Subject native shapes vary** (Z: 8-12, H: 162-246, W: always 256). The (12, 256, 256) canonical grid is chosen to roughly match the max native shape; subjects with smaller native extent occupy the central region of the canonical cube, with zero-padded surrounds (because `map_coordinates` uses `cval=0.0` outside the native volume). The eval region for 3D metrics is `V_gt > 1e-3` so background padding is excluded.

If per-subject native shape matters later (it doesn't right now — the canonical cube is just slightly larger than max native), make `gt_grid_shape` per-subject and pad to batch-max with a valid_mask. Not done in v1.

Diagnostic metrics that supplement the 3D PSNR:
- **`metric_coverage_frac`**: a healthy model should converge to a coverage fraction comparable to (number of distinct anatomical voxels) / (grid voxels). If it stays near 100%, the model is scattering pixels uniformly (trivial). If very low (<5%) with low loss, the model collapsed to a single point.
- **`metric_mean_disp_norm`**: how much canonical-space movement the model has learned. Near zero ⇒ model is acting as identity (`pos_pred = scanner_coords`). Should grow over training when mixed-phase batches are seen.

## How to run

```bash
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config mri_volume max_epochs=1 limit_train_batches=200
```

For overfit-on-one-subject sanity check, override `split_file`:
```bash
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config mri_volume \
    split_file=training/splits/overfit_p001.txt \
    max_epochs=5
```

## Known limitations / open questions

1. **No through-plane upsampling.** V_canon resolution z is 12, matching input z-coverage. To upsample (e.g. to D=64), Option 2 (splat + 3D U-Net in-painter) would be needed — explicitly deferred.
2. **Per-subject `grid_shape` is fixed.** Subjects with Z_total ≠ 12 get points truncated by the splat's in-bounds gate. Acceptable for first version; could be made per-subject in a v2.
3. **No volume export.** V_canon stays GPU-resident, gets dumped as a 2D MIP, then discarded. If you want NIfTI export for offline viewing, add a callback that writes every N epochs.
4. **No GT-volume baseline metric.** No way to say "PSNR is X dB" against a known target. We rely on self-consistency metrics plus visual inspection.
5. **Aggregator is frozen** (`frozen_module_names` includes `*aggregator*` in inherited config). Only the point head trains. If after a few hundred steps the loss stalls, consider unfreezing.
6. **Cardiac mask comes from `mask_frame_00.nii.gz`.** Per CLAUDE.md, this is frame-0 anatomy bounds; pixels that moved outside the mask at other phases lose supervision. Known limitation carried over from the supervised pipeline.

## Verification status

- ✅ 8/8 unit tests in `tests/test_splat.py` pass.
- ✅ End-to-end gradient flow verified: backward from `objective` reaches `pos_pred` with non-zero gradient.
- ✅ Hydra config composes; `loss.point.weight=0` and `loss.volume.weight=1.0` resolve correctly.
- ✅ All modified modules import cleanly.
- ❌ Not yet run on real CMRxRecon2024 data. Next session.

## If something looks wrong on the first real run

| Symptom | Likely cause | Quick check |
|---|---|---|
| `loss_intensity` near zero from step 1, but V_canon MIP looks like noise | Trivial unique-voxel collapse | Bump `loss.volume.tv_weight` from 0.1 → 0.5 |
| `metric_coverage_frac` ≈ 1.0 from step 1 | Model is splattering pixels uniformly across the whole grid | Same — increase TV |
| `metric_coverage_frac` very low and `loss_intensity` very low | Model collapsed to single voxel | Same — increase TV; also check that mixed-phase batches are actually being sampled |
| V_canon MIPs are completely blank | All splatted positions are OOB or all weights are zero | Print `coverage.sum()`; check `point_masks` is not empty |
| Loss explodes / NaN | grid_sample with `padding_mode="zeros"` near boundary, combined with bfloat16 | Disable AMP first (`optim.amp.enabled=false`); if it fixes it, investigate dtype |
| `metric_mean_disp_norm` stays ≈ 0 across phases | Model has collapsed to identity (no motion modeling) | OK for same-phase batches; for mixed-phase, raise `point` head LR or unfreeze aggregator |
