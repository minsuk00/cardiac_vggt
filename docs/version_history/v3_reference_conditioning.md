# v3 — Target-phase reference-slice conditioning (replaces the target_t index)

**Date:** 2026-06-23
**Config:** `training/config/mri_volume.yaml` (now the reference + aggft default)
**Status:** Implemented, unit-tested (new `tests/test_reference_conditioning.py` + updated `tests/test_freeze_pattern.py`); local smoke pending. NOT yet run to convergence — `sbatch/train_mri_volume_reference.sh` prepared (fresh-from-base), not launched. Confirmation = pred-EF-vs-true slope off ~0 via `tools/cmrxrecon_phase_analysis/`.

## Why

The model conditioned on a content-free `target_t` **index** (broadcast `target_t_embedder`), which caused the **flat-EF amplitude regression** (every patient reconstructed at ~48% EF, slope ≈ 0; docs/24) AND the `target_t=k/12` phase-timing ambiguity (docs/17). Root cause: an index can encode neither per-patient *state* nor per-patient *content*. Fix (docs/25): condition on a real **target-phase reference slice** using VGGT's native first-frame-anchor design.

## What changed (flag-gated; legacy path preserved)

Two new flags, both default **off** (so `mri_finetune.yaml` / archived configs are bit-identical); `mri_volume.yaml` turns them on.

- **Dataset** — `MRIDataset(reference_slot=bool)` (`training/data/datasets/mri_dataset.py`). When True, `get_data` forces **slot 0 = `(t_target, mid-ventricular z = (bbox_z0+bbox_z1)//2)`**; slots 1..S-1 stay scattered but draw z from the in-bbox planes **excluding** the reference plane. Val determinism, static mode, and the `n_forced_target` ablation hook are preserved. `gt_target_volume = phases[t_target]` and the (now inert) `target_t_indices` emission are unchanged.
- **Model** — `Aggregator(use_reference_token=bool)` + `VGGT(use_reference_token=bool)` (`vggt/models/aggregator.py`, `vggt/models/vggt.py`). When True, the per-slot conditioning token gains the native `slice_expand_and_flatten(self.camera_token, B, S)` (index 0 = slot 0 anchor, index 1 = the rest) — **no new module** (reuses the pretrained `camera_token`). The model reads the target phase from slot-0's image content; the anchor just marks *which* slot is the reference.
- **Config** — `mri_volume.yaml`: `use_reference_token=true`, `reference_slot=true`, `use_z_pose_embedding=true`, `use_t_pose_embedding=false`, `use_target_t_pose_embedding=false`; **aggft** (`optim.frozen_module_names=["*patch_embed*"]`) + `distributed.find_unused_parameters=true` so camera_token/z_embedder/attention specialize. Flag defaults added to `mri_finetune.yaml`. (Cascades to `mri_volume_bspline/diffusion`, which inherit `mri_volume`.)
- **Trainer** — `_log_cardiac_cycle_filmstrip` rebuilds slot 0 per swept phase (image = `phases_bundle[t, z_mid]`) in reference mode, since a single fixed-input sweep can't reconstruct an arbitrary phase here. `self.reference_slot` cached at init.
- **sbatch** — `sbatch/train_mri_volume_reference.sh` warm-starts **fresh from base VGGT-1B** (config default resume path, strict=false), aggft, `max_epochs=500`, requeue-safe. Not submitted.

## Unchanged
`training/loss.py` (no target_t dependency); `training/data/gpu_aug.py` (affine re-extracts slot 0 from `phases_aug` automatically; respiratory leaves slot 0 — moot for the clean first retrain); `training/data/composed_dataset.py` (`target_t_indices` kept inert).

## Follow-ups
- `eval/` + OOD adapters: rewrite to "reconstruct observed phases" (after in-dist confirmation).
- The ~20 `tools/*.py` that build `target_t_indices` (update lazily).
- After the retrain: confirm EF slope via `tools/cmrxrecon_phase_analysis/{measure,analyze}_model_contraction.py`.
