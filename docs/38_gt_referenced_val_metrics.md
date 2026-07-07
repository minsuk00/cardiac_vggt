# 38 — GT-referenced val metrics to quantify targeted improvements (implemented)

> **TL;DR & takeaway**
>
> Aggregate PSNR is dominated by static tissue and capped by the ~7-9 dB appearance wall, so a real
> targeted gain (breathing-z, motion, a loss tweak) is invisible — "even if something changes, we won't
> know." A 2-agent debate (process/DVF-GT vs outcome/reconstruction-GT) converged: **log BOTH, with a
> strict split** — outcome metrics decide whether a change ships, process metrics explain why. The
> design test is that the set must flag the stop-grad regression (docs/37: DVF slope ↑ while the volume
> got worse) — and it does.
>
> **Implemented (val-only, additive, training bit-identical) in `training/loss.py`** — new `metric_*`
> keys logged as `Val_Loss/metric_*` (each key **must be registered** in
> `logging.scalar_keys_to_log.val.keys_to_log`, `mri_volume.yaml` — the trainer only meters listed
> keys; forgetting this silently discards them):
> - **`recov_frac_heart`** = `(MSE_id − MSE_model)/(MSE_id − MSE_oracle)` on the cardiac-motion ROI —
>   the headline. Oracle = splat the TRUE target-phase content at Δ=0 (recoverable ceiling); this
>   **rescales the un-fixable appearance wall out of numerator AND denominator**, so a 0.5 dB fixable
>   gain shows as a big fraction jump. `=1` at ceiling, `<0` below the do-nothing floor (flags stop-grad).
>   Also logs `mse_heart_{identity,model,oracle}` for robust offline re-aggregation.
> - **`psnr_3d_static`** (control, should stay flat) beside the existing `psnr_3d_motion` (= heart).
> - **`hole_frac_heart`** = frac of heart voxels with `coverage<0.5` — the structural tripwire that
>   caught stop-grad (22%→33%) when slope didn't.
> - **Breathing** (no-op when resp off): `resp_slope_dz`, `resp_corr_dz`, **`resp_epe_dz_mm`** (report
>   EPE, not just slope — slope can hit 1.0 with scattered placement), `resp_frac_deep_ignored`.
>   Predicted Δz vs the EXACT applied `resp_disp_mm[...,0]`; brings `tools/exp_4wok_analysis.py` online.
>
> **Decision rule:** a change is a real win iff `recov_frac_heart` ↑ AND `psnr_3d_motion` ↑ WITHOUT
> `hole_frac_heart` ↑; the breathing metrics say *which* motion it fixed. Real EF/Dice stays offline
> (nnU-Net). See [[37_stopgrad_denominator_and_gather_aux_test]].

**Date:** 2026-07-07. **Status:** implemented + verified (208 tests pass; targeted grad/no-grad +
directional sanity check). **Where:** `training/loss.py`, val-only block after the motion metrics.

---

## 1. The problem

The team could not quantify whether a change helped. `metric_psnr_3d_{full,bbox}` is dominated by
static tissue (the heart is ~3-5% of voxels, so a 3 dB heart gain is ~0.1 dB in full PSNR — below
noise) and the absolute number is pinned ~14 dB under the appearance wall (docs 19-21), so fixable
improvements drown. `val_motion` PSNR helps (heart ROI) but its absolute scale still buries small gains.

## 2. The heart ROI needs no segmentation

The "heart ROI" is the **cardiac-motion mask** `compute_motion_mask(phases)` = voxels whose intensity
swings across the 12 GT cardiac phases > `tau`. Derived per-subject from `batch["phases"]` (the on-disk
NIfTIs we already have), no segmentation, no cached universal mask, no training. It is already the ROI
of the primary `val_motion` PSNR. It is a **motion proxy** for the heart (dynamic region), correct for
ROI-restricting reconstruction metrics but NOT an anatomical LV/myo/RV segmentation — real EF/Dice still
needs the offline nnU-Net (Task114, doc 33). A future upgrade is to cache nnU-Net masks per subject for
online chamber-level Dice/EF, but the motion mask is free and enough to start.

## 3. Why this set (2-agent debate synthesis)

Neither family alone is trustworthy, proven by the stop-grad case (docs/37): a **DVF-slope process
metric rated stop-grad an improvement** (0.36→0.66) **while the reconstructed volume got worse** (heart
PSNR −4 dB, holes 22%→33%). So:
- **Outcome/reconstruction-vs-GT metrics decide whether a change ships** (can't be gamed — they measure
  the objective). `recov_frac_heart<0`, `hole_frac_heart`↑, `psnr_static` flat while `psnr_motion`↓ all
  correctly flag stop-grad.
- **Process/DVF-GT metrics attribute** (which motion moved) and give breathing sensitivity — but report
  **EPE alongside slope** (Agent 1's fix: slope alone is foolable by scattered/wrong placement).
- **`recov_frac`'s oracle normalization** is the key sensitivity win: dividing by the recoverable span
  (~14 dB) instead of the total (~34 dB) makes a small fixable gain a large fraction move.

## 4. Implementation notes

- **Val-only gate:** `if not pos_pred.requires_grad:` — training runs the forward under grad (skipped
  ⇒ zero cost, loss numerics bit-identical), `val_epoch` is `@torch.no_grad()` (computed). Verified: the
  new keys are ABSENT in the grad path.
- **Additive + safe:** one block after the existing motion metrics; each part `try/except`-wrapped
  (logs a warning, never raises into the loop); existing metrics/loss untouched. **The keys must be
  listed in `scalar_keys_to_log.val.keys_to_log`** (`mri_volume.yaml`) — the trainer meters ONLY listed
  keys (`_get_scalar_log_keys`), so unlisted `metric_*` keys are silently dropped (a prove-it finding).
- **recov_frac guard:** only computed when the recoverable span `mse_id − mse_or > 1e-6` (skip when the
  oracle is not clearly better than identity — a signed `clamp(min=)` on that denominator would
  otherwise sign-flip a poor subject to the max "+1.5" and bias the mean up; prove-it finding).
- **Slot-0/reference + slope robustness:** slot 0 (reference anchor) excluded; per-subject slope
  clamped to [−3, 3] so one low-applied-variance subject can't dominate the meter average. `resp_corr_dz`
  is SIGNED Pearson (self-consistent with the signed slope) — it will NOT equal the offline abs-corr.
- **Cost:** 2 extra splats per val sample (identity + oracle), val-only, `limit_val_batches`-bounded.
- **Axis/sign convention** mirrors `tools/exp_4wok_analysis.py`: `dvf = world_points − scanner_coords`,
  `pred_dz_mm = dvf[...,2].mean()·66`, `applied = resp_disp_mm[...,0]`, slot 0 (reference anchor)
  excluded, deep = `|applied|≥12 mm`. Verified: identity → slope 0/epe=mean‖applied‖; perfect
  correction → slope 1/corr 1/epe 0.
- **Aggregation caveat:** slope/corr are per-sample (per-batch) then meter-averaged over val — a trend,
  noisier than the offline pooled fit; `resp_epe_dz_mm` and `recov_frac` are the robust headline numbers.
