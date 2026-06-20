# 07 — Predicted DVF analysis: is the model learning realistic motion?

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Question:** does the trained model predict *physically realistic* displacement fields (DVF) —
> for breathing (through-plane) and cardiac (in-plane) — or is the motion degenerate (≈0) or absurd?
> No motion GT exists, so this is a **magnitude + correlation sanity check**, not an accuracy metric.
>
> **What we did:** dumped the **true** predicted residual `Δ = world_points − scanner_coords` in
> physical **mm** for run **`t59w6nqy`** (`218747856_mri_volume_resp_allphases_aggft_z_no_t` — the
> doc-05 "best recipe": breathing ON, z-only, no input-t, aggregator-finetuned) over **all 30 val
> subjects**, breathing-augmented inputs, masked to anatomy pixels (`tools/dump_predicted_dvf.py`).
>
> **Conclusion: the model IS learning useful, realistic, breathing-aware DVF — with a ceiling.**
> - Predicted Δz tracks the applied breathing tightly: **corr(|d|, Δz) = +0.87** (n=297 slots),
>   correct sign, per-slot ~rigid.
> - Magnitudes are physically sane: median ~0.7–1 mm, p95 ~8–10 mm, through-plane max ~14 mm —
>   **neither degenerate nor blown up.**
> - **BUT it under-corrects deep breaths:** linear fit **slope ≈ 0.42** and Δz **plateaus ~7–8 mm**
>   while applied breathing reaches ~24 mm. This is the expected *shrinkage* of an **r-blind**
>   estimator (no input-t, no r-embedder) → real headroom for an r / input-t signal.
> - **One artifact:** in-plane has a heavy tail (rare pixels to ~45 mm) that is non-physical for
>   cardiac motion — likely slice-edge / low-coverage, not a motion estimate.
>
> **Also confirmed this session:** the respiratory *direction-jitter (tilt)* is a geometrically
> clean **SI-dominant cone about the through-plane axis** (domain randomization over the unknown
> SAX→true-SI angle, magnitude-preserving) — not a skewed direction. And the DVF wandb viz was
> switched from a single normalized ±0.05 colorbar to **per-axis mm** (in-plane ±15 mm, through ±25
> mm), because the old clip saturated at 2.2 mm in Z and hid exactly the breathing-scale motion.

**Date:** 2026-06-18
**Status:** Analysis complete (eval-only; no training change). Companion to
[`05_respiratory_variants_results.md`](05_respiratory_variants_results.md) (which established this run
as the best recipe via reconstruction PSNR; this doc inspects the *motion field itself*).
**Artifacts:** `result/predicted_dvf_ranges.png`, `result/respiratory_tilt_demo.png`,
`tools/dump_predicted_dvf.py`, `tools/demo_resp_tilt.py`.

---

## 1. The question

Doc 05 established (via target-phase reconstruction PSNR) that the breathing model
`t59w6nqy` beats its no-resp twin by +3.33 dB bbox / +2.24 dB motion. PSNR says the *output volume*
is better — it does **not** directly say the *predicted motion field* is physically meaningful. The
point head could in principle reach a decent PSNR with a degenerate or nonsensical Δ (e.g. exploiting
the splat). So we ask the field directly:

> Feed real (breathing-augmented) input slices, recover the model's predicted per-pixel displacement
> Δ in mm, and check: is it in a **reasonable range** — not ~0 (not learning), not absurd (e.g. half
> the heart)? Does the through-plane component actually respond to the breathing it's supposed to
> correct?

There is **no motion ground truth** in the intensity pipeline, so this is deliberately a
*magnitude + correlation* sanity check, not an accuracy number. (Cardiac realism cannot be scored
without an external reference — see §5 / `CLAUDE.md → Tagging`.)

## 2. Method

### 2.1 Recovering the true DVF in mm
The point head predicts `world_points` (normalized [−1,1]); the residual displacement is
`Δ = world_points − scanner_coords` (both [−1,1]). Convert to physical mm using the **splat's**
coordinate convention (`vggt/utils/splat.py`: `px = (pos+1)·0.5·(size−1)`, align_corners), so one
normalized unit = `(size−1)/2` voxels:

| axis | grid | mm per norm unit |
|---|---|---|
| Δx, Δy (in-plane) | 256 vox @ 1.4 mm | `(256−1)/2 · 1.4` = **178.5 mm** |
| Δz (through-plane) | 12 vox @ 8.0 mm | `(12−1)/2 · 8.0` = **44.0 mm** |

This is the same convention used by `respiratory.py:_norm_delta` and the new DVF colorbar.

### 2.2 The dump (`tools/dump_predicted_dvf.py`)
Faithful to the run and to the trainer's val path:
- **Model:** `use_z=True, use_t=False, use_target_t=True, train_on_residual_dvf=True`, only
  `*patch_embed*` frozen (aggregator-finetuned); load `ckpts/checkpoint_last.pt` (`missing=0,
  unexpected=0` → exact arch match). bf16 autocast, eval.
- **Data:** all 30 val subjects, `MRIDataset` val, S=12 slices each. For each subject, build the
  batch and apply `gpu_augment_batch(train=False, respiratory_cfg=RespiratoryConfig(enable=True))` —
  the **deterministic per-`seq_index` val breathing** (config defaults: amplitude 16±8 mm, Lujan
  `sin⁶`, AP=0.35·SI, direction-jitter 30°), exactly as the val metrics use. Target phase = the
  dataset's stratified `target_t = seq_index % T`.
- **Masking:** restrict Δ stats to **anatomy pixels** (`input intensity > 0.05`), since Δ on
  zero-padded / background pixels is meaningless (the splat gates `intensity > 1e-3`). Slots with
  <50 anatomy pixels skipped.
- **Breathing test:** per slot, signed mean Δz (mm, masked) vs applied `|d| =
  ‖resp_disp_mm‖` (mm). Pool all 297 surviving slots → correlation + linear fit + binned means.

### 2.3 Two supporting checks this session
- **Tilt geometry** (`tools/demo_resp_tilt.py` → `result/respiratory_tilt_demo.png`): using the
  actual `_rotate_disp`, verified the direction-jitter is a cone of half-angle θ~U(0,30°) about the
  through-plane axis D, azimuth φ~U(0,2π), magnitude-preserved. Rotating `e_D` by θ about
  `k=(0,−sinφ,cosφ)` gives `(cosθ, sinθ·cosφ, sinθ·sinφ)` — a clean SI-dominant cone, **not** a
  skewed direction. It is *domain randomization* over the unknown SAX→true-SI tilt (~20–45°, baked
  out of our recon), not a claim of the correct tilt. Caveat: 30° max slightly under-covers the
  literature 20–45° range, and azimuth is uniform (the heart's lean has a preferred direction) —
  both tunable, neither a correctness bug.
- **DVF colorbar (viz-only)**: `trainer._log_volume_and_dvf_to_wandb` previously plotted all three Δ
  rows with one normalized `±0.05` colorbar = **±8.9 mm in-plane / ±2.2 mm through-plane**. The Z
  clip (2.2 mm) saturated exactly the breathing-scale motion. Changed to **per-axis mm**: in-plane
  ±15 mm, through-plane ±25 mm; normalized `|Δ|` percentiles kept in the title. No training impact.

## 3. Results

### 3.1 Per-pixel predicted |Δ| over anatomy pixels (mm)

| component | mean\|·\| | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| Δx in-plane | 1.14 | 0.32 | 4.90 | 8.61 | 25.79 |
| Δy in-plane | 2.16 | 0.58 | 9.54 | 15.16 | 43.00 |
| **\|Δ\| in-plane** | 2.62 | 1.05 | 10.52 | 16.52 | 49.55 |
| **Δz through** | 2.80 | 0.69 | 7.90 | 8.53 | 14.06 |

### 3.2 Breathing test — per-slot signed mean Δz vs applied |d| (n=297 slots)
- **corr(|d|, mean Δz) = +0.870**
- **linear fit: mean Δz ≈ 0.420·|d| + 0.59 mm** (slope ≈ ±1 would be full rigid correction)

| applied \|d\| (mm) | n | mean Δz | mean \|Δz\| | mean in-plane \|Δ\| |
|---|---|---|---|---|
| [0, 2)  | 157 | +0.44 | 0.45 | 0.47 |
| [2, 8)  | 65  | +2.68 | 2.71 | 2.66 |
| [8, 16) | 47  | +7.03 | 7.03 | 5.04 |
| [16, 30)| 28  | +7.59 | 7.59 | 9.43 |

Figure `result/predicted_dvf_ranges.png`: (left) Δz vs |d| scatter — rises then **plateaus ~7–8 mm**;
(right) magnitude histograms — Δz is **bimodal** (a spike ≈0 for no-breath slots + a ~7–8 mm
correction mode), in-plane concentrated <5 mm with a tail.

## 4. Analysis

1. **Not degenerate, not absurd (bulk).** Median ~0.7–1 mm with clear modulation (not collapsed to
   0); p95 ~8–10 mm both in/through-plane; through-plane max 14 mm. All physically plausible
   (cardiac in-plane few mm–1 cm; breathing ~8 mm slice scale). The point head is learning real
   geometry, not gaming the splat with junk Δ.

2. **Breathing correction is real and well-directed.** corr +0.87 (vs a crude +0.67 read off the
   *clipped* DVF PNG earlier — the unclipped number is much tighter), correct positive sign
   (consistent with Lujan `d(r) ≥ 0` being one-signed), per-slot ~rigid. Small breaths (|d|<2 mm) →
   Δz ~0.4 mm, i.e. it correctly **leaves near-reference slices alone**.

3. **But it under-corrects deep breaths — slope 0.42, plateau ~7–8 mm.** For |d|∈[16,30) it applies
   only ~7.6 mm against ~20 mm of motion. This is the textbook **shrinkage of an estimator under
   uncertainty**: the model is **r-blind** (no input-t, no r-embedder), so it can only *content-infer*
   the breathing state and regresses toward a conservative mid-range correction, saturating. It is
   the expected, not pathological, behavior — and it is the concrete lever: feeding an `r` (or input
   `t`) signal should push the slope toward 1. (Note doc-05 found input-t *unnecessary for PSNR*;
   that does not contradict this — PSNR is dominated by the many shallow-breath slots, where the
   correction is already near-complete; the deep-breath tail is where the residual lives.)

4. **In-plane tail is the one yellow flag.** Bulk in-plane is realistic (median 1 mm, p95 10 mm), and
   part of it is *legitimate breathing correction*: in-plane |Δ| rises to 9.4 mm for deep breaths —
   the AP (0.35·SI) + tilt component re-projected in-plane, coherent with §2.3. But the tails are
   large (p99 16 mm, max ~43–49 mm; Δy worse than Δx). ~45 mm in-plane is not physical cardiac
   motion — almost certainly a small fraction of slice-edge / low-coverage pixels thrown far, not a
   motion estimate. Worth a targeted look (mask to bbox interior, inspect which pixels) but it is the
   rare tail, not the typical field.

5. **Tilt direction is concrete.** The geometry check rules out the "are we tilting in a weird
   direction?" worry: it is a symmetric SI-dominant cone about the through-plane axis, magnitude
   preserved. There is no single "correct" tilt to hit — randomizing over the plausible range is the
   point.

## 5. Conclusion, caveats, follow-ups

**Conclusion.** The model is learning **useful, realistic, breathing-aware** DVF: through-plane Δz
tracks the applied breathing tightly (corr +0.87), in the right sign and a physical magnitude range
(median ~1 mm, p95 ~8–10 mm), and correctly does ~nothing on near-reference slices. The honest
qualifier is a **ceiling**: it under-corrects deep breaths (slope 0.42, plateau ~7–8 mm), the
expected shrinkage of an r-blind model.

**Caveats.**
- Eval-only, no motion GT → this validates *plausibility and breathing-responsiveness*, not accuracy.
- One run (`t59w6nqy`), one checkpoint (`checkpoint_last`), val breathing at config-default
  amplitudes; target phase = stratified `seq_index % T`.
- In-plane "realism" is asserted from magnitude + breathing-coherence only; a true cardiac check
  needs an external field (tagging-derived `(dx,dy)` or registration DVF) — not done here.
- Breathing |d| is heavily weighted toward small values (157/297 slots <2 mm) because Lujan dwells
  at end-expiration; the deep-breath bins (n=28 for [16,30)) are smaller.

**Follow-ups (not done).**
1. **A/B the z+t sibling** (`use_t=true`) with the same dump — does input-t lift the breathing slope
   toward 1? Direct test of the §4.3 headroom.
2. **Add an r-embedder** (the doc-01 / doc-04 fork) and re-measure the slope — the principled fix for
   deep-breath under-correction.
3. **Chase the in-plane tail**: restrict to bbox interior, identify the ~p99+ pixels (edge vs
   coverage), decide if a coverage/edge regularizer is warranted.
4. **Cardiac validation against a reference** (tagging / elastix) on matched subjects.

## Files
- `tools/dump_predicted_dvf.py` — the DVF dump + breathing test (this analysis).
- `tools/demo_resp_tilt.py` — tilt-geometry demo (`result/respiratory_tilt_demo.png`).
- `training/trainer.py` — DVF wandb viz switched to per-axis mm colorbars (viz-only).
- `result/predicted_dvf_ranges.png` — scatter + magnitude histograms.
