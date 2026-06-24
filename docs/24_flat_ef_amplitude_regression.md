# 24 — The model reconstructs a flat ~48% EF for everyone (amplitude regression to the cohort mean)

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> Run the trained model at all 12 `target_t` for the 30 val subjects (scattered inputs held fixed), segment
> every reconstructed volume with nnU-Net Task114, and compute ejection fraction (EF) per patient. **The model
> gets cardiac TIMING right per-patient but regresses contraction AMPLITUDE to the cohort mean** — it
> reconstructs **~48 % EF for essentially everyone** (pred-EF 47.96 ± 9.23 % vs true 62.58 ± 7.37 %; mean gap
> **14.6 EF-points**), so reconstructed-EF-vs-true-EF **slope ≈ 0** (−0.03, r −0.02). Most extreme: a true-92 %
> heart reconstructed at 35 %. Timing is fine — predicted end-systole is within ±1 frame of GT for **93 %** of
> subjects, and the LV-volume curve shape matches (curve_corr 0.934). It knows *when* the heart contracts, not
> *how much*. This is clinically useless for function — a failing heart (low EF) and a healthy heart (high EF)
> both come out at ~48 %.
>
> **Proven cause chain:**
> 1. **NOT the pipeline.** An oracle-splat control (true target-phase slices, identity placement, *same* splat +
>    *same* nnU-Net, no model) reproduces true EF at **slope 1.03, r = 1.00** (range 50–92 % faithfully). The
>    renderer and segmenter are innocent — the flat EF is **the model's predicted motion**.
> 2. **It's an INFORMATION/OBSERVATION limit, and the architecture is SOUND.** A coverage ablation that forces
>    `k` input slots to OBSERVE the target phase flips the slope as `k` rises: **k=0 → −0.03, k=1 → 0.01,
>    k=6 → 1.03, k=12 → 0.84**. Observe the target phase at ~6 depths and per-patient EF snaps back.
> 3. **Mechanism.** In normal use, scattered one-frame-per-slice inputs *under-observe* the target phase; the
>    L1-optimal answer for an unobserved phase is the population-mean contraction → flat EF. And because the
>    model is blind to input phase, it uses what little it observes *inefficiently* (needs ~6 blind target-phase
>    slices; a toy shows explicit reference conditioning recovers from **1** — see [[25_target_phase_reference_design]]).
>
> **⇒ This is the AMPLITUDE half of the broader `target_t`-index problem.** The flat EF is a direct consequence
> of conditioning on a content-free `target_t` *index* (it carries no per-patient info, so the model can only
> learn `E[volume | k]`). The unified fix — a target-phase **reference slice** instead of the index — is in
> [[25_target_phase_reference_design]]. A reference also fixes per-patient EF but does NOT move per-voxel motion
> PSNR (that's a separate, information-limited channel — [[22_reference_slice_amplitude_vs_motion_psnr]]).
>
> **Report:** https://github.com/minsuk00/research-reports/blob/main/2026-06-22_1421_vggt-ef-amplitude-regression.html
> · local data `scratch/phase_analysis/` (`model_contraction.json`, `ablation_slopes.json`, `report.md`, `figs/`).

**Date:** 2026-06-22
**Status:** Cause chain **PROVEN** (oracle control + direct real-model intervention + leak-verified toy). The
*fix* ([[25_target_phase_reference_design]]) is supported by the toy and the ablation but its real-model efficacy
awaits a **retrain**. Scope: in-distribution CMRxRecon val (n=30), clean (no-resp), single z-only/target_t-on/
no-resp/aggregator-finetuned checkpoint. **Related:** [[17_cmrxrecon_target_t_phase_consistency]] (the *timing*
half of the index problem), [[22_reference_slice_amplitude_vs_motion_psnr]] (reference fixes EF not motion PSNR),
[[25_target_phase_reference_design]] (the unified design decision), [[21_motion_psnr_contract_levers]],
[[19_motion_correction_warp_ceiling]], and memory `project_vggt_flat_ef_amplitude`.

---

## 1. Question

The model reconstructs a 3D heart at a queried cardiac phase from a few scattered single-frame-per-slice
acquisitions. **Is the reconstruction quantitatively faithful per patient?** EF — `(EDV − ESV) / EDV`, the
fraction the LV blood pool shrinks from end-diastole to end-systole — is *the* clinical measure of cardiac
function (healthy ~60 %, dilated/failing ~20 %). If a low-EF and a high-EF patient both reconstruct at the same
EF, the recon reproduces the cohort average, not the patient. Four sub-questions:

1. **What** is the model doing (timing vs amplitude)?
2. **Because of what** — segmentation? splat renderer? `target_t` conditioning? the model itself?
3. **What kind of limit** — information/coverage (fixable by sampling/conditioning) or capacity (needs a new model)?
4. **How to fix it.**

## 2. The phenomenon (numbers)

Trained model (z-only, target_t-on, no-resp, aggregator-finetuned) run over all 12 `target_t` for the 30 val
subjects, scattered inputs held fixed; every reconstructed volume segmented with nnU-Net Task114 (M&Ms,
validated; see [[15_mnms_nnunet_segmentation_eval]]); EF from the LV blood-pool volume curve
(`VOX_ML = 1.4·1.4·8.0 / 1000`). Tool: `tools/cmrxrecon_phase_analysis/measure_model_contraction.py` →
`analyze_model_contraction.py`.

| Quantity | Model | GT |
|---|---|---|
| EF mean ± std | **47.96 ± 9.23 %** | 62.58 ± 7.37 % |
| pred-EF-vs-true-EF slope | **−0.03** (r −0.02) | 1.0 by construction |
| mean EF gap | **14.6 EF-points** | — |
| ES phase within ±1 frame | **93.3 %** | — |
| LV-curve shape corr (per-subject) | **0.934** | — |

**Read:** flat EF (slope ≈ 0 → reconstructed EF is *uncorrelated* with the patient's real contraction; it varies
20–62 % but the variation has nothing to do with the patient), correct timing (ES phase ±1 frame, 93 %), correct
curve *shape* (the LV curve bottoms out at the right phase) but wrong *depth* (too shallow → under-contracts).
Figures: `scratch/phase_analysis/figs/{ef_scatter,lv_curves}.png`.

## 3. Cause isolation

### 3a. Oracle-splat control — rules out the pipeline (slope 1.03, r=1.00)
If the splat or segmenter destroyed EF, even *perfect* data through them would come out flat. Feed them perfect
data: for each `(subject, target_t)` splat the **true** target-phase slices at identity placement (no model)
through the **same** splat and **same** nnU-Net. Result: oracle EF tracks true EF at **slope 1.03, r = 1.00**
(range 50–92 % reproduced). **The pipeline is innocent — the flat EF lives in the model's predicted motion.**
This is the single most important control: it converts "the EF looks flat" into "the *model* flattens EF."
Tool: `tools/cmrxrecon_phase_analysis/oracle_splat.py`.

### 3b. Coverage ablation — the decisive test (slope 0 → 1)
Force `k` of the model's input slots to OBSERVE the target phase (their image becomes the true target-phase
anatomy at the same depth; everything else held fixed), sweep `k`, re-measure the EF slope.
Tool: `tools/cmrxrecon_phase_analysis/coverage_ablation.py` → `ablation_slopes.json`.

| k = slots observing target phase | EF slope |
|---|---|
| 0 (scattered, current) | **−0.026** (flat) |
| 1 | 0.009 (flat) |
| **6** | **1.032** (recovers) |
| 12 (all-z) | 0.844 (recovers) |

**As soon as the model observes the target phase at ~6 depths, per-patient EF snaps back (slope 0 → 1).** This
is an **information/observation limit, and the architecture is SOUND** — the model *can* reconstruct per-patient
contraction; it doesn't, because the scattered inputs under-observe the target phase. (k=1 isn't enough: the
phase-blind model needs the phase observed at several depths to reconstruct the 3D cavity.)

### 3c. Toy — isolates the conditioning mechanism
A minimal synthetic stand-in (`tools/cmrxrecon_phase_analysis/toy_contraction.py`): Z=8-slice "ventricles" with
per-patient contraction amplitude `a` (= EF) and an apex→base gradient; phase-blind scattered observations; a
tiny set-network reconstructs at a queried phase. Compare conditioning schemes by pred-EF-vs-true-EF slope (1 =
recovers per-patient, 0 = regress-to-mean):

| scheme | conditioning | EF slope |
|---|---|---|
| **C0** | `target_t` index, blind input phase — **the current recipe** | ~0.26–0.29 (flat across coverage) |
| Clab | `target_t` index, *labeled* input phase | 0.26 |
| Ccov | `target_t` index + 1 guaranteed target-phase slot | 0.36–0.42 |
| **B** | **reference slice AT the target phase, no index** | **0.73** |

Coverage sweep: C0 stayed flat at ~0.3 from S=2 to S=20 — **general (random-phase) coverage does not help**; B
held 0.7–0.8 at every coverage, even S=2. Crucially, *labeling* input phases (Clab) didn't help — knowing *when*
each slice was taken isn't enough; you must **observe the target phase and condition on that observation**.

### How the toy and the real model reconcile
They look contradictory ("toy: coverage doesn't help; ablation: coverage fixes it") but say the same thing:
- toy coverage sweep added **random-phase** slices → no help (*general* coverage);
- ablation added **target-phase** slices → full recovery.

→ **What matters is observing the *target phase* specifically, not general coverage.** The real model uses
target-phase observation *inefficiently* because it's phase-blind (needs ~6 blind slices); the toy's *explicit
reference conditioning* (scheme B) recovers from **1**.

## 4. Conclusion & fix

**What the model does (proven):** per-patient cardiac *timing* (ES ±1 frame, 93 %), but contraction *amplitude*
regressed to the cohort mean (flat EF ~48 %, slope ≈ 0). **Why (proven):** not the pipeline (oracle slope 1.03,
r=1.00); the model, and it's an information/observation limit, not a fundamental flaw (ablation flips 0→1 once
the target phase is observed at ~6 depths). **Mechanism:** scattered one-frame-per-slice inputs under-observe the
target phase, and the phase-blind model uses what it observes inefficiently; for an unobserved phase the
L1-optimal answer is the population-mean contraction → flat EF.

**Fix (recommended): condition on a target-phase reference slice (the original VGGT camera-token design), not a
broadcast `target_t` index.** The toy shows this uses target-phase observation *efficiently* (recovers from a
single reference). Option A (separate ED/ES models) is unnecessary — the multiphase model is sound. Option C
(keep the index + denser sampling) needs ~6 target-phase observations the one-frame-per-slice regime can't
provide. The reason this single change also dissolves the `target_t`-as-fractional-time *timing* problem
([[17_cmrxrecon_target_t_phase_consistency]]), and the caveats, are written up in
[[25_target_phase_reference_design]].

**Caveat:** with a reference you reconstruct *observed* phases, not arbitrary unobserved queries — but the
ablation shows arbitrary-unobserved-phase amplitude is information-limited *anyway*, so "reconstruct observed
phases" is ≈ the recoverable limit. And a reference fixes EF/amplitude only — per-voxel motion PSNR on unobserved
planes stays information-limited ([[22_reference_slice_amplitude_vs_motion_psnr]]).
