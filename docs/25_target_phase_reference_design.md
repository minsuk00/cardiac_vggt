# 25 — One design change fixes both `target_t` problems: replace the index with a target-phase reference slice

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> The model conditions reconstruction on `target_t` — a **content-free index** (a number shared by all patients).
> That single choice is the **common root cause of two separate problems** the project found independently:
>
> | | why the index causes it | effect |
> |---|---|---|
> | **Issue 1 — phase timing** ([[17_cmrxrecon_target_t_phase_consistency]]) | the index `k` must stand for a cardiac *state*, but `k/12` is *fractional cycle time* → ES drifts across patients | `target_t=6` ≠ a fixed state; mild within CMRxRecon, **dangerous when mixing cohorts** (e.g. ACDC) |
> | **Issue 2 — amplitude** ([[24_flat_ef_amplitude_regression]]) | the index carries no per-patient info → the model can only learn `E[volume \| k]` = cohort-average contraction | regresses EF to the mean (~48 % for everyone) + under-contracts |
>
> **The fix (one change):** replace the index with a target-phase **reference slice** — a real acquired image at
> the target phase (the *original* VGGT camera-token design, where the special token carries an observed image,
> not a learned code). The target volume is just that slice's phase. **It dissolves both issues at once:**
> - **Issue 1 gone:** no index → no "`k` = ES" assumption. The reference *image* defines the phase per-patient, so
>   training is self-consistent — and **cross-dataset mixing becomes safe**: the reference is a *universal state
>   coordinate*, not a cohort-relative fraction.
> - **Issue 2 gone:** the reference carries the patient's *actual* target-phase content, so conditioning is
>   patient-specific → the model **can't** regress to the cohort average → per-patient EF recovers.
>
> **Measured support:** toy pred-EF-vs-true slope **0.26 → 0.73** (reference vs index; input-phase *labeling*
> does nothing, 0.26); real-model coverage ablation slope **0 → 1** once the target phase is observed
> ([[24_flat_ef_amplitude_regression]]). Use a **mid-ventricular** reference (most contraction signal).
>
> **Caveats:** (1) you reconstruct *observed* phases only — but unobserved-phase amplitude was information-limited
> anyway, so that's ≈ the recoverable limit; (2) per-voxel motion PSNR on *unobserved* planes stays
> information-limited — a separate problem the reference does **not** fix ([[22_reference_slice_amplitude_vs_motion_psnr]]).
> **Confirming end-to-end needs a retrain** with reference conditioning.

**Date:** 2026-06-23
**Status:** **Implemented (code), confirmation pending a retrain.** The synthesis is proven from its constituent
findings; the conditioning change is now wired into the codebase (flag-gated `reference_slot` dataset +
`use_reference_token` model; `mri_volume.yaml` is the reference+aggft default; `sbatch/train_mri_volume_reference.sh`
warm-starts fresh from base VGGT-1B). Tests guard the contract (`tests/test_reference_conditioning.py`,
`tests/test_freeze_pattern.py`). **Not yet retrained** → the real-model EF recovery (slope off ~0 toward 1) is
still to be confirmed via `tools/cmrxrecon_phase_analysis/`. Impl log: `version_history/v3_reference_conditioning.md`.
**Related:**
[[24_flat_ef_amplitude_regression]] (Issue 2 + the fix's amplitude evidence),
[[17_cmrxrecon_target_t_phase_consistency]] (Issue 1 + the ACDC-mixing hazard),
[[22_reference_slice_amplitude_vs_motion_psnr]] (reference fixes EF, NOT per-voxel motion PSNR),
[[04_inference_information_contract]] (the blind-input contract this lives inside),
[[21_motion_psnr_contract_levers]], [[19_motion_correction_warp_ceiling]] (the appearance/acquisition walls).
Memory: `project_vggt_flat_ef_amplitude`, `project_reference_fixes_ef_not_motion_psnr`,
`project_cmrxrecon_target_t_normalized`.

---

## 1. The root cause (one design choice → two issues)

Today the target phase enters the model as a **broadcast Fourier index**: the dataset emits
`target_t_indices = (t_target / T) · 2 − 1` for every slot, and the aggregator adds
`target_t_embedder(target_t_indices)` into the (replaced) camera/special token, identically for every input slice
(`vggt/models/aggregator.py:304-309`, `training/data/datasets/mri_dataset.py:362-364`). The embedding is the
*same function of `k` for every patient* — it can say "*which* phase (a number)," never "*what this patient's
heart looks like there*." Both project problems trace to exactly this:

- **Issue 1 (timing).** [[17_cmrxrecon_target_t_phase_consistency]] proved over all 301 subjects that
  CMRxRecon's 12 phases are a *resampling of each subject's own R–R interval* (native phase count 14–40, never
  12) ⇒ `target_t = k/12` is **normalized fractional cycle time, not a fixed state**. Only ED (t=0) is anchored
  (97 %); ES drifts frames 3–8 (std 0.72), worst in mid-diastole. An index forces a single `k → state` map; the
  drift makes that map a *cohort average*. Bounded within CMRxRecon, but the ACDC comparison showed mixing a
  wider-HR / cardiomyopathy cohort injects *contradictory* `target_t → state` supervision (ES spread 1.3–1.7×
  wider, peaks at a different `k`).
- **Issue 2 (amplitude).** [[24_flat_ef_amplitude_regression]] proved the index carries no per-patient info, so
  for an under-observed target phase the L1-optimal answer is `E[volume | k]` = the cohort-mean contraction →
  flat ~48 % EF for everyone (slope ≈ 0), with timing still correct.

These were discovered separately (timing on a phase-consistency audit; amplitude on a segmentation-EF sweep) but
are **the same defect viewed two ways**: a shared, content-free code can encode neither per-patient *state*
(Issue 1) nor per-patient *content* (Issue 2).

## 2. The fix and why it solves both

**Replace the index with a target-phase reference slice**: designate a real acquired image at the target phase
(mid-ventricular — most contraction signal) as the conditioning input; the reconstruction target is that slice's
phase. This is the *original* VGGT camera-token design — the special token carries an **observed image**, the
thing the architecture was built to consume — rather than the bolted-on learned `target_t` code.

- **Issue 1 dissolves.** There is no index, hence no "`k` = some fixed state" assumption. The reference *image*
  defines the phase, per-patient and self-consistently. **Bonus — cross-dataset mixing becomes safe:** a
  target-phase image is a *universal state coordinate* (an ES image is an ES image regardless of heart rate or
  cohort), not a cohort-relative fraction. The [[17_cmrxrecon_target_t_phase_consistency]] mixing hazard
  (ACDC) goes away because supervision is no longer "this fraction → average state of *this* cohort."
- **Issue 2 dissolves.** The reference carries the patient's *actual* target-phase content, so conditioning is
  patient-specific. The model literally cannot fall back on the cohort average — the answer is pinned to *this*
  heart's observed contraction → per-patient EF recovers.

## 3. Measured support (already in the repo)

- **Toy (`tools/cmrxrecon_phase_analysis/toy_contraction.py`):** pred-EF-vs-true slope C0 (index) ~0.26–0.29 →
  **B (reference, no index) 0.73**, robust across coverage S=2..20. Input-phase *labeling* (Clab) does nothing
  (0.26) — observing/conditioning on the target phase is what matters, not knowing when inputs were taken.
- **Real-model coverage ablation (`coverage_ablation.py`, `ablation_slopes.json`):** EF slope
  **−0.03 (k=0) → 1.03 (k=6)** as input slots are forced to observe the target phase — the architecture is
  sound and information-limited; explicit conditioning just makes the model *use* that observation efficiently
  (1 reference ≈ 6 blind target-phase slices). See [[24_flat_ef_amplitude_regression]] §3.
- **Reconciliation with "more frames don't help" ([[21_motion_psnr_contract_levers]], [[22_reference_slice_amplitude_vs_motion_psnr]]):**
  EF is volume-integrated over observed planes → the reference recovers it; per-voxel motion PSNR lives on
  *unobserved-plane appearance*, which doesn't propagate and is information-limited → the reference does NOT move
  it (≤0.25 dB even at a GT-fit oracle). No contradiction: the reference fixes the *amplitude* channel, which is
  both the wrong-and-fixable error and the clinically meaningful sense of "accurate motion correction."

## 4. Caveats / scope

1. **Observed-phases-only.** With a reference you reconstruct phases you *observed*, not arbitrary unobserved
   queries. But arbitrary-unobserved-phase amplitude is information-limited anyway (you need the target phase
   observed to get its amplitude right), so this is ≈ the information-theoretic recoverable limit, not a real
   loss of capability. It also aligns the query contract with what's physically recoverable from one-frame-per-slice.
2. **Per-voxel motion PSNR unchanged.** A separate, information-limited channel (appearance pattern on unobserved
   planes — [[19_motion_correction_warp_ceiling]], [[21_motion_psnr_contract_levers]]); the reference targets
   EF/amplitude, not this. Reconsider per-voxel motion PSNR as the headline metric accordingly.
3. **Interaction with the blind-input contract ([[04_inference_information_contract]]).** Input cardiac `t` and
   respiratory `r` remain assumed-unavailable; only the *target* query changes (index → reference image). The
   reference is itself a chosen/queried acquisition, consistent with "target queries stay free."

## 5. Implementation sketch + retrain recommendation

This is a **retrain**, not an eval tweak — the conditioning pathway changes. Rough shape (exact wiring is a
retrain-time decision):

- **Data (`mri_dataset.get_data`):** guarantee a designated reference slot observes the target phase at a
  mid-ventricular `z` (an actual `phases[t_target]` slice), and keep `gt_target_volume = phases[t_target]` as
  today. Drop `target_t_indices` from the emitted batch.
- **Model (`vggt/models/aggregator.py`):** set `use_target_t_pose_embedding=false` (remove the broadcast index
  add); the reference slice's own image + `z` provide the conditioning through the normal token path — the
  "original camera-token design." Keep `z`/(optionally `t`) embedders per the current contract.
- **Loss:** unchanged (`|V_canon − gt_target_volume|`).
- **Confirm:** re-run the EF measurement (`tools/cmrxrecon_phase_analysis/measure_model_contraction.py` →
  `analyze_model_contraction.py`) on the retrained model and check the pred-EF-vs-true slope moves off ~0 toward
  1, and that the LV-curve *depth* (not just timing) now tracks per-patient. Optionally re-run the ACDC-mixing
  check from [[17_cmrxrecon_target_t_phase_consistency]] to confirm mixing is now safe.

**What's proven vs expected:** the cause chain and the per-component fixes are proven (oracle control + real-model
ablation + leak-verified toy + full-cohort phase audit). The *integrated* end-to-end efficacy of reference
conditioning on the real model — and exactly how few target-phase depths it tolerates at one-frame-per-slice — is
**expected, not yet measured**. The recommended next step is the retrain above.
