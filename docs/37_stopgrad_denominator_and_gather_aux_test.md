# 37 — Stop-gradient denominator REFUTED; gather-placement auxiliary CONFIRMED (breathing-z)

> **TL;DR & takeaway**
>
> Conclusive toy-experiment test (2026-07-07) of the two coverage-free levers doc 34 proposed for the
> weak through-plane / breathing correction. **Verdict: the one-line "stop-gradient denominator"
> (`V = acc/(cov.detach()+eps)`) does NOT work as a drop-in, but the "gather-placement auxiliary loss"
> (lever B) does.**
>
> - **Stop-grad** restores the through-plane gradient (flat covdiv plateau → smooth ramp, 36× larger,
>   forward byte-identical) and it *transmits* — sighted direct-opt breath slope **0.36→0.66** (refutes
>   the Adam-washout worry). **BUT it craters reconstruction**: heart/motion PSNR **21.2→16.5 dB, ~4 dB
>   BELOW the do-nothing floor**, by tearing coverage holes in the heart (22%→33%). Coverage-division's
>   "damping" is the *correct* gradient of a forward that legitimately penalizes holes — detaching it
>   lets the `1/cov`-amplified surrogate over-move slices. **Do NOT ship `cov.detach()`.**
> - **Gather-placement auxiliary** (keep covdiv splat-L1 PRIMARY, add `λ·|sample_volume(V_gt, world+Δ) −
>   input|`, a coverage-free per-pixel "is this the right depth?" signal): slope **0.40→0.90 at heart
>   PSNR ~21** (λ≈0.2–0.5). The primary keeps reconstruction honest / hole-free; the auxiliary supplies
>   the un-damped placement gradient the splat's division destroys.
> - **Implementation is code-simple** (~15 lines, mirrors the existing `diffusion_weight` term exactly —
>   4 wiring points). **One real design wrinkle:** the gather target `V_gt` is the *target* cardiac phase,
>   but input slots are at *arbitrary* phases → the moving heart's content won't match `V_gt` (only static
>   tissue does). Breathing is a rigid whole-slice shift recoverable from the phase-invariant static
>   majority, so it should be fine, but validate / consider masking the aux to low-cardiac-motion voxels.
> - **Needs a SHORT WARM-START FINETUNE** from the 4wok ckpt (loss change ⇒ training-only; not
>   inference-time, not from scratch). **Bounded prize ~0.5–1 dB, secondary metric, EF unaffected.**
>
> Report: `research-reports/2026-07-07_0912_stopgrad-breathing.html`. Follows [[33_4wok_conclusive_analysis]]
> + [[34_breathing_z_fix_brief]] (this is the experimental test of that brief's levers A & B).

**Date:** 2026-07-07. **Status:** sighted-optimization proven + prove-it-audited; real-network retrain
NOT yet run. **Model context:** 4wokxzov (run 217720691, reference-slot, DPT, 1-frame).

---

## 1. What was tested

Doc 34 ranked two coverage-free levers to un-damp the through-plane placement gradient: **(A)** a
coverage-free *renderer* retrain and **(B)** a coverage-free *auxiliary placement loss* via
`sample_volume` gather. Separately, an in-conversation debate proposed a cheaper **stop-gradient
denominator** (`V = acc/(cov.detach()+eps)`) as a one-line variant of (A) — forward byte-identical,
backward-only, restoring the coverage-free gradient. This doc conclusively tests stop-grad and the
gather auxiliary.

**Method** (`scratchpad/toy1..exp4`, all reproduced under one common protocol):
- **Mechanism toy** (known-answer, single slice belongs at plane z\*): measures the exact autograd
  `dL/dz` per renderer. Confirms `faithful_scatter` == `splat_to_volume` forward (Δ 6e-8) and stop-grad
  forward is EXACTLY covdiv (`max|Δ|=0`).
- **Sighted direct-opt** (16 real val subjects, per-plane Lujan breath 16±8 mm sin⁶, fixed cardiac
  phase): directly optimize a per-plane Δz to minimize L1(render, V0), per renderer. This is the
  `tools/exp_covfree_diag.py` protocol (which the covdiv baseline reproduces: slope 0.41 on seqs 0-11
  vs the documented 0.44).

## 2. Results (measured, prove-it-audited — no conclusion-inverting bug)

**Mechanism (toy):** at the between-plane midpoint (away from the eps-singularity spikes covdiv has at
integer plane boundaries), covdiv `dL/dz` = **0.00057/mm** (flat plateau) vs stopgrad **0.0205/mm**,
nocovdiv **0.0098/mm** (≈ the E0 coverage-free 0.0103). Stop-grad's forward is byte-identical. The
covdiv plateau is the `(I_i−V)/(cov+eps)` term →0 for lonely low-coverage pixels; stop-grad →
`I_i/(cov+eps)`, which does not vanish.

**Sighted direct-opt (16 subj, do-nothing floor = 19.85 dB):**

| renderer | breath slope | motion (heart) PSNR | heart coverage holes |
|---|---|---|---|
| covdiv (current) | 0.36 | 20.8 | 22% |
| **stopgrad** | **0.66** | **16.5** (below floor!) | **33%** |
| covref (fixed-ref denom) | 0.15 | 15.6 (unstable) | — |
| **covdiv + gather-aux (λ0.5)** | **0.90** | **20.9** | 25% |

- Stop-grad **transmits** (0.36→0.66; raw signed slopes +0.31→+0.58, genuinely corrective — sign-flip
  never fired) → **refutes Adam-washout**. But it degrades reconstruction below the floor.
- **Why:** `cos(g_stopgrad, g_true)=+0.98` at the start, but the detached denominator diverges from the
  true objective as slices move; the un-damped `1/cov`-amplified surrogate over-moves slices and tears
  33% coverage holes in the high-contrast heart → heart PSNR collapses even as breath-slope rises.
- **Gather-aux** gets the slope (0.40→0.90) *without* the damage (holes 25% ≈ covdiv 22%, PSNR ~21)
  because the covdiv primary keeps penalizing holes while the aux only adds placement signal.

**Caveats (honest bounds):** the exact 0.90 is an *optimistic upper bound* — in the fixed-phase
breathing sim the input is a clean resample of V_gt, so the gather term has a zero-residual optimum;
real recovery will be weaker. These are *sighted-optimization* numbers (test the objective/renderer,
not the blind trained net). Prize is ~0.5–1 dB, secondary metric, EF unaffected.

## 3. IMPLEMENTED (2026-07-07, behind `gather_weight`, default 0.0 = no-op)

**Now wired into `training/loss.py`** (`compute_volume_intensity_loss`), mirroring `diffusion_weight`:
`if gather_weight>0: L_gather = |sample_volume(V_gt, world_points) − input_intensity|` (masked to real
pixels, fp32, `padding_mode='zeros'` so predicting outside the FOV self-penalizes), added to the
objective sum; `gather_weight` config knob + `loss_gather` logged (train+val). **Applied to the WHOLE
volume — NO mask** (resolved: the gather is the same "map any phase → V_gt" objective as the splat and
is correct on the heart too — a phase-t blood pixel belongs at the *target-phase* blood location, which
is where V_gt matches; masking is only a fallback ablation if the heart degrades). Cost: **one
`grid_sample`** — negligible vs the splat/forward. Verified: `gather_weight=0` bit-identical no-op;
`>0` finite, differentiable (grad reaches the point head), loss minimized at correct placement (0.00
perfect → 0.05 perturbed); 208 tests pass. **Turn on:** `loss.volume.gather_weight=0.1` (start ~0.1–0.2,
sweep toward ~0.5; docs/38). Below is the original design note.

The gather auxiliary mirrors the **existing `diffusion_weight` term** in `training/loss.py` exactly.
Four wiring points (~15 lines total):

1. **Config** (`mri_volume.yaml`, `volume:` block, next to `diffusion_weight`): add `gather_weight: <val>`
   (`**self.volume` auto-plumbs it as a kwarg — no other plumbing).
2. **Signature** (`compute_volume_intensity_loss`, loss.py:321): add `gather_weight=0.0`.
3. **Term** (mirror the `if diffusion_weight > 0:` block, loss.py:383): compute the intensity exactly as
   `splat_predictions` does (`images.mean(dim=2)`, /255 if needed), gather:
   `s = sample_volume(V_gt, world_points.reshape(B, S*H*W, 3))`, mask `input>1e-3`,
   `loss_gather = |s − input|.mean()` × `gather_weight`; add `"loss_gather"` to the out dict.
4. **Sum** (loss.py:110): `+ vol_loss_dict.get("loss_gather", 0.0)`.

`gather_weight=0.0` ⇒ exact no-op (bit-identical), like the diffusion term. `sample_volume` already
exists in `vggt/utils/splat.py`. Guard `gather_weight` in the freeze/DDP tests same as any loss knob.

**The one real design wrinkle (NOT surfaced by the fixed-phase sim):** the gather target `V_gt =
phases[t_target]`, but the multi-frame input slots are at *arbitrary* cardiac phases `t` (extraction-
only, blind). So the moving **heart's** content won't match `V_gt` at the correct depth (different
contraction) — only the **static tissue** (chest wall, liver; phase-invariant) matches. Because
breathing is a *rigid whole-slice SI shift* and static tissue is the slice majority, the breath depth
is still recoverable from the static content, so the aux should work — but the heart pixels get a
misleading per-pixel signal. **Recommended:** either accept it (rely on the static majority + the
smooth-Δ regularizer + primary loss to absorb heart noise) or **mask the gather aux to low-cardiac-
motion voxels** (`compute_motion_mask` complement) so it never fights cardiac motion. Validate on the
DVF slope + motion PSNR + EF; do not just watch full-volume PSNR.

## 4. What remains unproven / next step

The **real 941M-network warm-start finetune** with the gather-aux loss (CKPT_ONLY from 4wok, short) is
the one experiment that turns this from *sighted-proven* to *net-proven* — score on breathing DVF slope
(`tools/exp_4wok_analysis.py`) + motion PSNR + coverage holes, confirm EF (`tools/exp_4wok_p95.py`)
does not regress. Since the prize is a ~0.5–1 dB secondary lever and EF (the clinical headline) already
works, this is optional — pursue only if a through-plane endpoint (strain / regional wall motion)
becomes the target. See [[36_roadmap_future_enhancements]].

## 5. Finetune run (set up 2026-07-07): 4wok warm-start + gather_weight=0.5, 1-frame

Launcher: `sbatch/train_mri_volume_diffusion_ft_gather.sh` (copy of the diffusion sbatch). Setup:
- **Config** `mri_volume_diffusion` (4wok's own config: reference-slot, aggft, diffusion_weight=1000).
- **Warm-start** from 4wok (run 217720691) **weights-only**: `scratch/base_weights/4wok_weights_only.pt`
  (= `{'model': ...}` stripped from `checkpoint_last.pt`). Fresh exp dir + **new wandb run** (branches off).
- **Overrides:** `max_img_per_gpu=12` (approximate 1-frame — see below), `loss.volume.gather_weight=0.5`,
  `max_epochs=100`. LR = the proven default schedule (linear warmup→5e-5→cosine), fresh via the
  weights-only load.
- Logs the new metrics (docs/38) from epoch 0 — watch `metric_recov_frac_heart`↑ + `metric_resp_slope_dz`/
  `_epe_dz_mm`↑ **without** `metric_hole_frac_heart`↑ (the decision rule).

**Two gotchas found + handled (both smoke-caught):**
1. **CKPT_ONLY does a FULL resume, not weights-only.** `checkpoint.resume_checkpoint_path` loads
   optimizer + `prev_epoch` too, so a full `checkpoint_last.pt` resumes at epoch 191 → with
   `max_epochs<191`, **zero training** (smoke ran one val and exited). Fix: strip to `{'model': ...}`
   → epoch 0, fresh optimizer/schedule. (CLAUDE.md corrected.) Re-smoke confirmed `epoch: 0` + real
   `Train Epoch` steps with `train_loss_gather≈0.009` and flowing grads (~31 GB/A40).
2. **The sampler was rewritten (docs/28): the current default is a fixed S=20 multi-frame set, NOT
   4wok's one-frame-per-slice.** So the *exact* 1-frame regime can't be reproduced with the fixed-S
   sampler. `max_img_per_gpu=12` is the safe approximation — with `reference_slot`, S=12 guarantees full
   coverage (room=11 ≥ planes−1) and gives *exactly* one-frame-per-plane for full-12-plane subjects (a
   few extra frames only for small-FOV subjects).

**Metrics verified LIVE through the real trainer** (smoke, real data): all docs/38 keys populate
(`val_loss_gather`, `recov_frac_heart≈0.70`, `psnr_static≈35`, `hole_frac_heart≈0`, `resp_*`).

**Caveats:** not bit-exact to 4wok (sampler drift + fresh LR schedule), so a clean read wants a paired
`gather_weight=0` control in the SAME S=12 regime (not yet run). The epoch-0 val gives the 4wok-in-this-
regime baseline for a within-run trajectory. Submission was blocked at launch by the account job-submit
limit (`AssocGrpSubmitJobsLimit`) — resubmit once a slot frees:
`sbatch --job-name=vggt_ft_gather05 --output=slurm_logs/<ts>_%j.log sbatch/train_mri_volume_diffusion_ft_gather.sh`.
