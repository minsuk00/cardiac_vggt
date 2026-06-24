# 22 — A target-phase reference slice fixes contraction AMPLITUDE (EF), NOT per-voxel motion PSNR

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Toy-experiment** (`/toy-experiment`) testing the user's idea — *"give the model a slice with the target-phase
> LV state I want; would that make MOTION PSNR much better?"* — and reconciling the apparent contradiction
> between "more frames don't help" (docs 20/21) and "target-phase observation recovers EF" (the flat-EF finding).
> Real model, no retrain, leak-controlled, prove-it-clean ×2 + per-phase audit.
>
> **CONCLUSION (high confidence): the user's instinct is right, but for the RIGHT metric. A target-phase
> reference slice fixes the contraction AMPLITUDE the model gets badly wrong (per-patient EF) — but it does NOT
> meaningfully improve per-voxel held-out MOTION PSNR.** Two non-contradictory facts:
> 1. **AMPLITUDE (fixable):** the model regresses everyone's EF to the cohort mean (~flat); a reference slice
>    recovers per-patient amplitude — toy pred-EF-vs-true slope **0.26→0.73** (scheme B vs current index;
>    input-t labeling does nothing, Clab 0.26); prior real-model coverage ablation → 1 when target phase observed.
> 2. **PER-VOXEL MOTION PSNR (not fixable by a reference):** the held-out error is appearance **PATTERN**, not
>    magnitude — per-plane cosine(M_pred,M_gt) = **0.44**, CV of per-plane α = 0.52. Even a PERFECT amplitude
>    correction (per-plane-α oracle, GT-fit) adds **+0.25 dB**; the realistic 1-reference-plane α **HURTS −0.3 dB**;
>    **98%** of the held-out error remains after the oracle global-α (⇒ ~2% is amplitude). Holds at every phase
>    (0/100 subject-phase samples gain >1 dB) — no hidden peak-systole win.
>
> **Reconciliation:** EF is volume-integrated and recovers on the OBSERVED planes / global volume; per-voxel
> motion PSNR lives on UNOBSERVED-plane appearance, which doesn't propagate (doc 21) and is information-limited
> (the appearance wall, docs 19/20). No contradiction.
>
> **⇒ DIRECTION TO TAKE:** condition on a target-phase **reference slice (Option B)** to fix per-patient
> contraction amplitude/EF — the *clinically meaningful* sense of "accurate motion correction," and the model's
> real fixable error. Do NOT expect per-voxel motion PSNR to move (information-limited; only direct target-phase
> observation helps — the acquisition trade, doc 21). And reconsider per-voxel motion PSNR as the headline: it
> conflates a fixable amplitude error with an unfixable appearance-pattern error and is dominated by the latter.
>
> **Report:** `_html/22_reference_slice_amplitude_vs_motion_psnr.html` ·
> https://github.com/minsuk00/research-reports/blob/main/2026-06-22_1508_reference-slice-amplitude-vs-motion-psnr.html

**Date:** 2026-06-22
**Status:** Conclusive for the scoped claims (real model, no retrain, leak-controlled upper bound, prove-it-clean
×2 + per-phase audit). Scope: amplitude/magnitude channel; in-distribution; clean (no-resp); single noresp-aggft
checkpoint. **Related:** [[21_motion_psnr_contract_levers]], [[20_appearance_synthesis_test]],
[[19_motion_correction_warp_ceiling]], and the flat-EF memory (`project_vggt_flat_ef_amplitude`).

---

## 1. Question & method

Reconcile "more frames don't help motion PSNR" (docs 20/21) with "target-phase observation recovers EF"
(flat-EF), and test the user's reference-slice idea. Decisive experiment (`tools/toy_amplitude_propagation.py`,
real model 218643188 noresp-aggft, no retrain): run the model TWICE on the SAME blind input (val RNG seq-seeded
⇒ bit-identical inputs, runtime-verified max|Δ|=0): target=t → V_pred_t, target=ED → V_pred_0. M_pred =
V_pred_t − V_pred_0 (predicted motion), M_gt = phases[t] − phases[0] (true motion). Test whether the held-out
error is a propagatable global amplitude scalar α (what a reference supplies) by scoring the MOTION RESIDUAL
α·M_pred vs M_gt directly (no volume reconstituted ⇒ no static-swap confound; the red-team's fix to a fatal flaw
in the first design), baseline α=1, on HELD-OUT planes (α fit on z_ref, scored on z≠z_ref). Controls: global-α
oracle, per-plane-α oracle (regional ceiling, GT-fit upper bound), premise checks (per-plane cosine, CV).
EF side: `toy_contraction.py` scheme comparison.

## 2. Results

**EF / amplitude (toy):** C0 (current index) 0.26 · Clab (labeled input-t) 0.26 · Ccov (index+1 target slot)
0.36 · **B (reference slice, no index) 0.73** (robust across coverage S=2..20). Reference recovers amplitude;
input-t labeling does not.

**Held-out motion-residual PSNR (real model, n=20×5 phases, random z_ref):** baseline 20.42 · +1-plane α
**20.15 (−0.28)** · +global-α oracle 20.51 (+0.08) · +per-plane-α oracle **20.65 (+0.25)**. Premise: per-plane
cosine 0.44, CV 0.52, residual-after-oracle-α 0.98. Per-phase: 1-plane α hurts at every t; oracle ~flat; 0/100
samples gain >1 dB. z_ref∈{max,rand,worst} all agree (worst hurts most, −0.7).

## 3. prove-it (2 reviewers, clean)

No gain-suppressing bug. Linchpin (identical input ⇒ pure model motion) verified from code + runtime. The two
risky choices (model-ED anchor; GT-peeking global/per-plane oracles) both BIAS TOWARD a larger apparent gain ⇒
≤0.25 dB is conservative. Low cosine makes the null self-consistent (a scalar can't fix a wrong pattern).
**Mandated scoping caveats (honored in report):** the proxy isolates the amplitude/magnitude channel (static/ED
error differenced out) ⇒ bounds magnitude-rescale gain only; the appearance-propagation half is covered by
doc 21; per-plane-α oracle bounds the magnitude-rescale family, not pattern import; cosine is vs model-ED anchor.

## 4. Recommendation

1. **Option-B reference-slice conditioning → fix per-patient EF/amplitude** (the model's real fixable error;
   clinically meaningful motion-correction accuracy). The user's idea, proven — for this metric.
2. **Per-voxel motion PSNR is information-limited** (appearance wall); only direct target-phase observation
   moves it (acquisition trade, doc 21). A reference's amplitude buys ≤0.25 dB even at the oracle.
3. **Reconsider the headline metric** — EF/amplitude is both wrong and fixable; per-voxel PSNR is dominated by
   the unfixable appearance-pattern component.
