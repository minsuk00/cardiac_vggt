# 20 — Appearance synthesis is NOT a breakthrough (feature-splat decoder adds +0.03 dB)

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> **Toy-experiment** (`/toy-experiment`) testing doc-19's proposed fix — appearance synthesis (splat
> *features* + a 3-D decoder / learned renderer that *generates* target-phase appearance) — to see if it
> breaks the ~21 dB warp-only motion-PSNR ceiling on HELD-OUT subjects. Three code-verified (prove-it
> clean) experiments, all agree:
>
> **CONCLUSION (high confidence): NO — not a breakthrough, and as feature-splat not even a win.**
> 1. **Recoverability ceiling (real CMR, no training):** the recoverable target-phase appearance tops out
>    at **subject_temporal_interp = 28.1 dB even with DENSE adjacent-phase volumes** the model never has;
>    the **population template (avg heart at t) = 14.4 dB (below the 16.8 floor)** — hearts aren't
>    anatomically aligned in the canonical grid, so there's NO free population-prior gain. A sparse-slice
>    synthesizer is bounded *below* the breakthrough line (≥28) before training anything.
> 2. **Feature-splat + 3-D decoder (real CMR, trained, held-out val):** V_canon+features **19.38** vs
>    intensity-only (= the refiner) **19.36** → **features add +0.03 dB**. Feature-*only* is worse (15.8).
>    Scramble ablation (zero features) → floor, so the decoder *uses* the features (no target_t leakage) —
>    they just carry no extra recoverable target-phase appearance.
> 3. **Synthetic known-answer toy:** synthesis beats transport by +6.8 dB **only on the population-predictable
>    appearance** (recovers it, generalizes) and **HALLUCINATES** the subject-specific component (fabricates
>    at the population-mean location). On real unaligned hearts the predictable bucket is tiny ⇒ little to
>    recover, much to hallucinate.
>
> **Mechanism:** the 14 dB oracle gap = ~3-4 dB renderer/coverage (already the refiner, not synthesis) + a
> *small* population-predictable bucket (≈worthless here, template 14.4) + ~7-9 dB **subject-specific
> appearance that is NOT in the inputs** (r-blind, input-t-blind, ~10 sparse frames). No renderer synthesizes
> information the inputs don't contain.
>
> **⇒ The real lever is the INPUT CONTRACT, not the renderer:** more frames per slice / a temporal stream /
> input cardiac-t (ECG or content-inferred) / a 5-D motion-resolved reference — to make target-phase
> appearance recoverable. A fancier decoder on the current sparse, blind inputs cannot.
>
> **⚠️ This CORRECTS [[19_motion_correction_warp_ceiling]] §5**, which listed "appearance synthesis (learned
> renderer/decoder)" as the lever that breaks ~21 dB. Measured at **+0.03 dB**, and bounded below 28 by
> information — it does not. Doc 19's *diagnosis* (warp-limited, ~21 ceiling, appearance wall) stands; its
> *prescription* was wrong.
>
> **Report:** `_html/20_appearance_synthesis_test.html` ·
> https://github.com/minsuk00/research-reports/blob/main/2026-06-22_0530_appearance-synthesis-test.html

**Date:** 2026-06-22
**Status:** Conclusive (held-out, prove-it-clean on both load-bearing scripts). Bounds: in-distribution
simulator; a learned-projection / larger decoder unverified (could add a fraction of a dB, bounded by the 28
ceiling); the contract-relaxation recommendation is reasoned from the bounds, not yet trained.
**Related:** [[19_motion_correction_warp_ceiling]] (corrected prescription), [[11_unet_refiner_results]]
(the +1 dB refiner = coverage-fill not synthesis), [[04_inference_information_contract]] (the contract that
caps recoverability), [[13_limitations_and_improvements]].

---

## 1. Method & results

Breakthrough thresholds (set with the debate panel): held-out motion PSNR ≥28 = breakthrough, ≥25 = real
win, ≤22 = no.

| held-out motion PSNR | value | note |
|---|---|---|
| population template (avg heart at t) | **14.4** | below floor ⇒ no population-prior gain (hearts unaligned) |
| identity floor | 16.8 | |
| transport (V_canon, OLD ckpt @256) | 19.2 | warp output (prior warp ceiling ~21 @518) |
| intensity-decoder (= refiner) | 19.4 | known modest coverage-fill |
| **feature-decoder (V_canon+features)** | **19.4** | **+0.03 over intensity ⇒ features add nothing** |
| feature-decoder (features only) | 15.8 | random-projected features alone are a worse input |
| feature-decoder SCRAMBLED | 11.8 | features zeroed → floor ⇒ decoder uses features, no leakage |
| subject all-phase mean | 23.6 | subject anatomy, no phase appearance (dense data) |
| **subject_temporal_interp (UPPER BOUND)** | **28.1** | recoverable appearance with DENSE t±1 volumes |
| oracle (perfect placement) | 35.0 | |

Synthetic toy: transport 19.3 → synthesis 26.1 held-out (+6.8) where appearance is population-predictable
(intensity error 0.288→0.072 RECOVERED); subject-specific disocclusion HALLUCINATED (0.21 error at pop-mean
location). Train≈test ⇒ the recovered part generalizes.

## 2. Debate (3 agents) & adjudication

Optimist predicted 25-27 ("real win"); skeptic 22-24 ("beat-the-mean ≈0, OOD collapse"); red-team flagged
the `target_t`-in-backbone leakage risk → mandated the scramble ablation + beat-the-refiner bar + held-out
subjects + LR hygiene. Result landed **below even the skeptic** (+0.03), and the recoverability bound
explains why (predictable bucket tiny on unaligned hearts). All red-team guards passed (scramble shows no
leakage; beat-the-refiner is the bar and it's +0.03; held-out; LR converged).

## 3. prove-it

2 reviewers, both clean. Feature pipeline verified correct (hook chunk-order, feature↔world-point alignment,
shared full-rank JL projection, splat gate, scramble/eval) ⇒ +0.03 is a real finding, not a bug.
Recoverability reproduced exactly (14.39/23.62/28.14); 14.4 confirmed real (phase-shift invariant);
train/val disjoint.

## 4. Recommendation

Stop pursuing renderer/decoder changes for motion PSNR (warp ~21 is near the contract's ceiling; appearance
synthesis +0.03). Invest in **relaxing the input contract** so target-phase appearance becomes recoverable:
more frames per slice / temporal stream, input cardiac-t (ECG or an auxiliary content-inference head), or a
5-D motion-resolved reference. Scoped, not yet tested.
