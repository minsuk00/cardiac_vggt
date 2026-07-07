# 33 — Conclusive analysis of 4wokxzov: EF, breathing, in-plane & through-plane motion, + fixes

> **TL;DR & takeaway** *(human-facing; the rest is the agent-facing record)*
>
> Full per-axis probe of model **4wokxzov** (run 217720691: reference-slot, DPT head, 1-frame-per-slice,
> trained with an L2 DVF-smoothness "diffusion" regularizer — NOT generative; forward unchanged), n=30 val,
> DVF read directly in mm, EF via nnU-Net Task114, stress-tested by a 4-agent debate.
>
> **Headline (positive, and it corrects earlier in-session pessimism):**
> - **EF is RECOVERED.** pred-vs-true EF slope 0.773 / Pearson 0.765, but the debate showed this is
>   **overstated** — honest robust estimate **Spearman ≈ 0.55, leak-excluded Pearson ≈ 0.68, slope ≈ 0.74**:
>   the model recovers **~half the per-patient EF spread** with mild under-contraction (pred 54 vs true 63).
>   **Learned, not a reference-plane copy** (removing the reference plane barely changes it: 0.77→0.68).
> - **In-plane cardiac motion is real & localized** (mean 0.5 mm diluted; **p95 2.87 mm, max 6.7 mm** at the
>   myocardium) → drives EF.
> - **Through-plane cardiac motion is minimal** (p95 0.49 mm) but **barely matters** (LV volume is in-plane-dominated).
> - **Breathing: partial.** slope 0.35, deep breaths (≥12 mm) get only ~5.7 mm and **~54 % are ignored**;
>   residual ~1.9 dB in a ~12 % tail. Cause is SPLIT: fixable renderer (coverage-division weakens the
>   z-gradient ~2×) + info-limited blind-r (corr 0.52 ⇒ partial extraction, refutes doc-04 "unrecoverable").
> - **The ~7–9 dB (motion-scale) reconstruction gap is subject-specific appearance the 1-frame acquisition
>   never observed** — info-limited, head-independent, the accepted contract price (docs 19–21).
> - **The reference-SLOT design drives EF, not the regularizer/head:** L1-TV recovers EF equally (slope 0.79)
>   at its FINAL checkpoint; the earlier "flat -0.026" was an undertrained checkpoint. bspline applies 3× more
>   through-plane motion but same PSNR (cosmetic). Head is second-order.
>
> **Bugs caught this session (now fixed):** (1) eval built the dataset WITHOUT `reference_slot=True` → model
> got a random slot-0 → artifactual "no cardiac motion"; fixing it lifted the model +2 dB and made DVF vary
> with phase. (2) `recovery_pct=137%` is a broken statistic (use residual 1.93 dB / ~31 %). (3) "gap_to_ceiling"
> mislabeled the do-nothing FLOOR as a ceiling. (4) nnU-Net seg silently failed ×4 (wrong trainer name;
> correct = `nnUNetTrainerV2_MMS`).
>
> **Report:** `_html/33_4wok_conclusive_analysis.html` (self-contained, beginner-facing, many figures).

**Date:** 2026-07-04. **Status:** Conclusive on 4wok; two fixes proven-by-mechanism (EF multi-plane via doc-24
ablation; head second-order for EF via L1-TV final EF), breathing renderer-vs-info split argued not fully
isolated. **Model:** 4wokxzov = 217720691.

---

## 1. Protocol
n=30 val, `reference_slot=True` (slot 0 = target-phase mid-ventricular reference), S=12 (NOTE: trained S=20 →
absolute numbers mildly OOD; relative contrasts robust). resp OFF isolates cardiac; resp ON (group_by_burst)
= breathing. DVF = `world_points − scanner_coords` in mm. EF: `measure_model_contraction.py` dumps V_canon per
target_t → nnU-Net Task114 (2d, `-tr nnUNetTrainerV2_MMS`) → `analyze_model_contraction.py`.
Scripts: `tools/exp_4wok_analysis.py`, `exp_4wok_p95.py`, `render_4wok_qualitative.py`, `build_4wok_report.py`.
Data: `result/analysis_4wok/{summary,comparison_3way,p95_dvf,ef_honest}.json`, `scratch/phase_analysis/4wok_vols/ef_4wok.json`.

## 2. Per-axis findings (measured)
| Axis | Number (4wok) | Verdict |
|---|---|---|
| EF | slope 0.773 / Pearson 0.765 → **honest Spearman 0.55, leak-excl 0.68/0.74**; pred 54 vs true 63 | RECOVERED ~½ spread, mild under-contraction. PROVEN learned (leak-excl). |
| In-plane cardiac | mean 0.5 mm, **p95 2.87, max 6.7 mm** | Real, localized. PROVEN. |
| Through-plane cardiac | **p95 0.49 mm** | Minimal but low-impact on EF/PSNR. PROVEN minimal; harmlessness inferred. |
| Breathing | slope 0.35, deep 17.6→5.7 mm, 54 % ignored, residual 1.93 dB | Partial; deep tail under-corrected. PROVEN. Cause split (renderer vs info) OPEN. |
| Recon gap | model ~23 dB vs do-nothing ~21 vs perfect-placement oracle ~35 (motion scale ~14 dB) | ~7–9 dB info-limited appearance (imported from docs 19–21); ~3–4 dB renderer/coverage (partly fixable). |

## 3. 3-way head comparison
| head | breathing slope | in-plane p95 (mm) | through p95 (mm) | recon PSNR | EF slope |
|---|---|---|---|---|---|
| 4wok (diffusion L2) | 0.35 | 2.87 | 0.49 | 22.7–23.9 | 0.77 |
| reference (L1-TV) | 0.36 | 3.12 | 0.69 | 22.6–23.9 | **0.79 (final)** |
| bspline | 0.22 | 4.01 | 1.48 | 22.6–23.9 | **0.84** |
Head is **second-order**: same PSNR, EF driven by the reference slot (L1-TV ≈ diffusion), bspline's extra
through-plane motion is cosmetic for PSNR.

## 4. 4-agent debate outcome
Auditor found bugs #2/#3 above + the S=12/20 mismatch + a whole-anatomy (not cardiac) DVF mask caveat.
EF-skeptic quantified the overstatement (outlier + reference-plane leak → honest Spearman ~0.5) but confirmed
recovery survives (leak-exclusion 0.68; reference-plane is only ~15 % of LV volume). Limits-skeptic re-scoped:
breathing is NOT the top problem given EF recovered; the ~14 dB motion gap decomposes ~0.5 head + ~3–4
renderer/coverage (fixable) + ~7–9 appearance (info-limited); corr 0.52 refutes doc-04's "respiratory
unrecoverable". Adjudicator: EF/in-plane/through/breathing/head-second-order all PROVEN; two open items
(diffusion-vs-EF → resolved: L1-TV final 0.79; reference-plane leak → resolved: leak-excl 0.68).

## 5. Problems + fixes (with evidence)
- **EF under-contraction (slope <1):** more reference planes. Proven mechanism = doc-24 coverage ablation
  (slope 0→1.03 at k=6 observed planes); 4wok already at ~0.77 from k=1 reference. **Well-founded.**
- **Deep-breath under-correction:** coverage-free/inverse-warp renderer (doc-19 E0) recovers shallow/mid; deep
  tail is blind-r info-limited (54 % ignored, DVF caps at 5.7 mm) → **partial fix, bounded.**
- **Minimal through-plane cardiac:** bspline head (3× motion) — but same PSNR & barely affects EF → **cosmetic**
  unless bspline EF shows a lift (pending).
- **Appearance gap (~7–9 dB):** input-contract relaxation — proven info-limited (docs 20/21, decoder +0.03);
  only free win = proximity sampling +0.8 dB. **Report as accepted limit, don't chase.**

## 6. Open / caveats
Re-run at S=20 (training regime) to confirm absolutes. bspline EF DONE (slope 0.84, Spearman 0.59) — confirms head second-order for EF too.
Breathing renderer-vs-info split not fully isolated (coverage-free-renderer probe on fixed predictions un-run).
EF via segmenter on blurry recon + n=30 → report robust Spearman, not Pearson/slope.

## 7. Round-2 debate — ranked limitations + roadmap (2026-07-04)
A second 4-agent debate (ship-it minimalist vs renderer-champion vs input-contract-champion vs adjudicator)
argued the roadmap. Consensus: **good model; biggest residuals are info-limits of the fast-acquisition
contract, not fixable defects → confirm-and-bank, not redesign.**

**Ranked limitations:** (1) EF under-squeeze — bias is calibratable (pred_ef_std 8.15 ≈ gt 8.06; just fit
y=a·pred+b), rank (Spearman 0.55) is the soft part [FIXABLE]; (2) ~7-9 dB appearance gap [ACCEPTED info-limit,
dominant but least worth chasing]; (3) deep-breath tail [SPLIT: renderer-fixable shallow/mid + blind-r deep];
(4) through-plane cardiac ≈0 [ACCEPTED cosmetic — bspline 3× more Δz, same PSNR/EF]; (5) S=12-vs-20 eval
[ARTIFACT]; (6) EF-via-blurry-seg + n=30 [ARTIFACT → report Spearman+CI].

**Prioritized roadmap:** 0a re-eval @S=20 (free) · 0b robust metrics + linear EF calibration (free, removes
the 9-pt bias) · 0c proximity sampling (+0.8 dB, free eval) · **1 coverage-free renderer RE-TRAIN** (banks the
fixable half of breathing ~0.5-1 dB; won't break the appearance wall or deep-breath blind-r cap — must RETRAIN
not inference-swap, since it changes the learnable Δz gradient the downstream refiner can't) · 2 confirm
multi-frame (doc 28) · 3 more reference planes k=2-3 (EF slope 0.77→~0.9, doc-24 ablation→1 at k=6) · DEFER
multi-orientation LAX (the ONLY principled through-plane/appearance lever — a breathing SI shift is in-plane
for a LAX view; CMRxRecon raw has SAX+LAX+LVOT, reslice-simulatable — but partly breaks the 1-frame headline;
build only if a through-plane endpoint like regional wall-motion/strain becomes the goal).

**Disputes resolved:** deep-breath = half-fixable (renderer) + half-info-limited (blind-r); through-plane = not
worth fixing (cosmetic, EF in-plane-dominated); multi-orientation = correct diagnosis but defer (breaks goal,
EF already works). **#1 action:** re-eval @S=20 + bootstrap-CI Spearman + EF calibration (free), then the one
coverage-free-renderer retrain. Report §8-10 written from these two debates.
