# 05 — Respiratory-variant training results (5-run matrix)

> **TL;DR & takeaway.** We trained 5 aggregator-finetune runs to isolate three factors —
> respiratory-motion simulation, input-`t` conditioning, and aggressive affine aug — and
> re-evaluated every checkpoint under one common protocol with a standalone harness validated to
> reproduce the trainer's logged baselines (and a model's logged val numbers) to 3 decimals.
> **Result: training with simulated breathing is decisive.** On breathing-corrupted val, models
> never trained on breathing sit *at the do-nothing identity floor* (they cannot correct motion they
> never saw); the breathing-trained model clears the floor by **+3.5 dB bbox / +2.7 dB motion** and
> beats its no-resp twin by **+3.33 dB bbox / +2.24 dB motion**. It costs ~nothing on the clean task.
> **Input-`t` conditioning is unnecessary and slightly harmful** (dropping it is +0.47 dB better →
> validates the blind-input-`t` contract, [[04_inference_information_contract]]). **Aggressive aug
> hurts in-distribution** (it regularizes for OOD, not probed here). Best recipe so far: **v2 = resp,
> z-only, no aug.** Snapshot 2026-06-16; resp runs at epoch ~59 (stopped), no-resp at ~42 (running) —
> the gap is a conservative floor, robust to the epoch confound since no-resp is structurally at-floor.
> Full report + figures: `_html/07_respiratory_variants_analysis.html`.

---

## The runs

Five checkpoints share one recipe — aggregator-finetune (only DINOv2 `patch_embed` frozen,
aggregator+head trained), `use_z`+`target_t` on, all-12 multiphase targets, fresh from VGGT-1B,
200-epoch budget — differing only in three binary knobs. A 2×2 factorial of respiration × aug at
input-`t` off (v2/v3/v4/v5), plus v1 to isolate input-`t` (v1 vs v2).

| var | job | resp | input-`t` | agg-aug | state | epoch |
|----|----|----|----|----|----|----|
| 1 | 51695105 | ✓ | ✓ | · | stopped (crashed at maintenance) | 59 |
| 2 | 51695106 | ✓ | · | · | stopped | 59 |
| 3 | 51695107 | ✓ | · | ✓ | stopped | 60 |
| 4 | 51754121 | · | · | · | running | ~42 |
| 5 | 51754122 | · | · | ✓ | running | ~43 |

## Why a standalone re-eval was necessary

The runs' own training-time val metrics live on **different tasks** across the resp/no-resp boundary:
identity-Δ baseline = **24.7 dB** (resp runs score breathing-corrupted inputs) vs **31.9 dB** (no-resp
runs score clean inputs). The ~7 dB gap is the task, not the model — so wandb val curves are *not*
comparable across the boundary. `tools/eval_variants_matrix.py` re-evaluates all 5 checkpoints on the
same 30 val subjects × 12 phases (N=200, deterministic) under both protocols (clean / breathing).

**Harness validation (the proof).** Identity-Δ reproduces the logged `baseline_identity.json` exactly
(clean 31.896/30.443/20.740 full/bbox/motion; breathing 24.717/23.232/16.587), and the model path
reproduces var1's logged training-time val to the decimal (breathing bbox 26.272, motion 18.914). All
5 checkpoints loaded with 0 missing / 0 unexpected keys (t-embedder present only for v1).

## Cross-task matrix (N=200, dB; SSIM full-volume)

| | clean bbox | clean motion | breath bbox | breath motion | breath SSIM |
|---|---|---|---|---|---|
| identity Δ=0 | 30.44 | 20.74 | 23.23 | 16.59 | — |
| v1 resp z+t | 31.35 | 21.74 | 26.27 | 18.91 | 0.911 |
| **v2 resp z** | **31.65** | **22.00** | **26.74** | **19.28** | **0.918** |
| v3 resp+aug | 28.60 | 20.20 | 25.09 | 18.06 | 0.887 |
| v4 no-resp z | 30.91 | 21.81 | 23.41 | 17.04 | 0.846 |
| v5 no-resp+aug | 29.78 | 20.60 | 23.13 | 16.59 | 0.828 |

## Findings (all measured, not inferred)

1. **Breathing simulation is decisive on the breathing task.** Above the breathing identity floor:
   v2 **+3.51 bbox / +2.70 motion**; v4 **+0.18 bbox**; v5 **−0.10 bbox / +0.00 motion**. The no-resp
   models do essentially *nothing* to correct breathing. Direct resp−noresp: v2 vs v4 **+3.33 bbox /
   +2.24 motion**; v3 vs v5 **+1.96 bbox / +1.48 motion**. Robust to the epoch confound — v4/v5 are
   structurally at-floor (never saw breathing), so more epochs won't change the conclusion.
2. **Breathing training is ~free on the clean task.** v2 vs v4 on clean = **+0.74 bbox** (resp is
   *better*; partly the epoch gap). Simulated breathing acts as useful augmentation, not a tax.
3. **Input-`t` conditioning is unnecessary and slightly harmful.** v2 (no input-t) beats v1 (input-t)
   by **+0.47 bbox / +0.37 motion** on breathing val (and +0.30 on clean). The model content-infers
   cardiac phase → drop input-`t`, consistent with [[04_inference_information_contract]].
4. **Aggressive affine aug hurts in-distribution.** v2→v3 = **−1.65 bbox** (breathing), **−3.06 bbox**
   (clean); aug also tanks SSIM most. Expected — it regularizes for *out-of-distribution* robustness
   this gated→gated val does not probe. Resp+aug (v3) still beats no-resp+aug (v5) by +1.96 → breathing
   helps with or without aug.
5. **Best recipe so far: v2 (resp, z-only, no aug)** — top breathing-val and clean-val PSNR + SSIM.

## Caveats

- **Epoch confound:** resp@59 vs no-resp@42 (still climbing). The breathing conclusion is robust (no-resp
  at-floor); the clean-cost and aug magnitudes are snapshot estimates. Re-run epoch-matched when v4/v5
  finish (and after resuming v1–v3 to 200).
- **Mid-training snapshot;** none reached the 200-epoch budget. v1–v3 crashed at the 2026-06-15
  maintenance window and need resuming.
- **In-distribution only:** val is gated→gated (simulated breathing on gated cine). The aug payoff and
  true gated→real-time transfer are not measured here — see [[01_respiratory_motion_simulation]] §
  Future enhancements (bSSFP transient, single-shot artifacts, through-plane motion).

## Reproduce

`tools/eval_variants_matrix.py` (matrix) · `tools/pull_wandb_variants.py` (curves) ·
`tools/render_variants_panels.py` (panels) · `_html/build_variants_report.py` (report). Raw per-sample
JSON in `result/variants_eval/`.
