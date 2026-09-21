# 115 — The 24-frame rhythm arms (`regular24` / `hrv24` / `af24`): design, build, and campaign state

> **TL;DR & takeaway**
> The final-paper AF experiment uses **24 frames per slice (2 nominal beats)**, not 12. Measured:
> rhythm parameters are a weak lever on Fetal CMR 4D's bin coherence (~+2° `bin_phase_sd`),
> **frames-per-slice is the strong one (~+22°)**. At 24 frames the gate's uniform ramp wraps twice,
> so frames `f` and `f+12` share an output bin and the engine averages them: under a regular
> rhythm that pair is the same cardiac phase (denoising), under `af` it is ~2–4 of 12 phases apart
> (temporal blur). **The baseline gets MORE data, not less** — that is the fairness property.
> 540 bundles (180 test subjects × 3 arms) are built; scoring moved to nnU-Net `3d_fullres`;
> `af24` was rebuilt on 2026-09-21 under the hold rule (docs/114) and re-gated. **No reconstruction
> has run on any `*24` cohort yet.**
>
> ⚠️ **Provenance:** §1–§4 are transcribed from the 2026-09-20 session handoffs and were not
> re-derived here, except where marked **[re-verified 2026-09-21]**.

## 1. Why 24 frames (measured on the 12-frame cohort, 2026-09-20)

`bin_phase_sd` = for each output bin and slice, the true cardiac phase the gate put there; circular
sd across slices, averaged over bins. Higher = Fetal's bins are more incoherent. Ceiling (random ED
per slice) ≈ 91°.

| design | bin_phase_sd | empty (slice, bin) cells | note |
|---|---|---|---|
| 12 frames ×1 | 37.2° | 0 % | the docs/111–112 design |
| 12 frames ×1, CV 0.30 | 39.4° | 0 % | rhythm lever ≈ +2° |
| 12 frames, ×2 spacing | 58.3° | **50 %** | becomes a coverage experiment — rejected |
| **24 frames ×1** | **59.7°** | **0 %** | same damage, no coverage hole — **chosen** |
| 36 frames ×1 | 77.3° | 0 % | |

## 2. Locked spec

- Cohorts `{cmrx2023, cmrx2024, cmrx2025, acdc, mnms}_{regular24, hrv24, af24}`; 180 test subjects
  (26 / 38 / 46 / 21 / 49).
- 24 frames per slice; Fetal `-numcardphase` = **12**; GT = 24 volumes; VGGT sweeps 24 reference
  frames; VGGT companion slots draw their single frame from 0..23 (either beat).
- AF R-R ~ N(1.0, **0.25**) (Markl 2015 — a 5-patient conference abstract, docs/114 §2). CV stays
  0.25: the tempting CVRR 0.285–0.748 is a 24-h Holter measure including diurnal drift.
- Fetal: **self-gated only**, no oracle arms. `fetal4d_gate.py` stays at nnU-Net `-m 2d` — it is
  the baseline's own gating, not a metric.
- VGGT arms: `base`, `diff1000`, `nogather` (log dir `210498476`), `hw0`.
- Requirement: VGGT at least on par with Fetal CMR 4D on EF.

## 3. The two-number manifest contract

`manifest["T"]` = **24** = frame count = "how many phase volumes" for every consumer.
`manifest["n_cardphase"]` = **12** = nominal beat in frames = Fetal's `-numcardphase` = the gate
ramp's modulus. Conflating them gives every frame its own bin and silently deletes the mechanism;
nothing crashes. Guards, each fault-injected:

| where | what |
|---|---|
| `fetal4d_gate.py` | ramp `theta = 2π·((f − off) % NCP)/NCP`; **time pixdim = RR/NCP, not RR/T** (RR/T would halve the engine's temporal PSF width); `T % NCP` guard |
| `run_fetal4d.sh` | NF/NCP read from the manifest only — an inherited `T` env var used to force `-numcardphase 24`, leaving 12 of 24 bins empty (fixed 2026-09-21, docs/114 §5) |
| `run_vggt.py` | `build_batch` whole-multiple guard |
| `tools/check_gate_24.py` | GPU-free 7-check gate verifier (synthetic segs); **[re-verified 2026-09-21]** all 7 pass on rebuilt `cmrx2024_af24/CMRx24_Test_P017` |

## 4. Scoring: nnU-Net `3d_fullres`

The GT seg cache was keyed on `(gt_sha256, roi_sha256)` with the segmenter absent, so switching
config would have scored 3D predictions against 2D GT segs silently. Now keyed on `SEG_CFG`
(`seg_gt/` = 2d, `seg_gt_3d_fullres/`), with a fail-closed `nnunet_config` stamp check in
`ef_dice.score()`. Rationale (n = 33, cmrx2024, 12-frame, pre-hold-fix bundles): the
Fetal-vs-VGGT EF |err| interaction is non-significant on every cohort (p = 0.66 / 0.39 / 0.97 /
0.40), so 3D does not reorder the methods; it improves both.

## 5. Build + verification state

- **Build:** array 61609375 (20 CPU shards): 540/540 bundles, 24 stacks + 24 GT each, 45 GB.
  12-frame arms rebuilt with the new code are bit-identical (`max|diff| = 0` over 120 files).
- **Degeneracy is gone at 24 frames:** ref-plane LV excursion ≥ 50 % for all 180 `af24`, 0
  duplicate-target groups — the docs/111–112 exclusion rule is moot.
- **`af24` rebuilt 2026-09-21** under the hold rule + teleport fix (docs/114): array 61615769,
  180/180 verified, swapped in; old bundles at `scratch/eval/_archive_af24_stretch_20260921/`.
  `regular24` / `hrv24` re-simulate bit-identical (180/180 each) and were not rebuilt.
  **[re-verified 2026-09-21]**
- **Gate:** `af24` re-gated by array 61616097 into fresh `af24hold_s*` work dirs (stale-seg trap,
  docs/114 §5). `regular24` / `hrv24` have not been gated.
- **VGGT 24-frame path:** single-subject L40S smoke test — 24 distinct output volumes, two clean
  cardiac cycles, `f` vs `f+12` differ (max|diff| 0.57–0.65), 16.3 s/subject.

## 6. Campaign order (not yet run)

1. Fetal `af24` recon (CPU `standard`, the long pole).
2. In parallel on GPU: VGGT recons × 4 arms, then `3d_fullres` GT segmentation. GT seg must not go
   first — nothing depends on it until predictions exist.
3. Prediction segmentation → score → compare (target: VGGT ≥ Fetal on EF under `af24`).
4. Check the gate's `bin_phase_sd` on the rebuilt `af24` (do the holds help the self-gate?).
5. `hrv24` / `regular24`; later re-run all baselines under the 3D protocol.
