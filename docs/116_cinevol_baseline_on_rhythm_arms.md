# 116 — CiNeVol as a baseline on the 24-frame rhythm arms: what it is, how it is fed, build state

> **TL;DR & takeaway**
> CiNeVol (Vogt et al. 2026) = NeSVoR with the rigid per-slice poses replaced by two learned
> **deformable** fields, one conditioned on cardiac phase, one on breathing level, all fitted **per
> subject** to one canonical 3D heart. It never estimates the two states — they are **inputs** (paper:
> ECG + respiration belt). On the `*24` cohorts it is in its native regime (24 frames/slice), so it is
> a fair third method next to Fetal CMR 4D and VGGT. We feed it the simulator's **R-peak times**
> through the paper's own Feinstein phase rule (never the true cardiac position) and the true **breath
> level** (which, in our simulator, is the displacement up to one 3-vector per subject — disclose it).
> The pipeline is built and verified: labels checked on all 540 bundles (fault-injected), one full
> 500-step fit validated end to end — **149 s/subject on an L40S, 2.4 GB**, and on that one subject
> NCC 0.846 vs Fetal 0.689 vs VGGT 0.886 (**n = 1, a smoke result, not a finding**). **The 180-subject
> campaign has NOT been run.** Everything is new files; no existing data or result was touched.

## 1. The method in four lines
- One static canonical heart `I(x)` (NeSVoR's hash grid + MLP) and two warps `Δφ(x, φ)`, `Δψ(x, ψ)`
  (one-hidden-layer MLPs on Grid4D's decomposed x-y-state / x-z-state / y-z-state hash grids).
- A pixel at `x` in a frame labelled `(φ, ψ)` is explained by `I(x + Δφ + Δψ)`, PSF-blurred, plus
  NeSVoR's slice scale / bias / correction. Loss: MAE + TV + bias + correction + a Jacobian
  (near-rigidity) penalty per warp. 500 AdamW steps from random init. No training set.
- Output is a continuous function: query any `(φ, ψ)` on any grid. **One shape per label value** — this
  is the assumption AF attacks: beats with the same phase label but different true shapes get blended.
- Not periodic in φ; φ and ψ warps are additive with no interaction term.

## 2. How it is fed (decisions, with reasons)
| Input | Choice | Why |
|---|---|---|
| Frames | all 24 `breath/` frames of every slice (240 frames) | its native regime; same input as Fetal CMR 4D in these arms |
| Cardiac φ | paper Eq. 16 (Feinstein bilinear; systole = 546 − 2.1·HR ms per beat) on the simulated R-peak times; record means per subject | the method as published; an ECG gives R-peaks, not the heart's true state (which here depends on the LV-volume curve: volume-matched resume, hold) |
| Respiratory ψ | true breath level `sin^{2n}(πr)` ∈ [0,1], 0 = end-expiration | the method needs it; the paper's own phantoms used true states. **Caveat:** ≈ the displacement up to one 3-vector/subject → keep CiNeVol out of like-for-like breathing-correction claims |
| Training pixels | inside `mask_heart_pad10` | every other baseline's ROI; closest single-stack analogue of the paper's multi-stack intersection mask |
| Query for GT volume f | φ = reference plane's own label at frame f, ψ = 0 | same instant every method is asked for; same rule as the fit (a different rule addresses a different point on the learned axis) |
| Oracle-φ row (true position as label) | **not run** | would make `af24` ≡ `regular24` for CiNeVol by construction; only useful later to split "bad labels" from "model limit" |

Full deviation list: `baselines/cinevol/DEVIATIONS.md`.

## 3. Measured
**Label vs true cardiac position** (`tools/cinevol_verify_labels.py`, 180 test subjects/arm, all
planes × frames, circular, in gated phases of 12):

| arm | Feinstein mean / p95 / max | linear `elapsed/RR` mean / p95 / max |
|---|---|---|
| `regular24` | 0.000 / 0.000 / 0.00 | 0.000 / 0.000 / 0.00 |
| `hrv24` | 0.209 / 0.657 / 2.00 | 0.000 / 0.000 / 0.00 |
| `af24` | 1.228 / 4.100 / 6.00 | 1.446 / 4.264 / 6.00 |

`hrv24`'s mismatch is **not** physiology: `pos_compress` stretches the whole beat, Eq. 16 assumes
near-fixed systole — two heuristics disagreeing. It will cost CiNeVol a little in `hrv24` for a
reason unrelated to the method. The linear label is exact there and slightly worse in `af24`.

**Hard checks, all 540 bundles: 0 failures**; with an injected one-frame timeline shift: 540/540
fail. (A) rebuilt beat index == manifest; (B) regular/hrv position from our timeline == manifest;
(C) regular: label == true position; (D) breath level ∝ stored |displacement|, one constant/subject.

**Environment:** runs in `svr` unmodified (torch 2.13+cu130), nothing installed. 15/15 upstream
tests pass incl. CUDA-encoder-vs-reference 1st + 2nd gradients. Pure-torch encoder 10.3 s/step
(86 min/fit) vs Grid4D CUDA 0.21 s/step.

**One full fit, `cmrx2023_af24/CMRx23_Test_P005`, L40S** (arm dir `cinevol_debug`, kept):
prepare 8.9 s + fit 122.7 s + export 17.3 s = **148.9 s**, peak 2.42 GB, 1.30 M training pixels,
MAE 0.435 → 0.0185, no non-finite values. Scored through the standing `image_metrics.score_subject`:

| arm (n = 1) | PSNR | SSIM | NCC | fitted z-shift |
|---|---|---|---|---|
| `cinevol_debug` | 22.21 | 0.697 | 0.846 | ~4 mm, every frame |
| `fetal_cmr_4d` | 19.21 | 0.550 | 0.689 | ~8 mm (at the bound) |
| `vggt_final518_base_ep300` | 23.50 | 0.754 | 0.886 | 0–1 mm |

All 10 planes rendered (`temp/cinevol_trial/`): heart correctly placed everywhere, cavity shrinks
toward the ES-like frame, blurrier than GT (as the paper reports), basal planes look alike
(through-plane blur). The output grid was verified numerically — edges == mask bbox, same shape as
Fetal's grid — so the ~4 mm through-plane offset is the model's, not geometry. **Hypothesis, not
measured:** a single SAX stack cannot identify through-plane breathing, so the volume settles near
the mean breathing position (this subject's mean |disp| = 4.8 mm). The scorer's 6-DOF fit absorbs it,
as it does for every classical arm.

## 4. Files (all new; nothing existing modified except the index line in docs/README.md)
`baselines/cinevol/` (package, tests, paper, patched Grid4D encoder, `env.sh`, `DEVIATIONS.md`) ·
`tools/cinevol_prepare.py` (bundle → states + observations) · `tools/cinevol_verify_labels.py` ·
`tools/run_cinevol.py` (per-subject pipeline + sharded driver; publishes by atomic rename, refuses
to write into an existing dir) · `sbatch/rhythm24_cinevol.sh` (8 shards, A40-pinned; **not submitted**).

## 4b. `af24` campaign result (2026-09-21, jobs 61627963 recon + 61627975 score; supersedes "campaign not run" above)
180/180 subjects, 0 failures, **median 234 s/subject on an A40** (207–283; L40S trial 149 s), all fits
finite (final MAE median 0.021). Scored by the standing `score/run.py` + `ef_dice.py` (nnU-Net
`3d_fullres`), GT seg cache read-only. Table = `tools/cinevol_af24_compare.py` → `temp/cinevol_trial/compare_af24.json`;
paired on subjects where every arm has the value; p = paired Wilcoxon, CiNeVol vs VGGT base.

> ⚠️ **Correction (2026-09-22, docs/119):** the image-metric row (NCC/SSIM/PSNR) below is the
> **as-saved** CiNeVol export, which writes real (unconstrained) values outside its fitting mask
> where every other baseline writes zero — an apples-to-oranges comparison. The corrected
> **masked** numbers (CiNeVol volumes zeroed outside `mask_heart_pad10` before scoring, exactly
> like every other baseline) are given alongside; the functional rows (EF, EDV/ESV, Dice, HD95,
> stroke volume, LV mass) are **unaffected by the bug** — `ef_dice.py` masks every volume before
> segmentation regardless of what lies outside it, so those numbers were always correct. Full
> account of the bug, the fix, and a wrong intermediate mechanism claim that was retracted:
> docs/119 §1.

| metric (mean) | n | CiNeVol (as-saved) | CiNeVol (masked) | Fetal CMR 4D | VGGT base | VGGT diff1000 | p vs VGGT base (masked) |
|---|---|---|---|---|---|---|---|
| NCC | 180 | 0.900 | 0.896 | 0.795 | 0.883 | 0.883 | 6e-7 |
| SSIM | 180 | 0.750 | 0.720 | 0.590 | 0.710 | 0.715 | 0.05 (n.s.) |
| PSNR dB (unit-peak) | 180 | 25.30 | **25.12** | 22.11 | 24.48 | 24.52 | 1e-8 |
| EF abs error, points | 176 | 9.77 | 9.77 | 13.15 | 7.66 | **7.05** | 0.017 |
| EF signed error, points | 176 | −9.22 | −9.22 | −11.87 | −6.31 | **−4.85** | 8e-4 |
| Stroke volume abs error, % | 176 | 23.3 | 23.3 | 29.3 | 19.4 | **19.0** | 0.013 |
| EDV abs error, % | 180 | 8.7 | 8.7 | 10.9 | 7.9 | 7.9 | 0.29 (n.s.) |
| ESV abs error, % | 176 | 17.3 | 17.3 | 25.3 | 15.7 | **13.3** | 0.46 (n.s.) |
| LV mass abs error, % | 180 | 8.2 | 8.2 | 12.1 | 9.2 | **7.4** | 0.14 (n.s.) |
| LV Dice ED / ES | 180 | 0.867 / 0.824 | 0.867 / 0.824 | 0.807 / 0.729 | 0.903 / 0.850 | **0.905 / 0.856** | 5e-6 / 3e-3 |
| Myocardium Dice ED / ES | 180 | 0.747 / 0.793 | 0.747 / 0.793 | 0.635 / 0.661 | 0.791 / 0.827 | **0.796 / 0.836** | 4e-4 / 1e-3 |
| LV HD95 ED / ES, mm | 180/178 | 8.0 / 7.4 | 8.0 / 7.4 | 10.5 / 10.0 | **6.6** / 7.3 | 6.7 / **7.1** | 1e-3 / 0.60 |

Reading (corrected): CiNeVol still beats Fetal CMR 4D on every row and still **wins the image
metrics** against VGGT even after masking — PSNR +0.60 dB, NCC +0.013 (both p < 1e-6), SSIM a tie
(p = 0.05) — while **losing the functional ones** (EF, stroke volume, LV/myocardium Dice), same as
before the correction. The masking cost CiNeVol only ~0.18 dB / ~0.03 SSIM / ~0.004 NCC — the true
size of the as-saved bug was small, not the ~0.05 NCC an earlier (buggy) fix attempt suggested.
All three methods under-contract on EF (negative bias), VGGT least. EDV, ESV, LV mass: no
significant difference. **This picture holds at 24 frames/slice; at 12 frames/slice VGGT also wins
the image metrics (docs/119 §3) — see docs/119 for the full frame-count dependence of each method.**

**The AF mechanism, measured (CiNeVol only, kept per-frame segs, n = 180):** LV volume-curve error
12.3 % of EDV, curve correlation 0.727. Spearman vs per-subject label misplacement: EF abs error
+0.35, curve error +0.38, curve correlation −0.48. By tercile of misplacement (0.89 / 1.20 / 1.61
phases): EF abs error 7.3 → 8.2 → **13.9**; curve correlation 0.84 → 0.77 → **0.57**. So CiNeVol's
functional error grows with how badly the R-peak label misplaces frames, as predicted in §1.
Not yet known: the same terciles for VGGT/Fetal (would show whether they are flat), any
`regular24`/`hrv24` CiNeVol number (the paired control for "this is AF"), curve metrics for the other arms.

## 5. Not done
- The campaign (`sbatch --export=ALL,ARM=af24 sbatch/rhythm24_cinevol.sh`), then `cinevol` added to
  the method lists of `rhythm24_metrics.sh` / `rhythm24_seg.sh` (those are existing files — user's call).
- `cinevol` is not registered in `evaluation/src/engine/run_baselines.py` (evaluation/ is
  human-curated); `tools/run_cinevol.py` is the standalone driver meanwhile.
- A40 timing (only an L40S was measured); `regular24` / `hrv24` fits; any EF number.
