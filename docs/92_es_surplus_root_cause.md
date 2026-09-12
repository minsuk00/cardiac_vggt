# 92 — Root cause of the residual EF bias: ⅓ genuine ES amplitude, ⅔ mis-rendered ES wall band + real ED under-expansion

> **TL;DR & takeaway** — The −12.6 EF bias (no-reg ep300, n=143, shared by all arms) is NOT the
> single "ES under-contraction" it was assumed to be. Measured decomposition: **(a) EDV −13 ml is
> real** — the rendered ED blood pool is genuinely too small (pixel-space and nnU-Net agree);
> **(b) the ESV +12 ml surplus is a mixture** — only ~⅓ of the surplus volume is blood-bright
> (genuine residual under-contraction, ~3–5 ml/subject, bounding amplitude-side fixes at ~2–4 EF
> points — consistent with the TD loss buying exactly 1.8), while ~⅔ is gray/dark tissue that
> nnU-Net counts as cavity because the model renders the **compact myocardial wall half-missing in
> the contraction annulus** (the ring that is blood at ED and should be dark wall at ES sits at
> 0.44 normalized brightness; 97% of the surplus localizes there). Two oracle interventions were
> **refuted**: Gaussian-blurring GT to pred sharpness leaves nnU-Net's ESV unchanged (~0 ml), and
> synthetically *sharpening* pred's boundary makes ESV slightly WORSE (+13.8→+17.0) — nnU-Net
> reads wall POSITION, not softness, so "sharpness losses" are off the table. Arithmetic ceilings:
> fixing ED alone → bias −8.1; removing the ES artifact alone → −2.7; both → ≈0. Implied lever
> (docs/91-successor arm): graded temporal-swing L1 weighting (concentrates loss on the annulus —
> the shipped binary `heart_weight` 11× ROI already failed to fix this, so the *distribution
> within* the heart is the new content) + ED/ES-biased target-phase sampling using the
> pre-existing `scratch/data/whs/cardiac_phase.csv` labels. Evidence figures: `temp/ef_fix/`
> (01–07 walkthrough); repro scripts: `tools/es_surplus_diag/`.

All measurements: no-reg ep300 (`212682843_noreg224_pooled1337`, the chosen paper model), the
frozen `temp/dice_ef_sweep/mirror_full/` eval bundle, n=143 subjects (the 144-subject EF cohort
minus the baseline-only dropout). Session date 2026-08-28.

## 1. Starting point (what docs/91-era analysis had established)

- EF bias ≈ −12.6 for every arm; EDV/ESV seg errors antisymmetric (median −13.4 / +11.6 ml).
- Per-subject failure is subject-driven (cross-arm |EF err| Spearman 0.73–0.79), uncorrelated
  with breathing-draw severity, geometry (D, dz), heart size, FOV, or cohort/center/scanner.
- LV Dice at ED does not predict failure (r=−0.06, ns; median 0.89); Dice at ES does (r=−0.50,
  p<1e-4) — hence the working label "ES under-contraction". This doc revises that label.

## 2. The seg-vs-pixel contradiction (the key measurement)

Measure the same reconstructions' cavity volumes two independent ways:

1. **nnU-Net** (Task114, the EF pipeline's segmenter): ESV error **+11.6 ml** median.
2. **Calibrated pixel count** — bright-blood voxels inside a fixed GT-space ROI (dilated GT
   ED-LV), threshold = midpoint of each image's own blood/myocardium medians (so global
   contrast differences cancel): ESV error **−4.4 ml** median (pred pool slightly SMALLER
   than GT), Wilcoxon p=1e-8 in the OPPOSITE direction.

Per subject the two errors are **uncorrelated** (Spearman r=−0.08, p=0.33) — the seg surplus
carries no signal about pixel-space cavity size. At ED the two rulers AGREE (−13.4 vs −10.3 ml,
p=3e-18) → the ED deficit is real image content; the ES surplus is not.

Controls (all pass): pixel ruler tracks true ESV on GT (r=0.59, p=1e-14); pred blood pool is not
displaced out of the ROI (thresholded-pool Dice 0.78 median, CoM shift 1.2 mm, 0/143 below 0.5);
full 12-phase pseudo-volume curves give pred a contraction fraction of 107% of GT's ED→ES span;
the ED→ES annulus is NOT left blood-bright (paired annulus intensity Δ=−0.009 vs GT).

## 3. Localizing the surplus (nnU-Net re-run on pred ES frames)

Segmenting the pred ES frames directly reproduces the surplus (+13.0 ml — pipeline-consistent)
and lets the extra voxels (pred-LV ∧ ¬GT-LV) be characterized:

- **Where**: 97% (median) inside the GT ED-LV footprint — i.e. the contraction annulus.
- **Brightness** (per-voxel, 0=muscle / 1=blood, per-subject calibrated; robust to whether the
  references come from the pred or the GT image):

| surplus component | pred-ref | GT-ref |
|---|---|---|
| blood-like (>0.7) — genuine residual blood | 31% | 38% |
| gray (0.3–0.7) — smeared boundary band | 36% | 34% |
| dark (<0.3) — plainly mislabeled tissue | 34% | 28% |

So **~⅓ of the surplus is genuine under-contraction** (~3–5 ml median per subject — present in
most subjects, not just a tail; ~22–35/143 subjects have blood-like majority) and **~⅔ is
non-blood tissue** counted as cavity. Every subject mixes both components, which is why example
panels do not sort into clean "artifact" vs "genuine" categories.

## 4. Two refuted mechanisms (oracle interventions, same nnU-Net)

1. **Blur (REFUTED)**: Gaussian-blur GT ES frames to per-subject pred-matched gradient energy
   (median σ≈0.6 px; also σ=1.0) → nnU-Net ESV change ≈ **0 ml** (medians −0.16 / −0.57).
   Geometry identical by construction; the segmenter is robust to smoothness.
2. **Sharpness fix (REFUTED)**: sigmoid-commit every pred ES boundary voxel to its nearer
   blood/muscle pole (perfectly decisive boundary, same transition location) → ESV got slightly
   WORSE (+13.8 → +17.0 ml; EF bias −13.7 → −15.4). nnU-Net delineates the LV by finding the
   **compact myocardial wall** and takes everything inside it (its M&Ms training convention
   includes dark trabecular voxels in the cavity — which is also why ⅓ of the surplus being
   dark does not deter it). Sharpening does not move where the wall is.

Combined mechanism: at ES the model renders the wall **half-missing in the annulus** (0.44
normalized brightness where compact dark wall should be), so the segmenter finds the wall too
far out. The failure is the *rendered position/completeness of the wall*, not image softness and
not (mostly) bulk cavity motion — the bulk blood pool contracts fully (§2 controls).

## 5. Per-lever ceilings (arithmetic on the existing per-subject numbers)

| scenario | EF bias | EF MAE |
|---|---|---|
| current no-reg | −12.6 | 12.9 |
| ED fixed only (EDV:=GT) | −8.1 | 9.6 |
| ES artifact removed only (ESV −= gray+dark surplus) | −2.7 | 5.0 |
| ES fully fixed only (ESV:=GT) | −4.4 | 5.9 |
| ED fixed + artifact removed | **+1.1** | **4.5** |

Note the "artifact removed" ceiling is a *measurement* ceiling — §4.2 shows it is NOT reachable
by pixel-space sharpening; reaching it requires the wall actually rendered at its contracted
position. Amplitude-side losses are bounded by the ⅓ blood share at ~2–4 EF points (TD's
measured 1.8, docs/91 §6, sits inside that bound).

## 6. Implied fix (next arm; PREDICTION, not yet validated)

1. **Graded temporal-swing L1 weight**: `w = 1 + λ·normalize(swing)` with
   `swing = max_t − min_t` over the post-aug 12-phase stack (the graded form of the existing
   `compute_motion_mask`, computed on-the-fly — one std/minmax over the already-loaded phases,
   sub-ms). Rationale: the shipped binary `heart_weight: 0.5` (~11× effective per-voxel inside
   the whole-heart ROI) trained INTO no-reg and the failure persists — so coarse heart
   upweighting is insufficient; the annulus (max-swing voxels) needs the pressure specifically.
   Photometric aug is phase-consistent gamma/bias with noise disabled in all tiers
   (`gpu_aug.py`), so the swing pattern survives augmentation and per-sample normalization
   absorbs the scale.
2. **ED/ES-biased target-phase sampling**: wrapped bimodal Gaussian over the 12 phases (bumps
   at t=0 and per-subject t_ES, σ=1 phase, peak ~3× uniform — σ hedges the ±1-phase ES label
   uncertainty). ES labels ALREADY EXIST: `scratch/data/whs/cardiac_phase.csv` (1535 subjects,
   ED/ES on native T from the persisted per-phase `heart_seg.nii.gz` siblings, with
   `unimodal_ok`/`curve_mono_frac` quality flags) — needs only `t_ES12 = round(ES/T·12) % 12`
   and a coverage join against the training split.
3. Judge by the docs/91 §4 rule PLUS re-running this doc's diagnostics on the new ckpt
   (annulus brightness → toward muscle level; EDV deficit → shrinks; surplus decomposition).

## 7. Evidence artifacts

- Walkthrough figures: `temp/ef_fix/01…07*.png` + `temp/ef_fix/README.md` (numbered logical
  chain; 07 panels contrast typical gray-rim vs genuine blood-bright subjects).
- Repro scripts: `tools/es_surplus_diag/` (pixel-space diagnosis, alignment controls, blur
  dump, walkthrough figure generator).
- Data: `temp/dice_ef_sweep/es_diag{,2}_noreg.csv`, `temp/dice_ef_sweep/es_blurtest/`
  (blur/pred/sharpened nnU-Net inputs+segs, `blur_esv_results.csv`,
  `pred_surplus_analysis.csv`, `surplus_voxel_decomposition.csv`, `sharp_esv_results.csv`).
- All generated read-only against the frozen eval bundle; `evaluation/` untouched (its
  `run_seg.sh` executed as-is).
