# 88 — Cross-method metric hierarchy: NCC primary; intensity-gauge findings; pose_psf.py status

> **TL;DR & takeaway**
> **DECIDED: NCC is the primary cross-method metric; PSNR/SSIM are secondary.** Rationale: each
> baseline outputs intensities in its own gauge (NeSVoR pins `--output-intensity-mean=700`;
> NiftyMIC adds a ≈+1.0 pedestal; SVRTK and VGGT are **measured** scale-preserving), and NCC is
> invariant to any affine intensity map `a·x+b`, so the headline comparison never depends on the
> gauge normalization being perfect. PSNR/SSIM both do depend on it (SSIM's structure term is
> affine-invariant but its luminance/contrast terms are not). **Found, diagnosed & FIXED:**
> `prep_recon`'s self-norm takes NeSVoR's p99.9 over the **heart ROI**, but GT's preprocessing
> normalized over the **whole FOV** — on CMRx24_Test_P012 the heart only reaches 0.35 of GT's
> scale, so NeSVoR lands ~3× too bright → **6 dB PSNR with a healthy 0.82–0.84 NCC**. The
> obvious fix (percentiles over the FOV instead) is REFUTED — the baselines reconstruct a heart
> crop covering only ~6% of the FOV. The working fix (measured, **6.04 → 20.51 dB**, NCC
> invariant): anchor the recon's percentiles to the **input stacks'** percentiles over the
> coverage region — GT-free, restores exactly the input scale the method discarded. Same family
> as docs/84's ~13 dB OCMR flag. **WIRED IN + verified same day** (see §2): NeSVoR 20.60 dB,
> VGGT/SVRTK re-scored bit-identical (max diff 0.0).
> Explicitly REJECTED as overcomplicating: per-subject least-squares gauge fitting and symmetric
> percentile re-normalization of everyone (both are 2-DOF affine maps near-redundant with NCC;
> uniform rules measurably distort: −0.3 dB NeSVoR, −1.9 dB SVRTK). Also delivered:
> **`evaluation/src/score/pose_psf.py`** (3-DOF pose fit + PSF operator, standalone, verified —
> fault-injected shift recovered to 0.06 mm, PSF FWHM 8.01 mm vs 8.0 target) — **not yet hooked
> into scoring**; probe results below.

---

## 1. The gauge problem, measured

All methods reconstruct from the identical frozen `breath/stack_t*.nii.gz` inputs (already on the
canonical [0,1] scale from preprocessing) and are scored against the identical frozen
`gt/gt_t*.nii.gz`. The gauge question is only about each method's **output** intensity scale.

Why optimization does NOT automatically free the intensity gauge (unlike pose): SVR's data term
compares simulated slices against the observed slices **in intensity units**, so scaling the
volume ×2 doubles the residual — intensity is pinned by the objective. Pose floats because no
term references absolute position (slice-to-slice consistency only). But "pinned by the
objective" is a pull, not a guarantee — regularizers, bias-field variables, and output
conventions can still break it — so the treatment is keyed on **per-method measurement**
(linear fit vs GT, CMRx24_Train_P053, recorded in `image_metrics.py` comments):

| method | measured gauge | treatment in `prep_recon` |
|---|---|---|
| VGGT | identity (trained on the canonical scale) | none |
| SVRTK | ≈ identity (1.05·gt − 0.01; re-confirmed on Test_P012: in-ROI mean/max 0.085/0.384 vs GT 0.095/0.353) | clip the −1 sentinel only |
| NeSVoR | pure scale, ≈2065× (its `--output-intensity-mean=700` flag) | ÷ own p99.9, clamp [0,1] |
| NiftyMIC | scale + **pedestal** (constant offset ≈ +1.0) | − own p0.5, ÷ (p99.9−p0.5), clamp |

`SELF_NORM_METHODS = {nesvor, niftymic}` encodes the verdicts. The principle is ONE consistent
rule: invert each method's **measured** gauge, minimally. Uniform rules were tested and distort:
subtract-then-divide on NeSVoR injects an artificial offset (−0.3 dB); any self-norm on SVRTK
destroys real signal (−1.9 dB, 29.85→27.93).

## 2. The bug found this session (pre-existing, NOT yet fixed)

`prep_recon` computes the self-norm percentiles over the **heart∩FOV scoring ROI**. GT's
normalization (preprocess.py) used the **whole non-zero FOV** — and the heart is not the FOV's
brightest tissue. Measured on CMRx24_Test_P012: GT's in-ROI max is **0.35**, so mapping NeSVoR's
ROI-p99.9→1.0 makes it ~3× too bright → PSNR 6.04 dB (an earlier probe put the honest number
near 17–18) while NCC reads 0.82–0.84 (on par with SVRTK). Same disease as docs/84's OCMR
~13 dB flag (GT heart ROI peaking at 0.44 there).

**First fix idea REFUTED by measurement:** "take the percentiles over the content FOV instead of
the heart ROI" does nothing — the baselines are fed a heart-centered crop, so their recons cover
only **~5-6% of the content FOV** (measured, Test_P012: NeSVoR 6.0%, SVRTK 5.1%) and
FOV∩coverage ≈ heart ROI (p99.9 2372 vs 2384, PSNR unchanged at 2.5 dB). No self-referenced
percentile can recover a scale set by tissue the recon does not contain.

**Fix (WIRED IN, user-approved 2026-08-20):** anchor to the **input
stacks** instead — map the recon's percentiles to the `breath/stack_t*` percentiles over the
recon's coverage region — **ONE uniform two-point map for all self-norm methods**:
`(recon_p0.5, recon_p99.9) → (stack_p0.5, stack_p99.9)`. GT-free (the anchor is data the method
already received as input), coverage-proof (anchor lives on the same region by construction),
and it literally restores the input scale the method discarded — the exact defect. Measured on
Test_P012: NeSVoR **6.04 → 20.51 dB** scale-only, **20.60** two-point (plausible vs SVRTK 18.62
same subject, docs/83 stale ~17.7), NCC unchanged 0.817 (invariant, sanity). The two-point map
is safe for NeSVoR here where the old artificial-zero target was not (−0.3 dB): the target floor
is now the STACK's own p0.5 (≈0), so for a no-offset method the map degenerates to pure scale
(measured: 20.60 ≈ 20.51, both floors ≈0) — `PURE_SCALE_METHODS`' divide-only special case
becomes unnecessary (removed). SVRTK/VGGT untouched (not in the set).

**Implementation & verification (shipped):** `prep_recon(rec, method, content, stacks=None)` —
signature now takes the content mask + the variant's own input stacks (loaded lazily in
`score_subject` only for SELF_NORM methods, so scale-preserving arms pay no I/O); a self-norm
method called without stacks CRASHES (no silent fallback to the old scale). Verified by running:
VGGT and SVRTK re-scored on Test_P012 with pre/post-edit `metrics.json` diffed — **max abs diff
0.0** across every per-phase metric; NeSVoR scored 20.60 dB (matches the probe's two-point
prediction exactly, NCC 0.817 unchanged); the pose_psf probe re-run post-edit gives sane
three-column NeSVoR numbers for the first time (20.60 → 20.78 psf → 21.08 posed — note PSF now
HELPS NeSVoR on this bundle, opposite sign to docs/83's stale −0.65); `check_paths.py` ALL PASS.
Likely also resolves docs/84's ~13 dB OCMR flag (same mechanism) — verify on an OCMR subject
before checking that off.

## 3. The metric hierarchy (DECIDED)

- **NCC — primary cross-method metric.** Invariant to `a·x+b` by construction: needs no
  normalization decision at all, immune to any residual gauge error, standard in the SVR
  literature (the NeSVoR paper reports NCC). Already ranks sensibly straight through the broken
  gauge (probe: VGGT 0.879 > NeSVoR 0.82 > SVRTK 0.758). Weakness (accepted): blind to real
  global intensity errors; compresses near the top.
- **PSNR — secondary**, on the `prep_recon`-treated volumes. Valid only through the gauge rule;
  also structurally favors us (docs/83 §4.4: blur is MSE-optimal, we are smoother than GT) —
  another reason it is not the headline.
- **SSIM — secondary**, same dependence. Ours is a **global-moment** SSIM over the masked voxel
  set (NOT windowed — which is why an irregular heart ROI is fine; not comparable to skimage),
  L pinned to 1.0. Its structure factor is affine-invariant; luminance/contrast factors are not.
  Adds one thing NCC lacks: sensitivity to mean/contrast fidelity.
- **Rejected:** per-subject least-squares gauge fit and symmetric percentile re-normalization of
  both sides. Both are 2-DOF affine maps; after an optimal affine fit, residual MSE =
  var(gt)·(1−NCC²), so a gauge-fitted PSNR mostly re-encodes NCC — a third intensity treatment to
  defend, for a number we already have. Percentile matching is additionally distribution-shape
  sensitive (blur compresses tails → method-dependent residual), and renormalizing GT breaks
  continuity with every historical number.

One normalization, one place: `prep_recon` inside `score_subject`. The treated volume is what is
scored AND what is saved as `cine_*.nii.gz`, which viz and the EF/Dice seg chain consume — so any
gauge change propagates to metrics, GIFs, and Dice with zero extra plumbing. viz's display stack
(shared GT+preds ROI-p99.9 vmax → gamma 0.7) is a *display window* on top of the data scale and
needs no change (fixed vmin0/vmax1 would render hearts ~3× dark — the heart occupies only
~[0, 0.35] of the canonical scale even when correctly gauged).

## 4. pose_psf.py — built and verified standalone, NOT hooked in (paused)

`evaluation/src/score/pose_psf.py` (new file only; no existing `evaluation/` file or data
touched). Implements docs/83 §6: 3-DOF world-frame translation fit (coarse ±8 mm grid + Adam
sub-voxel refine), objective = masked NCC on ONE held-out phase (off-metric, frozen for the
other T−1), PSF = separable Gaussian (FWHM = stamp.json `thickness_mm` through-plane, 1.2×voxel
in-plane) applied on the recon's own 1.4 mm grid BEFORE sampling at GT voxel centers. VGGT runs
the same fitter, no PSF (docs/83 §4.4: already blurrier than GT). torch end-to-end (grid_sample
shift + fixed separable conv — blur commutes with translation so it is applied once outside the
loop); no registration library (none can host the shift→PSF→downsample forward model, and it is
3 parameters).

**Verified by running:** (1) fault injection — physically displaced the SVRTK recon by
(3, −2, 5) mm; fitter recovered it to within **0.06 mm** per axis; (2) PSF impulse response —
z-FWHM **8.01 mm** vs 8.0 target, unit mass preserved; (3) blur-vs-pose control — symmetric ±z
sweep shows VGGT's t00 preference is asymmetric (real, not interpolation-blur artifact).

**Probe (Test_P012, breath, fit on t00 → applied to all 12; `temp/pose_psf_probe/`):**

| arm | fitted shift (x,y,z) mm | PSNR anchored → psf → posed | NCC anchored → posed |
|---|---|---|---|
| svrtk3d | (0.2, −0.0, **+2.6**) | 18.62 → 19.55 → **20.09** | 0.758 → 0.809 |
| nesvor | (0.1, 0.6, +0.4) | 6.04 → 5.63 → 5.43 (gauge-broken, §2) | 0.819 → 0.839 |
| vggt_augaggr224hw2_ep300 | (0.3, −0.6, −2.0) | 22.39 → 22.39 → 22.14 | 0.879 → 0.875 |

Reads: SVRTK reproduces docs/83 directionally (+z float; honest gain smaller than the +0.62 dB
upper bound, as docs/83 predicted; different bundle — native-z rebuild + regenerated recons — so
exact numbers cannot match). NeSVoR's float is now near zero (docs/84 regenerated it with
`registration: none`; docs/83's −4.0 mm was a different run — confirms "measure per arm, never
derive"). VGGT: the t00 fit returns −2.0 mm, but **freeze-and-apply degrades the 12-phase mean**
(22.39→22.14) — the shift does not generalize, i.e. the global pose is ≈0 as the protocol
expects; presentation wording is an open decision.

## 5. Open / next (when pose/PSF resumes)

1. NeSVoR gauge region fix (§2) — one line, needs go.
2. Hook pose_psf into `score_subject`: three columns (anchored/`_psf`/`_posed`), record fitted
   shifts; decide whether `cine_*.nii.gz` becomes the **posed** volume (recommended — seg/viz
   then inherit the fair treatment, the original docs/85 design intent).
3. Score the 144 baseline volumes; re-score VGGT arms once for the same pose columns.
4. Decisions: pose_psf.py location blessing + commit; VGGT −2 mm wording (report
   fitted-but-degrades, vs a multi-phase fit that would likely return ≈0 directly).
