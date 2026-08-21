# 89 — Status: gauge fix shipped, pose_psf.py built-not-wired, full next-steps checklist

> **TL;DR & takeaway**
> **Shipped and verified this session:** the intensity-gauge bug (NeSVoR/NiftyMIC self-normalizing
> against the wrong reference → NeSVoR scored **6 dB at a healthy 0.82 NCC**) is **fixed and wired
> into `image_metrics.py`** — NeSVoR now scores **20.60 dB** on the probe subject, **17.93 dB** on
> the previously-flagged OCMR subject (was 7.2, docs/84); VGGT and SVRTK re-scored **bit-identical
> (max diff 0.0)**, proving the fix cannot have moved their numbers. `resp_corr` now pools into
> cohort summaries too (additive-only, verified). **`pose_psf.py` (3-DOF pose registration + PSF
> acquisition operator, docs/83's protocol) is written and self-verified (fault-injection,
> analytic PSF check) but NOT wired into scoring, NOT yet reviewed by the user, and NOT committed.**
> That is the entire remaining task: **user-verify `pose_psf.py` → hook it into
> `image_metrics.score_subject` → score the 144 baseline volumes → re-score VGGT arms → commit.**
> Full checklist in §4. Everything else from the docs/85 restructure handoff is done.

---

## 1. Why this doc exists

This session (2026-08-20/21) picked up from `_agent/handoff-2026-08-20-1056.md`, whose one open
item was `pose_psf.py` — rigid registration + PSF resampling for the classical SVR baselines, so
they can be fairly compared against VGGT. This doc is the **continuation point**: it summarizes
what happened, what's proven, what's guessed, and exactly what's left, so a future agent (or
future you) doesn't have to re-derive any of it from the conversation transcript.

**Read first if picking this up cold:** `docs/83_baseline_scoring_protocol_pose_gauge_and_psf.md`
(the pose/PSF protocol design — still authoritative, unchanged this session) and
`docs/88_metric_hierarchy_ncc_primary_and_gauge_findings.md` (the gauge-fix deep-dive this doc
summarizes at a higher level). `docs/85` is the scoring-restructure doc that created the
`engine/score/analysis` split this all builds on.

## 2. What shipped this session (DONE, verified by running, NOT committed)

### 2a. Intensity-gauge fix — `evaluation/src/score/image_metrics.py`

**The bug:** `prep_recon` self-normalized NeSVoR/NiftyMIC's arbitrary output units by dividing by
**their own in-ROI (heart) p99.9**. But GT was normalized by preprocessing over the **whole FOV**,
and the heart is not the FOV's brightest tissue (Test_P012: GT heart max is 0.35 of the [0,1]
scale). So the recon's heart-max got mapped to 1.0 while GT's heart-max sits at 0.35 — NeSVoR came
out ~3× too bright → **6.04 dB PSNR** despite a healthy **0.82–0.84 NCC** (NCC is affine-invariant,
so it was never wrong — it's what exposed the bug).

**First fix idea, REFUTED by measurement:** "normalize over the content FOV instead of the heart
ROI" — sounds right, does nothing. The baselines reconstruct a heart-centered crop, not the full
FOV: measured coverage on Test_P012 is **NeSVoR 6.0%, SVRTK 5.1% of the content FOV**, and FOV
percentiles over that crop ≈ heart-ROI percentiles (p99.9 2372 vs 2384 — no real difference).

**Working fix (shipped):** anchor to the **input stacks** instead of any recon-only region — the
method already received these as input, so they're GT-free and coverage-safe by construction. ONE
uniform two-point map for every self-norm method:
`(recon_p0.5, recon_p99.9) → (stack_p0.5, stack_p99.9)`, percentiles taken over the recon's own
coverage region (`rec > 1e-6`) intersected with content. `PURE_SCALE_METHODS`'s old divide-only
special case is **removed** — measured redundant, since a genuinely no-offset method's stack floor
is already ≈0 (20.60 dB two-point ≈ 20.51 dB scale-only on Test_P012 — the two-point map
degenerates to pure scale automatically when there's no real offset to correct).

**Signature change:** `prep_recon(rec, method, roi)` → `prep_recon(rec, method, content,
stacks=None)`. `score_subject` now loads the variant's own input stacks (`breath/stack_t*` for the
breath arm) **lazily, only for `SELF_NORM_METHODS`**, so VGGT/SVRTK pay zero extra I/O. A self-norm
method called without `stacks` **raises** — no silent fallback to a wrong scale.

**Verified by running (not by inspection):**
- VGGT (`vggt_augaggr224hw2_ep300`) and SVRTK (`svrtk3d`) re-scored on `CMRx24_Test_P012`
  (cmrx2024): pre-edit vs post-edit `metrics.json` diffed field-by-field over every per-phase
  metric — **max abs diff = 0.0**. Both methods take the early-return branch in `prep_recon`
  (`if method not in SELF_NORM_METHODS: return clip_sentinel(rec)`), so this is expected by
  construction, but it was proven, not assumed.
- NeSVoR (`nesvor`) re-scored on the same subject: **PSNR 6.04 → 20.60 dB**, NCC **0.817**
  unchanged (as it must be — NCC is invariant to the normalization by design; this is a sanity
  check, not evidence of the fix, and it passed).
- `check_paths.py` → ALL PASS after the edit.
- The **OCMR ~13 dB flag from docs/84** (`OCMR_fs_0012_3T`, nesvor) — same disease, different
  cohort — verified **resolved by the same fix**: PSNR **7.2 → 17.93 dB**, NCC **0.767**
  (matches docs/84's recorded 0.768 almost exactly — strong corroboration this is the same bug,
  not a new one). Docs/84's own affine-fit reference number was ~20.1 dB (an upper bound, since it
  used a linear fit against GT); 17.9 GT-free is the honest, expected-lower neighborhood.

### 2b. `resp_corr` cohort pooling — `evaluation/src/score/aggregate.py`

Per-subject `resp_corr` (Pearson correlation between predicted and applied per-plane breathing
displacement) was already computed and stored per-subject but never pooled into the cohort `all`
summary block, unlike `resp_epe_dz_mm`/`resp_slope`. Added two mirrored lines in
`summarize()` (stat + dict key), same pattern as `resp_slope`. **Verified additive-only**: miitt
cohort re-aggregated, full JSON diffed key-by-key against the pre-edit copy —
**only new key added** (`all.resp_corr = [0.986, 0.009]`), nothing else changed.

### 2c. `pose_psf.py` — new file, `evaluation/src/score/pose_psf.py` (~300 lines)

Implements the docs/83 §6 protocol end-to-end, but **is a standalone module — nothing calls it**.
Public surface:
- `fit_shift(vol_t, vol_affine, gt_xyz, mask_xyz, gt_affine, ...)` — 3-DOF world-frame translation
  fit. Objective = masked NCC on ONE phase (off-metric — never fits by maximizing the reported
  PSNR, per docs/83 §3.3). Coarse ±8 mm grid search (2 mm steps) then Adam sub-voxel refinement
  (150 steps). Pure torch, differentiable throughout (`F.grid_sample`).
- `blur_psf(vol_t, vol_affine, thickness_mm)` — separable Gaussian PSF, FWHM = `thickness_mm`
  through-plane (from the recon's own `stamp.json`, never hardcoded), 1.2×in-plane-voxel
  in-plane (Kuklisova-Murgasova 2012 convention). Kernel doesn't depend on the shift, so it
  commutes with translation and is applied once, outside the fit loop.
- `apply_and_downsample(path, shape_xyz, gt_affine, shift_mm, thickness_mm, apply_psf)` — the
  intended `load_canon` drop-in: load on the recon's own grid → optionally PSF-blur → sample at
  (shifted) GT voxel centers. `shift=(0,0,0)` + `apply_psf=True` = the `_psf` column;
  fitted shift + `apply_psf=True` (classical only) = the `_posed` column.
- `fit_subject_arm(ds, subj, method, variant="breath", fit_phase=0)` — orchestrates one
  (subject, arm) fit; returns a dict, no writes.
- `is_classical(method)` — `CLASSICAL_METHODS = ("svrtk3d", "nesvor", "niftymic")` get the PSF;
  VGGT/FC-SVR do not (docs/83 §4.4: VGGT is already blurrier than GT in z — PSF would
  double-blur).
- CLI `python pose_psf.py --subject <s> --methods <m...>` — probe/validation only, writes its
  JSON to `temp/pose_psf_probe/` (gitignored scratch), touches nothing under `evaluation/`.

Geometry note: all recon/GT affines on disk are diagonal (axis-aligned, no rotation) — verified by
inspection across VGGT/SVRTK/NeSVoR/GT on the probe subject. `pose_psf.py` asserts this
(`_diag_zooms` raises on a non-diagonal affine) rather than silently mishandling a future
rotated dataset.

**Self-verified this session (by the agent — NOT yet by the user, see §3):**
- **Fault injection**: physically displaced the SVRTK recon's affine by a known (3, −2, 5) mm and
  re-ran `fit_shift` — recovered the injected delta to within **0.06 mm per axis**.
- **PSF impulse response**: fed a unit impulse through `blur_psf` with `thickness_mm=8.0`,
  measured the z-profile's FWHM by linear interpolation across the half-max crossings —
  **8.01 mm** vs the 8.0 mm target; mass preserved (`sum ≈ 1.0000001`).
- **Blur-vs-pose control**: swept a symmetric range of pure z-shifts on VGGT and confirmed the
  fitted −2 mm z-preference on t00 is a real (if small, non-generalizing) NCC asymmetry, not a
  trilinear-interpolation artifact of the blur.
- Signature updated to the new `prep_recon` (§2a) — `pose_psf.py`'s probe CLI was re-run
  post-gauge-fix and produces sane output (see the NeSVoR three-column numbers in §2a's parallel
  entry in docs/88 §4).

**Probe results (`CMRx24_Test_P012`, breath, fit on t00 → frozen, applied to all 12 phases; JSON
in `temp/pose_psf_probe/cmrx2024_CMRx24_Test_P012_breath.json`, gitignored):**

| arm | fitted shift (x,y,z) mm | PSNR anchored → psf → posed | NCC anchored → posed |
|---|---|---|---|
| svrtk3d | (0.2, −0.0, **+2.6**) | 18.62 → 19.55 → **20.09** | 0.758 → 0.809 |
| nesvor (post-gauge-fix) | (0.1, 0.6, +0.4) | 20.60 → 20.78 → **21.08** | 0.817 → 0.838 |
| vggt_augaggr224hw2_ep300 | (0.3, −0.6, **−2.0**) | 22.39 → 22.39 → 22.14 | 0.879 → 0.875 |

Reads: SVRTK reproduces docs/83's direction (positive z float; smaller than the +0.62 dB
metric-fitted upper bound, as docs/83 predicted for an off-metric fit — different bundle though,
so magnitudes aren't meant to match exactly). NeSVoR's float is near-zero this generation (docs/84
regenerated it with `registration: none`; docs/83's −4.0 mm was measured on a different,
pre-native-z run — reinforces "measure per arm, never derive from the corruption"). VGGT: t00
alone fits −2.0 mm, but applying that shift to all 12 phases makes the mean **worse** (22.39 →
22.14) — i.e. the shift doesn't generalize, which is itself the evidence that VGGT's global pose
is ≈0, just noisier at n=1 phase than a clean 0.0 would suggest. (Agreed in conversation: report
both anchored and posed for every arm in the eventual table and let the numbers speak — no
narrative decision needed here.)

## 3. What is explicitly NOT done

- **`pose_psf.py` has not been reviewed/verified by the user.** All verification above is
  agent-run. User said (2026-08-20/21) they will verify it together with wiring it in — do not
  treat the fault-injection/PSF-FWHM checks as a substitute for that.
- **Not wired into `image_metrics.score_subject`.** No `_psf`/`_posed` metric columns exist in any
  real `metrics.json` yet; `pose="none", psf="none"` placeholders are still literally what's
  written for every arm, including baselines.
- **The 144 SVRTK/NeSVoR baseline volumes (docs/84) are not scored under the new pipeline at all**,
  except the two subjects touched by verification this session (`CMRx24_Test_P012`,
  `OCMR_fs_0012_3T` — both now HAVE `metrics.json`/cines under the new pipeline as a side effect
  of testing the gauge fix, but WITHOUT pose/PSF columns yet — they'll need re-scoring once
  hook-in lands, which is fine, `metrics.json`/cines are designed re-runnable outputs).
- **VGGT arms are not yet re-scored** for the pose columns (not needed for the gauge fix — VGGT
  never touched `prep_recon`'s self-norm branch — but will be needed once pose/PSF is hooked in,
  so every arm's `metrics.json` carries the same three-column shape for the cross-method table).
- **Nothing from this session is committed.** `git status` currently shows (besides pre-existing
  unrelated untracked docs/86, docs/87 from an earlier session):
  - Modified: `evaluation/src/score/image_metrics.py`, `evaluation/src/score/aggregate.py`,
    `docs/README.md`
  - New: `evaluation/src/score/pose_psf.py`, `docs/88_...md`, this doc (`docs/89_...md`)
  - Regenerated (side effect of re-scoring/re-aggregating during verification):
    `evaluation/metric_results/*/vggt_augaggr224hw2_ep300.json` (all 7 cohorts, `resp_corr`
    added), plus new `metrics.json`/`cine_*.nii.gz` under `CMRx24_Test_P012/{vggt_...,svrtk3d,
    nesvor}` and `OCMR_fs_0012_3T/nesvor`.
- **Two design decisions deferred to hook-in time, not blocking anything now:**
  1. Which cine file(s) get saved for classical arms once there are 3 metric variants
     (anchored/psf/posed). Current lean from conversation: save **two** cines —
     `cine_breath.nii.gz` = anchored **+ PSF** (the physically-correct *unregistered* volume,
     since PSF applies to both the anchored and posed columns per the user's explicit
     requirement — the plain-trilinear `anchored` PSNR number is a legacy/continuity row, not a
     saved artifact) and `cine_breath_posed.nii.gz` = PSF + registered (the fair headline). VGGT
     needs only the one cine (no PSF, shift ≈0).
  2. Which cine feeds the nnU-Net EF/Dice segmentation chain for the cross-method Dice/EF table —
     lean is **posed** (Dice under the same fair treatment as the headline PSNR), but the user
     wants to "see how each one does" once scored before deciding, possibly with eyes on the
     GIFs too.
  3. Bless `pose_psf.py`'s location (`evaluation/src/score/` — matches the handoff's original
     plan, but never explicitly re-approved this session per the standing
     nothing-added-to-evaluation/-without-approval rule) vs moving it to `tools/`.

## 4. Next-steps checklist (in order)

1. **User verifies `pose_psf.py`** — read it, sanity-check the math/design against docs/83 §6,
   decide if the fault-injection/PSF-FWHM checks in §2c are sufficient or if more is wanted
   (e.g. running the probe on a second subject, or a NiftyMIC arm if one exists).
2. **Bless the file location** (§3, decision 3) and **commit** `pose_psf.py` + the gauge fix +
   `resp_corr` pooling + docs/88 + this doc, as separate logical commits if that matches the
   project's existing two-commits-per-session-work pattern (see `docs/85` §"How to get
   oriented" for the git-staging gotcha if splitting commits touches any renamed file).
3. **Hook `pose_psf.py` into `image_metrics.score_subject`**: for every arm, run the fitter
   (`fit_subject_arm`); for classical arms, compute all three PSNR/SSIM/NCC columns via
   `apply_and_downsample`; write the fitted shift (extend `metrics.json` or a new
   `pose_breath.json` per arm — pick one, `paths.py` needs a new helper either way); replace the
   `"pose": "none", "psf": "none"` placeholders with real values; resolve decisions 1–2 from §3.
4. **Score the 144 baseline volumes**: `run.py --method svrtk3d --split val` and
   `--method nesvor --split val` (also naturally resolves the 2 previously-unscored smoke arms
   noted in the original docs/85 handoff, §2 "Known minor gap").
5. **Re-score all VGGT arms** so they carry the same three-column shape (needed for a clean
   cross-method comparison table even though VGGT's own numbers won't move).
6. **Look with eyes**: regenerate GIFs for a few subjects per arm/variant and visually sanity
   check the posed volumes before trusting the headline numbers (user's explicit plan).
7. Write up the full-cohort pose/PSF results (extend docs/88 or a new doc) once real numbers
   exist across more than n=1 subject.

## 5. Deferred parking lot (unchanged from the docs/85 handoff, no urgency)

Bland-Altman plot in `ef_dice.py`'s `plot()`; `D` (frame count) as an explicit cohort-level
"input frame efficiency" stat; `svrtk3d_debug` motion-transform rerun (needed only if baseline
breathing-EPE numbers are wanted); the stats.py-equivalent (paired Wilcoxon + bootstrap CI) into
`compare_table.py`, deliberately deferred to paper-writing time; the clean-arm baseline campaign;
whether/when to push the local commits to `origin/main` (currently all local-only).

## 6. Already resolved — do not re-open

- **EF/Dice chain real-data run**: done via `docs/86` (seven-arm sweep) — pooled EF slopes
  0.74–0.83 reproduce the docs/33 anchor. Was flagged not-done in the docs/85 handoff; is done.
- **OCMR ~13 dB intensity-gauge flag (docs/84)**: verified resolved by the same fix as NeSVoR/
  general (§2a) — do not design a separate per-source OCMR fix.
- **Metric hierarchy**: NCC is the primary cross-method metric (gauge-invariant by construction);
  PSNR/SSIM are secondary and depend on the gauge rule being right. Decided and documented in
  `docs/88` — do not revisit unless new evidence surfaces.
