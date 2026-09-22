# 119 — CiNeVol mask-zero correction, a frames-per-slice probe for both classical baselines, and the new `af12` arm

> **TL;DR & takeaway**
> **(1) Corrected `af24` CiNeVol image metrics** (docs/116 §4b): the first mask-zero fix had an
> off-by-half-slab bug that (wrongly) suggested the scorer's PSF blur penalizes every mask-limited
> baseline — that explanation is **retracted**, it was a bug in this session's own script. The true
> corrected numbers: CiNeVol still beats VGGT on PSNR/NCC at 24 frames (+0.60 dB, +0.013 NCC,
> p < 1e-6) and ties on SSIM; VGGT still wins EF/Dice. **(2) A 10-subject nested frames-per-slice
> probe** shows the two classical baselines have **opposite** dependence on frame budget: more
> frames make **CiNeVol better** on every metric (PSNR +3.5 dB, EF MAE −5 pp going 12→24→48) and
> make **Fetal CMR 4D's EF worse** (+4.3 pp going 12→24, self-gate temporal blur) while its image
> metrics stay flat. VGGT is flat on everything at any frame count. **(3) A full 180-subject `af12`
> arm** (the same af24 records' first 12 frames — no new simulation) confirms the probe at scale:
> VGGT beats both baselines on every image metric and on EF vs CiNeVol; EF vs Fetal is a
> **statistical tie** (7.47 vs 7.79 pp, p = 0.43) — unlike at 24 frames, where VGGT wins EF vs Fetal
> by 6.1 pp (p = 3e-13). **Recommendation: report both `af12` and `af24` rows** — no single frame
> budget is fair to both baselines, and VGGT's numbers don't depend on which one is picked, which
> is itself the finding worth stating.

## 0. Context

This continues docs/116–118 (CiNeVol added as a third baseline on the `af24` simulated-AF arm) and
answers two questions raised after that campaign: (a) is CiNeVol's as-saved image-metric edge over
VGGT real once scored fairly, and (b) is the 24-frame budget neutral between the two classical
baselines, or does it favour one? Short answers: (a) mostly yes but smaller than as-saved, (b) no —
it's near Fetal's worst case and short of CiNeVol's best, which is why this doc adds a second,
12-frame arm rather than defending one number.

## 1. The `cinevol_masked` correction, and a wrong intermediate explanation (retracted)

**The bug this corrects (docs/116 §4c, brought forward):** `cinevol/reconstruct.py` (upstream,
unmodified) queries the fitted implicit function over the WHOLE rectangular bounding box of the
fitting mask, not just inside it — a continuous representation happily extrapolates outside the
mask with made-up-but-plausible values. Every other baseline (NeSVoR, SVRTK, Fetal CMR 4D) writes
hard zero outside its recon mask. So the as-saved `cinevol` arm's image metrics (NCC 0.900, PSNR
25.30 dB unit-peak) benefit from never paying the "scored against nothing" penalty the zero-outside
baselines pay after registration shifts them a few mm — an apples-to-oranges comparison.

**First attempted fix, `cinevol_maskzero` (now deleted — see §5), had its own bug:** it resampled
`mask_heart_pad10.nii.gz` onto CiNeVol's 1.4 mm grid with
`nibabel.processing.resample_from_to(..., order=0, cval=0)`. That resample treats anything beyond
the LAST slice's voxel *centre* as outside the mask — but a 12 mm stack slice is a 12 mm *slab*, so
this zeroed the outer 6 mm (half) of the mask's first and last slice, voxels that are legitimately
inside the mask. It produced a large, spurious drop in image metrics.

**That spurious drop was mis-attributed, across several turns of this session, to "the scorer's
8 mm through-plane PSF blur inherently penalizes every mask-limited baseline"** — a whole
mechanistic story that was **wrong**, disproven only once a visualization
(`temp/cinevol_trial/psf_edge_demo.png`) showed the damage was already present in the *unblurred*
read. The explanation was given to the user for several turns before being retracted. **Lesson,
matching the project's standing rule (`feedback_measure_dont_reason_from_mechanism.md`): visualize
or measure the actual mechanism before proposing one, especially after a "that doesn't make sense"
pushback — that is a signal to stop and measure, not to elaborate the existing story.**

**The actual fix** (`tools/cinevol_maskzero_score.py::make_zeroed`): resample the mask with
`scipy.ndimage.map_coordinates(..., mode="nearest")` — clamps to the nearest stack slice instead of
falling off the edge. Verified on one subject: end-slice intensities now match the as-saved values
exactly (0.107 vs 0.106, 0.134 vs 0.134), and unregistered NCC-inside-mask drops only 0.831→0.828
(the true cost of zeroing, ~0.003 NCC) — not the ~0.05 the buggy version showed.

### Corrected `af24` numbers (n = 180, paired, unit-peak PSNR — matches docs/118's convention)

| metric | CiNeVol as-saved | CiNeVol masked (corrected) | VGGT diff1000 | Δ (masked − VGGT) | p |
|---|---|---|---|---|---|
| PSNR dB | 25.30 | 25.12 | 24.52 | **+0.60** | 1e-6 |
| SSIM | 0.750 | 0.720 | 0.715 | +0.005 | 0.92 (tie) |
| NCC | 0.900 | 0.896 | 0.883 | **+0.013** | 4e-5 |

Masking cost CiNeVol only −0.18 dB / −0.030 SSIM / −0.004 NCC vs as-saved — consistent with the
single-subject check above. **CiNeVol keeps a real image-metric edge over VGGT at 24 frames** (PSNR,
NCC significant; SSIM a tie), while VGGT keeps its functional-metric win (EF, Dice — unaffected by
this bug throughout, since `ef_dice.py` masks every volume before segmentation regardless of what
lies outside it). docs/116 §4b's table has been updated with these numbers.

**Why does CiNeVol win on image metrics at all, if VGGT wins on EF/Dice?** Not smoothness — blurring
VGGT's own output at increasing sigma never closes the gap (measured, 20 subjects × 4 frames,
scratchpad-only, not persisted — rerun if needed). CiNeVol is a per-subject fit over 240 frames
(20× VGGT's ~11 frames + 2 physiological signals VGGT never gets); the edge is concentrated in
**static-tissue appearance** (VGGT's static-region MSE is 36% higher than CiNeVol's, vs only 15%
higher in the moving-heart region) — consistent with the pre-existing project finding that ~88% of
VGGT's error is shared appearance synthesis (`project_breathing_not_the_bottleneck.md`), not a
motion/timing problem. CiNeVol's own weakness is elsewhere: its cardiac field is indexed by a given
*label*, not true position, so under AF the same label maps to different true positions across
beats and the model averages them — the EF/Dice loss.

## 2. Frames-per-slice probe: 10 subjects, nested inputs, both classical baselines

**Design.** `build_af_bundle.simulate()` draws each plane's beats from a private RNG stream keyed
only on `[seed, z]` — it does not depend on `n_frames` (`tools/build_af_bundle.py:339-345`, a
deliberate invariant so a longer recording window is a strict superset). Adding `"af48"` to `ARMS`
(2×24 frames = 4 nominal beats) therefore gives a record whose first 24 frames are **bit-identical**
to the real `af24` bundle — verified per-subject (byte-equal `breath/`+`gt/` stacks, matching
`rr_per_plane`/`pos_per_plane`/`beat_per_plane`) for all 10 subjects (`tools/cinevol_nframes_probe.py
setup`). This gives three nested inputs (first 12 / 24 / 48 frames of one record) with only the
frame count varying, scored against the same 24 GT volumes throughout — so the "more frames"
question can be answered without a confound.

10 subjects, 2 per source cohort (cmrx2023/24/25, acdc, mnms), same subjects for both probes.
CiNeVol: `tools/cinevol_nframes_probe.py` (fit at N ∈ {12,24,48}, mask-zeroed, scored). Fetal CMR
4D: `tools/fetal4d_nframes_probe.py` (self-gate rebuilt from the af24 gate's cached nnU-Net LV
segmentations restricted to the first N frames — no re-segmentation — verified the N=24 rebuild
reproduces the live campaign gate's ED frames and thetas exactly). Both probes' N=24 point
reproduces the corresponding real arm within noise (mean |Δ| ≤ 0.07 dB PSNR, ≤ 0.03 pp EF), i.e.
the probe machinery is faithful to the campaign. EF/Dice via the standing `ef_dice.py` chain
(`tools/nframes_probe_ef.py`), nnU-Net `3d_fullres`, reading the shared GT seg cache read-only.
Everything lives under `temp/{cinevol,fetal4d}_nframes_probe/`; nothing under `evaluation/` or
`scratch/eval` was written by the probes.

### 2a. CiNeVol: more frames help, unanimously, on every metric

| N frames/slice | PSNR (unit-peak) | SSIM | NCC | EF MAE (pp) | Dice LV ED/ES |
|---|---|---|---|---|---|
| 12 | 23.56 | 0.649 | 0.837 | 12.34 | 0.839/0.778 |
| 24 | 25.48 | 0.732 | 0.896 | 7.03 | 0.866/0.847 |
| 48 | 28.99 | 0.814 | 0.952 | 6.27 | 0.883/0.906 |

48 vs 24: +3.50 dB PSNR (range +1.4 to +6.8), +0.083 SSIM, +0.056 NCC — **10/10 subjects, p = 0.002**.
12 vs 24: −1.93 dB, −5.31 pp EF MAE (p = 0.020, 8/10 worse). **Mechanism** (measured, first subject,
consistent across the rest): the gain is not cardiac denoising (12→24 only bought +0.6 dB) — it's
**respiratory-field identifiability**. `T_BREATH` = 5 nominal R-R, so 24 frames covers only 0.4 of a
breath cycle per slice (mean per-plane ψ span 0.67 of [0,1]); 48 frames covers 0.8 (span 0.99).
At 24 frames the ψ=0 (end-expiration) query point is extrapolation for most slices — the scorer has
to shift the recon ~3 mm in z to register it, and the volume stays blurry (raw PSNR 21.1 dB); at 48
frames the required shift collapses to ~0 mm and raw PSNR jumps to 27.8 dB. **A 48-frame CiNeVol arm
would therefore beat VGGT on image metrics by several dB, not by 0.6 — do not build one hoping to
close CiNeVol's gap; it would open it.**

### 2b. Fetal CMR 4D: more frames hurt EF, image metrics stay flat

| N frames/slice | PSNR (unit-peak) | SSIM | NCC | EF MAE (pp) | EF bias | Dice LV ED/ES |
|---|---|---|---|---|---|---|
| 12 | 21.84 | (see report.json) | | 8.68 | −5.2 | 0.875/0.668 |
| 24 | 22.31 | | | 12.94 | −11.9 | 0.840/0.707 |

12→24: PSNR +0.48 dB (n.s., p = 0.11 on plain PSNR), but **EF MAE worsens by 4.3 pp** (8/10 subjects,
p = 0.23 at n = 10 — not significant at this sample size but the sign and magnitude match the
locked design rationale in docs/115: at 24 frames the self-gate's uniform phase ramp wraps twice, so
frames `f` and `f+12` land in the same output bin and get averaged; under AF those two frames are
2–4 of 12 phases apart, i.e. temporal blur, not denoising). This is measured on the **current**
(post-hold-rule, docs/114) simulator — the earlier 12-frame pilot (docs/111–113, "Fetal still
marginally ahead at 12 frames, EF MAE 8.94 vs VGGT 9.95") predates that fix and used both a
different, buggy AF position model and a different VGGT checkpoint; it does not transfer.

### 2c. VGGT: flat, on the same 10 subjects (af24 arm, diff1000)

PSNR 21.96 / SSIM 0.721 / NCC 0.881 (unaffected by which classical-baseline frame count is being
probed, since VGGT's own input is fixed at one frame/slice regardless).

## 3. The `af12` arm: full 180-subject campaign, all three methods

Given §2's opposite dependence — af24 (24 frames) sits close to Fetal's worst case and short of
CiNeVol's demonstrated ceiling — a second, 12-frame arm was built and fully campaigned rather than
picking one number to defend.

**Build:** `tools/build_af12_from_af24.py` — `<src>_af12` = the first 12 frames per slice of the
corresponding `<src>_af24` bundle. Every image/GT file is a relative **symlink** into the af24
bundle (no new simulation, zero new bytes for inputs); only the manifest is rewritten (`T=12`, the
scatter draw and rhythm block truncated to the same prefix). Verified byte-identical to af24's first
12 frames for all 180 test subjects (`--check`). The af24 GT segmentation cache (frames 0–11, same
files by content) is linked in read-only so nothing needs re-segmenting; `cine_gt.nii.gz` is
deliberately not linked (each arm writes its own 12-frame one on first score). One line added to
`evaluation/paths.py` (`RHYTHM_ARMS += ("af12",)`) — the only change to `evaluation/`.

**Recon + scoring, all fresh** (nothing reused from af24 except inputs and the GT seg cache):
Fetal CMR 4D self-gate rebuilt from the af24 gate's cached LV segmentations restricted to frames
0–11 (`tools/af12_fetal_gate.py`, no nnU-Net re-run; verified the N=24-equivalent path reproduces
the live af24 gate before trusting the pattern), then SVRTK recon on 24×`standard` CPU shards
(16 CPU/48 GB each, matching af24's allocation). CiNeVol: fresh 12-frame fits, A40 (`spgpu`, matching
af24's guard). VGGT `diff1000` and `base`: fresh `run_vggt.py` recon (L40S this session, vs A40 for
the af24 campaign — affects wall-clock only, not the scored metrics). All four arms scored with the
unmodified standing scorer (`run.py` image metrics, `ef_dice.py` EF/Dice, nnU-Net `3d_fullres`) and
CiNeVol additionally mask-zeroed exactly as in §1.

### 3a. Full three-method table, `af12` vs `af24` (n = 180 image; n = 178/176 EF)

| | af12 PSNR/SSIM/NCC | af12 EF MAE | af12 Dice ED/ES | af24 PSNR/SSIM/NCC | af24 EF MAE | af24 Dice ED/ES |
|---|---|---|---|---|---|---|
| CiNeVol (masked) | 23.28 / 0.648 / 0.842 | 10.21 | 0.849/0.804 | **25.12** / 0.720 / **0.896** | 9.77 | 0.866/0.825 |
| Fetal CMR 4D | 22.23 / 0.595 / 0.797 | 7.79 | 0.837/0.760 | 22.11 / 0.590 / 0.795 | 13.15 | 0.812/0.739 |
| **VGGT diff1000** | **24.50** / **0.715** / **0.883** | **7.47** | **0.903/0.857** | 24.52 / 0.715 / 0.883 | **7.05** | **0.906/0.856** |

Paired Wilcoxon, VGGT vs each:
- **af12** — vs CiNeVol: wins every image metric (PSNR +1.23 dB, p = 6e-25, 157/180) and EF
  (−2.74 pp, p = 2e-4). vs Fetal: wins every image metric (PSNR +2.27 dB, p = 3e-31, 179/180) and
  Dice, but **EF is a tie** (−0.32 pp, p = 0.43, VGGT better in only 93/178).
- **af24** — vs CiNeVol: **loses PSNR** (−0.60 dB, p = 1e-6), SSIM tie, loses NCC (−0.013, p = 4e-5);
  wins EF (−2.72 pp, p = 8e-4) and Dice. vs Fetal: wins everything, including EF by a wide margin
  (−6.10 pp, p = 3e-13).

**The one-line summary that survives both rows:** VGGT is the only method whose numbers don't
depend on which frame budget is chosen (PSNR 24.50↔24.52, EF 7.47↔7.05). Each classical baseline
has a frame count that specifically hurts it — Fetal at 24 (self-gate temporal blur), CiNeVol
nowhere in this range yet (its ceiling from §2a is beyond 24) — so *any* single-budget comparison
can be read as favouring VGGT for a reason unrelated to the method, unless both rows are shown
together. **Recommendation: report af12 + af24 as a paired row, exactly like this table**, with one
sentence on why each baseline moves the direction it does (§2a/§2b).

### 3b. Frame-count sensitivity, restated as one number per method

| method | Δ EF MAE, 12→24 (pp) | Δ PSNR, 12→24 (dB) |
|---|---|---|
| CiNeVol | −0.44 (10-subj probe: −5.3 at 12→24; full 180: −0.44, direction consistent, magnitude probe-noisy) | +1.84 |
| Fetal CMR 4D | **+5.36** (worse) | −0.12 (flat) |
| VGGT diff1000 | −0.42 | +0.02 (flat) |

### 3c. VGGT ablations on `af12` (added 2026-09-22): `base`, `nogather`, `hw0` — all six arms

`nogather` and `hw0` were generated after the three-method campaign (`nogather` inline on an L40S,
`hw0` on an A40 via `sbatch/rhythm24_vggt.sh PER_MODEL=1` so `af12` has one A40-timed VGGT arm) and
scored on the same 180 subjects with the same chain (`rhythm24_metrics.sh` → `rhythm24_seg.sh
STAGE=pred`, 0 failures). n = 180 image / 178 EF (Fetal's two failed self-gates excluded from the
paired EF set), unit-peak PSNR:

| | PSNR | SSIM | NCC | EF MAE (pp) | EF slope | Dice LV ED/ES |
|---|---|---|---|---|---|---|
| Fetal CMR 4D | 22.23 | 0.595 | 0.797 | 7.79 | 0.75 | 0.834/0.752 |
| CiNeVol (masked) | 23.28 | 0.648 | 0.842 | 10.29 | 0.72 | 0.848/0.801 |
| VGGT base | **24.53** | 0.711 | **0.884** | 8.89 | 0.88 | **0.903**/0.850 |
| **VGGT diff1000** | 24.50 | **0.715** | 0.883 | **7.49** | **0.91** | 0.902/**0.857** |
| VGGT nogather (ablation) | 23.73 | 0.680 | 0.860 | 37.6 | 0.40 | 0.837/0.747 |
| VGGT hw0 (ablation) | 23.61 | 0.679 | 0.856 | 42.9 | 0.29 | 0.828/0.731 |

- `base` vs `diff1000`: image metrics identical (±0.03 dB); `diff1000` is better on function (EF
  MAE −1.4 pp, ESV MAE 7.9 vs 9.7 mL, slope 0.91 vs 0.88). Same ordering as af24 (docs/118).
- `nogather`/`hw0` **replicate their af24 collapse almost to the decimal** (af24: 37.45 / 42.45 pp,
  docs/118 §3): predicted EF ≈ 11–13 % (e.g. CMRx24_Test_P001: EDV 111 / ESV 97 mL vs GT 165 / 71)
  — a near-static heart — while PSNR drops only ~0.9 dB. The gather mechanism is load-bearing for
  cardiac function, not for image fidelity, and that holds at 12 frames as at 24.
- Timing caveat: `af12` VGGT wall-clock is only A40-comparable for `hw0`; `base`/`diff1000`/
  `nogather` ran on an L40S (the scored metrics are GPU-invariant).

Files: `temp/rhythm24_ef/af12_summary.json` (6 arms; `tools/rhythm24_ef_table.py af12 --json …`
— the `--json` flag is what saves it), `figs/rhythm24/af12_results_table.png`,
`evaluation/metric_results/test/*_af12/vggt_final518_{nogather,hw0}_ep300.json`.

**CiNeVol `cinevol_motion` refit (checkpoints only).** A second from-scratch fit of all 180 `af12`
subjects with `tools/run_cinevol.py --keep-checkpoint` (arm dir `cinevol_motion`, `last.pt` ~170 MB
per subject, 180/180 landed) for the respiratory-motion EPE analysis of docs/119 §6. Seeded
(`seed=7`) but not bit-reproducible on GPU, and run on mixed L40S / RTX 6000 Ada, so it is kept as
its own arm: its volumes and timing are NOT scored and it never replaces `cinevol`/`cinevol_masked`.

## 4. Cleanup done this session

- `scratch/eval/*_af24/out/*/cinevol_maskzero/` (23 subjects, 895 MB) — the buggy first mask-zero
  attempt from §1 — **deleted**. Unreferenced by any result file; only mentioned in code comments
  explaining it is superseded.
- `/home/minsukc/vggt/baselines/CiNeVol/` and `CiNeVol_paper/` (main tree, not this worktree) —
  **deleted**. Both were pure duplicates: the code dir was an unpatched clone of the same upstream
  commit (`e30aecc`) the worktree's `baselines/cinevol/` already contains (patched) with no local
  edits of its own; the paper dir held the same `article.pdf`/`supplement.pdf` already under
  `baselines/cinevol/paper/`, plus two OCR `.txt` extracts (not kept — regenerable from the PDFs).

## 5. Files

New this session: `tools/cinevol_nframes_probe.py`, `tools/fetal4d_nframes_probe.py`,
`tools/fetal4d_probe_recon.sh` (patched copy of `evaluation/src/engine/run_fetal4d.sh` for the
probe — three deliberate diffs, documented at its top), `tools/nframes_probe_ef.py`,
`tools/build_af12_from_af24.py`, `tools/af12_fetal_gate.py`, `sbatch/cinevol_nframes_probe.sh`,
`sbatch/fetal4d_nframes_probe.sh`. Modified: `evaluation/paths.py` (+1 line, `af12` in
`RHYTHM_ARMS`), `tools/build_af_bundle.py` (+`af48` entry in `ARMS`, probe-only, never built into
`scratch/eval`), `tools/rhythm24_table_png.py` (+`cinevol_masked` display name, frame-count-aware
title, a guard for an empty column), `docs/116_cinevol_baseline_on_rhythm_arms.md` §4b (corrected
table, see §1). Results: `figs/rhythm24/af12_results_table.png`,
`temp/rhythm24_ef/af12_summary_3methods.json`, `temp/{cinevol,fetal4d}_nframes_probe/report.json`,
`temp/rhythm24_ef/af12/*.json`, `evaluation/metric_results/test/*_af12/*.json` (git-tracked),
`scratch/eval/*_af12/` (GPFS).

## 6. Not done / open

- ~~VGGT `base`/`nogather`/`hw0` on `af12`~~ — done, §3c.
- Motion EPE column for `af12` (`tools/rhythm24_resp_epe.py af12` not yet run), and the CiNeVol
  ψ-conditioned EPE (`tools/cinevol_resp_epe.py`, not yet written; its inputs, the `cinevol_motion`
  checkpoints, are complete — §3c).
- `regular24`/`hrv24` CiNeVol arms (would give the paired regular→AF control CiNeVol currently
  lacks, matching what docs/118 already did for Fetal-vs-VGGT) — not started, ~1 day each.
- A `regular12`/`hrv12` pair, if the paper ends up wanting the full dose-response at both frame
  budgets — not started.
- §2's 12-vs-24 Fetal EF degradation (docs §2b) is directionally consistent with the full-180 af12
  vs af24 result (§3b) but was only n=10 at the probe stage; the full-180 numbers in §3 are the ones
  to cite.
