# 105 — Fetal CMR 4D as a v2 baseline: rolled 12-phase input + LV-area self-gate + 4D-joint SVRTK

> **TL;DR & takeaway** (2026-09-15, updated 2026-09-16). Fetal CMR 4D (van Amerom et al., MRM 2019)
> is now a runnable arm of the v2 harness (`fetal_cmr_4d`), 3-subject pilot done. **Input regime
> decided by measurement:** on the stored 12-phase cine the method's self-gating is a no-op (period
> forced to 12 by the window, ED at frame 0 in 112/167 slices because every source is stored
> ED-first — the gater reads the index back), so the arm reads a new **`rolled/`** bundle stack:
> `breath/` with each plane's 12 phases circularly shifted by a frozen random per-slice roll. That
> removes the one quantity the paper's inter-slice synchronisation exists to recover (the per-slice
> start phase) while keeping every other corruption byte-identical to the other arms. Multi-beat
> pseudo-real-time streams were rejected (trivially-solved rate estimate + duplicate frames at N×
> compute); real-RT-only (MIITT/OCMR) was rejected as a separate-input row. **Gating on the roll is a
> real but solvable task:** 380 slices / 42 subjects, 94 % of anchored slices within one frame, median
> cross-slice ED scatter 13.9° (real-RT MIITT pilot: 15.6° / 6.7°; unsynced: 106°). **Two defects
> found and fixed on the first pilot (§5a):** (1) the gate step wrote the 4D stack's frame-duration
> header at nibabel's default 1.0 s instead of the true `RR/12`, so `reconstructCardiac`'s temporal
> window spanned the whole cardiac cycle and every output phase blended all 12 frames (stepwise, not
> smooth, cine) — **the July MIITT export has the identical bug, so docs/36 §7's "temporal PSF
> flattens contraction to 9 %" is likely this artifact, not the method, and is unverified until
> re-run**; (2) a smoothing step biased the ED anchor one frame early on most slices, removed.
> **Robust statistics — decided ON** (author default): confirmed better than OFF on every image
> metric and EF error on the one subject tested, at the cost of freezing motion in a few
> single-redundancy slices (disclosed as a limitation, not tuned away). **Pilot after both fixes**
> (2 CMRx24 + 1 MIITT val subjects, author params, 12 output phases): 9–11 min and ≤4.4 GB per
> subject on 4 CPU cores, coherent smoothly-contracting cine, PSNR 17.6–18.7 dB (≈ `svrtk3d_scatter`,
> below VGGT), EF error 2.6–22.0 pts (comparable to VGGT's 10.6–19.5, both n = 3). Paper row:
> "Fetal CMR 4D†", cite van Amerom 2019 + Åkesson 2025, disclose the sync substitution and the
> 12-frames-per-slice privilege.
>
> **UPDATE 2026-09-16 (late) — §5c/§5d supersede everything above about recon quality.** A
> `-debug` run on P012 showed (1) the §5a header "fix" was still wrong: tagging the time pixdim as
> `sec` makes MIRTK read it ×1000 (83.3 ms) against an R-R in seconds, so the temporal window was
> `2π·83.3/1.0` = flat over the whole cycle — the initial 4D estimate dipped 9 % at ES vs 46 % in GT;
> (2) the container's `reconstructCardiac` reads `-cardphase N v…` with the COUNT as frame 0's phase,
> shifting every frame's phase by one (the authors' scripts hit the same thing); (3) robust
> statistics then rejected exactly the systolic frames of the well-gated slices (per-frame trust ≈0
> on z5 f5–9 etc.), which is the "frozen slice". **Both input defects fixed** (unit code unset; count
> = 710 ≡ 0 mod 2π with slice 0's ED frame first, no `-rrintervals`). **P012 after the fix, one
> scorer, padded ROI, unit-peak PSNR:** fetal robust ON 27.5 dB / NCC 0.78 / EF 51.0 % (GT 59.3; old
> 28.4 %) vs the five `final518_*_ep300` VGGT arms 30.6–31.2 dB / NCC 0.89–0.91 / EF 36–54 %. Robust
> ON beats OFF on every metric now (freeze gone) — **decision ON stands, ablation dropped.**
> Cohort self-gating (291 subjects, 2484 anchored slices): 79 % of slices anchorable, 92 % of those
> within one frame of ED, 8 % gross. Output is now rolled so phase 0 = the GT's seg-derived ED (a
> no-op for ~90 % of subjects). **Test-split campaign DONE + scored (§8, 180 subjects):** Fetal
> CMR 4D 22.64 dB unit-peak / NCC 0.810 / **EF MAE 6.6, bias −1.3** vs the five `final518_*_ep300`
> arms 24.2–24.7 dB / NCC 0.88–0.89 / EF MAE 10.2–13.5, bias −8…−12; same-input 1-frame classical
> arms 20.0–21.5 dB / EF MAE 37–42. VGGT wins images and Dice, the 12-frames-per-slice baseline wins
> EF. The §5/§5a/§5b tables and the MIITT 43.2 % "confirmation" (docs/36) were all produced with
> the flat window + phase shift and are superseded / unverified.

---

## 1. Why this doc

docs/34 proved the paper's cross-slice synchronisation cannot run on single-orientation SAX
(no slice overlap → 106° ED scatter), docs/35 designed and docs/36 demonstrated the replacement
(per-slice LV-area ED anchor, Åkesson 2025) on MIITT real-time data, and docs/98 §11 left "needs the
docs/35 self-gate" as the open item for v2. The 2026-09-11 decision memo said "use the stored
12-phase cine, but do image-based self-gating" without resolving what that input should be. This doc
records the measurements that resolved it, the build, and the pilot.

## 2. The input-regime decision (measured, not argued)

**The stored cine makes self-gating vacuous.** Every source in the pooled cohort is stored ED-first
(`tools/convert_to_sax_layout.py::pick_frames` for ACDC/M&Ms/MIITT/OCMR with per-dataset ED
metadata; CMRxRecon ships ED-first). Verified on all 291 v2 bundles: whole-volume LV voxel count
peaks at phase 0 in 245/291 subjects and at its wrap neighbour (11 or 1) in 45; 1 elsewhere.
Running the gater on `breath/` (18 subjects, 167 slices, scratchpad probe):

| step | result |
|---|---|
| x-f Fourier period | 157/167 slices return period = 12 frames (the window length; 10 lock a harmonic) |
| LV-area ED index | 112 at 0, 40 at 11/1 (ED plateau), 15 elsewhere (base/apex) |

Both unknowns the method exists to recover — period and per-slice offset — are fixed by the storage
convention. A row built that way is "SVRTK 4D-joint on gated cine", not Fetal CMR 4D.

**A random per-slice roll restores the hard half.** Period stays trivial (one cycle per slice; the
paper's per-slice rate step is the easy half even on real data, docs/35), but the cross-slice ED
offset becomes unknown. Probe with a random roll on 42 subjects / 380 slices (LV areas from the
rolled GT seg = perfect-segmenter bound):

| quantity | result |
|---|---|
| ED offset exact | 52 % |
| within ±1 frame (30°) | 94 % |
| ≥3 frames off (≥90°) | 3 % |
| per-subject cross-slice ED scatter, median | 13.9° (CMRx/M&Ms/MIITT 9–25°; ACDC 13–78°, worse — unexplained, native cine has fewer phases) |

This is the same order as the real-RT MIITT result (15.6°, 6.7°), far from the unsynced 106°: the
method behaves on the rolled bundle as it does on real-time data.

**Rejected alternatives.** (a) Raw 12-phase: vacuous, above. (b) Multi-beat pseudo-RT (tile 12 phases
N times, per-frame breathing drift): the extra unknown (period) is trivially solved, the extra frames
are copies (no new information for SR/robust stats), cost N×. (c) Real-RT-only (MIITT 13, OCMR 13
exams): a separate-input, EF-only row with no GT — kept as the docs/35/36 pilot record, not the paper
row. **Breathing stays one state per plane** (`breath/`): a 12-frame real-time burst is ~1 s, well
inside a breathing cycle, and it keeps the corruption byte-identical to every other arm.

## 3. Protocol (what the arm is)

| item | value | status |
|---|---|---|
| input | `rolled/stack_t{00..11}`; `rolled[k][z] = breath[(k − r_z) mod 12][z]`, `r_z ~ U{0..11}` from `default_rng(seed+1)`, frozen in `manifest["rolled"]` (audit only) | new (`pooled.py add_rolled`) |
| gating | nnU-Net Task114 (2d, `nnUNetTrainerV2_MMS`) on the **rolled input frames** (full FOV, never the GT seg — it carries GT frame order); LV area per (slice, frame); ED = argmax of circular 3-tap-smoothed area (Åkesson local-max rule on a single-beat series); θ = 2π·((f − f_ED) mod 12)/12; slices with no LV keep offset 0 | docs/35 §10 rule |
| R-R | nominal 1.0 s per slice (no timing in the sources; sets only the temporal-PSF time axis) | disclosed |
| engine | `mirtk reconstructCardiac`, same `svrtk.sif` as the SVRTK arm; `-iterations 4 -rec_iterations 10 -rec_iterations_last 20`, robust statistics ON, temporal PSF | author verbatim |
| resolution | `-resolution 1.4` (author default is 1.25; overridden to match svrtk3d/nesvor/niftymic, §5b) | decided, not author-literal |
| output phases | `-numcardphase 12` (= GT; the authors' 25 is a free output length, doc 34 §3.1) | disclosed |
| ROI | bundle `mask_heart.nii.gz` (same as SVRTK/NeSVoR/NiftyMIC) | shared |
| scoring | unchanged: `fetal_cmr_4d` already in `PSF_METHODS` (blur-back), 6-DOF NCC registration, scored as-is (SVRTK-style −1 sentinel clip), heart-crop nnU-Net EF/Dice; **index-to-index phases** — recon phase 0 = detected ED, GT phase 0 = ED (§2), so any gating error is scored, as a mis-registration would be | none |

Deviations from the paper, in one place: `baselines/fetal_cmr_4d/DEVIATIONS.md` §F.

## 4. Build

- `evaluation/paths.py`: `rolled` added to `BUNDLE_DIRS`/`_STACK_PREFIX`.
- `evaluation/src/engine/build_inputs/pooled.py`: `add_rolled()` + `--add-rolled` (retrofit like
  `--add-scatter`; gt/clean/breath/scatter untouched). Roll verified bit-exact against `breath/`.
- `evaluation/src/engine/fetal4d_gate.py`: `dump` (stack4d + per-frame nnU-Net inputs, shared dir
  `scratch/eval/_fetal4d_gate/<split>/`) → `seg` (one batched `nnUNet_predict`, GPU) → `assemble`
  (`<subject>/fetal_cmr_4d/gate/{stack4d.nii.gz,cardphase.txt,rrintervals.txt,gate.json}`).
- `evaluation/src/engine/run_fetal4d.sh`: one `reconstructCardiac` call, splits the 4D cine into
  `vol_t00..11`, provenance/timing/stamp contract of the sibling shells; `ROBUST=0` env for the
  fallback that was not needed.
- `run_baselines.py`: `SHELLS["fetal_cmr_4d"]`; refuses `--input scatter`/`--variant clean`, checks
  `rolled/` and the gate files exist.

## 5. Pilot (val, 2026-09-15, this L40S node: 4 CPU cores / 48 GB, nnU-Net on the GPU)

Gating (nnU-Net on the rolled input, real segmenter): P012 9/11 slices anchored, all within one
frame, scatter 12.5°; P020 9/11, 14.2°; MIITT ARVC 8/13, 13.0° — matching the perfect-segmenter
probe. Robust statistics did **not** reject the single stack (unlike `mirtk reconstruct`, docs/98):
0 outlier messages, output on the input intensity scale (max 0.73 vs GT ROI max 0.35; scored
as-is), coherent LV across z with visible ES contraction (`temp/fetal4d_pilot_CMRx24_Test_P012.png`).

| subject | wall | fetal_cmr_4d | svrtk3d_scatter | nesvor_scatter | svrtk3d (gated) | VGGT curated ep300 |
|---|---|---|---|---|---|---|
| CMRx24_Test_P012 | 552 s | 17.82 / 0.669 | 19.27 / 0.764 | 20.24 / 0.799 | 20.60 / 0.822 | 22.18 / 0.873 |
| CMRx24_Test_P020 | 666 s | 18.54 / 0.645 | 18.92 / 0.657 | 19.26 / 0.676 | 19.88 / 0.719 | 21.75 / 0.806 |
| MIITT ARVC | 777 s | 17.50 / 0.741 | 17.24 / 0.719 | 17.53 / 0.719 | 18.51 / 0.791 | 19.25 / 0.818 |

(PSNR dB / NCC, registered, breath.) Peak RSS 4.4 GB (P012, `/usr/bin/time`). The 4D-joint arm with
12 frames per slice lands at or just below the 3D per-phase SVRTK given one frame per slice — the
temporal PSF's blending across phases (docs/36 §7) costs about what the extra frames buy. n = 3;
not a result, a sanity check that the arm is sound and scorable.

## 5a. Two defects found on the first pilot, fixed 2026-09-16 (§5's table is SUPERSEDED)

**(1) Frame-duration header — ours, in the export.** The first pilot's cine stepped from ED to ES
instead of moving smoothly (whole-volume LV voxels 5512, 5291, 5140 → 3957, 3671, 3715, 3477, 3717 →
5144: two plateaus). Cause: `reconstructCardiac` reads each frame's duration from the 4D NIfTI's
time pixdim (`ReconstructionCardiac4D.cc`: `_slice_dt = attr._dt`, temporal-PSF width
`dtrad = 2π·dt/rr`, sinc(π·Δθ/dtrad)·Tukey(0.3)). `fetal4d_gate.py` saved `stack4d.nii.gz` with
nibabel's default 1.0 s = one whole R-R, so the window spanned the entire cycle and every output
phase was a near-uniform average of all 12 frames. Fixed: `dt = RR/T` (0.083 s) in the header. The
engine is correct; the authors' k-t SENSE export writes the true 72 ms. **The July MIITT export
(`export_miitt.py`, `s01_rlt_ab.nii.gz`) has the same 1.0 s header for 25 ms frames — a 40×-too-wide
window — so docs/36 §7's "4D temporal PSF flattens contraction to 9 %" is very probably this bug,
not the method. Not yet re-run; treat that finding as unverified until it is.**

**(2) ED-anchor smoothing bias — ours, in the gater.** The 3-tap circular smoothing before the
argmax put ED one frame early on 7/9 anchored slices of P012 (the diastolic plateau is asymmetric:
slow late filling, fast ejection). Removed; plain argmax (the Åkesson rule on a single beat).
P012 afterwards: 5 exact, 2 at −1, 2 at −2 (the two apical slices).

**After both fixes** (buggy recons kept as `<subject>/fetal_cmr_4d/_recon_breath_dt1s_bug/`):

| subject | wall | PSNR / SSIM / NCC | EF (GT) | EF err: fetal / svrtk3d_scatter / VGGT |
|---|---|---|---|---|
| CMRx24_Test_P012 | 548 s | 18.09 / 0.643 / 0.687 | 47.9 (57.7) | 9.9 / 27.4 / 10.6 |
| CMRx24_Test_P020 | 678 s | 18.65 / 0.559 / 0.645 | 62.7 (60.1) | 2.6 / 47.6 / 12.3 |
| MIITT ARVC | 637 s | 17.57 / 0.455 / 0.742 | 29.5 (51.5) | 22.0 / 30.7 / 19.5 |

Image metrics moved +0.1–0.3 dB; the LV-volume curve is now continuous (P012: 5093, 5034, 4718,
3890, 3705, 3913, 2888, 3203, 4007, 4826, 4827, 5538 vs GT 5924 … 2504 … 5573). **Still open:** on
P012 the correctly-gated mid slice z5 stays at an ED-like state through all 12 phases (area
752 → 684 → 754 vs GT 756 → 368 → 749) while z3/z7 contract.

**(3) Robust statistics freeze correctly-gated slices — CONFIRMED on P012** (`ROBUST=0` arm
`fetal_cmr_4d_norobust`, same gate files). Mean intensity inside the GT-ED LV mask at z5 over the
12 phases: GT 0.131 → 0.066 → 0.139; robust ON **0.153 → 0.148 → 0.148 (flat)**; robust OFF
0.137 → 0.069 → 0.135 (tracks GT). Mechanism = docs/98's single-stack failure: one frame per
(slice, phase) gives the voxel/frame EM outlier model no redundancy, so the frames that differ most
from the smooth initial 4D estimate — the contracting ones — are down-weighted. The trade, n = 1 (**decision: keep robust ON, the author default — better on every image metric and on EF error**):

| P012 | PSNR / SSIM / NCC | EF (GT 57.7) | LV Dice ED/ES | look |
|---|---|---|---|---|
| robust ON (author default) | 18.09 / 0.643 / 0.687 | 47.9 (−9.8) | 0.76 / 0.64 | smooth, some slices frozen |
| robust OFF | 15.77 / 0.528 / 0.606 | 75.5 (+17.8) | 0.79 / 0.77 | contracts everywhere, noisier, higher contrast, per-phase pose jitter (z-shift −4…+4 mm) |

The 3D SVRTK arm runs `-no_robust_statistics` for the analogous single-stack reason (docs/98), but
there it was needed to avoid a crash/dimming; here the failure mode is local (a few slices freeze),
not global, and ON wins on every reported metric except Dice. **Decision (user, 2026-09-16): keep
robust ON (author default) for the full campaign** — `run_fetal4d.sh`'s default (`ROBUST=1`) needs
no change. Disclose the frozen-slice effect on single-stack input as a known limitation; the
`fetal_cmr_4d_norobust` pilot arm is kept on disk as the ablation, not run at scale.

## 5b. Resolution: 1.25 mm (author) vs 1.4 mm (match SVRTK/NeSVoR/NiftyMIC) — decided 2026-09-16

`fetal_cmr_4d` was the only SVR arm left at its tool's literal default (1.25 mm, `recon_cine_vol.bash`);
`svrtk3d`/`nesvor`/`niftymic` all deliberately override THEIR OWN defaults to 1.4 mm specifically to
match each other. **Decision (user): switch to 1.4 mm** for cross-baseline consistency — scoring
resamples every recon onto the GT grid regardless of native resolution (docs/83/98), so nothing is
lost at comparison time, and 1.4 mm is ~1.4× fewer voxels (measured: 437–544 s vs 548–678 s per
subject, all three pilots faster).

**Re-ran all 3 pilots at 1.4 mm** (1.25 mm outputs archived as `_recon_breath_res125_v1/`). Image
metrics essentially unchanged (PSNR 17.6–18.7 dB, same as 1.25 mm to within noise). Two real,
opposite effects on P012, measured directly:

| | GT | 1.25 mm | 1.4 mm |
|---|---|---|---|
| mean per-phase centroid step (mm) — the "unstable GIF" jitter | 2.1 | 5.6 (2.6×) | 3.2 (1.5×) |
| LV-volume range (voxels), contraction amplitude | 2504–5924 | 2888–5538 | 3873–4933 |
| EF (GT 57.7%) | — | 47.9% (err 9.9) | 21.5% (err 36.2) |

**Coarser resolution reduces the pose jitter** (fewer DOF per phase, closer to GT's own 2.1 mm) —
this is likely why the motion looked less smooth at 1.25 mm; part of the instability is a real
resolution effect, not purely the single-frame-per-phase redundancy issue hypothesized in §5a's
close. **But it also blunts the contraction** (more partial-volume averaging at the thin ES cavity
boundary), which hurt EF badly on P012 and only slightly on the other two (P020: 62.7→54.0, still
close to GT 60.1; MIITT: 29.5→35.0, actually closer to GT 51.5). n = 3, one subject dominates the
EF swing — **not a settled trend, kept at 1.4 mm regardless per the consistency decision**; the EF
number for the paper table should come from the full-cohort run, not this pilot.

## 5c. Debug run on P012 (2026-09-16): two input defects + the robust-freeze mechanism, measured

`mirtk reconstructCardiac -debug` dumps every outer/SR iteration (`init_mc*`, `super_mc*sr*`,
per-pixel `weight*`, per-frame `info_mc*.tsv` with the phase, R-R, scale and trust the engine
actually used). Run robust ON and OFF on P012 with identical inputs (SLURM 61262298, 8 cores;
`temp/fetal4d_debug/`, `analyze.py`). Three things fell out of the engine's own table:

**(1) Time-unit tag — ours, and §5a's fix was still wrong.** `info.tsv` said `TemporalResolution
83.3`, not 0.0833, and the stack MIRTK writes back carries `msec`: MIRTK multiplies a `sec`-tagged
pixdim by 1000 on read, while `-rrinterval` stays in seconds. `dtrad = 2π·83.3/1.0` = 523 rad ⇒
`sinc(π·Δθ/dtrad)` ≈ 1 everywhere ⇒ the temporal window is the Tukey taper alone, i.e. the whole
cycle. Direct consequence, measured: the phase-binned initial estimate at z5 dips 9 % at ES (GT
46 %); all contraction has to be carved back out by the SR iterations. The authors' writer
(`ktrecon/mrecon/mrecon_writenifti.m`: `pixdim(5) = frameDuration` in seconds, `xyzt_units = 0`)
leaves the unit code unset, which is why it works for them. **Fix:** `set_xyzt_units("mm", None)`
in `fetal4d_gate.py` (dump + assemble) and `export_miitt.py`. Verified: probe run reports 0.0833.

**(2) `-cardphase` / `-rrintervals` off-by-one — the container binary.** Log: "R-R intervals are
0:11, 1:1, …, 11:1" for 11 passed values; "Read cardiac phase for 133 images" for 132; `info.tsv`
frame 0 phase = 132 (the count), frame 1 = our θ₀, frame 2 = our θ₁ … Every frame got its
predecessor's phase (a one-bin rotation + one mislabelled frame per slice boundary), slice 0 got
R-R = 11 s. The current SVRTK git source parses correctly; the sif binary is another build, and
the authors' `recon_slice_cine.bash` passes the lists exactly as we did (their MIITT logs show
"0:13 … 2341 images" too). **Workaround, since the binary is fixed:** pass count 710 = 113·2π
(0.0004 rad) so entry 0 is ≈ phase 0, re-roll slice 0 in the stack so its detected ED is frame 0
(`assemble`), pass θ₁…θ_{N−1} zero-padded to 710 tokens, drop `-rrintervals` (engine fallback sets
every slice to `-rrinterval`). Verified in the probe's `info.tsv`: frame i ↔ our θᵢ for all i.

**(3) Robust statistics reject the systolic frames — measured, not inferred.** Final per-frame
trust (robust ON, flat-window run): z5 (ED f1) frames 5–9 ≈ 0; z3 (ED f3) frames 8–10 ≈ 0; z2 (ED
f8) frames 2–4 ≈ 0; z8 nearly all ≈ 0; z4/z6 all 1 but per-pixel LV weight 0.4–0.6 at ES. The EM
E-step fits two Gaussians to the per-frame residuals; with one frame per (slice, phase) and a
cycle-averaged starting volume, the ES frames are the ones that disagree, so they are labelled
outliers. `-no_robust_statistics` only skips the E/M-steps (intensity matching is a separate flag
and stays on). With the flat window gone the effect largely disappears (next table).

**P012 after both fixes** (SLURM 61268308; padded ROI, registered, `psnr_unit_peak`; EF/Dice via
Task114 on the registered cine, same crop; `temp/fetal4d_debug/compare_p012.py`, `ef/`):

| arm | PSNR unit-peak | SSIM | NCC | EF (GT 59.3) | LV Dice ED / ES |
|---|---|---|---|---|---|
| fetal robust ON, fixed | 27.52 | 0.738 | 0.780 | 51.0 | 0.81 / 0.84 |
| fetal robust OFF, fixed† | 26.51 | 0.711 | 0.741 | 48.8 | 0.81 / 0.84 |
| fetal old (flat window, phase shift) | — | — | — | 28.4 | 0.79 / 0.72 |
| VGGT final518_hw2_ep300 | 31.15 | 0.851 | 0.905 | 35.8 | 0.90 / 0.67 |
| VGGT final518_motion10_hw2_ep300 | 31.03 | 0.847 | 0.903 | 46.1 | 0.88 / 0.76 |
| VGGT final518_diff1000_ep300 | 30.81 | 0.844 | 0.897 | 50.6 | 0.93 / 0.84 |
| VGGT final518_motion10_ep300 | 30.66 | 0.837 | 0.894 | 54.1 | 0.87 / 0.81 |
| VGGT final518_base_ep300 | 30.59 | 0.836 | 0.892 | 49.1 | 0.93 / 0.83 |

†`fetal_cmr_4d_norobust` is not in `pose_psf.PSF_METHODS`, so it was scored without the PSF step.
LV-blood intensity at ES/ED on the well-gated mid slices z3–z6: GT 0.45–0.55, robust ON 0.43–0.55,
OFF 0.44–0.55, old 0.62–0.84 — ON now contracts as much as OFF and wins every metric. Apical
z1–z2 reach their minimum two phases late in both arms, matching their gating error (−1/−2
frames; an early ED pick delays that slice in the output cycle). **Decision: robust ON (author
default) stands; the OFF ablation is dropped.** The five VGGT arms were run on P012 for this
comparison only (val subject; the 518 arms otherwise exist for test).

GIFs / figures: `temp/fetal4d_debug/figs/p012_gt_fetal_vggt518_diff1000_hw2.gif`,
`p012_fixed_gt_on_off_old.gif`, `gating_*.png`.

## 5d. Self-gating accuracy on the whole cohort + the ED index convention

`fetal4d_gate.py` ran on all 291 subjects (val 111 + test 180; `_fetal4d_gate/<split>/`). Per
anchored slice, `ed_err_frames_diag` = detected ED − manifest roll (i.e. vs the cine's frame 0):

| source | n | slices anchored | ED exact | within ±1 | ≥2 off |
|---|---|---|---|---|---|
| cmrx2023 | 39 | 84 % | 82 % | 96 % | 4 % |
| cmrx2024 | 57 | 82 % | 79 % | 95 % | 5 % |
| cmrx2025 | 69 | 78 % | 73 % | 90 % | 11 % |
| acdc | 32 | 91 % | 42 % | 92 % | 8 % |
| mnms | 73 | 76 % | 61 % | 90 % | 10 % |
| miitt | 13 | 65 % | 68 % | 96 % | 5 % |
| ocmr | 8 | 80 % | 78 % | 95 % | 5 % |
| **all** | 291 | 79 % | 68 % | 92 % | 8 % |

Error histogram over 2484 slices: −1: 328, 0: 1749, +1: 278, |e|≥2: 205. Mid-ventricular slices
are mostly exact; apical slices drift −1/−2 on a flat diastolic plateau; base/apex with no LV
(21 %) get offset 0 (arbitrary on a rolled stack) and are left to the engine's robust statistics.

**The reference is a convention, not a measurement, for CMRx.** Frame 0 of the 12-phase cine is
the organizers' trigger frame (verified ED for 97 % of CMRx23/24 in docs/17), the annotated ED
for ACDC/M&Ms, and the nnU-Net LV-volume argmax for MIITT/OCMR. By the whole-heart csv
(`scratch/data/whs/cardiac_phase.csv`), seg-ED ≠ 0 for 8/195 CMRx23, 10/294 CMRx24, **51/359
CMRx25** (e.g. `CMRx25_train_Center006_Siemens_30T_Prisma_P021`: seg ED = 7, frame 0 has the
smallest LV on every slice — the gater is right and the reference wrong). Re-scoring the gating
against the seg ED changes little (all-cohort ≥2-off 8.1 % → 8.8 %; MIITT/OCMR unaffected).

**Consequence and fix (user decision 2026-09-16):** fetal's output phase 0 = its own ED; on a
subject whose GT frame 0 is not ED, phase-indexed metrics (PSNR/SSIM/NCC per phase, Dice at GT
ED/ES) penalise only this arm for a labelling convention. `run_fetal4d.sh` now rolls the output
cine by the GT's seg-derived ED (`<subject>/seg_gt/` LV-label argmax — the same index `ef_dice.py`
uses), records `gt_ed_roll` in `stamp.json`, and refuses to run without `seg_gt/`. One scalar
index convention, no slice-level GT information, no effect on per-slice gating errors or on EF
(shift-invariant); a no-op when GT ED = 0. Verified: P012 output byte-identical (GT ED = 0), a
synthetic seg with ED = 3 puts the engine's phase 0 at index 3.

Same-input fairness: `rolled/` and `scatter/` are both pure re-assemblies of `breath/`, whose
respiratory displacement is one frozen value per plane, so plane z carries bit-identical breathing
in fetal's input and VGGT's. The only input difference is the disclosed 12-frames-per-slice privilege.

## 6. Paper framing

Row "Fetal CMR 4D†" citing `vanamerom2019fetal` (pipeline/engine) and `akesson2025retrospective`
(the synchronisation step; bib entry added, author list to complete). Footnote: single-orientation
SAX input, 12 frames per slice with unknown per-slice phase (privileged vs VGGT's 1), inter-slice
synchronisation replaced by a per-slice LV-area ED anchor (public nnU-Net) because parallel slices
have no overlap (docs/34), 12 output phases, nominal R-R; engine and parameters unchanged; output phase index aligned to the
GT's seg-derived ED (§5d). Do NOT claim "the temporal PSF flattens contraction" (docs/36 §7): that
was the flat-window artifact (§5c); the corrected arm contracts on well-gated slices and its EF sits
inside the VGGT-arm spread on P012. Report per-slice self-gating accuracy (§5d) alongside.

## 7. Compute for the full run

~7–9 min per subject on 4 cores at 1.4 mm (437–544 s measured, faster than 1.25 mm's 548–678 s), OpenMP-parallel (more cores, not more memory); 291 subjects ≈ 35–40 core-hours at 4 cores. Gating is seconds per subject on a GPU after one nnU-Net model load per
split. Sequence: `pooled.py --add-rolled` (all sources, val+test) → `fetal4d_gate.py dump/seg/assemble`
on a GPU node → `run_baselines.py --method fetal_cmr_4d` as a `standard` array → `score/run.py` +
`ef_dice.py`. Measured with the fixed inputs: P012 robust ON 1209 s at 4 OMP threads on a shared
8-core allocation; 16 cores do not help (docs handoff: 493 s@16c vs 437 s@4c on the old config)
but are used anyway to match SVRTK/NiftyMIC's reported allocation. **Submitted 2026-09-16: test
split only, `sbatch/eval_baseline_fetal4d_v2.sh`, job 61280727, 16 shards × 16 cores × 48 GB,
`jjparkcv98`/`standard`** (≈ 11 subjects/shard ≈ 3 h). Prerequisites already on disk for all 291:
`rolled/`, fixed gate files, `seg_gt/` (test; val cache built 2026-09-16 via `ef_dice.py dump
--gt-only`). Val is not submitted.

## 8. Test-split campaign result (2026-09-16, 180 subjects, jobs 61280727 + 61285878)

All 180 test subjects reconstructed (median 492 s, mean 537 s per subject on 16 cores / 48 GB,
`standard`; 26.9 node-hours; `gt_ed_roll` = 0 for 138, ±1 for 36, ±2 for 6). One shard-level
incident: the first subject of each of the 16 shards lost its split step to a `micromamba run`
cache-lock collision (recon itself complete); `run_fetal4d.sh` now calls the env python directly
and the 16 were redone by a name-restricted array. Scored with the current harness (padded ROI,
registration, `psnr_unit_peak`; `ef_dice.py` Task114 on the registered cine), same as every arm
below; all nine arms have all 180 subjects.

| arm (test, n = 180) | PSNR unit-peak | SSIM | NCC | EF MAE | EF bias | EF r | LV Dice ED / ES | MYO Dice ES |
|---|---|---|---|---|---|---|---|---|
| VGGT final518_hw2_ep300 | **24.67** | **0.714** | **0.889** | 10.2 | −8.2 | 0.71 | 0.882 / 0.796 | **0.790** |
| VGGT final518_motion10_hw2_ep300 | 24.65 | 0.712 | 0.889 | 10.7 | −8.4 | 0.69 | 0.881 / 0.803 | 0.789 |
| VGGT final518_motion10_ep300 | 24.39 | 0.702 | 0.882 | 13.1 | −12.1 | 0.68 | 0.881 / 0.782 | 0.763 |
| VGGT final518_diff1000_ep300 | 24.23 | 0.700 | 0.876 | 10.9 | −9.6 | 0.76 | **0.884 / 0.813** | 0.776 |
| VGGT final518_base_ep300 | 24.21 | 0.695 | 0.876 | 13.5 | −12.2 | 0.73 | 0.874 / 0.780 | 0.759 |
| **Fetal CMR 4D† (12 frames/slice, self-gated)** | 22.64 | 0.620 | 0.810 | **6.6** | **−1.3** | **0.78** | 0.848 / 0.782 | 0.712 |
| SVRTK 3D scatter (1 frame/slice) | 21.52 | 0.561 | 0.769 | 36.9 | −36.9 | 0.43 | 0.774 / 0.673 | 0.549 |
| Dangi scatter | 21.08 | 0.520 | 0.738 | 41.8 | −41.8 | 0.33 | 0.765 / 0.649 | 0.525 |
| NiftyMIC scatter | 19.95 | 0.545 | 0.784 | 42.2 | −42.1 | 0.41 | 0.766 / 0.669 | 0.548 |

GT EF 57.0 ± 13.5. Per cohort (cmrx2023 / 2024 / 2025 / acdc / mnms): fetal PSNR 25.2 / 24.7 /
25.2 / 18.4 / 19.1 vs hw2 26.7 / 26.7 / 26.6 / 21.9 / 21.4; fetal EF MAE 5.7 / 6.2 / 7.1 / 7.9 / 6.5
vs hw2 10.1 / 10.2 / 10.3 / 8.8 / 10.7.

**Reading.** (i) On image metrics and Dice every VGGT 518 arm beats Fetal CMR 4D (+1.6–2.0 dB,
+0.07 NCC, LV-ED Dice 0.88 vs 0.85, MYO 0.79 vs 0.71); LV-ES Dice is a wash. (ii) On EF Fetal CMR
4D is the best arm by a wide margin and nearly unbiased (MAE 6.6, bias −1.3) where every VGGT
arm under-contracts by 8–12 points — docs/24's amplitude regression, now measured against a
baseline that observes all 12 phases of every slice. (iii) The same-input 1-frame-per-slice
classical arms collapse on EF (MAE 37–42), so the EF advantage is bought with the disclosed
12-frames-per-slice privilege, not by the reconstruction engine. Paper framing: report both, state
the privilege in the row, and cite the per-slice self-gating accuracy (§5d). Per-subject numbers:
`evaluation/metric_results/test/{<ds>,_ef}/fetal_cmr_4d.json`; GIF of a median-EF-error test subject
(`CMRx24_Val_P031`): `temp/fetal4d_debug/figs/test_P031_gt_fetal_vggt518hw2_svrtk.gif`.

## 9. Why Fetal CMR 4D's PSNR is below VGGT's while its EF is better (2026-09-16/17, n = 180 test)

Measured from the scorer's registered cines (`<arm>/cine_breath`, `cine_gt`), ROI = `mask_heart_pad10`
∩ FOV, VGGT = `final518_hw2_ep300`, controls = `svrtk3d_scatter` and the raw `breath/` input. PSNR is
unit-peak pooled over the named voxels (differs slightly from the per-phase headline; gaps are what
matter). Scripts + JSON: `temp/fetal4d_debug/why_psnr{,_align,_intensity,_breathing}.py`.

**Hypotheses tested and their measured size (fetal's pooled deficit to VGGT = 2.65 dB):**

| candidate | test | result |
|---|---|---|
| self-gating timing | gap by slice gating class; oracle per-slice cyclic time shift | gap 2.39 dB even on exactly-gated slices (59 % of ROI), 2.16 / 2.59 on ±1 / ≥2, 3.64 on unanchored (14 %); oracle shift recovers **+0.29 dB** |
| in-plane slice misplacement | oracle integer in-plane shift per slice (±8.4 mm) | fetal **+0.42**, VGGT +0.13, SVRTK-scatter +0.36; 25 % of SVRTK-engine slices want ≥ 2.8 mm vs 6 % for VGGT; gap identical at stack ends (3.1) and interior (2.8) |
| global intensity gauge | oracle affine a·V+b per subject | +0.24 (VGGT +0.06) |
| noise / blur | blur recon AND GT σ = 1.5 vox; gradient-energy ratio | gap unchanged (2.7 dB) after blurring; fetal is *sharper* than VGGT (0.73 vs 0.63 of GT) |
| per-slice intensity scaling (engine) | oracle a_z·V+b_z per slice; + smooth bias field | fetal **+1.12 / +1.65**, VGGT +0.26 / +0.76 ⇒ **≈ 0.9 dB net**; fitted per-slice gains span 0.53–1.11 (5–95 %) for fetal, 0.47–1.08 SVRTK-scatter, 0.86–1.15 VGGT |
| **uncorrected breathing** | static-voxel PSNR vs the do-nothing floor (raw `breath/` input), split by applied displacement | see below — **dominant** |

**The gap is largest where nothing moves.** Static voxels (GT swing ≤ 0.10, 58 % of ROI): fetal 23.44,
SVRTK-scatter 23.20, VGGT 27.42 (gap 3.98); moving voxels: 18.00 / 16.92 / 20.22 (gap 2.22). Twelve
frames per slice buy fetal nothing over SVRTK's one frame on static tissue, so the error there is
systematic, not noise.

**Static-tissue PSNR vs the raw corrupted input, by applied breathing displacement (thirds of 180):**

| | raw `breath/` input | fetal | SVRTK-scatter | VGGT |
|---|---|---|---|---|
| all | 22.89 | 23.44 | 23.20 | 27.42 |
| lowest third (3.4 mm) | 27.09 | 26.60 | 26.14 | 29.32 |
| middle third (5.4 mm) | 24.05 | 24.58 | 24.33 | 27.74 |
| highest third (8.6 mm) | 21.34 | 22.22 | 22.03 | 26.66 |

Per-subject static PSNR correlates with the applied displacement at r = −0.59 (raw input), −0.49
(fetal), −0.47 (SVRTK-scatter), −0.38 (VGGT); the VGGT − fetal static gap grows with displacement
(r = +0.39). Fetal is only **+0.3 dB above doing nothing** on static tissue (VGGT +4.5 dB), and on
low-breathing subjects it is *below* the raw input (the engine's per-slice intensity scaling costs
more than its registration gains).

**Reading.** The bundle's respiratory sim reslices each plane at a displaced position (mean 5.8 mm,
|dz| 5.0 mm), so the input slice shows anatomy from a few mm away; all 12 frames of a plane share
that displacement, and a single parallel SAX stack gives slice-to-volume registration no information
about z. Classical SVR therefore reproduces the displaced anatomy (docs/31's single-orientation
limit), which costs ~2 dB of voxel-wise agreement but leaves EF intact — a displaced slice still
contains a ventricle that contracts by the right amount. VGGT is trained to undo the displacement
(memory: breathing costs ~3 dB, the model recovers ~72 %), hence its image/Dice lead; it loses EF for
the unrelated amplitude-regression reason (docs/24). Self-gating error is a minor PSNR factor.
Error-map figure: `temp/fetal4d_debug/figs/test_P031_error_maps_fetal_vs_vggt.png` (one subject; its
"fetal beats VGGT on mid slices" pattern is NOT representative of the cohort).

## 10. Robust statistics OFF as a separate test arm (`fetal_cmr_4d_norobust`, 2026-09-16/17)

Run at the user's request as its own arm (job 61295512, 24 shards × 16 cores / 48 GB, `standard`,
`jjparkcv98`; `METHOD=fetal_cmr_4d_norobust ROBUST=0`, same gate files and padded mask; 180/180
stamped, 0 failures, mean 440 s / subject). Writes only `<subject>/fetal_cmr_4d_norobust/` and
`metric_results/test/{<ds>,_ef}/fetal_cmr_4d_norobust.json`. Scorer change required for a like-for-
like comparison: `pose_psf.base_method()` now strips a `_norobust` suffix (as it does `_scatter` /
`_debug`), so the arm gets `fetal_cmr_4d`'s PSF + gauge treatment — §5c's P012 OFF row predates this
and was scored without the PSF step.

| test, n = 180 | PSNR unit-peak | SSIM | NCC | EF MAE | EF bias | EF r | slope | LV Dice ED / ES | MYO ES |
|---|---|---|---|---|---|---|---|---|---|
| robust ON (`fetal_cmr_4d`) | 22.64 | 0.620 | 0.810 | 6.62 | −1.28 | 0.78 | 0.82 | 0.848 / 0.782 | 0.712 |
| robust OFF (`fetal_cmr_4d_norobust`) | 22.61 | 0.631 | 0.809 | 6.34 | −1.26 | 0.82 | 0.90 | 0.859 / 0.788 | 0.718 |

Paired: PSNR ON − OFF = +0.03 dB (ON better on 51 % of subjects); |EF err| ON − OFF = +0.28 pts (OFF
better on 46 %). Per cohort PSNR ON / OFF: cmrx2023 25.18 / 25.30, cmrx2024 24.68 / 24.81, cmrx2025
25.24 / 25.31, acdc 18.40 / 18.40, mnms 19.09 / 18.77; EF MAE ON / OFF: 5.65 / 5.59, 6.23 / 5.78,
7.07 / 6.33, 7.86 / 5.46, 6.48 / 7.57. **With the inputs fixed the robust setting is nearly
irrelevant** (the P012 "freeze" was the flat temporal window, §5c): OFF is a touch better on EF
overall and clearly on ACDC, worse on M&Ms; images identical. Reported arm stays robust ON (author
default; choosing OFF would be a post-hoc pick on the test set); OFF is the ablation row.

## 11. Does the engine track breathing at all? Direct measurement from its own per-frame pose (2026-09-17)

§9 inferred "uncorrected breathing" from PSNR behaviour. Direct test: `mirtk reconstructCardiac`
writes `info.tsv` every run (not only `-debug`) with each input frame's final rigid pose
(`TranslationX/Y/Z` mm, `RotationX/Y/Z` deg) — confirmed identical to the last SR iteration's table
on the P012 debug run. **Campaign runs never kept this file** (`$WD` deleted before scoring); fixed
in `run_fetal4d.sh` (`cp` before cleanup) and re-run as a new arm `fetal_cmr_4d_motion` (job
61305887, 24×16 cores, robust ON, test split, 180/180, 0 failures) so the existing `fetal_cmr_4d` /
`fetal_cmr_4d_norobust` results are untouched. **Caveat:** an earlier verbal preview in this session
("through-plane EPE ≈ predict-nothing") used `temp/fetal4d_debug/robust/work/info_mc03.tsv` — job
61262298, run BEFORE the §5c unit-tag and `-cardphase` off-by-one fixes — and is superseded by this
section, which uses the fully-fixed pipeline.

Method (`temp/fetal4d_debug/motion_epe.py`): average a plane's `TranslationX/Y/Z` over its 12 input
frames (one physical displacement per plane, matching VGGT's per-slot `resp_diag.json` convention,
run_vggt.py:258); compare against the bundle's applied `disp_dhw_mm` (dz, dy, dx) per plane. MIRTK's
translation axes are its own reconstruction-frame axes with no assumed 1:1 mapping to (dz,dy,dx), so
the best-correlated axis+sign is fit **pooled over the whole cohort** (1970 subject×slice samples),
not per subject (which would overfit noise).

| applied axis | best MIRTK axis (pooled corr) | demeaned EPE | predict-nothing baseline |
|---|---|---|---|
| dz (through-plane, mean \|·\| 5.0 mm) | TranslationY, r = 0.09 | 4.79 ± 1.83 mm | 4.82 ± 1.84 mm |
| dy (in-plane, mean \|·\| 2.1 mm) | TranslationY, r = 0.11 | 2.22 ± 1.33 mm | 2.03 ± 1.42 mm |
| dx (in-plane, mean \|·\| 1.3 mm) | TranslationX, r = 0.06 | 1.57 ± 0.98 mm | 1.26 ± 1.04 mm |

**Every correlation is ≤ 0.11 and every demeaned EPE is at or above the predict-nothing baseline.**
The engine's per-frame rigid registration carries no measurable signal about the applied breathing
displacement — its reported per-frame pose is indistinguishable from noise relative to ground truth.
This is a stronger, more direct claim than §9's PSNR-based inference and is consistent with it:
`SliceToVolumeRegistrationCardiac4D` registers each of a plane's 12 frames independently against the
current volume estimate frame-by-frame, with no mechanism to solve for one shared per-plane offset,
so its output is dominated by frame-to-frame cardiac-content differences, not the physical slice
shift. (§9's oracle-shift result — the *reconstructed volume* carries a residual in-plane offset
correlated with true displacement, r ≈ 0.56–0.60 — comes from spatial averaging across slices in the
final volume, not from the engine explicitly estimating motion; the two are not in tension.)
Data: `temp/fetal4d_debug/motion_epe.json`.

**⚠️ Caveat on the axis mapping, not yet resolved.** MIRTK's `TranslationX/Y/Z` are in the tool's
own reconstruction frame; there is no documented 1:1 mapping to our (dz,dy,dx) convention, so the
"best-correlated axis" above was found by pooled correlation across the whole cohort, not by a
verified geometric decoding. Zero correlation on every candidate axis is consistent with "the
engine tracks nothing" (the reading given above) but is also consistent with a wrong axis mapping
washing out a real signal. A cheap, decisive check — not yet run — is a synthetic calibration: one
tiny stack where a single plane is shifted by a known mm amount and every other plane is identical,
fed through `reconstructCardiac`, then read off which `info.tsv` column and sign responds. No
campaign re-run needed (existing `info.tsv` data stands regardless of the outcome); this only
affects which axis is called "through-plane". Do this before citing §11's "no tracking" claim as
settled in the paper.

## 12. Motion-comparability audit across baselines (2026-09-17) — not yet built

Question: can the same demeaned-EPE recipe (§11) be applied to every baseline for one cross-method
motion-correction comparison? Survey of what each method already outputs, no campaigns re-run to
produce this table:

| baseline | raw per-slice motion output | on disk without a re-run? | what's needed |
|---|---|---|---|
| VGGT | `resp_diag.json`, physical mm via `MM_PER_NORM` | yes (native) | already correct; only reports dz today, could extend to dy/dx |
| Fetal CMR 4D | `info.tsv` (`fetal_cmr_4d_motion` arm) | yes | axis-mapping caveat above |
| Dangi | `centres.json`, `shifts_mm` **already in physical mm** | yes | simplest conversion; **dz is not attempted by the method at all** (in-plane only by design), so a dz EPE number for Dangi is not a real finding — only dy/dx are meaningful |
| NiftyMIC | per-slice `.tfm` (ITK) under `transforms_t*/`, kept by default | yes | decode geometrically (SimpleITK, physical LPS/RAS translation vs the reference), not by correlation-guessing as was done for fetal |
| NeSVoR | `--output-slices` per-slice NIfTIs; each affine encodes the estimated 6-DOF pose | yes | diff each slice affine against the reference affine (nibabel), decode geometrically |
| SVRTK 3D | `.dof` per frame, **only with `DEBUG=1`** (default runs don't emit them, ~6× slower) | **no** for the standing `svrtk3d`/`svrtk3d_scatter` results | needs a `DEBUG=1` re-run as a separate arm (same pattern as `fetal_cmr_4d_motion`) — **done**: user ran `svrtk_debug_motion` (job 61308670) → arm `svrtk3d_debug_scatter`, test split, 180/180 subjects, 22392 `.dof` files, new arm name (`svrtk3d`/`svrtk3d_scatter` untouched); MIRTK axis-mapping caveat above applies here too since it's the same `reconstructCardiac`-family output |

**Gauge-freedom note.** Fetal, SVRTK, and NiftyMIC register each slice onto a *jointly
reconstructed* volume that has no externally fixed reference frame, so the whole recon can sit at
an arbitrary global pose; per-subject demeaning removes a constant offset but may not fully separate
a richer (e.g. rotation-correlated) gauge ambiguity from a real breathing signal. NeSVoR (registers
against a fixed INR field) and Dangi (explicitly anchored) don't have this problem, nor does VGGT
(absolute scanner-frame coordinates).

**Status: survey only, nothing built.** Converters for NiftyMIC/NeSVoR/Dangi are straightforward
(no re-run needed) but not written; the axis-calibration test for the MIRTK family (§11's caveat)
is not run; the cross-method comparison table does not exist yet.

## References
- van Amerom et al., MRM 82:1055–1072, 2019 — `baselines/fetal_cmr_4d/fetal_cmr_4d_paper.pdf`.
- Åkesson et al., Clin Physiol Funct Imaging 2025, doi 10.1111/cpf.70027 (PMC12406293).
- docs/34 (single-orientation failure), docs/35 (self-gating method), docs/36 (MIITT RT pilot,
  temporal-PSF flattening), docs/98 (v2 harness), docs/101 (paper fact sheet row).
- Probes: scratchpad `selfgate_probe.py`, `selfgate_roll_probe.py`, `ed_check.py` (session
  2026-09-15; numbers reproduced above).
