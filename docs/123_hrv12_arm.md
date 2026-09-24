# 123 — The `hrv12` arm: hrv24's first 12 frames, all 7 methods

> **TL;DR & takeaway**
> `<src>_hrv12` (180 test subjects) is the physiological-HRV twin of `af12` (docs/119): the first
> 12 frames per slice of the existing `hrv24` bundles, built by the same tool with no new
> simulation. A direct 12-frame simulation is identical to the truncation (measured on the old
> `cmrx2024_hrv` cohort: positions + scatter draw 38/38, voxels byte-identical), so the arm is
> reproducible either way. Recons: VGGT ×4, Fetal CMR 4D, SVRTK, NiftyMIC, Dangi and CiNeVol
> **done 180/180** (2026-09-23); NeSVoR still running (166/180 on 2026-09-24). **Scored (§5, all
> 180/180, NeSVoR pending):** VGGT `diff1000` wins every image metric against every baseline
> (unit-peak PSNR 24.59 vs CiNeVol-masked 24.26, +0.34 dB, p = 5e-4; vs Fetal +1.89 dB) and tracks
> breathing (dz EPE 0.92 mm vs ≥ 4.19 for all baselines, predict-nothing 4.73). **But on hrv12 it
> LOSES EF to both CiNeVol (7.88 vs 5.58 pp, p = 4e-5) and Fetal (7.88 vs 5.98 pp, p = 3e-4)** —
> unlike af12, where it beats CiNeVol and ties Fetal (docs/119 §3d). SVRTK / NiftyMIC / Dangi collapse
> on function (EF MAE 38–42 pp, near-static hearts), like VGGT `nogather`/`hw0`. Side finding: two
> CiNeVol fits of the same subject differ from each other about as much as from GT (§4), so hrv12
> uses ONE CiNeVol fit (checkpoint kept) for image metrics, EF and EPE.

## 1. Build

`tools/build_af12_from_af24.py --rhythm hrv` (the af12 builder, generalised; `--rhythm af` is
unchanged — it regenerates the on-disk af12 manifests exactly, 180/180). Per subject: every
`breath/`, `gt/` and mask file is a relative symlink into `<src>_hrv24`; `rolled/` → `breath/`;
the GT seg cache (`seg_gt_3d_fullres`, frames 0..11) is linked with a T=12 `src.json`; only the
manifest is rewritten (T=12, rhythm rows truncated, scatter draw folded `phase % 12`, beat 0).
hrv manifests have no `physio` block, so the af-only `physio_note` is skipped.

Checks run: `--rhythm hrv --check` 180/180 (breath + GT byte-identical to hrv24 frames 0..11);
`tools/build_af_scatter.py hrv12 --check` 180/180; hrv24 untouched (0 files modified).

**Truncation = direct simulation.** `build_af_bundle.simulate` seeds each plane with
`default_rng([sha256(subject) seed, z])`, independent of `n_frames`, and for `nf == T` the scatter
beat draw is all zeros, which equals folding hrv24's draw by `% 12`. Measured against the
directly-simulated 12-frame `cmrx2024_hrv` (built 2026-09-19 with arm `hrv`): `pos_per_plane`
38/38 equal, scatter `phase_per_plane` 38/38 equal, breath + GT frames 0..11 byte-identical
(120/120 files, 5 subjects). A future rebuild needs the same simulator commit and an untouched
`_hrv24` (hrv12 holds no image bytes of its own).

## 2. Fetal CMR 4D gate

`tools/af12_fetal_gate.py --rhythm hrv` links the cached per-frame nnU-Net LV segs of hrv24
(frames 0..11 are the same `rolled/` images) into work tag `_fetal4d_gate/hrv12_from_hrv24g3`
and runs the standing `fetal4d_gate.py assemble` (ED = LV-area argmax over 12 frames, ramp mod 12).

hrv24 has TWO seg sets: `hrv24_g3_s*` (4320 = 180 × 24, unique) and `hrv24_local_s0` (a partial
pilot, 14 complete subjects, 81/180 overlapping files differ from g3). The shipped hrv24
`gate.json` per-slice `ed_frame` + `lv_area_max` are reproduced by g3 on **180/180** subjects and
by local_s0 on only 11/14 → g3 is the set the shipped hrv24 Fetal arm used; hrv12 uses g3.
Assembled 180/180 (T=12, n_cardphase=12, beats_in_window=1).

## 3. Recon runs (2026-09-23)

| method | arm dir | where | status |
|---|---|---|---|
| VGGT ×4 | `vggt_final518_{base,diff1000,nogather,hw0}_ep300` | gl1722, 1× L40S, `rhythm24_vggt.sh PER_MODEL=1 ALLOW_ANY_GPU=1` | 180/180 each |
| Dangi | `dangi_scatter` | gl1722 L40S, `run_dangi.py --input scatter` | 180/180 |
| Fetal CMR 4D | `fetal_cmr_4d` | `standard`, `rhythm24_fetal.sh` (24 tasks) | 180/180 |
| SVRTK 3D (-debug, .dof kept) | `svrtk3d_debug_scatter` | `standard`, `af12_svrtk_debug.sh ARM=hrv12` | 180/180 |
| NiftyMIC | `niftymic_scatter` | `standard`, `af12_niftymic.sh ARM=hrv12` | 180/180 |
| CiNeVol (checkpoint kept) | `cinevol` | caesar, RTX 6000 Ada, torch cu126, 1 GPU/shard | 180/180, ~160 s/subject |
| NeSVoR | `nesvor_scatter` | `spgpu2` (`jjparkcv_owned1`), `eval_baseline_nesvor_v2.sh` 11 shards, `N_SHARDS=11` | running |

Scatter baselines read `scatter/` (the same input VGGT sees); all baselines go through
`run_baselines.py`, i.e. per-subject PSF thickness + `mask_heart_pad10`. (The af12 NeSVoR timing
launcher had bypassed this — THICK=8 + unpadded mask on every subject — fixed in `a4290b5`.)
`eval_baseline_nesvor_v2.sh` gained an `N_SHARDS` override so one 11-way split can be spread
over two submissions (shards 6–10 and 0–5 were moved from `spgpu` to `spgpu2`); the 11 shards
cover 180 subjects exactly once (16–17 each, checked by dry-run).

**Cluster gotcha:** `jjparkcv98` has no limits of its own, but its parent `jjparkcv_root` caps
**cpu=500, gpu=16, mem=3500G** for everyone under it. 24 Fetal + 30 SVRTK + 30 NiftyMIC tasks at
16 CPUs hit `AssocGrpCpuLimit`, which also blocks spgpu jobs on the same account (NeSVoR needs
5 CPUs/GPU). `jjparkcv_owned1` sits under a different parent (`jjparkcv_owned_root`, cpu=320,
gpu=40).

## 4. CiNeVol: one fit, not two

af12 has two CiNeVol fits per subject: `cinevol` (image metrics; A40, torch cu130) and
`cinevol_motion` (EPE; RTX 6000 Ada, cu126), both seed 7, same settings. On 30 af12 subjects
(phases 0/4/8) they are **not** close: fit-vs-fit PSNR 26.2 dB median (min 20.5), relative mean
|diff| 0.112, while their final data loss agrees (0.0181 vs 0.0182). That spread is about the
size of CiNeVol's own error vs GT (~25 dB), so the two af12 CiNeVol numbers come from two
different, equally-good solutions. Cause: the Grid4D backward is non-deterministic (docs/121 —
even same-GPU refits differ), compounded here by a different GPU and CUDA build. Hypothesis, not
measured: the fit is only constrained on the acquired slices, so the solutions diverge in the
inter-slice gaps.

hrv12 therefore runs CiNeVol **once** with `--keep-checkpoint` under the default arm name
`cinevol`, and both image metrics and motion EPE (`motion_epe/cinevol.py --arm-name cinevol`)
read that fit. It ran on caesar with the same GPU/CUDA build as af12's `cinevol_motion`. For a
like-for-like af12 row, af12's image metrics and EF were re-scored from `cinevol_motion` as arm
`cinevol_motion_masked` (2026-09-24, docs/119 §3d): pooled it equals the A40 fit (PSNR 23.27 vs
23.28, EF MAE 10.50 vs 10.29) even though the two fits differ per subject.

## 5. Results (2026-09-24; n = 180, NeSVoR pending)

Scored on caesar (RTX 6000 Ada, 3 GPUs) except the VGGT EF, which ran on one L40S (`spgpu2`).
Image metrics on caesar reproduce the A40 scorer to ≤ 1.3e-6 dB PSNR, identical SSIM/NCC
(5 af12 VGGT subjects re-scored and compared, originals restored). Unit-peak PSNR (the headline),
breath variant, mean over the 5 source cohorts pooled. EF MAE over subjects with a defined ES
volume (`n_ef`; the segmenter found no LV in some frame for the rest — `tools/rhythm24_ef_table.py`'s
rule, same for every method).

| | PSNR | SSIM | NCC | EF MAE (pp) (n_ef) | EF bias | SV MAE (mL) | Dice LV ED/ES |
|---|---|---|---|---|---|---|---|
| VGGT base | **24.61** | 0.715 | **0.886** | 8.74 (180) | −7.77 | 18.1 | 0.899/0.852 |
| **VGGT diff1000** | 24.59 | **0.719** | 0.884 | 7.88 (180) | −6.54 | 17.0 | **0.900/0.861** |
| VGGT nogather (ablation) | 23.87 | 0.686 | 0.864 | 39.00 (180) | −39.00 | 61.1 | 0.834/0.750 |
| VGGT hw0 (ablation) | 23.76 | 0.686 | 0.860 | 43.67 (180) | −43.67 | 66.2 | 0.819/0.737 |
| CiNeVol (masked) | 24.26 | 0.696 | 0.872 | **5.58** (179) | −2.72 | **12.4** | 0.897/0.856 |
| Fetal CMR 4D | 22.71 | 0.621 | 0.814 | 5.98 (179) | +0.02 | 12.6 | 0.856/0.789 |
| SVRTK 3D (`svrtk3d_debug_scatter`) | 21.94 | 0.584 | 0.787 | 38.45 (177) | −37.96 | 60.7 | 0.760/0.690 |
| NiftyMIC | 20.09 | 0.562 | 0.794 | 40.81 (178) | −40.44 | 63.8 | 0.762/0.695 |
| Dangi | 21.33 | 0.537 | 0.748 | 42.39 (179) | −42.25 | 64.8 | 0.771/0.694 |
| NeSVoR | pending | | | | | | |

(Unmasked `cinevol`: PSNR 24.43 / SSIM 0.723 / NCC 0.876 — as-saved, not the reported row, docs/119 §1.)

Paired Wilcoxon, VGGT `diff1000` vs each (image n = 180; EF on subjects both define):
- vs CiNeVol: PSNR +0.34 dB (p = 5e-4, 104/180), SSIM +0.023 (p = 3e-8), NCC +0.013 (p = 9e-5);
  **EF |err| +2.32 pp worse** (p = 4e-5, VGGT better in 65/179).
- vs Fetal: PSNR +1.89 dB (p = 5e-29, 170/180), SSIM +0.098, NCC +0.071 (all p < 1e-28);
  **EF |err| +1.93 pp worse** (p = 3e-4, 67/179).
- vs SVRTK / NiftyMIC / Dangi: PSNR +2.66 / +4.51 / +3.26 dB, EF |err| −30.5 / −32.9 / −34.7 pp
  (all p < 1e-29).

**hrv12 vs af12 (docs/119 §3d).** VGGT's numbers barely move between the two rhythms (PSNR 24.59 vs
24.50, EF 7.88 vs 7.49). The baselines move a lot: CiNeVol PSNR 24.26 vs 23.27 and EF 5.58 vs 10.50;
Fetal EF 5.98 vs 7.79. The two arms differ only in rhythm, so this is consistent with the near-regular
HRV rhythm letting the cardiac-phase baselines recover function at 12 frames, and AF taking it away
while VGGT is unaffected. Hypothesis for the mechanism, not measured: with near-regular beats, 12 frames of one beat sample the cycle evenly enough for
CiNeVol's phase model and Fetal's self-gate, while AF's irregular RR breaks that.

**Breathing-motion EPE** (docs/122 toolkit, `common_set.py`, 1803 common slices, 180 subjects;
demeaned EPE mm, corr):

| method | dz | dy | dx |
|---|---|---|---|
| VGGT `base` | **0.87** (0.98) | — | — |
| VGGT `diff1000` / `hw0` | 0.92 / 0.92 | — | — |
| VGGT `nogather` | 2.40 (0.81) | — | — |
| CiNeVol | 4.19 (0.34) | — | — |
| NiftyMIC | 4.71 (0.07) | 1.97 | 1.21 |
| SVRTK 3D | 4.80 (0.12) | 1.96 | 1.21 |
| Fetal CMR 4D | 4.80 (0.17) | 2.02 | 1.28 |
| Dangi | — | 3.14 | 3.24 |
| predict nothing | 4.73 | 1.98 | 1.22 |

Within 0.05 mm of af12 on every row (docs/122 §3) — the breathing simulation does not depend on the
rhythm arm; unverified whether the traces are byte-identical.

**Files** (all git-tracked): image `evaluation/metric_results/test/<src>_hrv12/<arm>.json`; EF
`.../<src>_hrv12/ef/<arm>.json` (per-frame seg masks on GPFS, `scratch/eval/_rhythm24_segs/hrv12/`);
EPE `.../_motion_epe/hrv12/` (per-method slice dumps, `rows.json`, `common_set_8methods.txt`);
EF table `.../_tables/hrv12_ef_table.json` (`tools/rhythm24_ef_table.py hrv12 --json …`).

## 6. Scoring notes (2026-09-24)

- **PSF bug fixed before scoring (`0b1fbe8`).** `pose_psf.base_method` stripped only ONE arm-name
  suffix, so `svrtk3d_debug_scatter` → `svrtk3d_debug`, `nesvor_4gpu_scatter` → `nesvor_4gpu`,
  `cinevol_motion` → itself: none in `PSF_METHODS`, so they would have been scored **without the PSF
  blur** (and NeSVoR also without its intensity self-normalisation). It now strips repeatedly and
  knows `_4gpu` / `_motion`. Every arm name already in `metric_results/` resolves as before, so no
  existing score changes. (`fetal_cmr_4d_mb3` is still unresolved — pre-existing, untouched.)
- Drivers: `rhythm24_metrics.sh` / `rhythm24_seg.sh` take `METHOD_LIST="…"` (`eaf6041`); EF now lands
  in `evaluation/metric_results/test/<cohort>/ef/<method>.json` (was gitignored `temp/rhythm24_ef/`).
  `tools/cinevol_maskzero_score.py --src-arm` masks any CiNeVol fit (`12f57b9`).
- Speed on caesar: image metrics are CPU-bound per process (~20 s/subject); 3 processes per GPU gave
  ~25 subjects/min. nnU-Net `3d_fullres` ran at ~1.7 s/volume (vs 4.0 on an A40).
- nnU-Net on caesar: `micromamba` env `nnunet` (python 3.10, torch 2.3.1+cu121, nnunet 1.7.1,
  numpy 1.26.4); Task114 weights read over the sshfs GPFS mount through
  `/home/minsukc/vggt/tools/nnunet_mnms_eval/env.sh`.

## 7. Next

1. NeSVoR (`nesvor_scatter`) once 180/180: `run.py`, `rhythm24_seg.sh STAGE=pred`,
   `motion_epe/nesvor.py hrv12 --arm-name nesvor_scatter`, then re-run `common_set.py`.
