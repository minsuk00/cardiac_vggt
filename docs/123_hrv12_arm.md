# 123 — The `hrv12` arm: hrv24's first 12 frames, all 7 methods

> **TL;DR & takeaway**
> `<src>_hrv12` (180 test subjects) is the physiological-HRV twin of `af12` (docs/119): the first
> 12 frames per slice of the existing `hrv24` bundles, built by the same tool with no new
> simulation. A direct 12-frame simulation is identical to the truncation (measured on the old
> `cmrx2024_hrv` cohort: positions + scatter draw 38/38, voxels byte-identical), so the arm is
> reproducible either way. Recons: VGGT ×4, Fetal CMR 4D, SVRTK, NiftyMIC, Dangi and CiNeVol
> **done 180/180** (2026-09-23); NeSVoR running (spgpu2, 11 shards, ~9 h). **No scores yet** —
> image metrics, EF/Dice and motion EPE are the next step (§5). Side finding: two CiNeVol fits of
> the same subject differ from each other about as much as from GT (§4), so hrv12 uses ONE
> CiNeVol fit (checkpoint kept) for both image metrics and EPE.

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
like-for-like af12 row, rescore af12's image metrics from `cinevol_motion` (not done).

## 5. Next (not done)

1. Image metrics (`evaluation/src/score/run.py --method <arm> --datasets <src>_hrv12 --split test`,
   A40) for all 8 arms; CiNeVol via `tools/cinevol_maskzero_score.py --arm hrv12`.
   `rhythm24_metrics.sh` / `rhythm24_seg.sh` hard-code Fetal + VGGT only — SVRTK, NiftyMIC,
   NeSVoR, Dangi need adding (the same four are also still unscored on af12).
2. EF/Dice (`rhythm24_seg.sh STAGE=pred`, after 1; GT seg cache already linked from hrv24).
3. Motion EPE (`motion_epe/<method>.py hrv12` → `common_set.py`), results into docs/122.
4. NeSVoR scoring once its run finishes.
