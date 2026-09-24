# 122 — Breathing-motion EPE for every method: one slot contract, fixed axes, common slice set

> **Update 2026-09-23:** an `hrv12` arm now exists (docs/123) and every script here runs on it
> unchanged (`<method>.py hrv12`). Its CiNeVol arm is the checkpointed `cinevol` fit, so pass
> `--arm-name cinevol` (not `cinevol_motion`). af12 NiftyMIC is now 180/180; af12 NeSVoR's timing
> arm is `nesvor_4gpu_scatter` after the `a4290b5` launcher fix.

> **Update 2026-09-24:** hrv12 EPE is done for every method but NeSVoR (§3, 1803 common slices):
> within 0.05 mm of af12 on every row, VGGT `base` 0.87 mm dz. All EPE outputs now live git-tracked
> in `evaluation/metric_results/test/_motion_epe/<arm>/` (af12's docs/122 dumps were imported there
> from the cluster; `common_set.py` on them reproduces the §3 af12 table exactly).

> **TL;DR & takeaway**
> Every reconstruction method is now scored for breathing-motion tracking by the same rule:
> predicted per-slice shift vs the TRUE simulated shift (mm), error demeaned per subject, on the
> **same slices for every method** — the `scatter/stack_t00` input VGGT receives (plane z at frame
> `phase_per_plane[z]`, reference plane at frame 0). One script per method in
> `evaluation/src/analysis/motion_epe/`, shared rules in `common.py`, cross-method table on the
> slices every method kept in `common_set.py`.
> **Result (af12, 1804 common slices, 180 subjects, through-plane dz):** VGGT `base` **0.87 mm**
> (corr 0.98) vs CiNeVol 4.23 (given the TRUE breath level as input), NiftyMIC 4.69, SVRTK 3D 4.79,
> Fetal CMR 4D 4.80, predict-nothing 4.72 — **no classical method tracks through-plane breathing**,
> and none beats predict-nothing in-plane either (Dangi has direction signal but is over-scaled).
> **Plain cohort (1807 common slices):** same picture; NeSVoR tracks in-plane only (dy 1.67 / dx
> 1.09 vs 1.97 / 1.23). af12 NeSVoR (`nesvor_4gpu_scatter`) is still running — its row is pending.
> Supersedes docs/105 §11's axis caveat and numbers, and the "not yet run/written" items of docs/119 §6.

## 1. The contract

- **Slots** (`common.slots`): per subject, D slots `(z, f)` = plane z at breath/ frame
  `manifest["scatter"]["phase_per_plane"][z]`, the reference plane at frame 0. This is exactly
  `scatter/stack_t00` (`pooled.add_scatter`), and on af12 it equals VGGT's saved `ed_dvf`
  `(slot_z, slot_t)` on 180/180 subjects. Non-reference planes show the same image for every
  target phase k; only the reference plane changes with k.
- **Who sees what:** VGGT and the scatter baselines (SVRTK, NiftyMIC, NeSVoR, Dangi) get only these
  images; Fetal CMR 4D and CiNeVol get all frames of every plane, which include them. Scatter
  baselines are scored on their **t00 run only** — t01..t11 re-estimate the same non-reference
  slices, so averaging them would give a baseline 12 estimates where VGGT gets one.
- **Truth** (`common.applied`): rhythm manifests breathe frame-wise, `r = r0[z] + f·DT/T_BREATH`
  (or `r0[z]` when `frozen_breath`); plain manifests carry one frozen shift per plane.
- **Axes are fixed, not searched.** Every bundle stack has a diagonal affine (1248/1248 checked) and
  every engine's pose is in world mm (SVRTK source: pixel → `ImageToWorld` → transform), so world
  Z/Y/X = true dz/dy/dx. Only one pooled **sign** per axis is fitted, on per-subject-demeaned values.
  docs/105 §11's best-|corr| axis search is gone: simulated breathing couples dz and dy, so it could
  (and for NeSVoR did: 0.38 via in-plane y vs −0.01 via its real z) credit in-plane tracking as
  through-plane.
- **Metric:** `epe` = per-subject-demeaned mean |error| (raw in `epe_raw`), same convention across
  all scripts; `predict_nothing` = pred 0 on the identical slices.

## 2. Per-method decode

| method | script | source | notes |
|---|---|---|---|
| VGGT | `vggt.py` | `resp_diag.json` pred_dz + `ed_dvf.npz` slots | `resp_diag` silently drops blank slots (no pixel > 0.05 — end planes shifted out of the volume); they are re-found from the input slice, only they are dropped (was: whole subject, 37/180) |
| CiNeVol | `cinevol.py` | re-queries `last.pt`'s breathing offset at each slot | GPU; TRUE breath level is an input (oracle) |
| Fetal CMR 4D | `fetal_cmr_4d.py` | `info.tsv` `MeanDisplacementX/Y/Z` | SVRTK's own mean over slice pixels of T(p)−p; plain cohort maps rolled frame `(f + r_z) mod T`; **plane 0 is re-rolled by the gate** (`gate.json` `slice0_frame_roll`) — verified 720/720 frames pixel-exact |
| SVRTK 3D | `svrtk3d.py` | t00 `transformation{z}.dof` (big-endian, decoded without mirtk) | shift at the heart-mask centroid: `R p + t − p`, R = `Rotation.from_euler("XYZ", −angles, deg)` |
| NiftyMIC | `niftymic.py` | t00 ITK `.tfm` | Euler3D ZXY (checked vs SimpleITK), LPS world, centroid shift |
| NeSVoR | `nesvor.py` | t00 `slices_t00/{i}.nii.gz` affines | pred = slice affine − stack affine at the heart-mask centroid; slice i = i-th plane with mask ∧ intensity > 0 (count matches 180/180; no subject shows a systematic 1-pitch misindex) |
| Dangi | `dangi.py` | `centres.json` `t00` shifts_mm | in-plane only (no dz by design) |

**Rotation about the world origin.** MIRTK and ITK rotate about (0,0,0), ~300 mm from the heart, so
raw `Translation*` is not the slice's shift (af12: mean |TranslationZ| 3.77 vs |MeanDisplacementZ|
2.18 mm). The MIRTK convention above reproduces SVRTK's `MeanDisplacement` to 0.03/0.03/0.12 mm
(X/Y/Z, 2544 af12 rows). NiftyMIC's angles reach 0.024 rad.

## 3. Results

af12 (`common_set.py`, 1804 common slices, 180 subjects, 8 methods; truth agreed across methods on
every slice). Arms: `svrtk3d_debug_scatter`, `niftymic_scatter`, `dangi_scatter_4gpu`,
`cinevol_motion`, `fetal_cmr_4d`. EPE demeaned (corr):

| method | dz | dy | dx |
|---|---|---|---|
| VGGT `base` | **0.87** (0.98) | — | — |
| VGGT `diff1000` | 0.92 (0.98) | — | — |
| VGGT `hw0` | 0.92 (0.97) | — | — |
| VGGT `nogather` | 2.42 (0.81) | — | — |
| CiNeVol | 4.23 (0.32) | — | — |
| NiftyMIC | 4.69 (0.08) | 1.97 (0.14) | 1.21 (0.18) |
| SVRTK 3D | 4.79 (0.13) | 1.96 (0.07) | 1.21 (0.10) |
| Fetal CMR 4D | 4.80 (0.15) | 2.02 (0.12) | 1.30 (0.04) |
| Dangi | — | 3.09 (0.33) | 3.25 (0.19) |
| predict nothing | 4.72 | 1.98 | 1.22 |

(VGGT / CiNeVol predict dz only. Restricted to VGGT + CiNeVol + Fetal the common set is 1839 slices
and the dz numbers move ≤ 0.06 mm.)

hrv12 (added 2026-09-24; 1803 common slices, 180 subjects, 8 methods; arms `svrtk3d_debug_scatter`,
`niftymic_scatter`, `dangi_scatter`, `cinevol` (the checkpointed fit, docs/123 §4), `fetal_cmr_4d`):

| method | dz | dy | dx |
|---|---|---|---|
| VGGT `base` | **0.87** (0.98) | — | — |
| VGGT `diff1000` | 0.92 (0.98) | — | — |
| VGGT `hw0` | 0.92 (0.97) | — | — |
| VGGT `nogather` | 2.40 (0.81) | — | — |
| CiNeVol | 4.19 (0.34) | — | — |
| NiftyMIC | 4.71 (0.07) | 1.97 (0.11) | 1.21 (0.17) |
| SVRTK 3D | 4.80 (0.12) | 1.96 (0.07) | 1.21 (0.07) |
| Fetal CMR 4D | 4.80 (0.17) | 2.02 (0.11) | 1.28 (0.05) |
| Dangi | — | 3.14 (0.32) | 3.24 (0.20) |
| predict nothing | 4.73 | 1.98 | 1.22 |

Every row is within 0.05 mm of af12: breathing is simulated independently of the rhythm arm
(unverified whether the traces are byte-identical).

Plain cohort (1807 common slices, 180 subjects; Fetal arm `fetal_cmr_4d_motion`):

| method | dz | dy | dx |
|---|---|---|---|
| Fetal CMR 4D | 4.65 | 1.97 | 1.25 |
| NiftyMIC | 4.71 | 1.97 | 1.22 |
| SVRTK 3D | 4.69 | 1.97 | 1.23 |
| NeSVoR | 6.36 | **1.67** | **1.09** |
| Dangi | — | 3.06 | 3.22 |
| predict nothing | 4.73 | 1.97 | 1.23 |

Dangi has real in-plane direction signal (corr 0.34 / 0.21) but its magnitude is badly
miscalibrated (unverified cause).

## 4. Verification

- 3-reviewer `/prove-it` pass + adversarial checks; fixed: Fetal plane-0 re-roll, `vggt.py`
  ignoring `frozen_breath` (latent: `*_regular_frozen24`), `vggt.py --json` overwriting the shared
  file, raw/demeaned "epe" mixing across methods, sign fitted on raw instead of demeaned values,
  NeSVoR measured at the FOV centre instead of the heart centroid, empty-input crash, unnamespaced
  CiNeVol predict-nothing key.
- Checked: af12 `scatter/` (`tools/build_af_scatter.py`, 180/180 byte-identical to the claimed
  breath frames; the checker also passes on pooled-built plain `scatter/`); `common_set.py` aborts on
  an injected truth mismatch (fault-injected).
- **Not exercised:** NeSVoR on af12 (still running), the `regular_frozen24` path (no VGGT outputs
  there).

## 5. Usage

```bash
export PYTHONPATH=training:.
D=evaluation/metric_results/test/_motion_epe/af12; mkdir -p $D   # git-tracked since 2026-09-24 (was temp/motion_epe/af12)
python evaluation/src/analysis/motion_epe/vggt.py af12 --slices-dir $D --json $D/rows.json
python evaluation/src/analysis/motion_epe/fetal_cmr_4d.py af12 --slices $D/fetal.json --json $D/rows.json
source baselines/cinevol/env.sh && $CINEVOL_PY evaluation/src/analysis/motion_epe/cinevol.py af12 --slices $D/cinevol.json --json $D/rows.json
# scatter baselines (after they run on af12; pass --arm-name matching the run dir):
python evaluation/src/analysis/motion_epe/{niftymic,svrtk3d,nesvor,dangi}.py af12 --arm-name <arm> --slices $D/<m>.json
python evaluation/src/analysis/motion_epe/common_set.py $D/*.json   # exclude rows.json
```
Runtime: `vggt.py` ~8 min (5 arms, GPFS reads), `cinevol.py` ~3 min on an L40S, the others minutes.
SVRTK must run with `DEBUG=1` so the per-slice `.dof` files are kept.

## 6. Open

- NeSVoR on af12 (`nesvor_4gpu_scatter`, 154/180) and hrv12 (`nesvor_scatter`, 166/180, both
  2026-09-24): when 180/180, run `nesvor.py <arm> --arm-name <dir>` and re-run `common_set.py`.
  Both arm names now get PSF + self-norm in image scoring (`pose_psf` fix `0b1fbe8`, docs/123 §6).
- ~~af12 image metrics + EF/Dice for SVRTK / NiftyMIC / Dangi~~ — done, docs/119 §3d.
- docs/118's `af24` "motion EPE" column was produced by the pre-fix `vggt.py` (subject skip, raw
  epe); re-run `vggt.py af24` before citing it.
