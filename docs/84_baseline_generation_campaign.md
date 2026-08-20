# 84 — Baseline generation campaign: per-subject PSF thickness, generation/scoring split, and the val-cohort SVRTK + NeSVoR run

> **TL;DR & takeaway**
> The classical-baseline campaign (docs/83's prerequisite) is **COMPLETE**: SVRTK + NeSVoR over the
> **144-subject val cohort, `breath` arm only**, run as two SLURM arrays (58259799 CPU /
> 58259800 V100×8, 2026-08-19) — **144/144 subjects × 12 phases for both methods, 0 failures**,
> verified on disk (stamps, volume counts, gzip integrity, provenance/timing completeness, and an
> independent re-derivation of every subject's thickness: 0 mismatches). Measured compute cost:
> **SVRTK 3.1 min/subject = 49.6 CPU-h**, **NeSVoR 38.1 min/subject = 90.8 GPU-h** (190 s/phase on
> V100, reproducing the archived 192 s). Generation is **cleanly separated
> from scoring**: the recon shells persist only the **raw isotropic volumes** (+ provenance +
> per-variant `stamp.json`); canonical/PSF/pose-registered versions are deferred to a Phase-2
> scoring step. The PSF slice **thickness is resolved per subject** by a new driver
> (`evaluation/src/engine/run_baselines.py`) from each bundle's `dz_mm` + a per-dataset rule —
> because **NIfTI headers carry only the pitch**, and the one per-subject "SliceThickness" sidecar
> (CMRx2025 csv) is **pitch-valued and wrong** (says 12.0 where the measured pitch is 10.0).
> Verified along the way: **`clean == GT` exactly** in the current builder (max |gt−clean| = 0.0),
> so baseline `clean_psnr` is a copy-your-input number — dropped from the campaign. Smokes:
> SVRTK 12/12 phases → **18.62 dB** breath vs VGGT's **22.39** on the same frozen bundle
> (CMRx24_Test_P012); NeSVoR 12/12 (its **first run ever on the native-z layout**) — which also
> **exposed a real Phase-2 scoring defect**: the scorer's GT-free intensity self-norm is ~0.3 dB
> on cmrx but **~13 dB wrong on OCMR** (GT heart-ROI peaks at 0.44, not ~1.0 → PSNR 7.2 dB with
> healthy structure, NCC 0.768, affine-corrected 20.1 dB). Raw volumes are unaffected.

## 1. Decisions locked (user, 2026-08-18/19)

| decision | choice |
|---|---|
| cohort | **all 144 val subjects** (acdc 15, cmrx2023 19, cmrx2024 29, cmrx2025 37, mnms 33, miitt 3, ocmr 8) |
| methods | **SVRTK + NeSVoR** first (NiftyMIC deferred — no engine shell exists) |
| variants | **`breath` only.** `clean == GT` (verified, §3) makes baseline `clean_psnr` degenerate; the clean→breath robustness Δ can be added later incrementally (frozen bundles + idempotent shells). |
| generation vs scoring | recon shells save **only raw isotropic volumes**; canonical (trilinear *and* PSF) + pose-registered versions are materialized later, at scoring, always from these frozen raws |
| SVRTK debug | **`DEBUG=0`** (fair timing; `-debug` is ~6× slower, volume identical). `.dof` motion transforms later via `METHOD=svrtk3d_debug` into a separate dir — a naive rerun would skip all cached phases (no `.dof`) while still clobbering `provenance.txt`/`total_wall.sec`. |
| accounts | CPU → `jjparkcv0`/`standard`; GPU → **`jjparkcv_owned1` / `gpu` partition (V100)**, **max 8 GPUs** |
| docs numbering | the colliding second "79" renamed → `docs/83_baseline_scoring_protocol_pose_gauge_and_psf.md` |

## 2. The dz→thickness problem and its resolution

The recon shells pass `-thickness` (SVRTK) / `--thicknesses` (NeSVoR) — the **PSF through-plane
FWHM** (Gaussian slice-profile model; in-plane FWHM ≈ 1.2× pixel is tool-internal). `THICK=8` was
hardcoded, correct only for dz=12 subjects (72 of 144 at campaign scope).

**Why it cannot come from the NIfTI:** a NIfTI header has ONE z number (`pixdim[3]`); this repo
standardizes it to the centre-to-centre **pitch** (docs/27 relabel). Thickness+gap as separate
fields is a DICOM concept that did not survive conversion. **Why it cannot come from the CMRx2025
sidecar:** `cine_sax_info.csv` has a per-subject `SliceThickness`, but it is **pitch-valued** —
Center006 records 12.0 where the measured pitch is 10.0 (thickness > pitch is impossible), i.e. it
is the very value docs/54 §10c measured against and rejected. Same field name as 2024's csv,
different semantics.

**The rule table** (implemented in `run_baselines.py::thickness_mm`, stamped into each recon's
`provenance.txt` + `stamp.json`):

| source | val dz (mm) | thickness | evidence tier |
|---|---|---|---|
| cmrx2023 | 12 | **8.0** | challenge site: 8 mm + 4 mm gap (dataset README) |
| cmrx2024 | 12 | **8.0** | source csv measured 305/305 @ 8 mm; +4 gap (docs/27) |
| cmrx2025 | 10, 12 | **dz − 4** (6 / 8) | docs/54 §10c: +4 mm gap rule measured to hold; csv untrustworthy (above) |
| acdc | 10 | **5.0** | Bernard 2018 / ACDC site: 5 mm thickness + 5 mm gap |
| acdc | 5, 8 | **= dz** | documented contiguous |
| acdc | 7 (patient093 only) | **7.0** | ⚠️ ASSUMPTION (contiguous). Raw header genuinely reads 7.0 — outside the documented 10/5/8 table; no source states its composition |
| mnms | 8.8, 9.6, 10 | **= dz** | Campello 2021 **Table III** (read from the paper PDF): per-centre thickness 9.2/9.9/9.9/10/10/9.7 ≈ the header spacings ⇒ (near-)contiguous. ⚠️ Weakest rule: paper gives centre *averages*, and the 9.6=8×1.2 gap hypothesis (MNMs README) remains possible — residual ±1–2 mm in PSF width, second-order (docs/83 measured the whole operator question at ~0.5 dB) |
| miitt | 10 | **8.0** | data author: 8 mm + 2 mm gap (docs/78) |
| ocmr | 7.8, 8, 10 | **per-subject** (6.0 fs_0012; 8.0 others) | the only source with REAL per-subject thickness metadata: ISMRMRD acquisition headers → source `meta.json:slice_thickness_mm`, resolved via `rel_path → sax/convert_meta.json → source_file → meta.json` |

143/144 subjects rest on documented or measured values; the residuals (patient093, MNMs averages)
cannot move a ranking at the ~0.5 dB operator scale.

## 3. `clean == GT` — verified exact

`build_inputs/pooled.py:200-203` writes the same array `Vt` to `gt/gt_tNN` and `clean/stack_tNN`;
empirically on CMRx24_Test_P012, max |gt−clean| = **0.0** for all 12 phases (breath differs by
0.97, so the probe isn't vacuous). Consequence: any method's `clean` arm is scored against its own
input — a sanity/deconvolution-fidelity number, **never a headline** (docs/83 §6.3 confirmed for
the current builder, not just the archived one).

## 4. What was built/changed

- **`evaluation/src/engine/run_svrtk3d.sh`** — `DEBUG=0` default; `MASK_FILE` exported to the
  xargs subshell; per-variant `stamp.json` written **only when all T phases have valid volumes**
  (partial run stays unstamped = "cannot verify"; no timestamps, so config-identical invocations
  count as the same run and clean/breath stamps from separate submissions can match).
- **`evaluation/src/engine/run_nesvor.sh`** — same stamp logic.
- **`evaluation/src/engine/run_baselines.py`** (NEW — the shells' only caller): iterates
  sources × val subjects (split-enforced via `paths.filter_by_split`), resolves `THICK` (§2) and
  `T` per subject from the manifest, sanity-guards `0 < thick ≤ dz`, invokes the shell with the
  env contract. **Skips any subject whose `recon_<variant>/stamp.json` exists** — the shells are
  phase-level idempotent, but a no-op re-invocation still rewrites `provenance.txt`/
  `total_wall.sec` and would clobber the fair-timing record. `--shard I N` for SLURM arrays,
  `--dry-run`, per-subject failure reporting, nonzero exit on any failure.
- **`sbatch/eval_baseline_svrtk.sh`** — `standard`/`jjparkcv0`, 8 CPUs, array 0-3 (shard count
  derived from `SLURM_ARRAY_TASK_COUNT`, so resizing the array rebalances automatically), 6 h.
- **`sbatch/eval_baseline_nesvor.sh`** — `gpu`/`jjparkcv_owned1`, 1×V100 per task, array 0-7
  (**8 GPUs**), **J=1 deliberately** (J=1 per-phase times are the fair compute-cost unit;
  J>1 is contention-inflated per the provenance contract), 16 h.
- `docs/79_baseline_scoring_protocol_pose_gauge_and_psf.md` → **renamed `docs/83_...`** (header +
  README index updated) resolving the numbering collision with `79_eval_harness_nativez_rebuild.md`.

## 5. Smoke tests (both passed end-to-end)

| | SVRTK (CMRx24_Test_P012) | NeSVoR (OCMR_fs_0012_3T) |
|---|---|---|
| phases | 12/12 OK, stamped | 12/12 OK, stamped (**first NeSVoR run ever on the native-z layout**) |
| hardware | 4 CPUs, J=4, DEBUG=0 | L40S, J=1 |
| wall | **259 s/subject** (per-phase 64–121 s) | **3058 s/subject** (~240–250 s/phase steady; t00 356 s = container start + JIT) |
| output | 1.4 mm iso, [0,1] scale, −1 sentinel | 1.4 mm iso, NeSVoR's own gauge (max ≈ 2455) |
| scored (current scorer, SKIP_GIF=1) | breath **18.62 dB** / SSIM 0.770 / **NCC 0.758** (VGGT arm: 22.39 dB on the identical bundle) | breath **7.18 dB** / **NCC 0.768** → §6 |

NeSVoR is slow **by nature**: each phase is a from-scratch INR training run (tiny-cuda-nn hash-grid
MLP, 6000 Adam iterations, poses refined jointly; K=256 Monte-Carlo PSF samples per pixel — K is
the integral-approximation sample count, NOT the PSF width, which `--thicknesses` sets). The V100
is the container's *fastest* arch (192 s/phase archived vs 242 s on L40S — it ships V100-era
builds). This cost is itself a logged result for the compute-cost comparison vs our sub-second
feed-forward model (`time_t*.sec`, `provenance.txt`).

⚠️ The SVRTK 18.62 vs docs/83's archived 20.07 on "the same subject" are **not comparable** —
different bundle generation (native-z rebuild, new breathing realization).

## 6. Finding for Phase 2: the scorer's intensity gauge breaks on OCMR

The NeSVoR smoke scored **7.18 dB with NCC 0.768** — structure fine, intensity gauge wrong.
`prep_recon`'s GT-free self-norm (divide by the recon's own in-ROI p99.9 → [0,1]) implicitly
assumes GT's heart-ROI intensities span to ~1.0. True-ish on cmrx (documented ~0.3 dB residual);
on OCMR_fs_0012_3T **GT's heart ROI peaks at 0.44** (GT is normalized over whole-FOV percentiles
and the heart is not the brightest structure) → the recon lands 2.7× brighter than GT → ~13 dB
artifact. Measured: `gt ≈ 0.373·rec + 0.022`, corr 0.765, affine-corrected PSNR **20.08 dB**.
**Raw generated volumes are unaffected** (stored unnormalized). Phase 2 must revisit the gauge
rule per source; NCC remains the gauge-invariant cross-check, exactly as docs/83 §6.2 recommends.

## 7. The campaign — **COMPLETE** (submitted + finished 2026-08-19)

```
58259799  eval_svrtk_val   standard/jjparkcv0        array 0-3   COMPLETED  1.7-2.0 h/shard
58259800  eval_nesvor_val  gpu/jjparkcv_owned1 V100  array 0-7   COMPLETED  10.8-11.7 h/shard
```

**Result: 144/144 subjects x 12 phases for BOTH methods, 0 failures, 0 leftover `work_t*` dirs.**
Verified on disk (not from exit codes): 144/144 stamped per method, 1728 volumes each, gzip
integrity spot-check 30/30 clean per method, and every subject has `provenance.txt`,
`total_wall.sec`, and T `time_t*.sec` files. Every stamp's `thickness_mm` was re-derived
independently from the rule table and cross-checked against both the stamp AND the provenance
text: **0 mismatches**. Realized thickness histogram (sums to 144, matching Section 2 exactly):
`{5.0: 14, 6.0: 14, 7.0: 1, 8.0: 82, 8.8: 2, 9.6: 2, 10.0: 29}`.

### 7.1 Measured compute cost (the headline for the cost comparison)

Excluding the 2 smoke subjects, which ran under different conditions and are correctly recorded
as such (SVRTK P012 at J=4 not J=8; NeSVoR fs_0012 on an L40S not a V100) — n=143 each:

| | per phase | per subject (12 phases) | cohort total |
|---|---|---|---|
| **SVRTK** (CPU, J=8 x OMP=1, DEBUG=0) | mean 104 s, median 102 s | **3.1 min** | **49.6 CPU-h** |
| **NeSVoR** (1x V100, J=1) | mean 190 s, median 190 s | **38.1 min** | **90.8 GPU-h** |

NeSVoR's 190 s/phase on V100 reproduces the archived pre-native-z 192 s almost exactly —
independent confirmation the container behaves as before, and that V100 is its right arch
(242 s/phase measured on L40S). ⚠️ SVRTK per-phase times are throughput at J=8, not
single-core times (8 concurrent phases contend); `provenance.txt` records the parallelism per
subject so this is never mis-read. All 144 SVRTK runs record `debug_mode: 0`; all 144 NeSVoR
runs record `J=1`; the host/GPU is recorded per subject (12 distinct nodes used).

### 7.2 Disk footprint (⚠️ NeSVoR is heavy)

`scratch/eval` now totals ~125 GB. Per subject: SVRTK **28 MB**, NeSVoR **793 MB** — of which
**756 MB is `model_t*.pt`** (the per-phase INR weights, ~110 GB cohort-wide). The recon volumes
are ~11 MB and `slices_t*/` (the per-slice pose records = NeSVoR's `.dof` analog, wanted for
motion analysis) only 26 MB. **`model_t*.pt` is safely deletable** for everything downstream if
space is needed; nothing in the scoring path reads it.

⚠️ **First submission (58259633/34) failed at t=0 on every task — a real sbatch gotcha:** slurmd
runs a COPY of the batch script from its spool dir, so the repo-locating pattern
`REPO=$(dirname "${BASH_SOURCE[0]}")/..` resolves to `/var/spool/slurmd.spool` under `sbatch`
(it is only correct when the script runs via `bash`). Fixed with a validated fallback:
BASH_SOURCE-derived path → `SLURM_SUBMIT_DIR` → the main tree, accepting the first that contains
`evaluation/src/engine`. **`sbatch/eval_pooled_val.sh` carries the same latent
BASH_SOURCE-derived-REPO pattern** (how its past runs succeeded was not investigated — possibly
run via `bash` inside an allocation, or with `REPO` exported). Apply the same fix when next touched.

Output per subject: `evaluation/volumes/<ds>/out/<subject>/<method>/recon_breath/`
(`vol_t00..11.nii.gz` + logs + timings + `provenance.txt` + `stamp.json`; NeSVoR also
`model_t*.pt` INR weights and `slices_t*/` per-slice motion-corrected outputs — its `.dof`
analog). The two smoke subjects are already stamped and are skipped by the arrays. **Failure
recovery = resubmit the same array**: only unstamped subjects rerun; finished subjects' timing
provenance is never touched.

## 8. What is left to do

**Phase 2 — scoring (docs/83 §6.2, none of it implemented yet):**
1. Split `assemble_and_gif.py` into a **canonicalize** step that persists, per arm, the
   canonical-grid volumes under all four treatments — {anchored, pose-corrected} × {trilinear,
   PSF} — plus the fitted `pose_breath.json`, and a **slim scorer** that only reads them (metric
   iteration then never re-resamples). Current metric field names stay mapped to
   anchored+trilinear so `aggregate.py`/analysis scripts work unchanged.
2. PSF operator: Gaussian, through-plane FWHM = the same per-subject thickness used at recon time
   (read it back from `stamp.json`), 1.2× voxel in-plane; **classical arms only, never VGGT**
   (docs/83 §4.4).
3. Pose-gauge fit per arm, **off-metric** (mask centroid / NMI on a held-out phase — NOT
   metric-maximizing, NOT derived from `manifest["breath"]["disp_dhw_mm"]`), applied on the native
   1.4 mm volume, one resample from the original (docs/83 §3.3–3.4).
4. **Fix the intensity gauge** (§6) — per-source handling or a principled replacement; report NCC
   alongside regardless.
5. Rebuild the docs/83 measurement probes into `tools/` (operator A/B, through-plane sharpness,
   gauge search — the originals died with a session scratchpad).
6. Aggregate + compare (both columns × both operators), drop `clean` from headlines, and
   **re-measure docs/83's n=1 numbers cohort-wide** (pose ±4 mm, PSF sign, ~1.2 dB margin).
7. Baseline GIFs/panels via `tools/render_all_gifs.sh` worklist (optional, qualitative).
8. Results doc (docs/85): protocol, thickness table, both-column results, compute-cost table.

**Phase 3 — flagged, unscheduled:** the `clean` anomaly (VGGT ~24.9 dB when input IS GT); the OOD
real-free-breathing comparison (OCMR-RT / MIITT-RT — docs/83 §6.6's load-bearing experiment);
EF/volumes/Dice-led headline (nnU-Net seg on baseline recons slots into `ef_dice.py`).

**Bookkeeping:** commit the shell edits + driver + sbatch scripts + the docs/83 rename + this doc.
Failure triage is DONE — there were none (0 failed across all 12 shards). Optional/deferred:
`.dof` motion analysis via a small `DEBUG=1 METHOD=svrtk3d_debug` subset; reclaiming ~110 GB by
deleting NeSVoR's `model_t*.pt` (§7.2) once it is certain no INR-weight analysis is wanted; adding
the `clean` arm for the breathing-robustness Δ (cheap and purely additive — frozen bundles,
stamp-skipping driver); NiftyMIC as a third baseline (needs an engine shell written).
