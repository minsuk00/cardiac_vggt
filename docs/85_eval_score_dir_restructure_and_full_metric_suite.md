# 85 — evaluation/src/score/: fair-scoring restructure + full paper metric suite

> **TL;DR & takeaway** (2026-08-20). The eval scoring path was rebuilt around a three-way split —
> `engine/` makes volumes, **`score/` makes numbers**, `analysis/` makes figures — retiring
> `assemble_and_gif.py`/old `aggregate.py` to `_archive/` and replacing them with
> `score/{run,image_metrics,ef_dice,aggregate}.py` + `analysis/viz.py`. One summary file per arm
> (`metric_results/<ds>/<arm>.json`) now carries the FULL paper metric suite: image
> PSNR/SSIM/NCC, breathing EPE (raw + demeaned) / slope, recon wall-clock, and (via the nnU-Net
> chain) EF/EDV/ESV/LVM (+RV) + Dice/HD95. The new scorer reproduces the old VGGT numbers
> **bit-exactly** (144/144 subjects, all 7 cohorts, max abs diff 0.0), and two prove-it passes
> (3+2 reviewers) confirmed the formulas and killed 15 robustness bugs. Still open: `pose_psf.py`
> (docs/83's registration + PSF for the classical baselines) and the baseline scoring runs.

## Why

Three forces: (1) docs/83 requires pose-gauge registration + a PSF resampling operator for a fair
baseline head-to-head, which the old scorer had nowhere to live; (2) all metrics were scattered
(image metrics + GIFs fused in one 450-line `main()`, EF/Dice off in `analysis/`, timing/breathing
never aggregated) while the paper needs ONE citable file per arm; (3) hard read-only rule — the
144 generated-but-unscored baseline recons and every existing record must never be overwritten.

## What changed

- **Layout**: `engine/` (volumes; unchanged) / `score/` (numbers) / `analysis/` (figures).
  `run.py` = the entry point (`--method <arm> [--datasets ...] [--split val]`), calling
  `image_metrics.py` per subject then `aggregate.py` per dataset. `assemble_and_gif.py` + old
  `aggregate.py` → `evaluation/_archive/` (git mv); its GIF half became the thin
  `analysis/viz.py` (reads the scored `cine_*` files; marks the VGGT reference slice red-★ from
  `ed_dvf.npz:slot_z[0]`).
- **Data migration**: old scorer outputs → `<arm>/_old_scorer/` (703 files), old summaries →
  `metric_results/_archive/` (7); nothing deleted, verified 0 leftovers / 0 clobbers.
- **`aggregate.py` is the single collector**: image metrics + `resp_diag.json` breathing
  (EPE raw/demeaned, slope; demeaned = per-subject mean error removed, the pose-gauge-invariant
  number — baselines join after the transform-saving `svrtk3d_debug` rerun) + wall-clock
  (`timing.json` | `total_wall.sec`) + the EF block from `metric_results/_ef/<arm>.json`
  (merge-if-present, null-if-absent).
- **`ef_dice.py`** (moved from `analysis/`): now consumes the **scored cines** (so Dice/HD95
  inherit gauge/pose/PSF), and adds EDV/ESV (mL), LVM (g, MYO×1.05 at GT-ED), RV EF/EDV/ESV,
  HD95 (MedPy-definition symmetric surface distance, spacing-aware) with paired MAEs.
- **Write-safety**: `image_metrics._guarded_write_path` refuses any pre-existing path outside its
  own three outputs; `cine_gt` refreshes only when older than the bundle's `gt_t00` (atomic).
  Aggregate summaries overwrite freely — derived views, git is the guard.

## Verification (all RUN, not read)

Bit-exact equivalence old-vs-new on all 144 VGGT subjects + 7 cohort summaries; read-only proof
by mtime snapshot; overwrite-guard fault-injection; synthetic seg phantom with analytic answers
(EF 70.37%, EDV 1.728 mL, HD95 = exactly the injected 2 mm shift, Dice 360/432); merge/absent-EF
hook probes; `check_paths.py` ALL PASS. Prove-it round 1 (3 reviewers) fixed: stale-row leak into
summaries on scoring failure (+ nonzero exits), zero-subject false success, malformed-resp_diag
crash/broadcast, cine_gt stale/race, viz vmax drift (was full-FOV → hearts ~3× too dark vs the
heart∩FOV record), render_all_gifs exit code. Round 2 (2 reviewers) fixed: `_contz` probing
breaking the `_ef` join key (now exact-name only), partial-cohort `_ef` clobber (now per-cohort
merge), sidx stale-seg mis-attribution (dirty-dir refusal + manifest-mtime check), vanishing-seg
EF=100%/ESV=0 (→ None, dropped from MAEs), NaN tokens in git-tracked JSON, silent cohort
shrinkage, LV-failure suppressing valid RV metrics.

## Known state / open items

`pose_psf.py` unbuilt (design settled: 3-DOF translation first, off-metric fit, PSF = Gaussian
FWHM=thickness through-plane + 1.2×voxel in-plane; three PSNR columns — anchored-trilinear /
anchored-PSF / registered-PSF — VGGT runs the same fitter, expected ≈identity, PSF column null).
Baselines (svrtk3d, nesvor; 144 val subjects × 12 phases, breath arm) generated but unscored;
two migration-era smoke arms (`CMRx24_Test_P012/svrtk3d`, `OCMR_fs_0012_3T/nesvor`) score with
the baseline run. `check_paths.py` metrics check relaxed to resolver-vs-glob (generation and
scoring are decoupled now). Stats layer (paired Wilcoxon + bootstrap CI + pre-set margins) folds
into `compare_table.py` at paper time.
