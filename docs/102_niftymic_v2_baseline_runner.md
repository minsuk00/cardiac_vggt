# 102 — NiftyMIC v2 baseline runner: full-cohort engine, wired into `run_baselines.py`

> **TL;DR & takeaway**
>
> Built `evaluation/src/engine/run_niftymic_v2.sh`, the NiftyMIC sibling of
> `run_svrtk3d.sh`/`run_nesvor.sh` — same env/positional contract, same CPU setup as
> SVRTK (J=8 phases × OMP=2 threads on a 16-CPU allocation), same output layout, per-phase
> idempotency, atomic write, `provenance.txt`, and stamp-only-on-full-success. Added
> `"niftymic": "run_niftymic_v2.sh"` to `run_baselines.py`'s `SHELLS` dict — the only
> driver change; sharding, split-filtering, thickness resolution, and stamp-skip are all
> already generic. Scoring needs zero changes: `pose_psf.py` and `image_metrics.py`
> already list `"niftymic"`.
>
> **Live-tested end-to-end on the docs/29 pilot subject** (`CMRx24_Train_P053`, val
> split), both `gated` and `scatter` (same-input) arms, all 12 phases each, on an actual
> compute-node allocation (not a login node): **gated ~120 s/phase, scatter ~80 s/phase**
> at J=2/OMP=2 (4 CPUs) — in line with docs/29's per-phase pilot time once corrected for
> the isotropic-resolution difference (see below). Scored with the real
> `evaluation/src/score/run.py`: PSF 6-DOF registration ran (z-shift reported per phase),
> self-norm gauge engaged (raw 11.0 dB → calibrated 17.75 dB gated / 17.4 dB scatter —
> a real correction of the same shape docs/29 found), metrics landed in
> `evaluation/metric_results/val/cmrx2024/niftymic.json`, identical layout to SVRTK/NeSVoR.
>
> **`--iter-max` (LSMR cap inside the TK1L2 solve) stays at the engine default, 10.**
> Probed 20 on one phase: still hits the cap in all 3 two-step cycles (does not converge
> either), and the two outputs differ non-trivially (corr 0.82, mean abs diff 0.27 on a
> ~0–2.3 intensity range) — so raising it trades one non-converged answer for a different
> non-converged answer, with no GT evidence either is better and roughly double the SR
> time. Not worth it; shipped runner uses the default.
>
> **Two flags ARE overridden from the engine's own defaults** (not "everything is
> default"), for the same reason SVRTK/NeSVoR already override their own tool defaults:
> `--slice-thicknesses` (engine default reads header spacing; wrong per docs/27 — real
> thickness ≠ pitch) and `--isotropic-resolution 1.4` (engine default is 1 mm iso; matched
> to SVRTK/NeSVoR's output grid so all three baselines land on the same canonical spacing
> for scoring). Everything else — `TK1L2`, `alpha=0.015`, `two-step-cycles=3`,
> `outlier-rejection=ON`, `intensity-correction=ON` — is the engine default.
>
> **Not yet done:** the full val+test × gated+scatter campaign (`sbatch/eval_baseline_niftymic_v2.sh`,
> account `jjparkcv98`, `--partition=standard`, `--array=0-15`) has not been submitted —
> holds on explicit user go-ahead per project convention. EF/Dice scoring
> (`sbatch/eval_ef_dice.sh`) not yet run for this method either.

## What was built

- `evaluation/src/engine/run_niftymic_v2.sh` — CPU engine, Singularity
  `niftymic_reconstruct_volume` inside `renbem/niftymic:latest` (v0.9), with the
  docs/29 bind-mount-depth fix (short `/data` mount) and the SimpleITK-2.x patch
  (`scratch/niftymic/patch/joint_image_mask_builder.py`, bind-mounted over the broken
  file at runtime — confirmed still present and correct, patch id `1ea4c3e1579e`).
  `module load singularity` is now required on this cluster (`/usr/local/bin/singularity`
  is a stub that just prints the module-load instruction and exits 0) — the script
  resolves the real binary the same way `baselines/fetal_cmr_4d/bin/mirtk` does
  (`/opt/singularity/*/bin/singularity`). The `.sif` (2.85 GB) is staged to node-local
  `/tmp` once per node, same pattern as the mirtk shim / `run_nesvor.sh`.
- `evaluation/src/engine/run_baselines.py`: one line, `SHELLS["niftymic"] = "run_niftymic_v2.sh"`.
- `sbatch/eval_baseline_niftymic_v2.sh` — copy of `eval_baseline_svrtk_v2.sh`'s CPU setup
  (account `jjparkcv98`, `--partition=standard`, `--cpus-per-task=16`, J=8/OMP=2,
  `--array=0-16`, both splits × both inputs by default, restrictable via
  `SPLITS`/`INPUTS`/`SOURCES` env vars).

## Engine flags (from `niftymic_reconstruct_volume --help` inside the container, v0.9)

```
--filenames /data/$INPUT/stack_t${pp}.nii.gz
--filenames-masks /data/${MASK_FILE:-mask_heart.nii.gz}
--output /data/$METHOD/recon_$VAR/work_t${pp}/vol.nii.gz
--slice-thicknesses $THICK
--isotropic-resolution $RES        # 1.4, matches SVRTK -resolution / NeSVoR --output-resolution
--iter-max $ITERMAX                # 10, engine default
```
Everything else is engine default: `--reconstruction-type TK1L2`, `--alpha 0.015`,
`--two-step-cycles 3`, `--outlier-rejection 1` (threshold 0.8 final cycle),
`--intensity-correction 1` (no-op for a single stack — logs "Reference image. Skipped"),
`--extra-frame-target 10`.

Thread caps forwarded into the container via `SINGULARITYENV_*`
(`ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`), all set to `$OMP`, mirroring SVRTK's `OMP_NUM_THREADS` per-phase cap.

## Per-slice motion record

NiftyMIC writes `motion_correction/*.tfm` (rigid slice transforms) +
`rejected_slices.json` per phase. The runner keeps only these (~7 KB/phase) under
`recon_<var>/transforms_t{pp}/`, dropping the rest of the (larger) work directory —
the analog of SVRTK's `.dof` transforms and NeSVoR's `slices_t*` directories, for a
later motion metric.

## Live test evidence (compute node, not login node)

Subject `CMRx24_Train_P053` (cmrx2024, val split, T=12, D=9, THICK=8mm):

| arm | phases | time | stamp |
|---|---|---|---|
| `niftymic` (gated) | 12/12 OK | ~120 s/phase (J=2, OMP=2) | written |
| `niftymic_scatter` | 12/12 OK | ~80 s/phase (J=2, OMP=2) | written |

Scored via `evaluation/src/score/run.py --method niftymic --datasets cmrx2024 --split val`:

```
CMRx24_Train_P053 [niftymic]:         breath PSNR 17.75 dB (raw 11.01)  SSIM 0.453  NCC 0.818 (raw 0.738)
CMRx24_Train_P053 [niftymic_scatter]: breath PSNR 17.38 dB (raw 10.84)  SSIM 0.378  NCC 0.724 (raw 0.633)
```

The raw-vs-calibrated PSNR gap (11.0 → 17.75 dB) confirms the self-norm gauge fired
(`SELF_NORM_METHODS` in `image_metrics.py` already lists `"niftymic"`); per-phase z-shift
values in the score log confirm the PSF 6-DOF registration ran (`PSF_METHODS` already
lists `"niftymic"`). Both were true without touching the scorer.

These numbers are **not directly comparable to docs/29's 23.6/25.3 dB** — different
metric (heart-ROI-masked canonical-grid NCC/PSNR with PSF blur-back, not the old ad hoc
anatomy-mask MAE/PSNR), different subject count (1 vs the same subject in docs/29 too,
but t=0 only there vs all 12 phases averaged into `breath` here), and this run used the
`breath` variant (respiratory-corrupted) rather than docs/29's clean t=0 export. Treat
this only as "the v2 pipeline plumbing works end to end," not as a headline number —
that comes from the full campaign.

## `--iter-max` probe (not adopted)

Ran phase t00 of the gated arm at `--iter-max 20` (vs the shipped default 10):

```
iter10: mean/std 0.880 / 0.509   "iteration limit has been reached" x3 (all two-step cycles)
iter20: mean/std 1.006 / 0.384   "iteration limit has been reached" x3 (all two-step cycles)
corr(iter10, iter20) = 0.823   mean|diff| = 0.272   max|diff| = 1.39   (range ~0–2.3)
```

LSMR does not converge at 20 either — same "iteration limit reached" message every
cycle — so this isn't "did it converge yet," it's two different non-converged states,
substantially different from each other, with no ground truth to say which is closer.
Decision: **keep the engine default (10)**, matching how SVRTK/NeSVoR are also run at
their engine defaults wherever the project doesn't have a specific reason to override
(docs/29's original "consider raising iterations" note is superseded by this measurement).

## Open items

- Full val+test × gated+scatter campaign not submitted (`sbatch/eval_baseline_niftymic_v2.sh`).
- EF/Dice scoring (`sbatch/eval_ef_dice.sh METHOD=niftymic` / `niftymic_scatter`) not run.
- Whether NiftyMIC gets a full v2 paper row, per docs/101's PENDING list, is still an open
  decision (not this doc's call).
