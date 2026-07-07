# 29 — NiftyMIC baseline: first real end-to-end run

> **TL;DR & takeaway**
>
> First actual execution (not just research) of a runnable external SVR baseline from
> `_html/24_svr_baselines.html`. Ran **NiftyMIC** (classical, no ML) on 2 val subjects'
> real, clean, per-phase SAX stacks (t=0/ED — the same tensor our own `V_gt` uses, single
> orientation, single stack, per docs/24 §4's "hand the baseline the clean stack" protocol).
>
> **Result: it works, and the reconstructed structure is real** — correlation with true
> anatomy **0.886 / 0.901** (2 subjects), **PSNR_anat ≈ 23.6 / 25.3 dB after intensity
> calibration** (mean 24.4 dB). **PSNR_full stays low (~7 dB)** — not a quality problem,
> but NiftyMIC's default 50 mm reconstruction-space padding spilling nonzero values into
> our zero-padded canonical background; **use `_anat`, same convention as our own
> `_bbox` metric** (CLAUDE.md logging section).
>
> **Three real bugs found + fixed to get here** (all now baked into
> `baselines/niftymic/`, not one-off): (1) Singularity bind-mount to a deep nested host
> path silently fails inside the read-only SIF — bind to a short mount point (`/data`)
> instead. (2) NiftyMIC's shipped image (`renbem/niftymic:latest`/`v0.9`, SimpleITK
> 2.1.1.2) has a **real, unconditional dependency-drift bug**: `joint_image_mask_builder.py`
> calls the old positional-args `BinaryThresholdImageFilter.Execute(img, lo, hi, in, out)`
> signature SimpleITK 2.x removed — patched via a bind-mounted fixed copy of that one file
> (setter-based API), no Docker tag has this fixed. (3) NiftyMIC's Tikhonov SRR solve does
> **not** preserve absolute input intensity scale (`pred ≈ 1.03·gt + 1.0`, not `pred ≈ gt`)
> — standard in the SVR literature; score with a linear intensity calibration fit on the
> anatomy region, not raw MAE/PSNR (raw PSNR_full was a meaningless 1.7 dB before this).
>
> **Runtime: ~18–25 min/subject, CPU-only** (registration ~3–4 min, Tikhonov SR
> ~14–20 min, did not fully converge at the default 10-iteration cap). Confirms the
> report's predicted caveat ("with truly one-slice-per-z it has little to register")
> empirically — correlation is decent (~0.89) but not the >0.95 you'd expect from
> multi-orientation fetal-native input.
>
> **Same-subject head-to-head:** the current in-progress reference-conditioning checkpoint
> (epoch ~182/500, still training) scores **32.70 dB mean PSNR_anat** on these same 2
> subjects/phase — **+8.3 dB over NiftyMIC**. Not a same-*input* comparison though (see
> §Comparison below) — NiftyMIC got a clean stack, VGGT got its own scattered (but
> respiratory-uncorrupted, since this eval bypassed the trainer's aug step) input.
>
> **Status:** n=2, proof-of-concept only. Not yet run: NeSVoR, SVRTK (next); more
> subjects for a real sample; whether reducing the padding frame or cropping before
> scoring recovers `PSNR_full`.

## What was run

`baselines/niftymic/` (repo, git-tracked):
- `export_stack.py` — pulls the real, clean `phases[t_target=0]` (D,H,W)=(12,256,256) +
  `content_mask` for 2 val subjects straight from `MRIDataset`'s canonical cache (NOT our
  synthetic scattered/respiratory-corrupted training input — per docs/24 §3/§4, that's the
  fair "clean input" way to feed a classical SVR baseline). Writes NIfTI with the canonical
  (1.4, 1.4, 12.0) mm diagonal affine (axis order: splat `(D=Z,H=Y,W=X)` →
  `transpose(2,1,0)` → nibabel `(X,Y,Z)`, the reverse of `MRIDataset.get_data`'s
  `permute(0,3,2,1)`).
- `run_niftymic.sh` — Singularity wrapper (see bugs below for the real flags that work).
- `score.py` — resamples NiftyMIC's output onto the exact canonical grid
  (`nibabel.processing.resample_from_to`), fits+inverts a linear intensity calibration on
  the anatomy region, reports both raw and calibrated MAE/PSNR (full and anatomy-masked).

GPFS (via the existing `scratch` symlink, per project convention — big artifacts never
live in the git-tracked repo dir):
- `scratch/niftymic/sif/niftymic.sif` — the pulled image (2.85 GB, `docker://renbem/niftymic:latest`).
- `scratch/niftymic/data/` — exported input stacks + masks.
- `scratch/niftymic/recon/<subject>_t0/` — NiftyMIC's raw output + logs.
- `scratch/niftymic/patch/joint_image_mask_builder.py` — the one patched file (see below).
- `result/niftymic/` — reserved for future scored/rendered comparison output.

Subjects: `Train_P053` (val split, name inherited from CMRxRecon's own directory naming —
our split file re-splits their `Train_*`-named dirs into our own train/val/test, so a
`Train_*` name landing in our val split is expected, not a bug) and `Val_P055`, both t=0 (ED).

## The three bugs (mechanism, so future runs don't rediscover them)

**1. Bind-mount to a deep host path silently fails.** `singularity exec --bind
/home/minsukc/vggt/scratch/niftymic ...` (bind source == dest, no explicit `:dest`) failed
with `FileNotFoundError` when NiftyMIC tried `os.makedirs` on the output dir — the deep
nested path doesn't pre-exist inside the read-only SIF's filesystem, and the bind didn't
create it. Fix: `--bind /home/minsukc/vggt/scratch/niftymic:/data`, explicit short
container-side destination, reference paths as `/data/...` inside the container.

**2. `renbem/niftymic:latest` (= `v0.9`) has a real SimpleITK API-drift bug**, unconditional
(happens with or without `--filenames-masks`, confirmed by testing both). NiftyMIC (last
released ~2019) was written against an older SimpleITK where
`BinaryThresholdImageFilter.Execute()` accepted `(image, lower, upper, inside, outside)`
positionally; the image ships SimpleITK 2.1.1.2 (compiled 2022), which removed that
overload — `Execute()` now only takes the image, parameters must be set via
`SetLowerThreshold`/`SetUpperThreshold`/`SetInsideValue`/`SetOutsideValue` first. Crashes at
`niftymic/utilities/joint_image_mask_builder.py:52`, inside "Reconstruction Space
Generation" — this step is unconditional (builds a "unity mask" even with no
`--filenames-masks`), so there is no flag-based workaround. Fix: extracted the file from
the container, patched the one call to the setter-based API, bind-mounted the patch over
the original path at runtime (`--bind .../joint_image_mask_builder.py:/app/NiftyMIC/niftymic/utilities/joint_image_mask_builder.py`).
Grepped the rest of `niftymic`/`pysitk` for the same call pattern (multi-positional-arg
`.Execute(`) — no other hits, but the sweep wasn't exhaustive (regex-based, single pattern).

**3. NiftyMIC's output intensity scale doesn't match the input's.** Best linear fit on the
anatomy region (`gt > 1e-3`): `pred ≈ 1.03·gt + 1.02` (corr 0.886, subject 1) / `pred ≈
1.09·gt + 1.01` (corr 0.901, subject 2) — near-unity slope but a large additive offset, not
a structural mismatch. `--intensity-correction` didn't apply (log: "Reference image.
Skipped" — a no-op with only 1 stack, since it normally harmonizes *multiple* stacks to a
common reference). This is expected/documented behavior for Tikhonov-regularized SRR
solves in general, not specific to our off-label cardiac use — cross-method PSNR/SSIM
comparisons in the SVR literature routinely calibrate intensity before scoring for exactly
this reason. `score.py` fits+inverts the calibration on the anatomy region before computing
final metrics; raw (uncalibrated) numbers are also printed for transparency.

## Results (n=2)

| Subject | corr | PSNR_full (raw / calibrated) | PSNR_anat (raw / calibrated) | Wall time |
|---|---|---|---|---|
| Train_P053_t0 | 0.886 | 1.68 / 5.41 dB | −0.16 / **23.59 dB** | 18m11s |
| Val_P055_t0 | 0.901 | 0.67 / 8.69 dB | −0.18 / **25.28 dB** | 24m39s |
| **mean** | **0.893** | — / 7.05 dB | — / **24.43 dB** | — |

`PSNR_anat` (voxels where `gt > 1e-3`) is the honest number — same reasoning the project
already applies to `_bbox` vs `_full` (small-FOV padding inflates/deflates `_full`;
here NiftyMIC's own 50 mm `extra_frame` default writes real nonzero values into what our
canonical grid treats as zero-padding, dragging `_full` down independent of reconstruction
quality).

## Comparison to VGGT-MRI (same subjects, same phase, same metric)

Ran the current in-progress reference-conditioning checkpoint
(`scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt`,
epoch ~182/500, **still training** — not a final number) on the identical 2 subjects at
t=0, scored with the identical `PSNR_anat` (`gt > 1e-3` mask) via
`baselines/niftymic/eval_vggt_same_subjects.py`:

| Subject | NiftyMIC (calibrated) | VGGT-MRI | Gap |
|---|---|---|---|
| Train_P053_t0 | 23.59 dB | 31.96 dB | +8.4 dB |
| Val_P055_t0 | 25.28 dB | 33.45 dB | +8.2 dB |
| **mean** | **24.43 dB** | **32.70 dB** | **+8.3 dB** |

**Caveat — not a same-input comparison.** NiftyMIC got the clean, single, real, already-
aligned t=0 stack (this doc's established protocol). VGGT-MRI's eval called
`MRIDataset.get_data()` directly, bypassing the trainer's GPU-augmentation step — so its
S=20 scattered-slot input (still gating-free/phase-scattered across the 19 non-reference
slots) had **no synthetic respiratory corruption applied**, i.e. also not its full target
operating point. Both numbers are real and fairly *scored*, but the *inputs* aren't
matched in either direction. A stricter test would run VGGT-MRI through the actual trainer
aug path (respiratory on) on these same 2 subjects.

## Open questions / not yet done

- Only 2 subjects — no real sample size yet.
- NeSVoR, SVRTK not yet run (same protocol should apply; SVoRT-vs-`stack` registration
  question already resolved in favor of `--registration stack`, see prior conversation —
  single-orientation input gives either mode little to register against).
- Whether NiftyMIC's `--isotropic-resolution` / `--extra-frame-target` flags can be tuned
  to stop the padding spillover into `_full`, rather than relying on `_anat` to route
  around it.
- LSMR hit its 10-iteration cap both times ("iteration limit has been reached", not
  converged) — unclear whether more iterations would materially change the correlation, or
  whether single-stack input is simply information-starved regardless of iteration count.
