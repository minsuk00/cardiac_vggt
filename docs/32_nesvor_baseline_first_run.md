# 32 — NeSVoR baseline: first real end-to-end run

> **TL;DR & takeaway**
>
> First actual execution (not just research) of NeSVoR (`github.com/daviddmc/NeSVoR`,
> Docker `junshenxu/nesvor:v0.5.0`) — the only usable **ML** SVR baseline in the roster
> (`docs/31`). Same protocol as NiftyMIC's first run (`docs/29`): 2 val subjects
> (`Train_P053`, `Val_P055`), clean real per-phase t=0/ED SAX stack (the same tensor our
> own `V_gt` uses), single orientation, `--registration stack` (generic rigid, not
> SVoRT — SVoRT is fetal-brain-specific and out-of-domain for cardiac).
>
> **Result: it works, and clearly beats NiftyMIC on the identical protocol** —
> correlation **0.981 / 0.965** (vs NiftyMIC's 0.886/0.901), **PSNR_anat (calibrated)
> 32.00 / 30.28 dB** (vs NiftyMIC's 23.59/25.28 dB, **+6.7 dB mean**), and **faster**
> (~8.5–9 min/subject on GPU vs NiftyMIC's 18–25 min CPU-only). Consistent with
> NeSVoR being a more capable method (learned pose + neural field + robust fitting)
> than classical Tikhonov+RegAladin, even off-label on cardiac data. **Do not read
> this as "NeSVoR solves single-orientation through-plane recovery"** — both tools
> face the identical fundamental limit (`docs/31` §3); this only shows NeSVoR
> interpolates/regularizes *better* within that limit, not that the limit is gone.
>
> **One real bug found + fixed** (now baked into `baselines/nesvor/run_nesvor.sh`,
> not one-off): NeSVoR's default **internal sampling-mask estimation**
> (`PointDataset.mask` in `nesvor/inr/data.py`) voxelizes all slice points into a
> grid, Gaussian-blurs, then thresholds — a heuristic implicitly tuned for the
> denser, less-anisotropic point clouds typical of multi-stack fetal-brain data. On
> our **single stack** with **extreme anisotropy** (1.4 mm in-plane vs 12 mm
> through-plane, ~8.6:1 — well beyond typical fetal-brain ratios), the blurred
> occupancy grid came out entirely below threshold → an all-`False` mask →
> `IndexError: amin(): Expected reduction dim 0 to have non-zero size` in
> `Volume.resample()` during the final "Results saving" step (after ~7 min of
> otherwise-successful data loading, registration, and INR training — the crash is
> late, easy to mistake for something else). **Fix:** pass `--sample-mask` with our
> own already-good `content_mask.nii.gz` (the same mask NiftyMIC already uses
> successfully), which bypasses NeSVoR's internal mask estimation entirely — this is
> a documented, intended escape hatch (`--sample-mask`: *"If not provided, will use
> a mask estimated from the input data"*), not a workaround hack.
>
> **Runtime: ~8.5–9 min/subject on 1× A40 GPU** (data loading ~4s once warm,
> registration ~3s, INR training ~7 min fixed at 6000 iters, results-saving ~1–1.5
> min) — the container needs `--nv` (Singularity's GPU passthrough) and the shipped
> image's `nesvor` entrypoint has its own correct `#!/usr/bin/python3` shebang, so
> no separate Python/conda activation is needed to invoke it directly.
>
> **Update (same day) — respiratory-corrupted re-run, the real test.** Built
> `baselines/export_resp_stack.py` (applies the trainer's own live `RespiratoryConfig`
> — rigid SI/AP shift, per real z-slice, via `sample_resp_disp`/`reslice_volume_vec`
> — to the clean t=0 stack; GT unchanged). Re-ran NeSVoR on the corrupted input
> (same `--sample-mask` fix, still required): **mean corr drops 0.973 → 0.859,
> PSNR_anat drops 31.14 → 22.99 dB (−8.15 dB)** — a real, non-trivial degradation:
> not so small the corruption is trivially absorbed, not so large registration
> collapses to garbage. This is the first baseline number that actually measures
> something (motion correction), not pipeline round-trip fidelity.
>
> **Status:** n=2. Clean-stack run is proof-of-concept only (`docs/29`'s own caveat
> — measures pipeline round-trip fidelity, not motion correction, since that input
> literally *is* `V_gt`); the respiratory-corrupted re-run is the meaningful result.
> Next: SVRTK on the same corrupted protocol; VGGT-side comparison deferred (which
> checkpoint/sampling-regime to use is still an open question, tracked separately).

## What was run

`baselines/nesvor/` (new, mirrors `baselines/niftymic/`'s structure — no separate
export step; reads NiftyMIC's already-exported clean stacks directly):
- `run_nesvor.sh` — pulls `docker://junshenxu/nesvor:v0.5.0` via Singularity, stages
  the `.sif` onto node-local `/tmp` (mirrors the project's monai-cache
  GPFS-vs-local-NVMe pattern), then runs `nesvor reconstruct` per subject with
  `--input-stacks`, `--stack-masks`, `--sample-mask` (see bug below),
  `--thicknesses 8.0` (see gotcha below), `--registration stack`, `--device 0`.
- `score.py` — same shape as `niftymic/score.py`: resamples NeSVoR's isotropic
  output onto the exact canonical grid (`nibabel.processing.resample_from_to`),
  fits+inverts a linear intensity calibration on the anatomy region, reports raw +
  calibrated MAE/PSNR (full and anatomy-masked). Self-contained (not a shared
  module with `niftymic/score.py`) — deliberate, see the design note in this repo's
  plan history: not speculative to share code across only 2 tools when a 3rd
  consumer (SVRTK) doesn't exist yet.
- Input: `scratch/niftymic/data/{Train_P053_t0,Val_P055_t0}_{stack,mask}.nii.gz` —
  reused directly, byte-identical to NiftyMIC's input, no NeSVoR-specific export.

GPFS (`scratch/nesvor/`):
- `sif/nesvor.sif` — the pulled image (5.39 GB, `docker://junshenxu/nesvor:v0.5.0`,
  pinned explicitly, not `:latest` — the repo is dead upstream since 2023-07).
- `recon/<tag>/` — `recon.nii.gz`, `model.pt` (the fitted INR checkpoint), `run.log`.

**A gotcha confirmed but not actually hit as a bug:** NeSVoR's `--thicknesses`
defaults to the input NIfTI's slice **gap** if omitted. Our exported NIfTI's
Z-spacing is the canonical **pitch** (12.0 mm = 8 mm true slice thickness + 4 mm
gap; `docs/27`), so omitting `--thicknesses` would silently model a PSF 1.5×
too thick. Passed `--thicknesses 8.0` explicitly to avoid this.

## The bug (mechanism, so future runs don't rediscover it)

**Single-stack + extreme anisotropy → NeSVoR's internal sampling mask comes out
empty.** Traced by reading the actual NeSVoR source inside the container (not
guessed from the traceback alone):

1. `nesvor reconstruct`'s `exec()` (`cli/commands.py`) calls `train()`
   (`inr/train.py`), which returns `mask = dataset.mask` — a `Volume` built by
   `PointDataset.mask` (`inr/data.py`): voxelize every slice pixel's transformed
   `(x,y,z)` point into a grid via `torch.bincount`, Gaussian-blur it, threshold at
   `mask_threshold` (a formula involving `resolution_min`/`resolution_max` meant to
   correct for anisotropy).
2. After training, `exec()` passes this `mask` into `_sample_inr()`, which calls
   `override_sample_mask(mask, sample_mask=None, output_resolution=0.8, ...)`. Since
   no `--sample-mask` was given and `--output-resolution` defaults to `0.8`,
   `override_sample_mask` calls `mask.resample(0.8, None)`
   (`Volume.resample()` in `image/image.py`).
3. `Volume.resample()`'s first line is `xyz = self.xyz_masked` — the coordinates of
   voxels where `mask`'s own boolean occupancy tensor is `True`. With our single
   stack and ~8.6:1 in-plane:through-plane anisotropy, the blurred occupancy grid
   from step 1 apparently never exceeded `mask_threshold` anywhere → `self.mask` is
   all-`False` → `xyz` has 0 rows → `xyz.amin(0)` raises `IndexError: Expected
   reduction dim 0 to have non-zero size`.
4. This crash happens **after** ~7 minutes of otherwise-successful work (data
   loading, `--registration stack`'s "use stack transformation", full INR
   training) — easy to misdiagnose as a training-time or GPU problem when it's
   actually a downstream sampling-mask degeneracy specific to sparse/single-stack
   anisotropic input.

**Fix:** `--sample-mask "<path>/{tag}_mask.nii.gz"` (our own `content_mask`, the
same file already passed as `--stack-masks`) makes `override_sample_mask` take the
`new_mask is not None` branch (`mask = load_mask(new_mask, ...)`), replacing the
degenerate estimated mask entirely — sidestepping the buggy internal heuristic
rather than patching it. This is a documented, intended CLI option (`--sample-mask`
help text: *"3D Mask for sampling INR. If not provided, will use a mask estimated
from the input data"*), not a container patch like NiftyMIC's SimpleITK fix — no
Singularity bind-mount override was needed here.

**Likely root cause, for future reference:** NeSVoR's occupancy-estimation
heuristic is implicitly tuned for the point-cloud density and anisotropy typical of
multi-stack fetal-brain SVR (multiple orientations → denser 3D coverage, milder
anisotropy). Single-orientation cardiac SAX at 1.4×1.4×12 mm sits well outside that
implicit assumption. **This is expected to recur for any single-stack NeSVoR run on
this data** (e.g. the future respiratory-corrupted protocol) — always pass
`--sample-mask` explicitly, don't rely on the default estimation.

## Results (n=2)

| Subject | Method | corr | PSNR_full (calibrated) | PSNR_anat (calibrated) | Wall time |
|---|---|---|---|---|---|
| Train_P053_t0 | NiftyMIC | 0.886 | 5.41 dB | 23.59 dB | 18m11s (CPU) |
| | **NeSVoR** | **0.981** | **34.63 dB** | **32.00 dB** | **8m32s (GPU)** |
| Val_P055_t0 | NiftyMIC | 0.901 | 8.69 dB | 25.28 dB | 24m39s (CPU) |
| | **NeSVoR** | **0.965** | **31.70 dB** | **30.28 dB** | **8m57s (GPU)** |
| **mean** | NiftyMIC | 0.893 | 7.05 dB | 24.43 dB | — |
| | **NeSVoR** | **0.973** | **33.16 dB** | **31.14 dB** | — |

Unlike NiftyMIC, NeSVoR's `PSNR_full` does **not** stay depressed relative to
`PSNR_anat` — NiftyMIC's low `PSNR_full` was specifically caused by its default 50
mm reconstruction-space padding spilling nonzero values into our zero-padded
canonical background (`docs/29`); NeSVoR's own reconstruction-space extent doesn't
appear to have the same artifact on this data. `PSNR_anat` remains the metric to
trust for cross-tool comparison regardless (project-wide convention, `_bbox`/`_full`
in CLAUDE.md's Logging section).

Raw (pre-calibration) intensity fits: `pred ≈ 12342·gt − 4.8` (Train_P053_t0),
`pred ≈ 11116·gt + 10.6` (Val_P055_t0) — NeSVoR's INR training targets an internal
`--output-intensity-mean` (default 700.0), so its absolute output scale is even
further from our `[0,1]`-normalized input than NiftyMIC's was; the same
calibrate-before-scoring rationale from `docs/29` applies, just with a larger `k`.

## Respiratory-corrupted re-run (docs/30 §4 step 2 / docs/31 §8 step 2)

**The clean-stack run above has nothing to correct** — the exported stack *is*
`V_gt`, so its numbers measure NeSVoR's round-trip fidelity on already-correct
data, not motion correction. This section gives NeSVoR (and future NiftyMIC/SVRTK
re-runs) an actual registration problem to solve.

**What was built:** `baselines/export_resp_stack.py` — loads the SAME clean t=0
stack `export_stack.py` already produces, then applies the trainer's own
`training/data/respiratory.py` simulation **per real z-plane**: `sample_resp_disp(1,
12, cfg, device, train=False, seq_index=idx)` draws one val-time-deterministic
`(d_D, d_H, d_W)` mm displacement per z-plane "slot" (seeded by the same `idx` used
for subject selection, so it's reproducible and matches what the trainer's own val
path would draw), then `reslice_volume_vec(V, disp[0, z])` reslices the WHOLE clean
volume by that plane's displacement and keeps only that plane's own resliced
content. `RespiratoryConfig` is loaded from the **live** `mri_volume.yaml` (Hydra
`compose()`, mirroring `inference/run_cmrxrecon.py`'s pattern) rather than hand-copied
defaults — confirmed `enable=True amplitude_mm=16.0 ap_ratio=0.35
direction_jitter_deg=30.0` (matches CLAUDE.md's stated trainer defaults). GT stays
the original clean stack; `content_mask` is reused unshifted for both variants
(acquisition-geometry property, doesn't move with a modest anatomical shift).
Output: `scratch/niftymic/data/<tag>_resp_stack.nii.gz` (same dir/mask as the clean
export). Confirmed nonzero corruption before running anything: mean
`|corrupted−clean|` = 0.014 / 0.017 (Train_P053_t0 / Val_P055_t0).

**`run_nesvor.sh`/`score.py` parameterization:** `STACK_SUFFIX=resp_stack bash
baselines/nesvor/run_nesvor.sh <tag>...` points `--input-stacks` at the corrupted
NIfTI while `--stack-masks`/`--sample-mask` stay on the unshifted `content_mask`;
writes to a separate `scratch/nesvor/recon_resp_stack/` so the clean-stack results
in `scratch/nesvor/recon/` are never touched. `score.py` reads an optional
`NESVOR_RECON_DIR` env var for the recon location while `DATA_DIR` (GT) stays
hardcoded to the clean stack regardless — GT must never change between variants.
**The `--sample-mask` fix (see bug above) is still required** here — same
single-stack/extreme-anisotropy mechanism, unrelated to whether the input is clean
or corrupted.

**Results (n=2):**

| Subject | Protocol | corr | PSNR_anat (calibrated) |
|---|---|---|---|
| Train_P053_t0 | Clean | 0.981 | 32.00 dB |
| | **Resp-corrupted** | **0.846** | **21.99 dB** |
| Val_P055_t0 | Clean | 0.965 | 30.28 dB |
| | **Resp-corrupted** | **0.873** | **23.98 dB** |
| **mean** | Clean | 0.973 | 31.14 dB |
| | **Resp-corrupted** | **0.859** | **22.99 dB** |

**Reading it:** a real ~8.15 dB / 0.11-corr drop under real misalignment — evidence
the corruption amplitude (16 mm mean SI breath-depth, per the trainer's default) is
a genuine, non-trivial registration challenge for NeSVoR's rigid stack-to-stack
mode: neither absorbed for free (would show ~0 drop) nor catastrophic (corr stayed
0.85–0.87, not collapsed). This is the first baseline number in the roster that
actually reflects motion-correction capability rather than pipeline round-trip
fidelity. No VGGT-side comparison yet (see the checkpoint/sampling-regime question
deferred above) — that's the next step to make this a full head-to-head.

## Why NeSVoR scores higher than NiftyMIC here

Not yet root-caused with certainty — plausible, unverified contributors: (1) NeSVoR
fits a continuous neural field rather than a fixed-grid Tikhonov solve, which can
represent smooth structure more efficiently at this resolution; (2) NeSVoR's
implicit-neural regularization + learned encoding may generalize better within a
single sparse anisotropic stack than a purely geometric registration+SR solve; (3)
NiftyMIC's LSMR hit its 10-iteration cap without converging (`docs/29`), while
NeSVoR ran its full fixed 6000-iteration training schedule to completion. **This
gap should not be over-interpreted** — both tools face the identical fundamental
single-orientation-through-plane limit (`docs/31` §3); this result says NeSVoR
handles the *interpolation regime* better, not that it escapes the limit.

## Open questions / not yet done

- Only 2 subjects — no real sample size yet (same caveat as `docs/29`).
- SVRTK not yet run, on either protocol.
- NiftyMIC not yet re-run on the respiratory-corrupted stack (the corrupted export
  already exists at `scratch/niftymic/data/<tag>_resp_stack.nii.gz` — but
  `baselines/niftymic/run_niftymic.sh` on disk is stale/broken, see `docs/29`/`31`;
  must be fixed to the real working bind-mount invocation first).
- **VGGT-side comparison on the respiratory-corrupted protocol not yet done** — open
  question about which checkpoint/sampling regime is the right match: the
  reference-conditioning checkpoint (`217721337`) used for the clean-stack VGGT
  number is pre-multi-frame (`num_slices=12`); the "deployment-realistic"
  multi-frame batch (`inference/run_cmrxrecon.py`'s `_build_multiframe_batch` — full
  cardiac cine at the mid-ventricular reference plane + 5-frame bursts at every
  other in-bbox plane, ~58 slots total) is a different, more realistic acquisition
  contract, but its own default checkpoint (`217720691`, diffusion variant) is
  ALSO pre-multi-frame — no checkpoint currently exists that's both
  reference-conditioning AND trained under the S=20/multi-frame or
  deployment-realistic regime. Deferred, not blocking further baseline work.
- Whether NeSVoR's `--deformable` mode (non-rigid, meant for uterus/body, not used
  here) would change anything on single-orientation cardiac input — not attempted
  since our respiratory sim is confirmed pure rigid (`docs/30`), so rigid
  `--registration stack` is the mechanistically correct match, not deformable.
- Whether NiftyMIC would show a similar or different-sized drop under the same
  respiratory corruption (its Tikhonov+RegAladin vs. NeSVoR's learned pose + INR
  might respond differently to the same misalignment amplitude).
