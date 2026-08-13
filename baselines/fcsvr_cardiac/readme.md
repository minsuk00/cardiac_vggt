# FC-SVR Cardiac Stage-1 Baseline

This directory is an isolated fork of the released FC-SVR repository at
`baselines/fc_svr`. It implements **FC-SVR Stage 1, GT-pose-normalized** for the
CMRxRecon2024 fixed-ED respiratory-motion experiment. It is not full FC-SVR,
does not include Stage 2 inpainting, and is not a deployable upper bound.

No VGGT source file or dataset is modified by this implementation. The cardiac
entrypoints read the existing `MRIDataset` preprocessing, respiratory simulator,
`cmrx24only.txt`, and `vggt/utils/splat.py` in place as read-only dependencies.
Fork-local code and tests live here. Large checkpoints, logs, W&B files, and
evaluation output default to the GPFS repository at
`/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/`, never the home
repository.

## Prove-it audit status: training code ready; long-job wrapper still required

A three-reviewer whole-target audit on 2026-08-09 compared the adapter against
the paper, released training/inference code, tests, and VGGT respiratory
contracts. Every confirmed implementation defect was corrected without
changing the released architecture, loss, compensation, or upsampler:

- dense motion remains `(B,D,256,256,3)`; only the four repeated slab planes
  are averaged at each pixel;
- intensity is zeroed outside the acquired heart foreground after coarse
  interpolation, and that same acquired foreground gates target channel 4;
- the adapter rejects `dz_mm != 12`; variable-pitch physical scaling is deferred;
- validation reports the exact released coarse L22/L21 metrics plus unambiguous
  component-MSE, paper-formula vector-MSE, and EPE in mm and 3-mm slab voxels;
- cardiac model imports no longer load legacy Lightning wrappers;
- the released value-gradient clip of 0.5 and shuffled subject order are used;
- strict CUDA determinism is enabled around no-gradient validation only. It is
  intentionally not enabled during training because PyTorch 2.13 has no
  deterministic CUDA backward for the released model's 3D `grid_sample`;
- W&B logs train loss/LR and aggregate raw/compensated validation metrics. Its
  16-character id is persisted at `<output>/wandb_id.txt`; checkpoint resume
  uses W&B `resume_from=<id>?_step=<checkpoint_step>` so server history is
  rewound to the same committed step before regenerated metrics are logged;
- `checkpoint_last.pt` is an atomic full model/Adam/step checkpoint, saved after
  each validation interval (default 5,000 steps) and the final step.

The real-data GPU gate passed on SLURM job `56753945`, NVIDIA L40S, with PyTorch
`2.13.0+cu130` and `torch-interpol==0.2.4`: one CMR24 forward/backward step,
full 3.846-GB model/Adam checkpoint, checkpoint resumes from step 1 through step 3,
one-subject raw/compensated validation, and two exactly equal validation runs.
Peak allocated GPU memory during the measured training/validation benchmark was
6.07 GiB.
Fork-local tests report 26 passed. `original_source_manifest.sha256` verifies
that every file in `baselines/fc_svr` remains unchanged.

One known comparison mismatch is intentionally unchanged per user direction:
FC-SVR validation currently uses seeds `42+index` (standalone evaluation) or
`10042+index` (training validation), while VGGT uses `seq_index`. The marginal
simulator distribution is the same, but subject-level corruptions are not paired.
Pair seeds or persist a shared displacement set before reporting paired tests.

The released foreground-centered `BoundingBox3d` crop remains a declared
adapter deviation because this protocol intentionally fixes the model grid at
`(2D,60,60)` as specified in the implementation plan. Held-out test selection,
aggregate provenance export, and persistent motion/reconstruction artifacts are
evaluation-harness work; they do not affect the approved 256,000-step training
path and must be completed before final paper tables are frozen.

### Complete deviation inventory

This is an **adapted FC-SVR Stage-1 baseline**, not a reproduction of full
FC-SVR. Every known difference from the paper or released pipeline is listed
here:

- Stage 1 only; the learned Stage-2 interpolator and artifact-free final FC-SVR
  reconstruction are omitted.
- Fixed-ED CMR24 replaces adult/fetal brain data and atlas registration.
- One native SAX direction replaces the adult three-direction models and fetal
  random-direction setting.
- Independent per-plane VGGT respiratory translations replace the authors'
  smooth B-spline slice rotations/translations and two-shot interleaving.
- The motion target contains translations only; no per-slice rotations.
- Global 3D augmentation is reduced to in-plane flip/rotation/translation/zoom;
  through-plane rotation is omitted.
- The synthetic four-voxel boxcar acquisition PSF is omitted because CMR24 is
  already acquired slice data.
- Physical 8-mm-thick/12-mm-pitch CMR slices are represented as four repeated
  approximately 3-mm slab planes.
- For native depths below eight, the adapter temporarily zero-pads only the
  coarse Stage-1 depth to 16; the released cohorts did not require this cardiac
  short-stack padding.
- After released upsampling, each group of four repeated slab predictions is
  averaged at matching H/W coordinates to recover one acquired physical slice;
  released inference instead splats all repeated foreground planes.
- The released foreground-centered `BoundingBox3d` crop is omitted; the full
  coarse `60×60` canonical FOV is processed.
- CMR percentile normalization replaces the released min-max/remap preprocessing.
- Paper-specified gamma/noise augmentation is enabled on every cardiac training
  sample, whereas the released source has its noise application and gamma term
  disabled/commented; zoom is also applied every training sample rather than
  through the released `MaybeTransform(..., 0.9)` gate.
- A dilated union-over-all-phases heart ROI replaces brain foreground; deriving
  it from complete gated cine is privileged segmentation information and must
  be disclosed alongside `GT-pose-normalized`.
- The earlier clean-coordinate loss-mask defect is corrected; the acquired
  foreground now gates both input and target.
- The earlier dense-to-translation projection is removed; dense native
  per-pixel motion is preserved.
- Native VGGT splatting replaces the released `torch-interpol` dense splatter.
- Output stays on native `(D,256,256)` instead of the paper's high-resolution,
  Stage-2-interpolated volume.
- Compensation uses simulated GT motion, making the official compensated result
  non-deployable.
- VGGT-style intensity volume metrics augment the paper metrics; motion MSE/EPE
  are now logged in mm and slab voxels, while slice PSNR remains missing.
- CMR24 split sizes are 235 train / 29 validation / 30 test instead of the
  paper's cohorts and four-fold validation protocol.
- Training uses a minimal float32 loop instead of released fp16 Lightning. The
  released value-gradient clipping threshold of 0.5 is preserved.
- Batch size 1 follows the released FeTA shell script and cardiac memory plan,
  while generic released defaults differ.
- The paper specifies 256,000 steps; the public FeTA shell script specifies
  300,000. This adapter follows the paper's 256,000.
- Subjects are shuffled once per logical epoch with a deterministic seeded
  permutation, matching the released DataLoader's shuffle behavior while
  remaining exactly resumable from a global step.
- Validation respiration seeds currently differ from VGGT's subject seeds
  (confirmed fairness defect).
- Cardiac entrypoints now avoid legacy Lightning imports.

What is preserved exactly: `Flow_SNet3d0_1024`, its dense output head, dormant
`project()`, `l22_loss_affine_invariant`, released compensation algebra, and
`upsample_flow()`. SHA comparison during the audit found the copied author model,
loss, transform, and dataset source files byte-identical to `baselines/fc_svr`.
The `D→4D→2D` construction, `/6` target units, released ×2 flow upsampling,
×3-mm conversion, `(D,H,W)↔(x,y,z)` axes, respiratory pull/splat sign, and
padding/unpadding are coherent for **exactly 12-mm pitch**.

## Original implementation plan (verbatim)

This is preserved verbatim as requested. Its “paper-faithful” language states
the original intent and is superseded by the prove-it audit above wherever the
current implementation does not satisfy it.

### FC-SVR Cardiac Stage-1 Baseline

#### Summary

- Keep `baselines/fc_svr` untouched.
- Copy it to `baselines/fcsvr_cardiac`, excluding nested `.git` and generated outputs.
- Implement Stage 1 only: original `Flow_SNet3d0_1024`, SVD-invariant loss, and compensation.
- First experiment: CMR24, fixed ED, current respiratory simulator, `cmrx24only.txt`.

#### Implementation

- Replace obsolete Lightning training with a minimal PyTorch loop:
  - batch size 1;
  - 256,000 steps;
  - Adam, LR `1e-4`, polynomial decay `0.9`;
  - float32;
  - resumable last checkpoint.
- Reuse `MRIDataset` preprocessing and `training/data/respiratory.py`.
- Use all native ED slices and generate one respiratory displacement per z-plane.
- FC-SVR working representation for CMR24:
  - native stack: `(D,256,256)`, spacing `(12,1.4,1.4)` mm;
  - repeated slab: `(4D,120,120)`, approximately 3 mm;
  - Stage-1 input: `(2D,60,60)`, approximately 6 mm.
- If `2D < 16`, symmetrically zero-pad only the temporary Stage-1 depth to 16. Set its mask to zero and remove it after prediction.
- Keep the original SVD loss unchanged. Do not activate the dormant `project()` function.
- Use `torch-interpol==0.2.4`; Cornucopia and old Lightning code are not needed by the cardiac entrypoints.

#### Reconstruction and Evaluation

- Produce both raw and paper-compensated motion.
- Apply the released `upsample_flow()` operation.
- Remove padding, group each four-plane slab back into one physical slice, and convert flow from voxel units to millimetres.
- Splat the original respiratory-corrupted `256×256` slices once using `vggt/utils/splat.py`.
- Final output remains native `(D,256,256)`—no high-resolution reconstruction or resampling back.
- Score with the existing VGGT volume metrics on the identical GT grid.
- Treat compensated metrics as the official paper-faithful result; log raw metrics as a diagnostic.
- Explicitly label this as **“FC-SVR Stage 1, GT-pose-normalized”**, not full FC-SVR or a deployable upper bound.

#### Tests and Documentation

- Test zero-motion identity reconstruction.
- Test millimetre/voxel conversion, axis order, and motion sign.
- Test `D=6`, `D=8`, and `D=10` padding/unpadding.
- Test that compensation removes one global transform but preserves relative slice errors.
- Run a real CMR24 one-sample forward/backward smoke test and deterministic validation twice.
- Confirm `baselines/fc_svr` remains byte-identical.
- Record the final protocol and corrections to doc 67 in a new numbered document.

#### Assumptions and Deferred Work

- CMR24 pitch is asserted to be 12 mm; pooled variable-pitch support is deferred.
- Fixed ED only for this first test.
- Random same-phase training is the next extension.
- Mixed cardiac phases and Stage 2 are out of scope.

**TL;DR:** Fork the original code, preserve the published Stage-1 architecture/loss/compensation, use a temporary padded coarse slab grid, and render directly back to native CMR24 volumes with the common splatter.

## Exactly what differs from upstream FC-SVR

The directory began as a file-for-file copy of `baselines/fc_svr`, excluding
the nested Git repository and generated directories (`checkpoints`, `outputs`,
`lightning_logs`, `wandb`, and Python caches). The released architecture files
remain copied locally. The cardiac path adds or changes only the following:

- `cardiac.py` is the geometry boundary. It creates the repeated/coarse slab,
  applies temporary depth padding, converts between external FC-SVR `(x,y,z)`
  flow channels and physical `(D,H,W)` millimetres, implements the released
  compensation algebra as a directly testable function, and performs the one
  final native-grid splat.
- `pipeline.py` is the read-only CMRx24 adapter. It constructs `MRIDataset` in
  fixed-ED/static/native-z mode, samples one respiratory vector for every
  native plane using the current VGGT simulator, forms model pairs, evaluates
  raw and compensated predictions, computes the same unit-range full/bbox
  MAE, MSE, and PSNR definitions used by VGGT, and calls the unchanged released
  L22/L21 losses for exact coarse-grid Stage-1 metrics.
- `train.py` replaces the copied Lightning trainer for cardiac entrypoints only.
  It is a batch-one float32 loop using the released model and unchanged released
  `l22_loss_affine_invariant`, Adam at `1e-4`, `(1-step/256000)^0.9` decay, and
  the author's value-gradient clip at 0.5. It writes an atomic resumable
  `checkpoint_last.pt` containing model, optimizer, and step.
- `evaluate.py` loads either a cardiac full checkpoint or weights dictionary,
  emits both raw and compensated metrics, and can run validation twice and fail
  if the JSON-serializable results differ.
- `requirements-cardiac.txt` pins `torch-interpol==0.2.4` and `wandb==0.25.0`.
  The cardiac model exports no longer import legacy Lightning wrappers.
- `runtime.py` owns seeded epoch shuffling, author-matched gradient clipping,
  and CUDA reproducibility policy without modifying model operations.
- `tracking.py` persists one W&B run ID and reuses it across restarts.
- `tests/test_cardiac.py` contains fork-local geometry/correctness tests.
- `tests/test_runtime.py` covers shuffle resume, gradient clipping, and strict
  validation-determinism scoping, positive CLI bounds, fresh-output safety, and
  torn-tail JSONL recovery.
- `tests/test_tracking.py` covers persistent W&B IDs and checkpoint-step rewind
  initialization.
- `original_source_manifest.sha256` is the audit record for the untouched
  `baselines/fc_svr` source tree.
- `readme.md` is this complete protocol record. The copied upstream `README.md`
  remains available as the upstream usage/history document.

The copied `models/flow_SNet4.py`, `models/losses.py`, and
`models/flow_UNetS.py` are not modified. Consequently:

- the network is exactly `Flow_SNet3d0_1024`;
- Motion SVD Loss is exactly the released `l22_loss_affine_invariant`;
- `Flow_SNet.project()` remains dormant because `rigid=False` and no cardiac
  code changes it;
- paper compensation uses the released least-squares/SVD rigid alignment;
- flow enlargement uses the released `model.upsample_flow()`.

## Data contract and preprocessing

`pipeline.make_dataset()` imports `data.MRIDataset` from the repository's
`training` path. It passes `t_target_fixed=0`, `mode="static"`,
`one_frame_per_slice=True`, `continuous_z=False`, and
`defer_input_images=True`. Thus the target and every acquired slice are ED,
all native z planes are used once, and no cached file is rewritten by this fork.

The default root is `/home/minsukc/vggt/scratch/data`; the default split is
`training/splits/cmrx24only.txt`. These are command-line overrides, not copied
or edited data. This first protocol is valid only for the asserted CMR24 pitch
of 12 mm. The loader returns each subject's `dz_mm`; the pipeline uses that
value when running the respiratory reslicer and rendering, but this experiment
must reject/non-combine other-pitch cohorts because the learned slab convention
is intentionally fixed at approximately 3/6 mm.

Respiration parameters match the current shipped moderate-pipeline simulator:
amplitude 18.8±7.35 mm, Lujan `sin^6`, AP ratio 0.35, burst grouping by z plane,
subject-level tilt uniformly 0–45 degrees, and no per-breath amplitude jitter.
`sample_resp_disp()` receives `group_ids = 0..D-1`, so exactly one independent
respiratory phase is drawn per physical slice. Its returned `(D,H,W)` value is
one **translation vector per slice**, not a dense DVF. The reslicer broadcasts
that one vector across the slice's pixels only to construct the dense sampling
grid required by `grid_sample`; all pixels still undergo the same rigid
translation. The vector is both the applied corruption and supervised target.
The simulator's sign is preserved: positive motion samples deeper anatomy at
the fixed plane, so the observed pixel is splatted to the positive physical
coordinate for correction.

The VGGT public extraction function appends a DINO-specific tail: native
256-pixel reslice → 518-pixel resize → RGB replication → `[0,255]`. FC-SVR does
not need that representation. `cardiac.extract_respiratory_slices_256()` uses
the same per-axis mm normalization, `grid_sample`, zero padding, bilinear mode,
and `align_corners=True` equations, but stops at the native grayscale
`(B,S,256,256)` reslice. This avoids the former, unnecessary
`256→518→256` double interpolation. A test compares it directly against VGGT's
`reslice_volume_vec()` for non-zero 3-axis translations.

## Faithfulness contract and augmentation

The governing rule is: reproduce released FC-SVR unless the cardiac experiment
requires a declared substitution. The model, SVD loss, compensation, slab
representation, and upsampler are preserved. No correction is permitted inside
`models/flow_SNet4.py`, `models/flow_UNetS.py`, or `models/losses.py`; cardiac
adaptation and compatibility code must stay at the data/geometry/entrypoint
boundary. Every task-level substitution is declared below.

There is one deliberate task-level substitution: the author's synthetic smooth
B-spline slice-motion trajectory is replaced by the current VGGT respiratory
translation simulator. The two motion generators are **not composed**; doing so
would train on a different, harder corruption than either FC-SVR or the common
VGGT comparison. Respiratory motion is translation-only, so its dense target is
spatially constant within each slice even though Flow_SNet retains its original
dense per-pixel output capacity.

The applicable author augmentations are train-only and implemented locally:

- one transform shared by all cardiac phases and z planes;
- horizontal flip with probability 0.5;
- in-plane rotation over ±180° (the released FeTA regime);
- translation over ±13 pixels;
- isotropic in-plane zoom over ±10% (the released `zooms=(-0.1,0.1)` regime);
- slice gamma exponent uniformly in `[0.9,1.0]` and Gaussian noise σ=0.01,
  following paper Appendix A.1.

Through-plane global rotation is not applied: the authors augment an isotropic
0.8/1-mm reference volume, whereas CMR24 has acquired 12-mm z pitch. Rotating
that native stack through-plane would interpolate missing anatomy and cease to
be a faithful reproduction. Their slice-varying rotation/translation and
two-shot B-spline trajectory are the acquisition-motion component replaced by
the declared respiratory simulator. The paper's synthetic four-voxel boxcar
PSF is also not re-applied because CMR24 inputs are already physical slice
acquisitions with finite thickness rather than dense isotropic atlas volumes.
Validation receives respiratory corruption only and no random author
augmentation, matching the released train/validation separation.

## Foreground masks

The authors use an anatomy foreground mask as input channel 1 and target channel
3, and Appendix A.1 states that background voxels are masked out of the motion
loss. The current adapter constructs both channels without changing Flow_SNet:

- input channel 1 is the heart foreground resliced with the **same respiratory
  translation as its acquired intensity slice**, matching the author's
  jointly-transformed image/foreground pair;
- target channel 3 is the same acquired/motion-corrupted heart foreground as
  input channel 1, matching released FC-SVR's coordinate frame.

The released `l22_loss_affine_invariant` uses target channel 3 both to select
points for its least-squares/SVD global rigid fit and to weight the final motion
residual. Voxels outside the heart mask therefore contribute exactly zero loss.
The adapter fails loudly if the ROI is absent, stale/wrong-shaped (the dataset
omits it), or empty. All 294 deduplicated CMR24 subject directories currently
contain a heart-ROI file; shape validity is still checked per loaded subject.
`content_mask` remains a geometric validity constraint: it is intersected with
the clean heart ROI before augmentation, but it is not substituted for the
author's foreground input channel. The unchanged forward path discards this
second channel, so the adapter also foreground-multiplies intensity after
coarse interpolation, matching released real-stack inference.

## Working-grid geometry

For a native corrupted stack `(D,256,256)`:

1. Bilinear/trilinear interpolation produces `(D,120,120)`.
2. Each physical plane is repeated four times, producing `(4D,120,120)`.
3. Taking every second voxel in all axes produces `(2D,60,60)`.
4. The intensity and respiratory-resliced heart foreground become the two input
   channels, matching the released intensity/foreground contract.
5. The physical displacement is reordered from `(D,H,W)` to the released
   external `(x,y,z)` channel order and divided by 6 mm per coarse voxel.
6. The target's fourth channel is the acquired, respiratory-resliced heart
   foreground (derived from the clean heart/content-mask intersection before
   reslicing); this gates the released SVD loss in observed-slice coordinates.
7. When `2D<16`, `16-2D` zero planes are split floor/ceil before/after. Both
   input channels, all target-flow channels, and the target mask are zero there.
   The unpadded depth is recorded in `Stage1Meta`.

For `D=6`, padding is two planes before and two after. For `D=8` and larger,
there is no padding (`D=8` is exactly depth 16; `D=10` remains depth 20).
Padding is removed from the coarse prediction before `upsample_flow()` so it
cannot become part of the physical slab.

The released upsampler maps `(2D,60,60)` to `(4D,120,120)` and scales flow by
two, leaving flow in the approximately 3-mm repeated-slab voxel convention.
The adapter averages each group of four repeated planes **at each H/W
location**, never across H/W. It multiplies by `(3,3,3)` mm, reorders `(x,y,z)`
back to `(D,H,W)`, and bilinearly returns the dense physical field to native
`256×256`.

## Reconstruction and metric protocol

The renderer constructs physical scanner coordinates for every pixel of every
original corrupted 256-pixel slice. X/Y are normalized across the native grid;
z uses the repository convention `z_mm / 90`. Each pixel's predicted dense
physical displacement is added once, and `vggt.utils.splat.splat_to_volume()` renders to
the identical native `(D,256,256)` target grid. Splat weights use the
respiratory-resliced heart foreground, matching the released inference code's
foreground mask rather than an intensity threshold. There is no
Stage 2, inpainting, high-resolution reconstruction, atlas alignment, or final
resampling.

Every validation subject emits two metric dictionaries:

- `raw`: direct Stage-1 motion, retained only as a diagnostic;
- `compensated`: Stage-1 motion after fitting/removing one global rigid transform
  against the known target motion. This is the official GT-pose-normalized
  Stage-1 result.

The record label is always `FC-SVR Stage 1, GT-pose-normalized`. Compensation
uses target pose and therefore must not be described as deployable or as a
motion estimator with an observable global frame. Metrics are unit-range
full-volume, geometric-bbox, and heart-foreground MAE/MSE/PSNR on the exact GT
tensor. Heart-foreground metrics are the intended primary adapted-baseline
result; full/bbox metrics remain diagnostics because foreground-masked FC-SVR
does not attempt to reconstruct intensities outside its splat mask.

Motion reporting deliberately separates two MSE conventions that the paper and
released implementation do not scale identically:

- `metric_released_coarse_l22_component_vox2` and
  `metric_released_coarse_l21_epe_vox` call the byte-identical released loss
  functions on the coarse `(2D,60,60)` field with `eps=0`;
- `metric_motion_component_mse_*` averages squared error over all `N×3`
  components, matching the released L22 normalization;
- `metric_motion_paper_mse_*` computes
  `mean_N(sum_xyz(error²))`, matching Appendix A.2's literal
  `N^-1 ||u-y||_F²` formula and therefore equalling three times component-MSE;
- `metric_motion_epe_*` computes `mean_N(||error||₂)`, matching Appendix A.2.

The native-grid physical diagnostics are calculated after released upsampling,
four-copy reduction, and in-plane interpolation; the compensated row first
applies released compensation, while the raw row intentionally does not. They
are reported in millimetres/mm² and 3-mm slab voxels/voxel². The exact released
coarse metrics are retained alongside them so a paper result cannot silently
conflate the two evaluation grids or the two MSE normalizations.

## Training, resume, and evaluation

### Measured runtime and SLURM requirement

An on-node L40S benchmark on 2026-08-09 timed 100 complete training steps across
50 real CMR24 subjects after their one-time MONAI cache construction. Each step
included cached data preparation, respiratory simulation, augmentation,
forward, SVD loss, backward, 0.5 clipping, and Adam:

- training: 40.369 s total = **0.404 s/step** = 2.48 steps/s;
- direct 256,000-step optimizer-path projection: **28.7 hours**;
- peak allocated GPU memory in the real-data smoke gate: **6.07 GiB**.

This supersedes two misleading microbenchmarks: a single small-depth cached
subject projected 8.9 hours, while 50 previously unseen subjects (including
lazy preprocessing/cache construction) projected 82.6 hours when that one-time
cost was incorrectly multiplied by all 256,000 steps. Neither represents the
full cached cohort.

Validation, 52 large GPFS checkpoint writes, per-step JSONL/W&B logging,
startup, and scheduling overhead are not included in the 28.7-hour projection.
Budget **roughly 32–36 hours** on one L40S; this range is an operational margin,
not a measured end-to-end run. A 48-hour allocation should fit, while the
current five-day allocation has ample margin. The 5,000-step checkpoint cadence
bounds a manual-resume rollback to about 34 minutes at measured compute speed;
no fork-local automatic requeue script has been implemented yet.

Run from the repository root in the `svr` environment:

```bash
PYTHONPATH=baselines/fcsvr_cardiac:training:. \
python baselines/fcsvr_cardiac/train.py
```

Resume the GPFS last checkpoint:

```bash
PYTHONPATH=baselines/fcsvr_cardiac:training:. \
python baselines/fcsvr_cardiac/train.py --resume
```

The default output is
`/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/fcsvr_cardiac_stage1_cmrx24_ed/`.
Training metrics append to
`train_metrics.jsonl`; raw and compensated per-subject validation records append
to `val_metrics.jsonl`; `checkpoint_last.pt` is atomically replaced after each
validation. Defaults are 256,000 optimizer steps, batch size one, float32, Adam,
LR `1e-4`, polynomial power 0.9, validation/checkpoint every 5,000 steps.
For safety, a fresh launch refuses an output directory containing any run
artifact, and `--resume` refuses to run without `checkpoint_last.pt`. Resume
truncates JSONL rows newer than the checkpoint step before continuing, preventing
duplicated metrics when a crash happened between checkpoint intervals. A torn
final JSONL record from an interrupted append is discarded; malformed interior
records still fail loudly as evidence of non-tail corruption.
GPFS checkpoints are staged to node-local `/tmp` through the repository's
read-only checkpoint-staging utility before `torch.load`; checkpoint writes
remain atomic on GPFS.

W&B project `fcsvr-cardiac` logs each training loss/LR and the mean of every
raw/compensated validation metric. The run id is minted once, persisted in the
output directory, and reused across `--resume`, crash restart, or manual
resubmission. On checkpoint resume, the pinned W&B 0.25 `resume_from` API
rewinds history to the checkpoint step, preventing stale or dropped metrics
when disk logs correctly replay post-checkpoint training. Set
`WANDB_MODE=online` for the full run and use
`WANDB_MODE=offline` only for local pilots. Offline restarts create separate
local `.wandb` containers but retain the same persisted run ID; online restarts
reattach to one dashboard.

Training uses the author's 0.5 value-gradient clip and a seeded shuffled
subject order. Strict deterministic-algorithm mode is scoped to validation;
enabling it across backward would crash in the unchanged model because
`grid_sampler_3d_backward_cuda` has no deterministic implementation in the
active PyTorch version.

At validation boundaries, disk rows and W&B metrics are written before the
atomic checkpoint is published. Therefore the checkpoint is the commit marker:
a failure before publication resumes from the preceding checkpoint, rewinds
W&B to it, truncates disk rows to it, and regenerates the whole interval.

Evaluate and explicitly prove repeatability:

```bash
PYTHONPATH=baselines/fcsvr_cardiac:training:. \
python baselines/fcsvr_cardiac/evaluate.py \
  /gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/fcsvr_cardiac_stage1_cmrx24_ed/checkpoint_last.pt \
  --verify-determinism
```

## Tests and invariants

```bash
PYTHONPATH=training:. pytest -q baselines/fcsvr_cardiac/tests
```

The tests cover:

- exact identity rendering at zero motion;
- positive through-plane motion landing on the positive output plane;
- respiratory corruption followed by oracle positive-motion splatting reducing
  reconstruction MSE relative to zero-motion splatting;
- external `(x,y,z)` versus physical `(D,H,W)` channel order;
- 3-mm slab voxel to millimetre conversion;
- preservation of dense per-pixel H/W flow without spatial averaging;
- `D=6`, `D=8`, and `D=10` padding metadata, zero masks, and shapes;
- removal of a global translation by compensation while relative slice error is
  unchanged;
- exact equality between wrapper-reported coarse metrics and the unchanged
  released L22/L21 functions;
- paper-vector versus released-component MSE scaling;
- shuffled subject-order resume, 0.5 gradient clipping, persisted W&B IDs, and
  strict validation-determinism scoping.

The shipped `original_source_manifest.sha256` captures every non-generated,
non-`.git` file under `baselines/fc_svr`. Verify the untouched source at any
time from the repository root:

```bash
sha256sum -c baselines/fcsvr_cardiac/original_source_manifest.sha256
```

The original directory must never be edited or used as an output directory.

## Deliberately deferred

- Random same-cardiac-phase training beyond ED.
- Mixed cardiac phases or target-phase inference.
- Variable-pitch pooled cohorts.
- FC-SVR Stage 2 and any learned intensity interpolation.
- Deployment-time global-pose recovery without GT motion.
- Any changes to VGGT code, VGGT configuration, splits, cached preprocessing,
  source MRI, or derived data.

**TL;DR:** This isolated fork preserves released FC-SVR Stage 1 and wraps it in a fixed-ED CMR24 respiratory adapter, temporary coarse slab geometry, GT-pose compensation, and one native-grid VGGT splat. All VGGT code and data remain read-only.
