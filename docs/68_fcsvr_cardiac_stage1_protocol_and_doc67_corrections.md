# 68 — FC-SVR Cardiac Stage-1 Protocol and Corrections to Doc 67

> **TL;DR & takeaway** (2026-08-09). The implemented baseline is **FC-SVR
> Stage 1, GT-pose-normalized**, an explicitly adapted cardiac baseline—not full
> FC-SVR. The released `Flow_SNet3d0_1024`, SVD loss, compensation algebra, and
> `upsample_flow()` are unchanged. A real CMR24 L40S forward/backward,
> checkpoint/resume, and exactly repeated validation passed. The authoritative,
> detailed protocol and complete deviation inventory are in
> `baselines/fcsvr_cardiac/readme.md`; this document records the final decision
> and corrects overclaims in doc 67.

## Final implementation boundary

The untouched release remains at `baselines/fc_svr`. The cardiac fork is
`baselines/fcsvr_cardiac`. The following author operations are immutable:

- `models/flow_SNet4.py`, including `Flow_SNet3d0_1024` and dormant `project()`;
- `models/flow_UNetS.py`;
- `models/losses.py`, including `l22_loss_affine_invariant`;
- released `compensate()` algebra and `upsample_flow()`.

`baselines/fcsvr_cardiac/original_source_manifest.sha256` verifies every
non-generated source file in `baselines/fc_svr`. Cardiac-specific behavior lives
only in data/geometry adapters, entrypoints, logging, and tests. VGGT source and
data are read-only dependencies; no VGGT code, configuration, split, cache, or
MRI file was modified.

## Implemented protocol

- CMRxRecon2024, fixed ED, `cmrx24only.txt`: 235 train / 29 validation / 30 test.
- Native CMR stack `(D,256,256)` at `(12,1.4,1.4)` mm.
- One current-VGGT respiratory translation vector per native z plane.
- Foreground-masked cardiac intensity and acquired-frame heart ROI loss mask.
- Repeated slab `(4D,120,120)` at approximately 3 mm, then coarse Stage-1
  `(2D,60,60)` at approximately 6 mm.
- Temporary symmetric depth padding to 16 only when `2D<16`; padding mask is
  zero and padding is removed before released upsampling.
- Dense H/W flow is preserved. Only each physical slice's four repeated slab
  copies are averaged at the same H/W coordinate.
- Original respiratory-corrupted 256-pixel slices are foreground-splatted once
  onto the identical native GT grid with `vggt.utils.splat.splat_to_volume`.
- Raw output is diagnostic. The official adapted result is compensated against
  known simulated target motion and labeled **FC-SVR Stage 1,
  GT-pose-normalized**.

Training is batch 1, float32, Adam `1e-4`, polynomial power 0.9, 256,000 steps,
released 0.5 value-gradient clipping, and seeded per-epoch subject shuffling.
W&B uses a persistent 16-character run ID and rewinds server history to the
loaded checkpoint step before logging regenerated steps. An atomic full
model/Adam/step checkpoint is saved every 5,000 steps and at completion. Large artifacts default
to `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/vggt/logs/`; GPFS
checkpoints are staged to node-local `/tmp` before loading.

## Evaluation definitions

Both raw and compensated rows report:

- exact released coarse `l22_loss_affine_invariant(..., eps=0)` and
  `l21_loss_affine_invariant(..., eps=0)`;
- adapted native-grid component-MSE, matching the released L22 normalization;
- adapted native-grid vector-MSE, matching Appendix A.2's literal
  `N^-1 ||u-y||_F²` definition and therefore 3× component-MSE;
- EPE, volume MAE/MSE/PSNR, bbox MAE/MSE/PSNR, and heart-ROI MAE/MSE/PSNR.

Strict deterministic CUDA algorithms are enabled during no-gradient validation
only. They cannot be enabled for training because PyTorch 2.13 has no
deterministic CUDA backward for the unchanged model's 3D `grid_sample`.

## Corrections to doc 67

Doc 67 is the pre-implementation architecture analysis. These statements must
not be carried into the methods/results text without the corrections below:

1. **Not full FC-SVR.** Stage 2 is omitted. Native Stage-1 splats cannot be
   called the paper's final reconstructed output.
2. **`project()` is dormant.** The released model has `rigid=False`; the cardiac
   baseline does not activate the optional per-slice projection.
3. **No core precision patch was needed or permitted.** Float32 is selected at
   the entrypoint. No SVD, warp, architecture, loss, or pooling code was edited.
4. **The implemented input is not native `(2,D,256,256)`.** It follows released
   real-stack ratio normalization through `(4D,120,120)` and
   `(2D,60,60)`.
5. **The implemented release-like batch size is 1.** The public FeTA shell uses
   batch 1 and 300,000 steps; this experiment follows the paper/approved plan's
   256,000 steps. Doc 67's 512,000-volume-pass and 2,133-epoch arithmetic does
   not describe this run.
6. **The training split is 235, not 240.** At batch 1, 256,000 steps correspond
   to about 1,089 logical passes over the 235-subject training split.
7. **The A40 2.5-hour runtime estimate was not validated for this adapter.** It
   must not be used for scheduling or reporting. Only the L40S functional smoke
   and 5.17-GiB validation peak are measured here.
8. **Full z-plane acquisition does not guarantee zero holes.** Respiratory
   displacement and foreground-only splatting can leave local/native-grid
   coverage holes even when every physical z plane is acquired once.
9. **The common VGGT splatter is a deliberate comparison adapter, not the
   released FC-SVR renderer.** This makes the final grid common but remains an
   explicit deviation.
10. **Cardiac motion differs materially from author training.** Translation-only
    respiratory samples replace smooth B-spline rotations/translations and
    two-shot interleaving; global 3D augmentation becomes applicable 2D
    augmentation; the boxcar PSF and foreground crop are omitted.
11. **Compensation is privileged.** It uses known simulated target motion, so
    compensated results are GT-pose-normalized and non-deployable.

## Verification evidence

- NVIDIA L40S, PyTorch `2.13.0+cu130`, `torch-interpol==0.2.4`.
- Real CMR24 forward/backward completed with finite loss.
- Full checkpoint size: 3.846 GB; consecutive resumes advanced step 1 → 2 → 3.
- Repeated one-subject validation was exactly equal after deterministic
  validation scoping.
- Representative cached-cohort L40S benchmark: 100 complete steps across 50
  real CMR24 subjects took 0.404 s/training step; the real-data smoke gate used
  6.07 GiB peak allocated memory.
- Runtime projection: 28.7 hours for the optimizer path; budget roughly 32–36
  hours including unmeasured validation, GPFS checkpoint, W&B, and startup
  overhead. Earlier single-subject and cold-cache extrapolations are explicitly
  rejected in the baseline README.
- Fork-local suite: 26 passed.
- Original manifest: every `baselines/fc_svr` entry reports `OK`.

## Remaining declared evaluation work

The training path is approved. Before final paper tables, add held-out test
selection, persist raw/compensated dense motion and native reconstructions, and
write aggregate provenance. The known FC-SVR/VGGT validation-seed mismatch is
temporarily accepted; pair the corruptions before a paired statistical claim.
Before launching on a 48-hour partition, add and smoke-test SLURM auto-requeue
or obtain a ≥96-hour allocation; the current Python resume path is complete but
no FC-SVR-specific signal/requeue wrapper is shipped.
