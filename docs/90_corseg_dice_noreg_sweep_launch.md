# CorSeg-Dice + no-regularization sweep: port, fixes, launch, and evaluation protocol

> **TL;DR & takeaway** — To attack the EF/stroke-volume deficit (SV ratio 0.750, EF bias
> −10.4 points at preserved per-patient r=0.816 — the deficit is purely in the time-varying
> component, docs/86 + the 2026-08-21 investigation), the CorSeg soft-Dice loss was ported
> from the stale `vggt-arm-corseg` sibling tree onto current main (worktree `vggt-dice`,
> branch `arm/corseg-dice`), **three real bugs were found by adversarial review and fixed**
> (zero-coverage-plane NaN in backward, ~0.44 px label/image grid misalignment, in-place
> clamp in the backward hook), clamp-saturation logging was added, and **four 300-epoch arms
> were launched 2026-08-21** (jobs 58421972–75, spgpu2/L40S): `corseg_weight ∈
> {0.1, 0.5, 2.0}` plus a no-regularization arm (`diffusion_weight=0`). Everything else is
> byte-identical to the awrobewn baseline (213340611 augaggr224), which is the comparison
> anchor. Win rule: SV-ratio/EF-bias improve AND per-patient r holds, no motion-PSNR/hole
> regression; decisive checkpoints must be re-scored with nnU-Net Task114, never CorSeg.

## 1. Why this experiment

The 2026-08-21 EF investigation (handoff `_agent/handoff-2026-08-21-0241.md`) established:

- The reference arm reconstructs static/DC geometry essentially perfectly (LV-mass ratio
  0.996, (EDV+ESV)/2 midpoint bias −0.03 ml, r=0.974) while EDV is −10.9 ml and ESV is
  +10.9 ml — antisymmetric. The entire deficit is contraction/relaxation **amplitude**
  (SV ratio pred/GT = 0.750, uniform 0.65–0.83 across all 7 eval datasets).
- Intensity-L1 is structurally near-blind to this failure: a 1-px cavity-boundary error is
  ~125 wrong pixels diluted into a 65 536-px plane mean, yet it is ~10% of cavity volume
  (~10 EF points). A region/boundary loss (soft Dice against per-phase GT segmentation)
  penalizes exactly the quantity that is wrong.
- The prior "Dice already failed" belief (docs/71) is **not established**: those arms were
  read at ~41k steps while docs/71's own control did not amplitude-bootstrap until ~43.9k,
  and the weights tried (0.002, 0.1 — verified from the runs' `run_meta.jsonl`) were likely
  too small (heart-L1 needed 0.5–2.0 to move anything). docs/71 §5 explicitly defers the
  CorSeg verdict. Verified from those runs' `metrics.jsonl`: zero NaN/Inf in ~11.5k logged
  points each through ~58k steps, so the NaN bug fixed below did NOT corrupt them either —
  they are inconclusive purely for the early-read/low-weight reasons.
- The no-reg arm is the user's standing untested hypothesis: the L2 diffusion term is only
  0.26% of the objective (measured, see the `diffusion_weight` comment in `default.yaml`),
  but a trained arm is different evidence than loss-magnitude arithmetic — and it tests the
  complementary question ("does the smoothness prior *suppress* contraction?") to the dice
  arms' ("does a boundary term *add* contraction pressure?").

## 2. What was ported (worktree `vggt-dice`, branch `arm/corseg-dice`)

The implementation was grafted hunk-by-hunk from `/home/minsukc/vggt-arm-corseg` (a plain
directory copy, NOT a git branch — main had diverged 83–245 lines/file since it was forked,
including the docs/73 native-render splat, so whole-file copies were not an option):

- **`training/corseg_dice.py`** (new) — frozen CorSeg-CineSAX (MedNeXt-L, 2D per-slice)
  segments every z-slice of V_canon; volume-level soft Dice (classes 1..3, CorSeg label
  space) against `heart_seg_canonical[..., t_target]`. Differentiable, branchless (no GPU
  sync under compile), fp32 island under autocast, chunked+checkpointed forward (peak 6.9 GB
  at D=21, measured), per-voxel grad clamp ±5e-6 on the Dice branch only (the job-57023101
  dead-ReLU post-mortem: Dice's boundary-spike gradient reached ~16 000× the L1's per-voxel
  max; the clamp caps the ~1% spike tail at ~3.5× L1 scale and is deliberately in absolute
  post-weight units so raising `corseg_weight` saturates voxels at the cap instead of
  re-scaling spikes). GT→CorSeg label remap LUT (GT 1=LV_cav/2=LV_myo/3=RV → CorSeg
  1=LV_myo/2=LV_cav/3=RV); absent classes contribute Dice=1 with no gradient.
- **`training/data/datasets/mri_dataset.py`** — loads `heart_seg_canonical.nii.gz`, slices
  at the sample's `t_target`, emits `batch["heart_seg_t"]` (same (D,H,W) splat-order
  transpose and warn-and-skip shape policy as `heart_roi_canonical`).
- **`training/data/gpu_aug.py`** — `heart_seg_t` rides the same affine as
  `phases`/`content_mask` (nearest interp) at all 3 sites (keys/mode_dict, aug_dict insert,
  uint8 writeback); the exception-fallback, aug-off, val, and respiratory paths all leave it
  consistent with `gt_target_volume` (respiratory touches only input slices, by design).
- **`training/loss.py`** — `corseg_weight` kwarg, guarded `loss_corseg` block
  (raise-not-skip on a missing key, same policy as heart-L1), objective sum, returned dict.
- **`training/config/default.yaml`** — `corseg_weight: 0.0` knob next to `heart_weight`;
  `loss_corseg` + `corseg_sat_frac` added to the scalar log key lists.
- **`training/config/exp_corseg.yaml`** (new) — documents the awrobewn recipe replication
  (img 224 / aggressive / 300 ep) + the corseg knob. NOTE: the launched jobs use the sbatch
  scripts' explicit override strings (the repo's requeue-safe convention), not this file;
  it is the composable-config record of the same recipe.

Nothing else was copied from the corseg tree (it is stale; e.g. it flips `heart_weight` to
0 — these arms deliberately KEEP `heart_weight: 0.5` to match awrobewn, so the dice term is
tested as an *addition* to the shipped recipe, not a replacement).

## 3. Bugs found and fixed before launch (prove-it review, 2 reviewers + GPU repros)

All three were latent in the sibling tree's implementation (i.e. also present in the old
docs/71 runs):

1. **[critical] NaN gradient from any zero-coverage z-plane** — `_zscore_nonzero`'s
   `sd = var.sqrt()`: at `var==0` (an exactly-constant or all-zero slice, e.g. a
   zero-coverage plane in early-training V_canon) `sqrt` backward emits NaN even though the
   forward `torch.where`s route around it, and the grad-clamp hook passes NaN through.
   Reproduced on GPU: one zeroed plane → 40 401 NaNs in the gradient while the loss value
   stays finite (silent — the trainer's NaN-guard would skip backwards without any NaN ever
   appearing in the logged metrics). Fix: `sd = (var + 1e-12).sqrt()`.
2. **[medium] GT labels ~0.44 px off the image grid** — the label path used legacy
   `F.interpolate(mode="nearest")` (floor convention) while the image path uses
   `bilinear, align_corners=False` (half-pixel centers): measured mean −0.444 px (up to
   −0.94 px ≈ 0.55–1.2 mm at 1.25 mm) systematic directional shift of the Dice boundary
   target relative to what the segmenter sees. Fix: `mode="nearest-exact"` (0.002 px).
   Still exact-label nearest-neighbor — no interpolation artifacts, labels stay integral.
3. **[low] in-place `clamp_` in the backward hook** violates the autograd hook contract
   (hooks must not mutate their grad argument). Works today (single-consumer grad) but UB.
   Fix: non-inplace `clamp`.

Verification: full 363-test suite green; end-to-end GPU smoke on a real subject
(get_data → aug co-warp → loss → gradient: loss(GT)=0.057 vs loss(zeros)=0.50, grad finite,
clamp saturating exactly at 5e-6); bf16-autocast + no-grad val paths; absent-class case;
batchaug shared-param-draw proven (seg fed as a copy of content_mask came out bit-identical
over 5 draws); **all 1350 pooled-split subjects have a T=12 `heart_seg_canonical` on
exactly the ROI's grid (0 missing/mismatched)** so the raise-on-missing cannot fire.

## 4. Saturation logging (new diagnostic)

`corseg_sat_frac` = fraction of Dice-branch grad elements pinned at ±`grad_clamp` in the
most recent backward (computed inside the clamp hook, module-level buffer, ONE-STEP LAG,
train-only — val never backwards, and the key is absent when `corseg_weight=0`). Purpose:
`corseg_weight` and `grad_clamp` are different knobs — weight scales the typical (median)
gradient, the clamp caps the spike tail and is intentionally weight-independent (that is
the dead-ReLU protection). When most elements saturate, raising weight no longer
strengthens the term (the gradient degenerates to a constant-magnitude push) — the correct
next lever is then `grad_clamp` (e.g. 3.5×→10–20× L1 per-voxel scale ≈ 1.5e-5–3e-5), as a
deliberate separate step. There is no external "common value" for this clamp; it is
calibrated relative to this pipeline's L1 per-voxel gradient max (~1.4e-6).

## 5. What was launched (2026-08-21)

| job | arm | tag | ONLY override on top of the awrobewn recipe |
|---|---|---|---|
| 58421972 | dice 0.1 | `corseg01` | `loss.volume.corseg_weight=0.1` |
| 58421973 | dice 0.5 | `corseg05` | `loss.volume.corseg_weight=0.5` |
| 58421974 | dice 2.0 | `corseg20` | `loss.volume.corseg_weight=2.0` |
| 58421975 | no-reg | `noreg224` | `loss.volume.diffusion_weight=0.0` |

- Scripts: `sbatch/train_corseg_{w01,w05,w20,noreg}.sh` — byte-identical siblings modeled
  on `train_pooled1337_dpt_augaggressive_224.sh` (the awrobewn launcher), differing only in
  `ARM_OVERRIDES`/`VARIANT_TAG`; they `cd /home/minsukc/vggt-dice` (the worktree — the port
  exists only on `arm/corseg-dice`), spgpu2/L40S/`jjparkcv_owned1`, requeue + wandb-resume
  logic carried over. Exp dirs land in the shared `scratch/logs/` as
  `<rev_ts>_<tag>_pooled1337`.
- Shared recipe (verbatim awrobewn): pooled1337 cohort, img_size 224, aggressive aug tier,
  respiratory ON, 300 epochs, peak LR 5e-5 (the three-knob schedule), gradient
  checkpointing off, heart_weight 0.5, gather 0.5, tv 0, diffusion 1000 (except noreg),
  fresh-from-base VGGT-1B, aggft freeze.
- **Baseline for comparison = awrobewn itself**
  (`scratch/logs/213340611_mri_volume_dpt_augaggr224_dynamic_axial_pooled1337`). A same-code
  w=0 control was considered and dropped: the 11 training/ commits since awrobewn's rev
  `a031ba7` are dinov3 plumbing/perf/config only, and the one dinov2-224-relevant commit
  (`bfca5ef`) *restores* the exact pre-merge behavior awrobewn ran with. Residual caveat:
  awrobewn's tree was dirty at launch and perf commits can shift numerics at noise level —
  if any arm ends within noise of awrobewn, run the w=0 control THEN before concluding.
- The no-reg arm's objective is exactly: full-volume L1 + heart-L1(0.5) + gather(0.5).
  Gather is a second data term, not a regularizer — it stays. TV was already 0.

## 6. How to evaluate (checklist, in the order it becomes readable)

Early, from each run's `log_dir` (`tools/load_run.py`; never wandb — docs/60):

1. **Health**: `train/loss/corseg` nonzero, finite, decreasing; no dead-ReLU signature
   (`grad_aggregator`→0 with frozen `loss_corseg`); no NaN streak. For noreg:
   `loss_diffusion == 0` exactly.
2. **Saturation**: `corseg_sat_frac` per arm — expect it to rise with weight. w=2.0 near
   1.0 ⇒ weight exhausted, next experiment raises `grad_clamp` instead (see §4).
3. **Amplitude-transition timing** (the question docs/71 never answered): does dice
   accelerate/deepen the amplitude bootstrap vs awrobewn's own curve at matched steps?
   Reference points (docs/71): unweighted control ~43.9k steps, heart-L1 0.5 ~8.9k. All
   four arms include heart-L1 0.5, so awrobewn's curve is the like-for-like reference.
   Do NOT judge any arm before ~44–50k steps — that mistake is what invalidated docs/71's
   dice reading.

Final, all arms + awrobewn under ONE protocol:

4. **Ship metrics** (docs/86 EF chain, `evaluation/src/score/ef_dice.py`): SV ratio
   (baseline 0.750), EF bias (−10.4), per-patient Pearson r (0.816). **Win rule: SV-ratio
   and bias improve AND r holds.** A bias improvement with degraded r is the trade the
   α-sweep showed is available for free (post-hoc `alpha=1.3` gives raw EF MAE 8.77→6.13
   while calibrated MAE worsens 4.78→5.87 and r drops 0.922→0.872) — an arm must beat that
   free trade-off frontier to count.
5. **Collateral-damage veto** (docs/38): motion PSNR and hole_frac must not regress. Dice
   boundary pressure could tear coverage holes (the stop-grad-denominator failure mode).
   recov_frac↑ & psnr_motion↑ WITHOUT hole_frac↑.
6. **Independent re-scoring is MANDATORY**: the in-run val EF/amp metrics run through
   CorSeg (`ef_seg_backend: "corseg"`), which the dice arms now train against — training
   signal and verdict must be decoupled. Decisive checkpoints get re-scored with nnU-Net
   Task114 via the `evaluation/` chain. In-run CorSeg numbers are steering signals only.
7. **No-reg arm additionally**: whether removing diffusion alone moves SV ratio, plus a
   qualitative look at DVF roughness / the V_canon panels (roughness is what the
   regularizer was suppressing — docs/44/46 ViT-patch ripple).

## 7. Provenance / repro

- Branch `arm/corseg-dice` in worktree `/home/minsukc/vggt-dice`, commits `5f27855` (port
  + fixes + sat logging + exp config) and the sbatch commit after it. Worktree has the 6
  data/eval symlinks recreated by hand (worktrees don't carry gitignored symlinks).
- Adversarial review: 2 reviewers (loss numerics; integration/data-flow) + GPU
  refutation/repro of every finding; probes in the session scratchpad (throwaway).
- Old-arm forensics: `scratch/logs/213515736_mri_volume_corsegdice_w{002,100}_*` —
  weights 0.002/0.1 from `run_meta.jsonl`, zero NaN in `metrics.jsonl`.
- EF-deficit numbers: `evaluation/metric_results/_ef/*.json` (docs/86 sweep) + the
  2026-08-21 handoff's midpoint/mass analysis.
