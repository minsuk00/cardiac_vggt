# v2 — Optional 3D UNet refiner on the splat

**Date:** 2026-06-16
**Config:** `training/config/mri_volume.yaml` (refiner OFF by default)
**Status:** Implemented, unit-tested (163 tests green), 3 smoke modes verified (OFF / ON-joint / ON-frozen, all RC=0). Not yet run to convergence on real data — two sbatch scripts prepared, not launched.

## Pipeline state this version builds on (everything that changed since v1)

v1 (2026-05-08) was the first splat-based unsupervised pipeline. A lot changed between v1 and this
refiner version; the refiner sits on top of all of it. Summary (detail in the linked docs):

- **Canonical-grid refactor (2026-05-24).** Every subject is resampled to ONE fixed canonical cube —
  `(1.4, 1.4, 8.0)` mm, `(256, 256, 12)` voxels — so canonical voxels have the same physical size for
  every subject (v1 used per-subject normalization). Input slices AND `V_gt` both live in this cube.
  The old supervised-DVF path is **fully removed** (`MRIDataset` no longer reads DVF/mask NIfTIs).
  → `CLAUDE.md` "CMR data notes" / "Canonical-grid refactor".
- **Multi-phase targets.** Each sample picks a target cardiac phase `t_target ∈ {0..11}`; the loss
  compares the splatted `V_canon` against the on-disk NIfTI **at that target phase** (`V_gt =
  phases[t_target]`), not just ED. Discrete-only (continuous-phase query not implemented).
- **Pose embedders** replace the camera token: `z_embedder` (linear depth), `t_embedder` (cyclic INPUT
  cardiac phase), `target_t_embedder` (the query phase). `use_z/use_t/use_target_t_pose_embedding`
  toggle them independently.
- **Respiratory (breathing) motion simulation** (`training/data/respiratory.py`, default OFF). Per
  input slice, a deform-then-reslice SI+AP rigid shift (Lujan `sin^{2n}` waveform, ~8–24 mm, randomized
  SI direction ±30°), applied AFTER affine aug and overwriting **only the input slices** — the target /
  `scanner_coords` / `V_gt` stay at the unshifted end-expiration reference, so the model learns to
  **correct** breathing, blind to the respiratory phase `r`. Train draws iid per epoch (private RNG);
  val draws **deterministically per `seq_index`** (so val measures the real corrupted→clean task).
  → `docs/01_respiratory_motion_simulation.md`, `_html/06`.
- **In-plane affine + photometric GPU augmentation** (`training/data/gpu_aug.py`, `batchaug`, default
  OFF; tiers conservative/moderate/aggressive) — one affine per subject across all phases.
- **Aggregator-finetune ("aggft").** `frozen_module_names=[*patch_embed*]` (freeze only DINOv2
  patch_embed) → the aggregator + point_head train (~637M / 941M, ~27 GB/A40), vs the head-only default
  (freeze the whole aggregator). Needs `find_unused_parameters=true`.
- **The 5-variant study (`docs/05`, `_html/07`, `_html/08`).** Trained resp × input-`t` × aggressive-aug
  and re-evaluated under one protocol. Findings: **breathing simulation is decisive** (models never
  trained on breathing sit at the do-nothing identity floor); **input-`t` is unnecessary** (drop it);
  best recipe = **resp, z-only, aggft** — this is the **t59w6nqy** model that seeds the frozen refiner
  run. Report 08 then showed the residual error is ~75% **splat blur** — which is *why* this refiner exists.

The breathing sim and the t59w6nqy seed are therefore part of "the current setup"; this doc's refiner is
the next increment on top.

## What this version does

Adds an **optional, default-OFF** 3D UNet that refines the splatted reconstruction
`V_canon → V_refined`, to recover the high-frequency detail the splat smooths away. Motivation:
the failure-mode analysis (`_html/08_breathing_failure_mode.html`) showed **~75% of the breathing
reconstruction blur is the splat renderer itself** (the trained model's sharpness ≈ the raw-splat
sharpness; the point head fixes *position* but cannot add detail the coverage-averaging discards).

- **When OFF (`enable_refiner=false`, the default): the pipeline is BITWISE IDENTICAL to v1** — same
  loss, same outputs, same logging (verified: an OFF smoke run has zero "refiner" mentions and the
  same `Grad/aggregator`+`Grad/point` console as before).
- **When ON:** `VGGT.forward` splats (`V_canon, coverage`), runs the refiner (`V_refined`), and the
  loss applies a **deep-supervised two-term L1**:
  `L = L_pre + λ·L_post = |V_canon − V_gt| + λ·|V_refined − V_gt|` (λ=`refiner_lambda`, default 1.0).
  L_pre keeps the point head's geometry directly supervised (so the refiner can't make it lazy);
  L_post trains the refiner. Image loss stays **L1 for now** (TODO: add SSIM/gradient/perceptual —
  L1 is mean-seeking and caps sharpness).
- Refiner input is **`[V_canon, coverage]`** (2 channels) — coverage tells the net where data is
  trustworthy vs under-covered, curbing hallucination in sparse regions.

## Pipeline (end to end, refiner ON)

```
images + scanner_coords → Aggregator → Point head → world_points = scanner_coords + Δ
        │  (all as v1)
        ▼
   splat_predictions(world_points, images, grid) → V_canon, coverage   [INSIDE VGGT.forward now]
        │
        ▼
   VolumeRefiner([V_canon, coverage]) → V_refined = V_canon + Δ_refine
        │   predictions += {V_canon, coverage, V_refined}
        ▼
   loss: L_pre = |V_canon − V_gt|  (keeps point head honest)
         L_post = λ·|V_refined − V_gt|  (trains refiner)
         objective += (L_pre + L_post) · volume.weight
        ▼
   backward → refiner params AND (via V_canon→splat→world_points) the point head/aggregator
```

The splat moved from the loss into `VGGT.forward` (gated on `enable_refiner`) so the refiner's params
are used inside the **DDP-wrapped forward** (DDP-safe). The OFF path still splats in the loss via the
SAME `splat_predictions` helper → byte-identical `V_canon`.

## Architecture (`vggt/models/refiner.py`, ~0.35M params)

Residual, **anisotropic** 3D UNet (wolny/pytorch-3dunet `DoubleConv` style):
- **DoubleConv** = `(Conv3d 3×3×3 → GroupNorm → GELU) × 2`. GroupNorm (not BatchNorm) because the
  volume batch is **B=1** — BatchNorm stats would be meaningless.
- **Anisotropic pooling**: `MaxPool3d((1,2,2))` pools H/W only — **D=12 is preserved** (12→12→12;
  H/W 256→128→64). Z is 8 mm (coarse, 12 planes) and must not be downsampled. Upsample via
  `F.interpolate(size=skip.shape[2:])` (handles any size, no checkerboard).
- 2 levels, channels [16, 32, 64], standard UNet skips.
- **Output `Conv3d(16,1,1)` zero-initialized** → `Δ_refine = 0` at init → `V_refined = V_canon`
  (starts as the identity; gentle warm-up, no early disruption to a good V_canon).
- **fp32**: runs under `autocast(enabled=False)` so `V_refined` matches `V_gt` for the L1/PSNR and the
  small 3D convs stay stable (V_canon from the splat is already fp32).

## Files changed / created

| File | Status | Purpose |
|---|---|---|
| `vggt/models/refiner.py` | **NEW** | `VolumeRefiner` (anisotropic residual 3D UNet) + `DoubleConv` |
| `vggt/utils/splat.py` | MODIFIED | Added `splat_predictions(predictions, batch, grid_shape)` — the splat-prep extracted verbatim from the loss so forward + loss share ONE path (byte-identical V_canon) |
| `vggt/models/vggt.py` | MODIFIED | `__init__` kwargs (`enable_refiner`, `grid_shape`, `refiner_*`); forward splats+refines when `self.refiner is not None`, adds `V_canon/coverage/V_refined` to predictions |
| `training/loss.py` | MODIFIED | `compute_volume_intensity_loss` consumes model-provided `V_canon`/`V_refined`; adds `loss_refiner = λ·L1(V_refined, V_gt)` + `metric_psnr_3d_*_refined`; `MultitaskLoss` adds the refiner term to the objective (only when present) |
| `training/trainer.py` | MODIFIED | `_log_refiner_viz_to_wandb` (`refiner_viz/` per-z panel); refiner cardiac gif; per-phase `val_psnr_{bbox,motion}_refined`; direct `Train_Loss/*_refined` scalars; `Grad/refiner` meter (refiner-only `has_params` guard) |
| `training/config/mri_finetune.yaml` | MODIFIED | model passthroughs; top-level `enable_refiner: false` / `refiner_use_coverage: true`; **`gradient_clip` `refiner` group** (the fix — see Bugs) |
| `training/config/mri_volume.yaml` | MODIFIED | `loss.volume.refiner_lambda: 1.0` |
| `tools/make_weights_only_ckpt.py` | **NEW** | strip a checkpoint to `{"model": ...}` (frozen-run seed) |
| `sbatch/train_refiner_frozen.sh` | **NEW** | run A: freeze all VGGT, train only refiner, seed from t59w6nqy weights-only |
| `sbatch/train_refiner_joint.sh` | **NEW** | run B: aggft + refiner from VGGT-1B base |
| `tests/test_refiner.py` | **NEW** | 13 tests (OFF bitwise, splat==helper, residual identity, two-term λ, freeze isolation, …) |
| `tests/conftest.py` | MODIFIED | register `slow` marker |

## What is logged to WandB (ADDITIVE — only when refiner ON; nothing new when OFF)

New keys: `Train_Loss/{loss_refiner, metric_psnr_3d_{full,bbox,motion}_refined}`;
`val_psnr_bbox_refined/{t*, mean*}` and `val_motion_refined/{t*, mean*}` (same identity baselines as
their V_canon counterparts, so V_canon vs V_refined are directly comparable);
`refiner_viz/{Train_Visuals,Val_Visuals_subj0,Val_Visuals_subj7}_Volume` (per-z V_gt/V_canon/V_refined/diff);
`refiner_viz/cardiac_cycle_gif` (V_gt | V_refined beating heart); `Grad/refiner` + `Train_Optim/grad_refiner`.
**No existing panel is replaced or altered.**

## The two training runs (prepared, not launched)

| | seed | trainable | freeze | purpose |
|---|---|---|---|---|
| **A frozen** (`train_refiner_frozen.sh`) | t59w6nqy weights-only (resp, z, no-t, aggft) | **only refiner** | `[*patch_embed*,*camera_token*,*aggregator*,*point_head*]` | isolate the pure splat-deblur gain (geometry fixed) |
| **B joint** (`train_refiner_joint.sh`) | VGGT-1B base | aggregator+point_head+refiner | `[*patch_embed*]` | let geometry co-adapt with the refiner |

Both: breathing ON, z+target_t on, no input-t, `find_unused_parameters=true`, λ=1, max_epochs=200.

## Bugs found during verification (and fixes)

1. **Gradient-clip crash (caught by the ON smoke, NOT static review).** `GradientClipper.setup_clipping`
   *requires every trainable param to match a configured `module_name`*; the new `refiner.*` params
   matched none → `ValueError: Some parameters are not configured for gradient clipping`. **Fix:** added
   a `refiner` group to `optim.gradient_clip.configs` in `mri_finetune.yaml`. Empty when OFF ⇒ skipped
   by the clipper (`gradient_clip.py` empty-group `continue`) ⇒ no effect on the OFF path.
2. **OFF console regression (caught by an adversarial review agent).** My first fix created `Grad/`
   meters only for clip groups with trainable params — but the **aggregator** group is *also*
   always-empty in mri runs (fully frozen) yet its `Grad/aggregator: 0.0000` meter was historically
   created+displayed. The broad guard silently dropped that console column. **Fix:** narrowed the guard
   to skip **only** the `refiner` group when empty; `Grad/aggregator`/`Grad/point` are created as before.
3. **Spurious OFF metrics (caught early).** Initially the refiner scalars were added to the val
   `scalar_keys_to_log` allowlist → unupdated AverageMeters logged `0.0` when OFF. **Fix:** removed them
   from the allowlist; refiner train scalars are logged directly via `_log_scalar` only when present,
   and val is covered by the per-phase `*_refined` panels (empty ⇒ skipped when OFF).

## How to run

```bash
# Train (cluster) — prepare-only; submit when ready:
bash sbatch/train_refiner_frozen.sh   # run A (only refiner trains)
bash sbatch/train_refiner_joint.sh    # run B (joint)

# Local override (any existing run can flip it on):
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py --config mri_volume \
  enable_refiner=true refiner_use_coverage=true distributed.find_unused_parameters=true
```

## Verification status

- 163/163 unit tests green (`tests/`), incl. `test_refiner.py` (OFF bitwise, splat==helper byte-identical,
  residual-identity-at-init, two-term λ scaling, freeze isolation for both modes).
- Smoke (offline, real data, A40): **OFF** byte-identical (`Grad/aggregator`+`Grad/point`, 0 refiner
  mentions); **ON-joint** trains (Grad/refiner=0.41, two-term objective, `val_psnr_bbox_refined` logs);
  **ON-frozen** trains only the refiner (Grad/refiner only). All RC=0, no DDP errors.
- 5 review subagents (two rounds) — confirmed correct; the 2nd round caught bug #2 above, since fixed.

## Notes for the next agent — how to evaluate after the runs finish

The headline question: **does the refiner deblur V_canon (recover detail) without hallucinating?**

1. **Primary signal — refiner beats the raw splat, same run:** compare `val_psnr_bbox_refined/mean` vs
   `val_psnr_bbox/mean` and `val_motion_refined/mean` vs `val_motion/mean`. **V_refined should be HIGHER
   than V_canon.** If refined ≤ V_canon, the refiner isn't helping (or λ too low / under-trained).
2. **Reference numbers** (breathing val, the deployment task; from `docs/05` + `_html/07,08`):
   - identity floor: bbox **23.23**, motion **16.59** dB.
   - the seed model v2 (resp, no refiner, epoch 59): bbox **26.74**, motion **19.28** (this is V_canon).
   - So **target: `val_psnr_bbox_refined` > ~26.7** for frozen mode (pure deblur on a fixed geometry);
     joint mode may lift both V_canon and V_refined.
3. **Sharpness** — the real point. Baseline (report 08): V_canon is **0.65× GT** sharpness (breathing),
   0.74× (clean). Extend `tools/measure_sharpness.py` to also measure `V_refined` (the model now returns
   it) and check the ratio rises toward 1.0. A higher PSNR *and* higher sharpness = genuine deblur.
4. **Hallucination check (critical):** clean ≠ correct. Evaluate on **held-out / val** subjects (not
   train) and confirm `val_psnr_*_refined` actually improves — if V_refined only *looks* crisp but PSNR
   doesn't rise, it's inventing detail. Eyeball `refiner_viz/Val_Visuals_subj{0,7}_Volume` (V_refined row
   vs V_gt) for fabricated structure, especially in the ~5% under-covered regions (report 08).
5. **Frozen vs joint:** frozen isolates splat-deblur (geometry fixed); joint lets the point head
   co-adapt. Compare the V_refined gains; if joint's V_canon *also* improves, the deep supervision is
   helping geometry too.
6. **Sanity:** `Grad/refiner` should be non-zero and finite; `loss_refiner` should fall over training;
   `loss_volume` (L_pre) should NOT degrade (the point head stays supervised).

Pointers: `_html/09_unet_refiner.html` (this version, human-facing + the eval guide), `_html/08`
(why the refiner — the blur decomposition), `_html/07` + `docs/05` (the 5-variant results that produced
the t59w6nqy seed).

## Known limitations / future steps

- **L1 only** — add a gradient/SSIM term to the refiner loss to actually push high frequencies (L1 is
  mean-seeking). Then optionally a light perceptual/adversarial term (raises hallucination risk).
- **Refiner vs learned decoder** — the refiner deblurs the splat output; a higher-ceiling alternative is
  replacing the splat with a learned decoder (regress V from features). Run head-to-head.
- **Coverage-conditioned behavior** — verify the refiner is actually using the coverage channel (ablate
  `refiner_use_coverage=false`).
- **enable_refiner=true requires enable_point=true** (the refiner runs in the point-head branch). Not a
  concern for mri configs (point always on); add an assert if other configs use the refiner.
