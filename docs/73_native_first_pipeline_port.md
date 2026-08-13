# Native-first pipeline port: `img_size` knob + always-native splat

> **TL;DR & takeaway** — The tracked training pipeline now (a) supports any model-input
> resolution `img_size` that is a multiple of 14 (518 default, 224 validated by the docs/72
> pilot), and (b) **always splats the original native 256² slice content**, never the
> DINO-resized copies: the slices are extracted ONCE at native resolution
> (`batch["images_splat"]`), the model input is a resample of them, and the loss resamples
> the predicted field back to 256² before splatting. Not a config knob. Verified by a
> 3-reviewer prove-it (two rounds), 347/347 tests, and GPU probes showing the render is now
> bit-invariant to `img_size`. **Series break:** ~0.1 dB PSNR at 518, and coverage-derived
> metrics (`hole_frac_heart`, `metric_coverage_*`) see ~4.1× fewer splatted points — never
> compare them across this port.

**Date:** 2026-08-13. **Motivation & pilot evidence:** docs/72 (224² pilot `efra0f3j`:
~3–6× faster steps, ~12 GB vs ~25 GB, EF slope ~1.06 at parity with 518).

## 1. The pipeline now (one step)

```
phases (T, D, 256, 256)                       [native canonical, per-subject D]
  → affine/photometric aug (whole subject)    [phases, mask, heart ROI; V_gt re-derived]
  → breathing shifts sampled once (disp)
  → extract input slices ONCE at native 256²  [resp-corrupted; = batch["images_splat"]]
  → batch["images"] = resize(native → img_size), RGB   [model input, derived view]
  → DINOv2 (img_size/14 patch grid) → alternating attention → DPT
  → world_points = scanner_coords + Δ at img_size
  → loss resamples the FIELD to 256² (bilinear, exact for the coord convention)
  → splat native intensities → V_canon (D, 256, 256)  → L1/heart-L1 vs V_gt
```

Key principle: **the render consumes measurements, the backbone consumes a view.**
Resampling the smooth field is ~free (measured −0.118 dB for 518→256, docs/72 §3);
resampling the image is lossy for any `img_size < 256`. `img_size` is therefore a pure
perception knob; the render is resolution-invariant (GPU probe: identical loss at
R=518/224/256 with the model factored out).

## 2. What changed where

- `training/data/gpu_aug.py` — `extract_slices_from_phases(..., out_size=None)`;
  `gpu_augment_batch` derives `R = batch["scanner_coords"].shape[-2]`, extracts native
  once per path (resp / affine-only / aug-off), sets `images_splat`, builds `images` via
  `_resize_to_model_res`. **On every path the batch is finalized for the loss** — see the
  updated docstring; mutating `images` afterwards no longer affects the render.
- `training/data/respiratory.py` — `extract_slices_with_respiratory_vec(..., out_size=None)`.
- `training/loss.py` — `_resize_field` + `_splat_preds_native` (the only splat path;
  falls back to the old `splat_predictions` when `images_splat` is absent). `V_id` and the
  oracle `V_or` use the same native point set + gate, so `recov_frac` compares one
  pipeline throughout.
- `training/data/datasets/mri_dataset.py` — `target_size` guard relaxed to `% 14 == 0`.
- `inference/run_cmrxrecon.py` + `training/trainer_viz.py` filmstrip — the no-breathing
  arms now call `gpu_augment_batch` with aug off so BOTH protocol arms render native.
- Tests: `tests/test_native_splat.py` (new) pins the contract (native content, resp
  corruption preserved, field-resample-equals-manual, fallback identity);
  `test_gpu_aug.py` pins single extraction; `test_respiratory_native_z.py` spy updated.

## 3. Contracts future agents must know

1. **`gpu_augment_batch` is mandatory batch finalization**, not just augmentation: it
   requires `phases`, `scanner_coords`, `timesteps`, `slice_indices` on every path and
   always writes `images_splat`. Batches built without it fall back to the model-res splat
   in the loss (the pre-port render) — fine for frozen harnesses, but numbers then sit on
   a slightly different render than train/val.
2. **Don't mutate `batch["images"]` after `gpu_augment_batch`** expecting the render to
   change — the splat reads `images_splat`. Rewrite that too, or re-call the function with
   aug off (the run_cmrxrecon clean arm shows the pattern). Known offender predating this
   rule: `tools/miitt_viz/gated_gather05_7row.py` (clean_ref mode) — its splat now uses the
   breathed slot 0; fix before reusing.
3. **Series break** (metrics, not weights): pre-port checkpoints load and run fine —
   architecture untouched. But PSNR shifts ~0.1 dB at 518, and `hole_frac_heart`'s
   absolute `coverage < 0.5` threshold + `metric_coverage_*` see ~4.1× fewer points at
   518. The docs/38 hole_frac veto must not be applied across the boundary. This also
   retroactively inflates the docs/72 pilot-vs-reference hole_frac comparison (0.26 vs
   0.11): part of that gap is threshold rescaling, not real holes.
4. **Live runs**: any run resumed/requeued from the repo after this port silently picks up
   the new render mid-series (loss changes slightly, metrics shift as above).
5. At 518 input the 518→256 field downsample leaves 8 of 518 rows/cols with only
   regularizer gradient (point-sampled bilinear) — structural, benign.
6. `img_size` values below ~224 are guarded only by `% 14`; tiny grids (14, 28…) will
   crash loudly in the DPT fusion chain. 224 and 518 are the validated points; 252 is the
   suggested intermediate (docs/72 §10.6).

## 4. Verification record

- prove-it round 1 (3 reviewers): found the flag-threading, oracle, and fallback-hole bugs
  → fixed by removing the flag (always-native) and porting `V_or` + the early return.
- prove-it round 2 (3 reviewers, final form): **zero training-path defects**; findings were
  offline-harness asymmetry + stale docs (fixed in this commit) and the metric-scale
  caveats recorded above.
- Runtime: 347/347 tests; GPU probe (A40) — 9 combos (R∈{518,224,256} × {affine+resp,
  resp-only, aug-off}), correct shapes/dtypes, finite losses, gradients through the field
  resize, and render invariance across R.
- 224 end-to-end evidence: the docs/72 pilot (equivalent logic, /tmp tree) trained ~240
  epochs with the native render from epoch 15.
