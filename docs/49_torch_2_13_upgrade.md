# 49 — Torch 2.13 / monai 1.6 / triton 3.7 upgrade

> **TL;DR & takeaway.** The `svr` env was unpinned from the frozen **torch 2.3.1** build to the
> latest stable stack — **torch 2.13.0+cu130 · torchvision 0.28.0 · torchaudio 2.11.0 · triton
> 3.7.1 · monai 1.6.0**, and subsequently **numpy 2.2.6** — to enable monai 1.6, a matched triton, and future
> `torch.compile`. It is **safe and requires no pipeline logic change** (only a docstring + an
> error-message string were touched — both cosmetic): the active training + inference pipeline
> behaves identically. Verified (parity vs the untouched baseline svr on the
> same code): `pytest` identical pass/fail on both envs; canonical preprocessing **float32
> byte-identical** across all 28 spacing tuples + 7 Z-paths (monai 1.4→1.6); every distinct config
> path (dpt+TV, dpt+diffusion, bspline, gather, one-frame, contz, dino-unfrozen, moderate-aug)
> trains with finite loss + grads; full-checkpoint resume loads under torch 2.6+'s new
> `weights_only=True` default. **`fused_ssim`** (CUDA ext) was rebuilt against torch 2.13 so the
> SSIM-3D metric + optional refiner-SSIM loss are restored. The old env is recoverable anytime from
> `envs/svr_torch231.freeze.txt` / `.env.yaml`. The `torch.cuda.amp.*` deprecation was **migrated to
> `torch.amp.*` on 2026-07-24** (71 call sites, 58 files — proven equivalent: bitwise-equal autocast
> matmul, matching `GradScaler` state).
>
> **numpy was then unpinned 1.26.4 → 2.2.6 (2026-07-24), one variable at a time.** Training is
> structurally immune and verified byte-identical (its numeric core is pure torch; RNG is stdlib/torch;
> checkpoints contain no numpy — six real ones scanned). **One real source fix was required**, in
> `inference/adapters/base.py percentile_scale`: numpy 2 returns a float32 scalar from `np.percentile`
> on float32 input, which silently killed the divide-by-zero guard for `vmin >= 32` and turned a finite
> fallback into an all-NaN cine. Fixed by widening with `float()` before the guard (zero cost). The
> accompanying ~1.9e-4 input drift is **deliberately accepted** — recovering it cost +34% time and 2.5x
> memory for a difference far below PSNR/Dice/EF noise.
>
> Re-running eval does not bit-match the numpy-1-era numbers in `evaluation/results/*.json`, but the
> difference is **not attributable to numpy**: measured end-to-end on ACDC, numpy-1-vs-numpy-2 differs
> by max 0.075 dB while **re-running the identical command in the identical env differs by max 0.252 dB**
> — the eval's own reproducibility floor is ~13× the numpy effect. (That floor is worth knowing in its
> own right: any per-subject OOD comparison below ~0.25 dB is noise.)
>
> Remaining caveats: slightly different forward numerics vs torch 2.3.1 (a fresh-series boundary, not a
> break); and `torch.Tensor.__array__` still lacks numpy 2's `copy=` kwarg, so monai emits
> DeprecationWarnings (warning-only, no extra copy).

## Why

torch 2.3.1 forced monai `<1.5` and triton 2.3.1, blocking monai 1.6 and a matched triton, and
`docs`-level `torch.compile` work was bottlenecked on 2.3.1-specific compile bugs. Upgrading unpins
all of it. `monai>=1.5` needs torch≥2.4 and monai 1.6 needs torch≥2.8, so the monai bump rides with
the torch bump. numpy was deliberately held at 1.26.x **for this step** (one variable at a time) and
migrated to 2.2.6 separately later the same day — see the numpy section below.

## Version matrix (all co-matched, verified)

| package | old | new |
|---|---|---|
| torch | 2.3.1+cu121 | **2.13.0+cu130** |
| torchvision | 0.18.1 | **0.28.0** (pins `torch==2.13.0`) |
| torchaudio | 2.5.1 (mismatched, unused) | 2.11.0 (unused) |
| triton | 2.3.1 | **3.7.1** (auto; dormant — batchaug forced to pytorch backend) |
| monai | 1.4.0 | **1.6.0** (needs torch≥2.8) |
| numpy | 1.26.4 | 1.26.4 at the torch step; **2.2.6** after the separate numpy migration (see below) |
| fused-ssim | built vs 2.3.1 | rebuilt vs 2.13 (same git commit) |

PyPI `torch==2.13.0` is the cu130 build (bundles cudnn-cu13); the A40 driver 595.71 supports CUDA
13. The cu12 nvidia stack + an orphaned `cupy-cuda12x` are purged; the cu13 closure is exact.

## How it was verified (parity framing)

Because a DDP-removal refactor was in flight, the bar was **parity with the untouched baseline svr
on the same code**, not an absolute "all green":

1. **`pytest tests/`** — identical pass/fail on baseline-svr (2.3.1) and the upgraded env.
2. **Preprocessing** — float32 (pre-`CastToTyped`) canonical cube **byte-identical** across a
   31-subject set spanning all 28 (X,Y) spacings + all 7 Z values (Z-crop=14, Z-pad=6, Z-identity=12).
   All 301 CMRx subjects are LPS, so monai 1.6's changed `Orientationd` `labels` default is moot.
3. **Config matrix** — a train-step smoke for each distinct code path (`mri_volume`,
   `mri_volume_diffusion`, `mri_volume_bspline`, `+gather=0.5`, `+one_frame`, `+continuous_z`,
   `+diffusion=100`, `+aug tier=moderate`, `+dino-unfrozen`): finite loss, diffusion/gather branches
   non-zero where enabled, grads flowing, no OOM.
4. **fused_ssim** — SSIM-3D metric logs a real value again (`~0.55`), was `0.0`/`nan` pre-rebuild.
5. **Checkpoint resume** — a real 8.3 GB full checkpoint loads under torch 2.6+'s `weights_only=True`
   default (schema is config-invariant tensors + scalars), so no code change needed there.

## No code change required

Every torch/monai function that *could* have forced a change was checked and stays compatible:
`Orientationd(axcodes="LPS")` (LPS data ⇒ default change moot), `Spacingd`/`ResizeWithPadOrCropd`
(byte-identical), `torch.load` (`weights_only` default flip — full ckpts still load),
`torch.cuda.amp.*` (at upgrade time: deprecated but functional — FutureWarnings only; since migrated,
see below).

**Update 2026-07-24 — the amp deprecation is now cleaned up.** All 71 `torch.cuda.amp.*` call sites
across 58 files were migrated to `torch.amp.*` (`autocast(...)` → `autocast("cuda", ...)`,
`GradScaler(...)` → `GradScaler("cuda", ...)`; `_archive/` left frozen). Equivalence was proven, not
assumed: old vs new autocast give the same enabled-state/dtype and a **bitwise-equal** matmul, and
`GradScaler` matches on enabled/scale/growth_interval (including raising the same `AttributeError`
when disabled). Test suite unchanged (205 pass at the time; 211 after a concurrent test file landed),
warnings 71 → 42 with zero amp deprecations left.

**Update 2026-07-24 — batchaug triton backend A/B'd: NO-SWITCH.** The other cleanup this doc listed
was "switch batchaug to the now-matched triton backend." Measured on an A40 (`tools/ab_batchaug_backend.py`)
and the answer is **keep pytorch**:

- **batchaug's triton backend is not a separate implementation.** `triton/__init__.py` starts with
  `from ..pytorch import *` and overrides only *intensity* transforms — `triton/geometric/__init__.py`
  is a **0-byte file**. Of 44 transforms, 28 are the literal same object; the 16 that differ are all
  intensity. Our tiers run 3 active transforms, so only `RandAdjustContrastd` + `RandBiasFieldd` can
  change; the expensive `RandAffined` (grid_sample over 12 phases) is identical either way.
- **triton is not faster anywhere** (seeded-paired, interleaved, 200 rounds): full pipeline
  **−0.048 ms ± 0.0044 SEM (−10.9σ, 0.993×)** — marginally *slower*; isolated at `prob=1.0`
  **−0.013 ms ± 0.0104 (−1.3σ, 0.990×)** — null.
- **Context kills it regardless**: aug is **0.166%** of a 3518 ms train step (docs/47), and
  `augmentation.enable` is `false` by default — so it is 0% on the default path.
- **It would cost reproducibility**: triton output is not bitwise equal (max |Δ| 1.9e-6).

⚠️ **RETRACTED (2026-07-24, prove-it):** an earlier version of this section claimed "triton's intensity
kernels ARE faster in isolation (1.18×, +139σ)" and explained the pipeline null as "the intensity ops
are prob-gated so they fire ~half the rounds." **Both were wrong**, and both were caught only by
re-measuring:
- The **1.18× isolated win does not reproduce** under seeded-paired timing (−1.3σ). It was
  cross-process GPU clock drift: pytorch's own median measured 2.84 ms in one process and 2.34 ms in
  another — a swing larger than the "effect."
- **Intensity-transform COST is not probability-gated.** Both backends run the full compute
  unconditionally and gate only the *output* via `torch.where` (`pytorch/intensity/contrast.py:116`,
  `bias_field.py:84`); triton's kernels early-return on `m==0`. So gating, if anything, *favours*
  triton — it cannot explain a null.

⚠️ **Three measurement traps**, all of which produced wrong numbers before being caught:
1. `gpu_aug` calls `set_backend("pytorch")` at *module import*, so importing it after `set_backend`
   silently resets the requested backend — an arm labelled "triton" ran pytorch (this produced a
   bogus "1.22× speedup"). Import `gpu_aug` FIRST.
2. **Per-process GPU clock drift** exceeds the effect. Arms must be **interleaved within one
   process**, never run back-to-back.
3. **Pairing must be seeded.** Drawing from the shared global CUDA RNG gives each arm a *different*
   Bernoulli mask, so the identical affine's on/off gate injects ~1.6 ms of noise: unseeded
   sd **1.139 ms** vs seeded **0.062 ms** (18× tighter, SEM 0.147 → 0.0044). The unseeded design sat
   exactly on its own detection threshold and reported a meaningless "null".

Verification hygiene: `assert resolve_backend() == backend` is a **tautology** (nothing can change it
in between) — assert on the count of transforms whose `__module__` contains `triton` instead, and do
it for the equivalence check too, not just the timing loop.

## numpy 1.26.4 → 2.2.6 (2026-07-24) — done, with ONE required source fix

numpy was unpinned separately from the torch bump (one variable at a time). It resolves to a single
package (`numpy-2.2.6`); nothing else changed, and nothing in the env pins `numpy<2` (bounds are
`<3` from monai/gradio/PyWavelets and `<2.5` from scipy/numba). `opencv-python 4.13` and
`torchkbnufft` actually *declare* `numpy>=2`, so we had been running below their stated requirement.

**Training is structurally immune, and that is not luck:** `preprocess.py` and `loss.py` contain
**zero numpy** (normalization is `torch.quantile`); val determinism is stdlib `random.Random(seq_index)`
plus `torch.Generator`, not numpy RNG; every accumulator is a Python float fed by `.item()`; no counter
is numpy-typed; `cache_signature()` hashes to `179caf2419` under both versions so the monai cache
cannot silently orphan; and six real checkpoints (8.8–11.3 GB) were pickle-scanned — they contain only
`OrderedDict`/`_rebuild_tensor_v2`/`FloatStorage`, all torch-allowlisted, so a numpy-1 checkpoint →
numpy-2 requeue is byte-identical work.

**Verified** (same bar as the torch upgrade): 24/24 numpy-C-API packages import (incl. `SimpleITK 2.2.1`,
which needs **no** bump — its extension has zero numpy C-API symbols and its Python layer uses only
stable dtype names); canonical preprocessing **62/62 arrays byte-identical** over the 31-subject
spanning set (28/28 spacings, 7/7 Z, negative-controlled: a deliberate 1-ULP tamper was detected and
exited 1); `pytest` 211 passed; `fused_ssim` live; two training smokes (default + `aug=moderate`
+respiratory) with finite loss and flowing grads.

**⚠️ ONE REAL DEFECT — `inference/adapters/base.py:42-44` `percentile_scale` (FIXED).** numpy 2 changed
`np.percentile` on a float32 array to interpolate in **float32** (numpy 1 used float64). Two consequences:
1. **Silent NaN.** `vmin` became a `np.float32`, so the divide-by-zero guard `max(vmax, vmin + 1e-6)`
   is a **no-op for `vmin >= 32`** — float32 cannot resolve 1e-6 at that magnitude. Measured: guard
   alive at 31, dead at 32. A degenerate constant-intensity cine at ≥32 then gives span `0.0` → an
   all-NaN normalized volume, which `_nanmean` (`run_cmrxrecon.py:381`) silently drops into a `nan`
   metric. Under numpy 1 the same input produced a finite `0.0`. **fail-loud → fail-silent.**
2. **Numerical drift on every OOD input** (22/30 random cines bit-different, max |Δ| 1.9e-4 on a
   [0,1] image) — below PSNR/Dice/EF noise, but it breaks bit-exactness vs numpy-1-era baselines.

Scope: `percentile_scale` has ~40 call sites incl. the git-tracked `evaluation/engine/build_inputs/*`.
The in-distribution training path never calls it.

**Fix = widen the percentile result with `float()` before the span guard** (`vmin = float(np.percentile(...))`).
No-op under numpy 1, revives the guard at every magnitude, and **costs nothing** (453 vs 449 ms, same
peak memory, measured on a 15.7M-voxel cine). Mirrored into `tests/_legacy_ocmr.py` (a frozen oracle
carrying an identical copy) so `test_eval_ocmr_equivalence` keeps comparing like with like.

**Consequence (1) — the silent NaN — is fixed. Consequence (2) — the 1.9e-4 drift — is deliberately
ACCEPTED.** Casting `nz` to float64 before `np.percentile` would also reproduce numpy 1's values
exactly (verified 30/30), but measured **+34% time and 2.5× peak memory** (602 vs 453 ms, +315 vs
+126 MB). Not worth it — because the drift is **far below the eval's own reproducibility floor**:

**MEASURED end-to-end on real ACDC** (`run_vggt.py` + `assemble_and_gif.py`, ckpt
`20260719_1frame_gather05_ep99`, same subjects as the committed `vggt_20260719_1f_gather05_ep99` arm):

| comparison | max \|Δ PSNR\| | mean \|Δ PSNR\| |
|---|---|---|
| numpy-1 committed vs numpy-2 re-run | 0.075 dB | 0.013 dB |
| **numpy-2 vs numpy-2 (SAME env, re-run twice)** | **0.252 dB** | **0.171 dB** |

The pipeline's own run-to-run nondeterminism is **~13× larger than the numpy difference**. So "numpy 2
changed the OOD numbers" is not a supportable statement — the delta is invisible against re-running the
identical command twice.

⚠️ **The second row is the more important finding, and it is independent of numpy:** this eval has a
**~0.17 dB per-subject reproducibility floor**. Cohort effects in docs/46 (e.g. aug +0.41 dB pooled,
p<0.001, n=61) survive it because they average over many subjects, but **any per-subject or small-n
OOD comparison below ~0.25 dB is noise.** Do not read a 0.1 dB per-subject difference as signal.
(An earlier version of this doc predicted "~1e-3 dB wobble" from the percentile change — wrong by two
orders of magnitude, and irrelevant once the real noise floor was measured.)

**Deliberately NOT done:** numpy stays `<2.5` (scipy/numba bound). `torch.Tensor.__array__` (torch 2.13,
`_tensor.py:1241`) still lacks numpy 2's `copy=` kwarg, so monai transforms emit a `DeprecationWarning`
(warnings 42 → 393). Warning only — `np.asarray(tensor)` still shares memory, no extra copy, no
per-step allocation. Nothing to do until torch adds the kwarg.

## Rebuilding `fused_ssim` (required to restore the SSIM-3D metric + refiner-SSIM loss)

It is a CUDA extension pinned to `git+https://github.com/rahul-goel/fused-ssim/@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38`.
torch 2.13 is cu130, so build with a **CUDA-13** toolkit and a host gcc in its supported range
(`gcc/11.2.0` was used). The env's bundled libstdc++ carries `GLIBCXX` up to 3.4.34, so the built
extension's C++ ABI resolves against the env at runtime (the original failure was the *system*
libstdc++ — GCC 8.5, no `GLIBCXX_3.4.29` — being loaded instead):

```bash
module load gcc/11.2.0 cuda/13.1.0
export CUDA_HOME=/sw/pkgs/arc/cuda/13.1.0
pip install --no-build-isolation --force-reinstall --no-deps \
  "fused-ssim @ git+https://github.com/rahul-goel/fused-ssim/@a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38"
```

It must import and run with **no modules loaded** (training runs in a plain env) — verified.

## Revert

Two rollback sources, in order of authority:
1. **Byte-exact:** `scratch/torch_upgrade/svr_torch231.tar` on GPFS (`tar xf … -C ~/micromamba/envs`,
   then fix `bin/` shebangs) — the literal old env.
2. **Reproducible:** the git-tracked `envs/svr_torch231.freeze.txt` (pip freeze) — this is the
   **authoritative version list** — using the documented special installs (cu121 index for
   torch/vision; batchaug `--no-deps -e`; vggt `-e .`; fused_ssim from its original commit).

**Caveat on `envs/svr_torch231.env.yaml`:** it is a faithful `micromamba env export` of an env that
was built pip-over-conda, so its *conda* layer lists a shadowed `pytorch=2.5.1 / torchvision=0.20.1 /
torchtriton=3.1.0 / numpy=2.2.6` while the pip layer (which wins at runtime, matching `freeze.txt`)
is the real `torch==2.3.1 / torchvision==0.18.1 / triton==2.3.1 / numpy==1.26.4`. Do NOT
`micromamba env create -f` it naively — trust `freeze.txt` (or the tarball) for the true stack.

## Forward-parity (old torch vs new torch, same code)

Restoring the old env from the tarball and running the identical deterministic val forward
(respiratory/aug off) under torch 2.3.1 vs 2.13.0 gives **numerically ~identical** output:
`val_loss_objective` 0.0488 = 0.0488, `mae`/`ssim` identical to 4 dp, `psnr_3d_full` 18.5583 vs
18.5589 (Δ 0.0006 dB — bf16/kernel rounding). So "same as before" holds at the numerics level, not
just functionally — the earlier "fresh-series" hedge overstated the drift.
