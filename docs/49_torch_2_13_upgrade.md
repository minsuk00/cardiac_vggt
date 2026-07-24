# 49 — Torch 2.13 / monai 1.6 / triton 3.7 upgrade

> **TL;DR & takeaway.** The `svr` env was unpinned from the frozen **torch 2.3.1** build to the
> latest stable stack — **torch 2.13.0+cu130 · torchvision 0.28.0 · torchaudio 2.11.0 · triton
> 3.7.1 · monai 1.6.0 · numpy 1.26.4** — to enable monai 1.6, a matched triton, and future
> `torch.compile`. It is **safe and requires no pipeline logic change** (only a docstring + an
> error-message string were touched — both cosmetic): the active training + inference pipeline
> behaves identically. Verified (parity vs the untouched baseline svr on the
> same code): `pytest` identical pass/fail on both envs; canonical preprocessing **float32
> byte-identical** across all 28 spacing tuples + 7 Z-paths (monai 1.4→1.6); every distinct config
> path (dpt+TV, dpt+diffusion, bspline, gather, one-frame, contz, dino-unfrozen, moderate-aug)
> trains with finite loss + grads; full-checkpoint resume loads under torch 2.6+'s new
> `weights_only=True` default. **`fused_ssim`** (CUDA ext) was rebuilt against torch 2.13 so the
> SSIM-3D metric + optional refiner-SSIM loss are restored. The old env is recoverable anytime from
> `envs/svr_torch231.freeze.txt` / `.env.yaml`. Only caveats: deprecation *warnings* (`torch.cuda.amp.*`
> etc. — still work), and slightly different forward numerics (a fresh-series boundary, not a break).

## Why

torch 2.3.1 forced monai `<1.5` and triton 2.3.1, blocking monai 1.6 and a matched triton, and
`docs`-level `torch.compile` work was bottlenecked on 2.3.1-specific compile bugs. Upgrading unpins
all of it. `monai>=1.5` needs torch≥2.4 and monai 1.6 needs torch≥2.8, so the monai bump rides with
the torch bump. numpy is deliberately held at 1.26.x (torch 2.13 runs on it; a numpy-2 migration is
kept separate).

## Version matrix (all co-matched, verified)

| package | old | new |
|---|---|---|
| torch | 2.3.1+cu121 | **2.13.0+cu130** |
| torchvision | 0.18.1 | **0.28.0** (pins `torch==2.13.0`) |
| torchaudio | 2.5.1 (mismatched, unused) | 2.11.0 (unused) |
| triton | 2.3.1 | **3.7.1** (auto; dormant — batchaug forced to pytorch backend) |
| monai | 1.4.0 | **1.6.0** (needs torch≥2.8) |
| numpy | 1.26.4 | 1.26.4 (held <2) |
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
`torch.cuda.amp.*` (deprecated but functional — FutureWarnings only). Optional future cleanup:
migrate the amp deprecations; switch batchaug to the now-matched triton backend (measured speedup
only, currently dormant by design).

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
