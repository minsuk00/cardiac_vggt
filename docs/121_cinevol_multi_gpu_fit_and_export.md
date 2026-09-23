# 121 — Multi-GPU CiNeVol: sharded fit step + sharded frame export (`run_cinevol.py --gpus`)

> **TL;DR & takeaway**
> `tools/run_cinevol.py --gpus 0,1,2` splits each fit step's 32,768-pixel batch across up to 3 GPUs
> (one chunk per GPU, one model replica + AdamW per GPU, one thread per GPU, gradients summed on GPU 0
> and copied back) and splits the 12-frame export across the same GPUs. The config is **unchanged**
> for any GPU count (500 steps, 32,768 px, 16 PSF samples); only the hardware split differs. Default
> `--gpus 0` = the original single-GPU path. On caesar (RTX 6000 Ada, 2 af12 subjects): **fit 116–123 s
> → 69–71 s, total 123–132 s → 75–80 s (~1.7×)**. Equivalence: step-1 gradient 3-GPU vs 1-GPU rel.
> diff 5e-6 (1-GPU vs itself 1.2e-6), replicas bit-identical after 20 steps; full 500-step fits differ
> from the original by the same amount the **original differs from itself** (Grid4D backward is not
> bit-reproducible; mean |Δvol| ~2e-2 run-to-run). Multi-GPU `--resume` is refused (would corrupt
> AdamW step counts). **Not done:** the clean-machine af12 timing run (use a NEW `--arm-name`), and
> whether the run-to-run spread moves the paper metrics.

## 1. Why / what

Timing comparison with VGGT's `--gpus` (docs/120). Unlike VGGT/NeSVoR/Dangi (12 independent
per-phase jobs), CiNeVol fits **one** model per subject, so the phase-level split doesn't apply; the
fit step itself is data-parallel.

Measured first (1 GPU, 40 steps, caesar): larger chunks alone barely help — `--microbatch` 8192 /
16384 / 32768 → 0.224 / 0.198 / 0.192 s/step (peak 1.30 / 1.92 / 3.09 GB). The GPU is already
nearly saturated, so a real speed-up needs more GPUs.

## 2. Implementation (`baselines/cinevol/cinevol/fit.py`, `tools/run_cinevol.py`)

- `fit(..., devices=[...])`: replicas = `copy.deepcopy(model).to(d)`, one AdamW each. Per step:
  sample on CPU with the same generator as the 1-GPU path (identical pixels + PSF noise) →
  `sharded_batch_gradient`: contiguous part per GPU; pass 1 (no grad) each part's `|bias|·n/count`,
  summed → the same `global_abs_bias`; pass 2 loss+backward per part with the same `n/count` weights;
  grads summed on GPU 0 and copied to every replica; every optimizer steps. Checkpoint, `losses.jsonl`
  and the non-finite check use replica 0 (replicas are identical). One device → the original code
  path, untouched.
- Every worker thread calls `torch.cuda.set_device` first: Grid4D launches its kernels on the
  current device with no device guard (same gotcha as VGGT).
- `run_cinevol.py --gpus` (1–3 distinct ids, relative to `CUDA_VISIBLE_DEVICES`; caesar must never
  use all 4). `--microbatch` default: 8192 on 1 GPU, `ceil(32768/N)` on N GPUs (one chunk per GPU;
  memory knob only). `export_frames` splits the frames into contiguous blocks, one model load per
  GPU; `query()` reseeds its CPU RNG per frame, so the split is exact and frame order is kept.
- Metadata: `stamp.json` gains `gpus`; provenance lists the GPUs; peak memory = max over GPUs;
  `environment.json` / `timing_fit.json` gain `devices`. File set, overwrite guard and atomic publish
  unchanged. `fit()` refuses `resume` with >1 device: each optimizer's `load_state_dict` of the same
  dict shares the CPU `step` tensors, so every step increments them N× (reproduced: resumed 3-GPU
  losses drift 1.5e-2 vs 4e-7 for the original 1-GPU resume). `run_cinevol.py` never resumes.

## 3. Verification (caesar, 3× RTX 6000 Ada, torch 2.13.0+cu126, private Grid4D build with CUDA 12.4)

All outputs redirected to `temp/cinevol_gpu/` (monkeypatched `paths.arm_dir`); snapshot of all
21,516 existing `cinevol*` files on GPFS identical before/after. Subjects `cmrx2023_af12/CMRx23_Test_P005`,
`acdc_af12/ACDC_patient102`.

| run | fit (s) | export (s) | total (s) |
|---|---|---|---|
| original code (HEAD copy), 1 GPU, 2 runs | 116–120 | 3.6–7.1 | 123–132 |
| new code, 1 GPU | 117–123 | 4.0–6.8 | 128–131 |
| new code, 3 GPUs | 69–71 | 3.0–4.6 | 75–80 |

CPU load average was ~27 (other users) — indicative, not a benchmark. Why 1.7× and not 3×:
**not profiled** (hypothesis: per-step CPU sampling, thread sync, cross-GPU grad copies).

- One step, same init + sample: loss 1-GPU vs 3-GPU equal to 2e-9; gradient rel. diff 5e-6
  (1-GPU vs 1-GPU: 1.2e-6). Replica grads bit-identical after the sum; replica params bit-identical
  after 20 steps; all params/buffers on the right device after deepcopy.
- 500-step fits: the **original** code is not reproducible — two identical runs start bit-equal and
  diverge (1e-10 at step 2 → 1e-4 at step 10 → 1e-2 by step 50); volumes mean |Δ| 1.6–2.6e-2, max
  0.4–0.6. 3-GPU vs original is the same size (mean |Δ| 1.7–2.5e-2). Last-50-step mean MAE agrees
  within ~1–3% across all runs (the stamped `final_mae` is one noisy minibatch). Likely source Grid4D
  backward non-determinism (unverified which op).
- Upstream CiNeVol tests 15/15 pass; `python -m cinevol.fit --devices cuda:0 cuda:1` runs.
- 3-subagent review (math equivalence, threads/devices, I/O+metadata): no defects besides the resume
  bug above (found by the run, not by the reviewers).

## 4. Open / next

- Clean-machine af12 run for the timing table (docs/120 §6), with a **new `--arm-name`** (e.g.
  `cinevol_3gpu`) — stamped subjects are skipped, so reusing `cinevol` would mix 1- and 3-GPU times.
  Timing span differs from VGGT's: CiNeVol `total_wall.sec` = prepare + fit + export (fit dominates).
- Whether CiNeVol's run-to-run spread moves NCC/PSNR: score two original-code runs against each other.
- Caesar's `svr` is torch cu126; the shared Grid4D build `scratch/cinevol/grid4d_build` targets
  cu130 and was partly rewritten by a failed caesar build on 2026-09-23 (`build.ninja`, `bindings.o`,
  ninja logs; `.so` untouched) — run `run_cinevol.py --build-only` once on Great Lakes before the next
  array job. On caesar use a private `GRID4D_BUILD_DIR` + `CUDA_HOME=/usr/local/cuda-12.4`.

## 5. Files

- `baselines/cinevol/cinevol/fit.py` — `sharded_batch_gradient`, `fit(devices=...)`, `--devices`.
- `tools/run_cinevol.py` — `--gpus`, `export_frames`, microbatch default, metadata.
- `temp/cinevol_gpu/` (caesar, gitignored) — `drive.py` (output redirect; runs new or HEAD copy),
  `run_tests.sh`, `compare.py`, `grad_check.py`, `resume_check.py`, `orig/`, `out/`, `logs/`.
