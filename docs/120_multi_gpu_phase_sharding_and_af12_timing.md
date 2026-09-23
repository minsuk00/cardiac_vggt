# 120 — Multi-GPU phase sharding for `run_vggt.py`, and the af12 inference-timing benchmark (in progress)

> **TL;DR & takeaway**
> `run_vggt.py --gpus 0,1,2` (commit `8de17e1`) splits each subject's T phases across up to 3 GPUs
> (one model copy + one thread per GPU, batch built once and copied). Verified on caesar with the 518
> `final518_diff1000` model: outputs are **bit-identical** to the unmodified script under
> `torch.use_deterministic_algorithms` (84/84 files, 3 subjects × clean/breath, 1-GPU and 3-GPU), and
> **~2.8–3.0× faster per subject**. Batching the 12 phases on one GPU was measured and rejected
> (slower: a single phase already saturates the GPU). **The af12 timing benchmark is NOT done:** the
> caesar run was cancelled because another user's job shared GPU 0 and ~18 CPU cores mid-run
> (timings contaminated). It must be redone on a quiet machine — §5 is the exact procedure. Also
> open: new vs existing af12 recons differ by more than float noise (worst-phase PSNR 40–48 dB,
> max voxel diff ~1.0) with **identical code** — the environment differed (torch cu126 vs cu130,
> RTX 6000 Ada vs L40S); cause unverified (§4).

Session 2026-09-22/23. Worktree `~/vggt-afsim`, branch `exp/af-sim`.

## 1. Goal (what the user asked for)

For the paper's speed comparison, per method, on the af12 cohorts:
a) generate VGGT `final518_diff1000` af12 recons **using 3 GPUs**, b) record the time, c) report
the average inference time, d) document here how it was done (model, script, machine, GPUs) so the
other methods' timings (NeSVoR, Dangi, CiNeVol, Fetal CMR 4D, SVRTK …) can be added to this doc
later under the same definition. GPUs must be verifiably free of other jobs during the run.

## 2. What was shipped: `--gpus` phase sharding (`8de17e1`)

`evaluation/src/engine/run_vggt.py`:
- `prepare_batch()` — the old inline `build_batch` + `pin_scatter`, moved into a function (no logic change).
- `reconstruct(..., phases=None, batch=None)` — sweep only `phases` (default all T); use a prebuilt
  `batch` if given. First line `torch.cuda.set_device(device)`: the current device is per-thread and
  the per-phase timing's `torch.cuda.synchronize()` uses it.
- `reconstruct_sharded(models, devices, ...)` — 1 device → plain `reconstruct` (no thread; exact old
  path). N devices → `np.array_split` of the phases into contiguous chunks (12 → 0–3/4–7/8–11),
  batch built ONCE and `.to(dev, copy=True)` per GPU, `ThreadPoolExecutor` (one thread per GPU),
  results stitched in phase order; `ed_pack` from the shard holding phase 0.
- `main()`: `--gpus` (default `0`; 1–3 distinct ids — **never 4 on caesar, the machine shuts down**),
  one model copy per GPU, `timing.json` gains `"gpus"`. File writing untouched.
- `--gpus` ids are **relative to `CUDA_VISIBLE_DEVICES`** (PyTorch `cuda:N`). Set
  `CUDA_VISIBLE_DEVICES` to exactly the GPUs you use, or PyTorch opens a small context on every
  visible GPU (seen on caesar: 424 MiB on GPU 3). `tools/af12_vggt_timing.sh` does this.

**Why threads, not processes / DDP / batching** (all measured on caesar, RTX 6000 Ada):
- *Batching phases on one GPU is slower*, not faster. 518 model, 6 phases: 6×B=1 3.34–3.42 s vs
  1×B=6 3.77–3.88 s; B=12 OOMs (DPT head, 8 GB alloc). At B=1 the GPU is already 100 % busy (wall ≈
  GPU-busy, zero launch overhead) and power-capped at 300 W; B=6 runs ~6 % lower SM clock and its
  bigger kernels are 12–27 % less efficient (elementwise +30 %, not fully explained). At 224 px the
  loss came from activations overflowing the 96 MB L2 (measured throughput cliff at ~100 MB).
- *DDP* is for synchronizing gradients; inference shards are independent.
- *Threads vs processes*: same outputs, same speed (process variant built and tested). Threads need
  no IPC/pickling; exceptions propagate. First design took a lock around `build_batch` → shards built
  serially → only 2.6×; building once and copying fixed it (2.8–3.0×).
- `torch.compile` (+10–15 %/forward, 22 s compile, ~0.3 mm output shift) and fp16 (7 % slower,
  2 mm shift) rejected.

**Verification** (`temp/gpushard/`, gitignored; outputs redirected via a monkeypatched
`paths.arm_dir`, never into `evaluation/volumes`): cmrx2024 test P001/P004/P017 (D 10/9/12), clean +
breath, 518 `final518_diff1000`:
- Default mode: the *unmodified* script is not bit-identical to itself (max 5.4e-7 on ~7.5 % of
  voxels) — the splat's float `scatter_add_` (`vggt/utils/splat.py`) sums in nondeterministic order.
  1-GPU / 3-GPU-thread / 3-GPU-process all differ from it by the same ~6e-7. Harmless.
- Deterministic mode (`torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`):
  **84/84 files bit-identical** (12 recons ×2 variants + `ed_dvf.npz` + `resp_diag.json` + stamps, ×3
  subjects) for original ×2, shipped `--gpus 0`, shipped `--gpus 0,1,2`. `metadata.json` identical
  except arm name; `timing.json` correct (12 `per_phase_ms` in phase order, `gpus`, `total_sec`).
- Speed, default mode, per subject `total_sec`: 1 GPU 6.6–9.6 s → 3 GPUs 2.2–3.4 s (**2.8–3.0×**;
  first subject has a one-time warm-up on GPUs 1–2).

## 3. Timing definition (use the SAME for every method added later)

`timing.json → breath.total_sec` = wall time of the reconstruction only: batch build (slice
extraction + scatter pinning + copy to GPUs) + all T forwards + splats + stitching. **Excluded:**
model load (`model_load_sec`, once per process), the one-subject dataset build (`make_dataset`),
reading the input bundle, writing outputs. This is the same span the pre-sharding script timed, and
what `evaluation/src/score/aggregate.py:83` reads. With 3 GPUs the `per_phase_ms` overlap in time —
**never sum them** for a per-subject time. Per-volume time = `total_sec / T`.

Note: on caesar each subject took ~11–15 s wall but only ~4 s is `total_sec`; the rest (dataset
build, GPFS I/O) is harness overhead, **not profiled yet**.

## 4. The cancelled caesar af12 run (2026-09-23 00:20–01:05)

`--gpus 0,1,2`, arm `vggt_final518_diff1000_ep300_3gpu`, launcher `temp/gpushard/af12_3gpu.sh`
(superseded by `tools/af12_vggt_timing.sh`). GPUs verified idle at start.
- **Contaminated:** another user's `python` held 2.5 GB on GPU 0 from 00:42 to 00:54 (most of
  `cmrx2025_af12`; one subject hit 53 s), and a `python -` of the same user ran at ~1800 % CPU (load
  avg 23). Since a subject waits for its slowest shard, one slowed GPU slows everything. Numbers
  below are **not reportable**:

  | cohort | new 3-GPU mean `total_sec` | old (L40S, 1 GPU) | speedup |
  |---|---|---|---|
  | cmrx2023_af12 (26) | 3.70 s | 7.06 s | 1.9× |
  | cmrx2024_af12 (38) | 4.86 s | 7.52 s | 1.6× |
  | cmrx2025_af12 (46, contaminated) | 8.00 s | 11.01 s | 2.0× |
  | acdc_af12 (15 when read) | 4.07 s | 9.39 s | 2.3× |

  (Old times are a different GPU — a same-machine 1-GPU reference is needed for a real speedup.)
- **Nothing overwritten:** all 28,440 pre-existing files under the 5 af12 cohorts have identical
  path/size/mtime before vs after (snapshot diff = 0 lines).
- **Partial arm left on GPFS** `evaluation/volumes/<cohort>_af12/out/*/vggt_final518_diff1000_ep300_3gpu/`:
  complete for cmrx2023 (26), cmrx2024 (38), cmrx2025 (46), acdc (21), mnms 17 of 49 (+1 partial dir).
  Recons valid, timings contaminated. **Not deleted — ask the user** before removing; the new
  launcher refuses to reuse that name, so a rerun needs a new `NAME` (or the user deletes it).
- **Volumes vs the existing `vggt_final518_diff1000_ep300`** (`tools/compare_af12_recons.py`,
  1776 phases): mean |diff| 6.5e-5 – 1.4e-4, **max |diff| up to ~1.0**, worst-phase PSNR(old,new)
  39.6–47.7 dB. Code is identical (`bcc0318` → `8de17e1` differ only in the `--gpus` change), but the
  old run used **L40S + torch 2.13.0+cu130**, caesar's `svr` env is **torch 2.13.0+cu126** (CLAUDE.md
  pins cu130 — caesar's env is off-spec). Hypothesis (unverified): different kernels → slightly
  different bf16 `world_points` → a few low-coverage voxels flip empty↔filled. Not measured: whether
  scored metrics (PSNR/NCC/EF/Dice) change materially. Sharding itself is ruled out (same-machine
  1-vs-3-GPU diff is ~6e-7).

## 5. TODO for the next agent (on the new machine)

1. **Machine:** 3 identical idle GPUs (same model), low CPU load, the repo's pinned env
   (`torch 2.13.0+cu130`, check `python -c "import torch;print(torch.__version__)"`). If on Great
   Lakes, request 3 GPUs (`--gres=gpu:3`) of one type and record which; `nvidia-smi` must show no
   other process on the allocated GPUs.
2. **Arm names** (the `_3gpu` name is taken by the partial caesar arm on the shared GPFS): e.g.
   `final518_diff1000_ep300_3gpu_t` and a same-machine reference `final518_diff1000_ep300_1gpu_t`.
3. **Run** (from repo root; each run refuses on existing arm / busy GPU / load > `MAX_LOAD`, monitors
   GPUs + load every 2 s, diffs a before/after snapshot of all existing af12 files):
   ```
   GPUS=0,1,2 NAME=final518_diff1000_ep300_3gpu_t bash tools/af12_vggt_timing.sh
   GPUS=0     NAME=final518_diff1000_ep300_1gpu_t bash tools/af12_vggt_timing.sh
   ```
   Log dir `temp/af12_timing_<NAME>/`: `monitor.csv`, `gpu_pids.txt` (must list only the run's own
   `run_vggt.py` PIDs), per-cohort logs, `run.txt`. Check `monitor.csv` for load spikes too.
4. **Average inference time** (180 subjects; report mean ± sd per subject, per volume = /T, and
   speedup vs the 1-GPU reference on the same machine):
   ```python
   import json, glob, numpy as np
   for name in ("final518_diff1000_ep300_3gpu_t", "final518_diff1000_ep300_1gpu_t"):
       t = np.array([json.load(open(f))["breath"]["total_sec"] for f in
                     glob.glob(f"evaluation/volumes/*_af12/out/*/vggt_{name}/timing.json")])
       print(name, len(t), f"{t.mean():.2f} ± {t.std():.2f} s/subject", f"{t.mean()/12*1e3:.0f} ms/volume")
   ```
   Consider excluding each cohort's first subject (GPU warm-up) and say so.
5. **Volume check:** `PYTHONPATH=training:. python tools/compare_af12_recons.py vggt_<NAME>`. If it
   still shows max diff ~1.0 in the cu130 env, investigate before trusting either set (e.g. score
   both arms with `score/` and compare metrics; locate the differing voxels).
6. **Write results into §6** (machine, GPU model, env, commit, n, mean ± sd, per-volume, speedup) and
   add a line to `docs/README.md` (already added for this doc).

## 6. Results table (fill in; other methods to be added under the §3 definition)

| method | machine / GPU(s) | env | n | per subject (mean ± sd) | per volume | notes |
|---|---|---|---|---|---|---|
| VGGT `final518_diff1000`, `--gpus 0,1,2` | — | — | — | — | — | pending (§5) |
| VGGT `final518_diff1000`, 1 GPU (same machine) | — | — | — | — | — | pending (§5) |

## 6b. Same GPU config for the other methods (2026-09-23)

- **CiNeVol** (one model per subject): data-parallel fit + sharded export, `run_cinevol.py --gpus` — docs/121.
- **Dangi**: `run_dangi.py --gpus 0,1,2` — one model per GPU, contiguous phase blocks, one thread per
  GPU; `timing.json` gains `gpus`. Caesar, 47 af12 test subjects (outputs in `temp/dangi_gpu/`):
  564/564 volumes + 94/94 JSONs **bit-identical** to the unmodified script, 1 GPU and 2 GPUs;
  `total_sec` 2.1 → 1.44 s with 2 GPUs (3 not tested: another user held GPU 2). `--gpus 1` alone
  touches only that GPU (nvidia-smi).
- **NeSVoR**: `GPUS=0,1,2 bash run_nesvor.sh ...` — one worker per GPU, phase p on GPU p mod N, one
  fit per GPU at a time (`CUDA_VISIBLE_DEVICES=<physical>` + `--device 0`); unset = original `J` path.
  Provenance records the mapping and the GPUs' names; a stamped, complete subject now exits before
  rewriting `total_wall.sec`/provenance (previously a rerun reset `total_wall.sec` to 0). GPU count is
  NOT in `stamp.json` (the scorer requires identical clean/breath stamps). **Tested only with a stub
  `nesvor`** (mapping, concurrency, id validation, failure, rerun): `nesvor-t2` is not installed on
  caesar, so no real NeSVoR fit has run with `GPUS`.
- For the table: use new arm names so 1-GPU and N-GPU runs never share an arm (stamped subjects are skipped).

## 7. Files

- `evaluation/src/engine/run_vggt.py` — `--gpus` (commit `8de17e1`).
- `tools/af12_vggt_timing.sh` — guarded, monitored af12 launcher (this doc).
- `tools/compare_af12_recons.py` — new-vs-existing af12 recon comparison, all phases.
- `temp/gpushard/` (caesar only, gitignored) — verification harness (`run_vggt.py` test copy with
  output redirect / deterministic switch / process-pool variant, `run_original.py`, `compare.py`,
  `magnitude.py`), and `af12_logs/` of the cancelled run (monitor CSV, before/after snapshots).
