# 120 — Multi-GPU phase sharding for `run_vggt.py`, and the af12 inference-timing benchmark

> **TL;DR & takeaway**
> `run_vggt.py --gpus 0,1,2,3` splits each subject's T phases across up to 4 GPUs (one model copy +
> one thread per GPU, batch built once per subject and copied; commit `8de17e1`, cap raised 3→4 on
> 2026-09-23). Outputs are **bit-identical** to the unmodified script under
> `torch.use_deterministic_algorithms` (84/84 files, caesar) and, on Great Lakes L40S with the pinned
> cu130 env, **equal to the existing 1-GPU af12 arm to float noise** (2160/2160 phases, max |diff|
> ≤ 7.8e-7, ≥ 155 dB — the 40–48 dB discrepancy seen on caesar was its off-spec cu126 env, §4).
> **af12 timing (§6, 180 test subjects, 4× L40S): VGGT `final518_diff1000` = 1.96 ± 0.54 s per
> subject, 164 ms per volume, 4.16× the same-GPU-type 1-GPU forward time (8.16 ± 2.48 s).**
> Pitfall (§6a): a first attempt read 3.10 s because the monai preprocess of each subject ran
> lazily *inside* the timed span on a cold cache (~1 s); `run_vggt.py` now warms it before the
> timer (`cache_warm_sec`), verified draw-identical. Batching phases on one GPU was measured and rejected
> (slower: one phase already saturates the GPU). Other methods get added to §6 under the §3
> definition, each split over the same 4 GPUs the way its structure allows (per phase for NeSVoR /
> Dangi; data-parallel fit for CiNeVol — §8).

Sessions 2026-09-22/23 (caesar, sharding + cancelled first timing run) and 2026-09-23 (Great Lakes
`gl1706`, the reported timing). Worktree `~/vggt-afsim`, branch `exp/af-sim`.

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

## 5. Procedure (done 2026-09-23 on gl1706 — results in §6)

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

## 6. Results (other methods to be added under the §3 definition)

Both runs 2026-09-23 on Great Lakes `gl1706` (spgpu2, SLURM job 61762307: 4× **NVIDIA L40S**,
16 cores), `svr` env **torch 2.13.0+cu130**, repo `22a72c7` + the 3→4 cap change (+ the §6a
cache warm-up for the reported run), ckpt
`210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt`, launcher
`GPUS=0,1,2,3 NAME=<arm> bash tools/af12_vggt_timing.sh` (`--split test --arms breath --input
scatter`, the same args as every af12 VGGT arm). n = 180 = all af12 test subjects (cmrx2023 26,
cmrx2024 38, cmrx2025 46, acdc 21, mnms 49), first subjects **included**. Logs
`temp/af12_timing_<arm>/` (gitignored).

**Reported run** = arm `vggt_final518_diff1000_ep300_4gpu_v2` (11:44–12:01, monai cache warm,
§6a). The first attempt, arm `vggt_final518_diff1000_ep300_4gpu` (11:07–11:29), is **kept as a
valid recon but its timings are cold-cache-inflated** (3.10 ± 0.84 s; §6a) — do not cite them.

| method | machine / GPU(s) | env | n | per subject (mean ± sd) | per volume | notes |
|---|---|---|---|---|---|---|
| VGGT `final518_diff1000`, `--gpus 0,1,2,3` (`_4gpu_v2`) | gl1706, 4× L40S | torch 2.13.0+cu130 | 180 | **1.96 ± 0.54 s** (median 1.78) | **164 ms** | 3 phases/GPU, sequential B=1 forwards; `cache_warm_sec` 0.06 s mean (outside the span) |
| VGGT `final518_diff1000`, 1 GPU — forward+splat only (`vggt_final518_diff1000_ep300`, existing arm, Σ `per_phase_ms`) | Great Lakes, 1× L40S | torch 2.13.0+cu130 | 180 | 8.16 ± 2.48 s | 680 ms | reference: its `total_sec` (9.74 ± 3.07) is cold-cache-inflated (§6a), so the forward sum (+ ~0.1 s warm overhead) is the fair 1-GPU number; not re-run |
| Dangi Stage A, `--gpus 0,1,2,3` (`dangi_scatter_4gpu`) | gl1706, 4× L40S | torch 2.13.0+cu130 | 180 | **0.132 ± 0.026 s** (median 0.123) | **11 ms** | 3 phases/GPU; span = preprocess+predict+translate; stack reads (0.68 s) / NIfTI writes (0.92 s) per subject are outside it (`io_load_sec`/`io_save_sec`); run 13:08–13:12, only our PID on the GPUs, existing files untouched. 1↔4 GPU outputs bit-identical (P001, 12/12) |
| NeSVoR, `GPUS=0,1,2,3` (`nesvor_scatter_4gpu`) | gl1706, 4× L40S | nesvor-t2 (docs/90) | **1** (P001 test; cohort pending) | 363 s | 30.3 s | 12 independent per-phase fits, phase p on GPU p mod 4, 3 sequential per GPU; per-fit 140 s (first wave, one-time warm-up) then 111–112 s; span = `total_wall.sec` (includes each `nesvor reconstruct` process start + stack read, inherent to the CLI). Cohort ≈ 180 × 6 min ≈ **18 h**, to run on the 3-day job after CiNeVol (`RESUME=1`, P001 kept) |
| CiNeVol, `--gpus 0,1,2,3` (`cinevol_4gpu`) | gl1706, 4× L40S | torch 2.13.0+cu130, Grid4D cu130 build | pending | (one-subject check: fit 65.9 s, export 3.5 s, total 78.0 s) | — | one joint 4D INR per subject: data-parallel fit (one 8192-px microbatch per GPU, grads summed) + export 3 frames/GPU (docs/121); the check subject was deleted so the cohort run regenerates it. Cohort ≈ 180 × 78 s ≈ **4 h** on the 3-day job (`tools/af12_cinevol_timing.sh`) |

Existing single-GPU references for the baselines (different GPU, indicative only): Dangi
`scratch/eval/*/out/*/dangi_scatter` ~3.5 s `total_sec` on A40 — a span that *included* the file
I/O; NeSVoR `nesvor_scatter` on the gated cohorts 157 s per fit at J=1 on A40 → ~31 min per
subject; CiNeVol `cinevol` on af12 fit 203–215 s + export 13–15 s on A40.

Speedup 4 GPUs vs 1: **4.16×** (1.96 s vs the 1-GPU forward sum 8.16 s; the forwards alone shard
8.16 → 1.88 s = 4.3×, slightly super-linear because each GPU runs only 3 phases). Per cohort (4
GPUs): cmrx2023 1.63 ± 0.24, cmrx2024 1.79 ± 0.22, cmrx2025 2.09 ± 0.67, acdc 1.82 ± 0.59,
mnms 2.21 ± 0.52 s — the spread follows each subject's slice count D. Non-forward work inside the
span (batch build + per-GPU copies + thread join + stitching) is **0.08 s/subject** (profiled per
component on 3 subjects: `prepare_batch` 0.06–0.09 s, copies ≤ 0.04 s, stitch 0.005 s).

### 6a. Pitfall found and fixed: lazy monai preprocessing inside the timed span

The first attempt measured 3.10 ± 0.84 s with ~1.2 s/subject of non-forward time, first
misattributed to batch build. Profiling the same subjects afterwards gave `prepare_batch` ≈ 0.06 s
and `total ≈ forwards`. The difference: `MRIDataset` builds its monai `PersistentDataset` entry
(GPFS NIfTI load, `Orientationd`, in-plane resample, normalize, ~1 s) **lazily on the first
`get_data`**, which `build_batch` calls *inside* the timed span; `make_dataset` (before the timer)
only constructs the object. On a node that has never seen the cohort every subject is cold —
evidence: 182 entries in `/tmp/vggt-mri_${USER}_monai_cache` with mtimes in the 11:07–11:29 window,
none after. The existing 1-GPU arm (`total_sec` 9.74 vs forward sum 8.16 → 1.58 s "overhead") and
the caesar numbers (§4) carry the same bias.

Fix (`run_vggt.py`, before the timer, once per subject): `dset.get_data(seq_index=seq,
img_per_seq=dset.num_slices)` and record its cost as `timing.json → cache_warm_sec`. Safe: for a
non-train split the draw is a local `random.Random(seq_index)` — verified on two subjects that two
calls return identical arrays for all 9 batch keys and leave the global `random`/torch/numpy RNG
untouched, and the `_4gpu_v2` recons equal the existing arm (below). **Rule for every method added
to this table: warm/preload the subject's inputs outside the timed span** (NeSVoR/CiNeVol read the
bundle stacks directly, but check for lazy first-touch costs the same way — profile one subject
twice).

**Run audit** (`monitor.csv`, 2 s cadence; 1991 samples first run, 1651 reported run): only the
run's own five `run_vggt.py` PIDs (one per cohort process) ever appeared on the four GPUs;
`before.txt`/`after.txt` snapshot: first run 0 lines; reported run **2160 added lines, 0
modified/deleted** — `<subject>/scatter/stack_t{00..11}.nii.gz` for all 180 subjects, written
11:55–12:00 by another process (`run_vggt.py` never writes stacks; provenance to be confirmed with
the user — presumably a baseline-input export). 1-min load ≤ 6.1 throughout (≤ 5.8 in that window;
acdc/mnms timings show no anomaly). The node was **shared** with three other users' jobs (their own
GPUs/cores under SLURM cgroups; only memory/PCIe bandwidth shared) — accepted by the user as not
material.

**Volumes vs the existing arm** (`tools/compare_af12_recons.py`, all 2160 breath phases, both
arms): max |diff| 4.8e-7 – 9.5e-7, mean |diff| ≤ 2e-9, worst-phase PSNR 154.9–163.8 dB per cohort —
i.e. the splat's `scatter_add_` float-order noise (§2), nothing else. This closes §4: on the pinned
cu130 env the sharded output equals the existing arm; caesar's 40–48 dB discrepancy was its cu126
env / different GPU, not the sharding.

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

**Follow-up on Great Lakes (gl1706, 4× L40S, 2026-09-23 13:00, commit `7267efc`):** both caps
raised 3→4. `run_dangi.py` additionally moved the stack reads before and the NIfTI writes after the
timer (`io_load_sec`/`io_save_sec`; the §3 span excludes input reads/output writes and for Dangi
the GPFS I/O was most of the old number) and gained `--arm-name`; 1↔4 GPU outputs re-verified
bit-identical (P001, 12/12 volumes + centres). `run_nesvor.sh` had `VGGT=/home/minsukc/vggt`
hardcoded (ignores the worktree; now defaults to the script's own checkout, overridable) — its
4-GPU mapping was re-verified with a stub through that override, then with a **real** fit
(P001: 12/12 phases, 363 s, §6 table). Cohort launchers `tools/af12_dangi_timing.sh`,
`tools/af12_nesvor_timing.sh` (same guards/monitor/snapshot as the VGGT and CiNeVol ones; guards
fault-injected; NeSVoR `RESUME=1`).

## 7. Files

- `evaluation/src/engine/run_vggt.py` — `--gpus` (commit `8de17e1`; cap 3→4 on 2026-09-23).
- `tools/af12_vggt_timing.sh` — guarded, monitored af12 launcher (this doc).
- `tools/compare_af12_recons.py` — new-vs-existing af12 recon comparison, all phases.
- `temp/gpushard/` (caesar only, gitignored) — verification harness (`run_vggt.py` test copy with
  output redirect / deterministic switch / process-pool variant, `run_original.py`, `compare.py`,
  `magnitude.py`), and `af12_logs/` of the cancelled run (monitor CSV, before/after snapshots).

## 8. Plan for the other GPU methods (same 4 GPUs, one patient at a time, §3 span) — implemented (§6b); Dangi cohort done, CiNeVol + NeSVoR cohorts pending on the 3-day job

The rule: every GPU method is timed on one patient using all 4 GPUs, split along whatever axis its
structure allows, with no change to the method's math. Status of the af12 arms as of 2026-09-23:
CiNeVol exists (1× A40, ~199 s fit + 14 s export per subject); **NeSVoR and Dangi have no af12
arms** (they exist only on the gated-cine cohorts `scratch/eval/<src>/out`).

- **Dangi** (`run_dangi.py`): 12 independent per-phase slice-CNN passes → 3 phases/GPU (one model
  copy + thread per GPU, as VGGT). I/O-bound, ~3.5 s/subject on 1 GPU (docs/100 §10a); sharding is
  for consistency, not speed.
- **NeSVoR** (`run_nesvor.sh`): already 12 *independent* INR fits per subject (one per phase,
  ~157 s each on A40 at J=1) → 3 fits/GPU: give each fit its own `CUDA_VISIBLE_DEVICES`, run 4 at
  a time (one per GPU). Exact (nothing is shared between fits). ~8 min/subject → ~24 h for 180
  subjects; needs its own allocation.
- **CiNeVol** (`baselines/cinevol/cinevol/fit.py`): ONE 4D INR per subject, phases share all
  parameters — no phase axis. Current step = 32,768 sampled pixels processed as 4 exact
  gradient-accumulation microbatches of 8,192 (`batch_gradient`, `--microbatch 8192`), preceded by
  a no-grad pre-pass computing the batch-wide `global_abs_bias`. 4-GPU version = data-parallel:
  identical model + AdamW replicas, every rank draws the same sample stream (same CPU `rng`), takes
  microbatch `rank`, all-reduce (sum) the bias scalar, then all-reduce (sum) the grads, identical
  `optimizer.step()` on every rank; logging/checkpoints from rank 0; export 3 frames/GPU. Same
  gradient as today up to float summation order (not bit-identical). Verify on 2–3 subjects against
  the existing single-GPU fit before running the cohort. Expected fit speedup < 4× (per-step
  all-reduce of the Grid4D hash grid + MLP at ~0.4 s/step).
