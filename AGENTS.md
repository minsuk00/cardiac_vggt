# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** real-time free-breathing cine — reconstruct the full 3D heart at any target cardiac phase from a few scattered single-frame-per-slice acquisitions. No real-time training data exists, so we **simulate** the sparse scattered acquisition from gated cine + motion aug and aim to generalize to true real-time cine. **Information contract:** the model may know only `z` per input slice — input cardiac `t` and respiratory `r` are unavailable (design stance, not fully implemented) — `docs/04`. Full statement: docs/65.

**Active pipeline: unsupervised intensity-based, multi-phase** (`default.yaml` / `exp_*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`).

**Target-phase conditioning = REFERENCE SLICE (current default, `default.yaml`).** Slot 0 is a real target-phase reference slice at the mid-ventricular plane (`reference_slot=true`, `use_reference_token=true`), marked via VGGT's native two-token `camera_token` (index 0 = anchor, 1 = the rest); the model reads the target phase from slot-0's *image content*. Requires the **aggregator finetune (aggft)** (freeze `*patch_embed*` only). Consequence: you reconstruct **observed** phases (≈ the recoverable limit). EF recovery confirmed on final ckpts (docs/24, 25, 33); the legacy content-free `target_t`-index path is gone — history in docs/25 + docs/65. Pre-refactor ckpts (e.g. the "4-day baseline") are weights-only warm-start seeds, not resumable — docs/37 + docs/65.

**Geometry: in-plane canonical grid + native-z (docs/58).** Every subject's **in-plane** axes are resampled to a shared grid; **z is never resampled** — each subject keeps its own native slice pitch/count (`dz`, `D`), see "CMR data notes" below. The old supervised-DVF path is fully removed (`gt_dvfs`/`scale_factors` gone from the batch); legacy DVF tooling and old configs/sbatch live under `_archive/`. Refactor history: docs/58 + docs/65.

- MRI data: `/scratch/data/CMRxRecon2024/` (symlinked, GPFS)
- Env: `micromamba activate svr`
- SLURM: `spgpu` partition for training (A40 GPUs), `standard` for CPU jobs. **Account: `jjparkcv0` is the default and every `sbatch/*.sh` header now says so** (`jjparkcv98` frequently hits `AssocGrpSubmitJobsLimit`; a few GPU-heavy recon jobs use `jjparkcv_owned1` for spgpu2/L40S).

## Setup

```bash
micromamba activate svr
pip install -r requirements.txt           # includes monai>=1.6,<1.7
pip install --no-deps -e /home/minsukc/MRI2CT/batchaug/  # GPU aug — see note below
```

**This repo is NOT installed as a package** — no `pip install -e .`. **Always run from the repo root with `PYTHONPATH=training:.`**. Both entries are load-bearing: `.` makes `from vggt…` imports resolve (the job an editable install would do), `training` makes Hydra `_target_` short names resolve (`loss.MultitaskLoss`, `data.*`, `train_utils.*`) and is required by the ~66 scripts in `tools/`/`baselines/`/`evaluation/` that import them directly. `tests/` needs neither (`tests/conftest.py` inserts both). Details/evidence: docs/65.

**Stack: torch 2.13.0+cu130 / torchvision 0.28.0 / triton 3.7.1 / monai 1.6.0 / numpy 2.2.6** (2026-07 upgrade — docs/49; pin rationale in the comment block at the bottom of `requirements.txt`). Re-verify any dependency bump with `bash tools/verify_env_migration.sh`. **batchaug** is not on PyPI — install editable from the MRI2CT clone with `--no-deps` (keeps pip from re-resolving the pinned torch stack); `gpu_aug.py` forces `batchaug.set_backend("pytorch")` for reproducibility. **fused_ssim** is a CUDA extension rebuilt against the active torch (`module load gcc/11.2.0 cuda/13.1.0`, then `pip install --no-build-isolation` from its pinned git commit).

## Training

Entry point: `training/launch.py` (Hydra).

```bash
# Active config
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config default

# NOTE: single-GPU only. DDP was removed in 284992c (no process group, device hardcoded to
# cuda:0, sampler pinned to num_replicas=1) — `--nproc_per_node>1` would run N duplicate
# trainings on GPU 0, not a data-parallel job.

# ED-only fallback (matches original pre-multi-phase behavior)
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config default t_target_fixed=0

# Override
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config default optim.base_lr=1e-4
```

**Cluster submission**: `bash sbatch/_archive/train_mri_volume_reference.sh` — self-submits via embedded `sbatch`, `WANDB_MODE=online`, SLURM auto-requeue (SIGUSR1 → checkpoint-and-resume). Resume modes (vars at the top of the script):
- both `RESUME_FROM`/`CKPT_ONLY` empty (**default**) → **fresh-from-base VGGT-1B** (config's base-weights resume path, `strict=false`), fresh exp dir + new wandb run.
- `RESUME_FROM=<exp_dir>` → continue same exp_name + reuse same wandb run id (crash/requeue recovery).
- `CKPT_ONLY=<ckpt_path>` → **fresh** exp dir + new wandb run, loading from `<ckpt_path>` via `checkpoint.resume_checkpoint_path` (`strict=false`). **GOTCHA (docs/37):** this is a **FULL resume** (weights + optimizer + `prev_epoch`), NOT weights-only — a full `checkpoint_last.pt` can silently do zero training. For a real warm-start, strip to `{"model": ...}` first (`torch.save({'model': torch.load(ckpt)['model']}, out)`). Full mechanics: docs/37 + docs/65.

**Configs** (`training/config/`) — **one complete config + thin experiment overrides** (flattened 2026-08-01):
- `default.yaml` — **THE config.** Complete and runnable on its own (`--config default`): cohort, sampling, logging, loss, optimizer, augmentation, aggft freeze. One file, one truth.
- `exp_bspline.yaml` — the one shipped experiment variant, inheriting `default.yaml`, ~20 lines, overriding only the warp head (and the only config still on the L1 TV regularizer, `tv=0.1`). There is **no `exp_diffusion.yaml`**: the L2-diffusion arm IS `default.yaml` (`tv=0`, `diffusion=1000`).
- The old three-layer config chain (`default → mri_finetune → mri_volume`) is gone; `exp_name`/`config_name` deliberately keep the `mri_volume` family name (log-dir + wandb continuity). Why it was removed + flattening verification: docs/61 + docs/65.
- Legacy variants (`mri_finetune_*`, `mri_p001_overfit`, `mri_volume_overfit`) and their sbatch scripts live under `_archive/legacy_configs/` and `_archive/legacy_sbatch/`.

**Key knobs:**
- `img_nums: [20, 20]` → **slot BUDGET/cap, not a slot count**: with `one_frame_per_slice: true` (default), S == this subject's own D (5–21); `get_data` raises if a subject needs more. **To cut memory, cut D or the model, not this knob** — batch size is pinned to 1 (same-D-different-pitch subjects collate silently). docs/59 + docs/65.
- `continuous_z: false` (default) | `true` → sample non-reference slots at **continuous physical z** (±`z_jitter`=0.5 off the integer plane + 2-plane interp); off ⇒ numerically identical to the discrete grid. docs/28.
- `t_target_fixed: null` (default → multi-phase, uniform per train call) | `0` (reproduces ED-only behavior) | any int K (force `t_target=K`).
- `t_target_phases: null` (default → all T phases) | list e.g. `[0,7]` → restrict the multi-phase target pool to that subset (train samples uniformly, val cycles it deterministically). **Mutually exclusive with `t_target_fixed`** (single-phase wins if both set).
- `optim.frozen_module_names` — two regimes, guarded by `tests/test_freeze_pattern.py`: **head-only** (legacy) freezes the entire aggregator; **aggft** (all shipped configs) = `["*patch_embed*"]` — attention blocks, `z_embedder`, `camera_token`, `point_head` all train (~2.8× slower, ~27 GB/A40). Exact patterns: docs/65.
- `model.train_on_residual_dvf: true` → point head outputs Δ; `world_points = scanner_coords + Δ`.
- `logging.filmstrip_every_n_val_epochs: 3` → cadence for the multi-phase cardiac-cycle visualization (also gates the ED/ES panels and the augmentation panel).
- `data.augmentation.enable: true` (default since 2026-07-31) | `false` → opt out of GPU augmentation. `data.augmentation.tier: conservative|moderate|aggressive` (default `moderate` — the docs/46 §3 C2 shipped arm). See "Augmentation" below.

## Volume pipeline (one forward pass)

0. **Preprocess (cached, one-time per subject; native-z, docs/58).** monai `PersistentDataset` (`training/data/preprocess.py`) resamples in-plane to `1.4`mm, crops/zero-pads to `256×256`; **z is never resampled or padded** (native `dz`, `D` kept). Intensity normalized against phase_00's 0.5/99.9 percentiles over non-zero FOV. Output: `(T=12, 256, 256, D)` float16 + `(256,256,D)` content mask + `dz_mm`, cached on `/tmp/vggt-mri_${USER}_monai_cache/`. Transform-level detail: docs/65.
1. **Sample (ONE frame per slice, the default).** Every in-FOV plane appears exactly once, so **S == D** — the sparse extreme the research goal targets; full z-coverage keeps the full-volume L1 valid. Per-slot `t` is random and **used only to extract slice content** — never a model input. Slot 0 = `(t_target, z_mid)` reference. Train samples from global `random`; val from a private `random.Random(seq_index)` (reproducible). Legacy multi-frame sampler (`one_frame_per_slice: false`) and continuous-z: docs/28; full text: docs/65.
2. **Aggregator.** DINOv2 patch_embed + 24× alternating frame/global attention. Per-slot special token = sinusoidal embeddings: `z_embedder(z_norm)` (linear, `z_norm = z_mm / Z_HALF_MM` — **physical**, `Z_HALF_MM=90` is a fixed constant shared by every subject regardless of that subject's own `D`/`dz`, docs/58) always on; reference default adds the two-token `camera_token` (slot 0 = target-phase anchor), legacy path adds `t_embedder`+`target_t_embedder` instead (see Project + Key knobs). Frozen vs aggft per the freeze-pattern note.
3. **Point head (trainable, DPT).** Outputs per-pixel residual Δ (3 channels) + confidence (1, unused). `world_points = scanner_coords + Δ`, all in normalized [-1, 1] (x/y index-normalized over the fixed 256×256 grid; z is physical, see above).
4. **Splat.** `splat_to_volume(world_points, intensity, (D,256,256), z_scale)` → `V_canon`, where `D` is THIS subject's own native slice count and `z_scale = Z_HALF_MM/dz` is required (no default — a missed call site must crash, not silently compress the volume). Differentiable trilinear scatter; divides by accumulated coverage (`vggt/utils/splat.py`). **`splat_weight = intensity > 1e-3` is kept** — padded X/Y slots are all-zero, and the gate prevents their zero-intensity pixels from diluting V_canon if the model's Δ ever moves them into content planes.
5. **Loss.** `loss_volume = (V_canon - V_gt).abs().mean()` + `0.1 * TV(pos_pred)` — **full-volume L1**, no anatomy mask.

**Input slices:** each canonical `(256,256)` slice is bilinear-resized to `518×518` for DINOv2 (no letterbox/padding). `scanner_coords[py,px] = (px/517·2−1, py/517·2−1, z_norm)` with `z_norm = (z_i − (D−1)/2)·dz/Z_HALF_MM` — x/y purely geometric and identical for every subject, z physical (mm). Every pixel has a valid coord (no invalid sentinel). Full text: docs/65.

**`V_gt`** = `phases[t_target]` from the cache (canonical frame, batch key `gt_target_volume`). **`anatomy_bbox`** = `(z0,z1,y0,y1,x0,x1)` geometric bbox of the content mask (used to restrict z sampling AND for the bbox metric). Both produced by `MRIDataset.get_data`.

## CMR data notes

Native cine shapes and spacings vary per subject (T=12 always). **Gotcha: CMRxRecon2024's header Z=8.0 is slice THICKNESS, not pitch** — true pitch is 8+4mm gap = **12 mm**; affines were relabeled 8→12 on disk (`docs/27`). Full shape/spacing/FOV stats: docs/65.

`MRIDataset` (`training/data/datasets/mri_dataset.py`) maps every subject onto the canonical grid described in "Volume pipeline" step 0: fixed in-plane extent (`1.4mm`, `256×256`, `358.4×358.4`mm), native per-subject z (no shared cube depth). Subjects with FOV < 256×256 get zero-padded in X/Y only; larger FOVs get center-cropped (the heart is near the acquisition center, so cropping loses only periphery).

**Axis-order gotcha:** monai/nibabel store volumes `(X, Y, Z)`; the splat consumes `(D, H, W) = (Z, Y, X)`. The single conversion site is the `permute(0, 3, 2, 1)` in `MRIDataset.get_data` right after the cache lookup — everything downstream is splat-order. Easy to break silently; tests in `test_canonical_invariants.py` guard it.

**Orientation = LPS everywhere** (training forces `Orientationd(axcodes="LPS")`). **ALL data must be LPS — training, val, AND every OOD adapter.** When adding a dataset/adapter, check axcodes and reorient to LPS — a mis-oriented heart still looks like a heart and silently degrades anatomical priors with NO crash (burned us on ACDC, which is mixed LPS/LAS). Per-dataset status + full text: docs/65.

`mri_mode: "axial"` means **native SAX z-slicing** — not anatomical axial. The slices are short-axis views.

**Slice order = APEX AT z0, for every subject** (standardised on disk 2026-07-31; full history, flip/roll ordering constraints, and revert paths: docs/56 + docs/58 §10a/§10b + docs/65). The rules that survive:
- **Never "fix" orientation by editing the affine** — flip the *array*; `Orientationd(axcodes="LPS")` would silently undo an affine edit.
- **Adding data?** Run `tools/render_slice_order_check.py` and flip to apex-at-z0 **per subject** — never by a per-source rule (CMRx2025 is ~50/50 within one scanner).
- ⚠️ **Do NOT run/revert `tools/fix_slice_order.py` / `fix_slice_roll.py` without reading docs/56 + docs/58 §10** — the fixes don't commute and a wrong ordering corrupts silently.
- ⚠️ **Anything derived from the cohort before 2026-07-31 12:19 is pre-flip stale** (frozen eval bundle, SVR baseline outputs, `*_recon_v1_espirit_imagedomain/`); rebuilding the eval bundle needs the three `evaluation/engine/build_inputs/cmrxrecon.py` fixes first — docs/58 §10b.

## Augmentation

GPU augmentation via `batchaug` (`training/data/gpu_aug.py`), **ON by default** (`data.augmentation.enable: true`, tier `moderate`), **train-only** (val never augments). One in-plane affine per subject across all 12 T-phases + content mask (phase-consistent); the trainer re-derives `gt_target_volume`, re-extracts input slices, and recomputes `anatomy_bbox` (`scanner_coords` unchanged — pure geometry). Tiers `conservative`/`moderate`/`aggressive` escalate affine + photometric; W-flip is **aggressive-tier-only**. Tier contents, flip history, and D-agnostic verification: docs/46 §3, docs/58 §10c, docs/65. Visual proof: `tools/render_augmentation_examples.py` → `result/augmentation_examples/`.

**Respiratory-motion sim** (`training/data/respiratory.py`) is a SEPARATE toggle (`data.augmentation.respiratory.enable`), **ON by default**. Per-input-slice deform-then-reslice SI+AP shift applied **after** affine, overwriting **only the input slices** — targets stay at the unshifted end-expiration reference, so the model learns to **correct** breathing (blind to `r`). Applies in **both train AND val** (unlike affine): train iid per epoch, val deterministic per `seq_index`. Disabling ⇒ bit-identical to pre-respiratory. Parameter-level detail (per-subject tilt/azimuth/amplitude sampling): docs/01, docs/05, docs/65. Visual proof: `tools/render_respiratory_examples.py` → `_html/06_*.html`.

## Architecture

```
VGGT (vggt/models/vggt.py) — ~941M total, base weights at ./scratch/base_weights/vggt1b_base.pt (download: huggingface.co/facebook/VGGT-1B/resolve/main/model.pt — NOT a regenerable cache, keep it)
├── Aggregator
│   ├── DINOv2 patch_embed (518² inputs, patch=14 → 37² tokens)    [FROZEN, ~304M]
│   ├── 24× frame_blocks + 24× global_blocks (alternating attn)    [FROZEN, ~605M]
│   ├── rope / camera_token / register_token                        [FROZEN, ~10K]
│   └── ZIndexEmbedder, TIndexEmbedder (sinusoidal Fourier)         [FROZEN, ~28K — see Key knobs]
└── point_head — DPT upsampler → 4-channel (Δ, conf)                [TRAINABLE, ~32.65M]

Camera / depth / track heads disabled in mri_volume config.
[FROZEN] labels show the legacy head-only freeze; the shipped default (aggft) also trains the attention blocks + z_embedder + camera_token — see `optim.frozen_module_names` under Key knobs.
```

Checkpoints save the **full 941M state dict** (~3.8 GB each), not just the trainable head. Optimizer + scaler state included.

## Inference / inspection

```python
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
preds = model(images, batch=batch)  # batch needs: z_indices, t_indices, scanner_coords
# To use compute_volume_intensity_loss: batch must also include gt_target_volume (already the t_target phase; t_target itself is only used for per-phase logging, not the loss).
```

Handy tools (full descriptions: docs/65): `tools/preview_canonical_preprocess.py` (canonical resample sanity-check), `tools/render_augmentation_examples.py`, `tools/render_volume_example.py`, `tools/test_sequential_sampling.py`, `baselines/eval_within_body_mask.py` (identity-Δ PSNR floor).

**Where new scripts go** (sort by *reuse potential*, not temp-vs-permanent):
- **Throwaway** one-off probe / sanity-check you won't rerun → scratchpad dir, NOT the repo.
- **Might reuse, or an experiment script backing a `docs/` finding** → `tools/` (git-tracked; several `tools/exp_*`/`toy_*` are cited by docs as repro provenance).
- **`evaluation/` is OFF-LIMITS for auto-adding** — it holds only standing eval code we always run. **NEVER add anything to `evaluation/` on your own initiative; the user decides what goes there.** Write to `tools/` and ask.

**Evaluation & SVR baselines**: eval harness in `inference/` (`run_cmrxrecon.py`, `run_rtfb.py`, `adapters/`), classical SVR baselines in `baselines/`, frozen breathing-simulated harness in `evaluation/` (see its README; heavy data on gitignored GPFS). Rationale/protocol/results: docs/24 + docs/29–35. The `evaluation/` off-limits rule above applies to `evaluation/analysis/` too.

## Logging (wandb + on-disk, project `vggt-mri`)

**To analyse a past run, READ THE FILES IN ITS `log_dir` — do NOT go to wandb.** Every run mirrors
all its numbers to disk (docs/60), so no network, auth, or run-id lookup is needed:

```python
from tools.load_run import load_run, load_identity_baseline
meta, scalars, subjects = load_run("scratch/logs/<exp_dir>")   # or: python tools/load_run.py <log_dir>
subjects.groupby("source")["metric_psnr_3d_bbox"].mean()       # any slice, offline
```

Files in `log_dir`: `run_meta.jsonl` (one line per process launch), `metrics.jsonl` (every scalar), `val_per_subject.csv` (per-subject metrics — **exists nowhere else**; join to `training/splits/manifest.csv`), `baseline_identity.json` (identity floors — normalise per-subject PSNR by these first). Only the image panels are wandb-only. Full field descriptions: docs/60 + docs/65.

**Prefer `metric_psnr_3d_bbox` as the honest number** (`_full` is inflated by X/Y padding for small-FOV subjects). **Don't compare PSNR across the canonical-grid or native-z refactors** — treat post-2026-07-31 runs as a fresh series.

- **GT-referenced ship-decision metrics (val-only, `docs/38`):** recov_frac_heart, psnr_3d_static vs psnr_3d_motion, hole_frac_heart, and the breathing `metric_resp_*` family. **Decision rule: a change wins iff recov_frac↑ & psnr_motion↑ WITHOUT hole_frac↑.** Gated val-only ⇒ training bit-identical.
- Per-metric definitions, panel/visual cadence, identity-Δ startup baseline, tags, and fixed-phase gating: docs/38 + docs/60 + docs/65. `save_val_volumes` (default true) dumps per-subject pred+GT NIfTIs to `${log_dir}/val_volumes/` (~360 MB, overwritten each epoch). All diagnostic logging is `try/except`-wrapped — never raises into training.

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.
- **Monai cache is node-local `/tmp`, rebuilt per job** (`/tmp/vggt-mri_${USER}_monai_cache/`, ~55 MB/subject). Lazy first-epoch rebuild ~3–10 min for ~270 subjects, overlaps GPU compute. Intentionally not on GPFS — cached GPFS reads are ~18–20× slower than /tmp, so persisting would slow every epoch to save one rebuild.

## Git / branches (multi-agent hygiene)

Multiple agents share this single working tree — a bare `git switch` with uncommitted changes drags them onto the new branch. **Do branch work in a dedicated worktree, not by switching HEAD in place:** `git worktree add ../vggt-<task> -b cleanup/<task>`; remove it after merge.

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- **Checkpoint loads auto-stage to node-local `/tmp`** (`vggt/utils/checkpoint_stage.py`, docs/50) — GPFS `torch.load` is ~266s vs ~5s from `/tmp`. Training stages only immutable base/seed weights; inference stages every load. Byte-identical; falls back to the original path on failure.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. The cluster scripts (`sbatch/train_mri_volume_*.sh`) set `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`, `phase_mode:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`; `OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))`; `OmegaConf.register_new_resolver('phase_mode', lambda t: 'multiphase' if t is None else f't{int(t)}')`.

## Testing

```bash
micromamba run -n svr python -m pytest tests/
```
Synthetic in-memory CMR dataset (`tests/conftest.py`, T=12) — no real data needed; each session gets an isolated monai cache dir. Per-file coverage map: docs/65.

## Docs

Research findings, design decisions, and experiment write-ups live in **`docs/`** (numbered). **When you make a non-trivial design choice, run a literature sweep, or run an experiment, record the choice AND the reasoning as a numbered `docs/NN_*.md`.** Every doc MUST open with a `> **TL;DR & takeaway**` blockquote (human-facing summary; everything below is the agent-facing record). **The index lives in `docs/README.md`** — read it to find docs, add a line when you create one. Keep CLAUDE.md pointers short; don't list individual docs here.

## Future enhancements (not implemented)

Roadmap / parking lot: **`docs/36_roadmap_future_enhancements.md`** — none are in the current pipeline; headline direction = realistic real-time acquisition simulation.
