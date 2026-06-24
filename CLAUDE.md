# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** enable fast real-time free-breathing cine by reconstructing the full 3D heart volume at any target cardiac phase from a *few scattered single-frame-per-slice* acquisitions (ideally one frame/slice), instead of the slow many-frames-per-slice + retrospective-sort/SVR route. No real-time training data exists, so we **simulate** the sparse scattered acquisition from gated breath-hold CMRxRecon2024 cine (each input slice = one frame at an arbitrary (phase t, z-depth)) + motion aug, and aim to generalize to true real-time cine. Currently *only* the scattered sampling + in-plane aug are simulated; realistic acquisition physics (bSSFP transient, single-shot artifacts, respiratory motion) is aspirational — see Future enhancements. **Target inference information contract:** at the one-frame-per-slice extreme the model is assumed to know only `z` per input slice — input cardiac `t` and respiratory `r` are *unavailable* (no ECG / no respiratory device / no self-gating); target-phase *queries* stay free. Design stance, not yet implemented — see `docs/04_inference_information_contract.md`.

**Active pipeline: unsupervised intensity-based, multi-phase** (`mri_volume*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`).

**Target-phase conditioning = REFERENCE SLICE (current default, `mri_volume.yaml`; docs/24, docs/25).** The query is *not* a content-free `target_t` index (that regressed every patient's EF to the cohort mean ~48% — flat-EF, doc 24). Instead **slot 0 is a real target-phase reference slice** at the mid-ventricular plane (`reference_slot=true`), marked via VGGT's **native two-token `camera_token`** (index 0 = first frame = the anchor, index 1 = the rest; `use_reference_token=true`). The model reads the target phase from slot-0's *image content*; `V_gt = phases[t_target]` = that slice's phase. This needs the **aggregator finetune (aggft)** so the camera_token/z_embedder specialize, so `mri_volume.yaml` freezes only `*patch_embed*` + sets `find_unused_parameters=true`. Consequence: you reconstruct **observed** phases (≈ the recoverable limit; doc 25). `use_t_pose_embedding`/`use_target_t_pose_embedding` are OFF; `target_t_indices` is still emitted but **inert**. Confirmation pending a fresh-from-base retrain (`sbatch/train_mri_volume_reference.sh`); the legacy `target_t`-index path survives behind the default-off flags (`mri_finetune.yaml`).

The "**4-day baseline**" (referenced below) is the prior ED-only run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` — 31+ dB PSNR at ED. Post-refactor it's used **weights-only as a warm-start seed** (the sbatch `CKPT_ONLY` default, `strict=false`), **not a true resume**: input normalization + V_gt frame changed, so its memorized codes are stale and the old PSNR won't reproduce — treat the canonical-grid pipeline as a fresh-retrain series.

**Canonical-grid refactor (2026-05-24).** Every subject is resampled to one fixed canonical cube (see "CMR data notes"); input slices and V_gt both live in it. The old supervised-DVF path is **fully removed** — `MRIDataset` no longer reads DVF/mask NIfTIs and `gt_dvfs`/`scale_factors` are gone from the batch. DVF NIfTIs remain on disk for repro; legacy DVF tooling (`compute_cine_dvf_elastix.py`, carmen/elastix verification) plus OCMR/CMRx4DFlow2026 recon explorations and legacy configs/sbatch all live under `_archive/`.

- MRI data: `/scratch/data/CMRxRecon2024/` (symlinked, GPFS)
- Env: `micromamba activate svr`
- SLURM: `spgpu` partition for training (A40 GPUs), `standard` for CPU jobs, account `jjparkcv98`

## Setup

```bash
micromamba activate svr
pip install -e .
pip install -r requirements.txt           # includes monai>=1.4,<1.5
pip install --no-deps -e /home/minsukc/MRI2CT/batchaug/  # GPU aug — see note below
pip install -r requirements_demo.txt       # demos only
```

**monai** is pinned `>=1.4,<1.5` — monai 1.5+ requires torch≥2.4 which would force-upgrade away from torch 2.3.1 (VGGT's frozen build). **batchaug** is not on PyPI; install editable from the MRI2CT clone with `--no-deps` to skip its `triton>=3.0` requirement (our env has triton 2.3.1 bundled with torch 2.3.1; triton 3.x risks breaking torch.compile/inductor paths). At runtime `gpu_aug.py` forces `batchaug.set_backend("pytorch")` so triton is never used. Full rationale is in the comment block at the bottom of `requirements.txt`.

## Training

Entry point: `training/launch.py` (Hydra).

```bash
# Active config
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config mri_volume

# Multi-GPU
PYTHONPATH=training:. torchrun --nproc_per_node=4 training/launch.py --config mri_volume

# Warm-start (weights only, strict=false) from the 4-day baseline — fresh series, not a true resume
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume \
    checkpoint.resume_checkpoint_path=./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt

# ED-only fallback (matches original pre-multi-phase behavior)
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume t_target_fixed=0

# Override
PYTHONPATH=training:. torchrun --nproc_per_node=1 training/launch.py \
    --config mri_volume optim.base_lr=1e-4
```

**Cluster submission**: `bash sbatch/train_mri_volume.sh` — self-submits via embedded `sbatch` call; sets `WANDB_MODE=online`; configurable resume modes inside the script:
- `RESUME_FROM=<exp_dir>` → continue same exp_name + reuse same wandb run id (overwrites old ckpts).
- `CKPT_ONLY=<ckpt_path>` → load model weights only into a **fresh** exp dir + new wandb run. Bumps `max_epochs=500` (keeps the config default `limit_val_batches=200` for multi-phase per-phase coverage; the ED-only `train_mri_volume_ed.sh` is the one that sets `=30`). Default points at the 4-day baseline ckpt for multi-phase fine-tune.

**Configs** (`training/config/`):
- `mri_volume.yaml` — **active** unsupervised intensity pipeline. Inherits `mri_finetune.yaml` via `defaults:` and disables the deprecated DVF loss. Sets `config_name: "mri_volume"` (used as one of the wandb tags).
- `mri_finetune.yaml` — base/parent config (shared optimizer / data / freeze pattern); `mri_volume.yaml` inherits it and is what you actually run. Running `mri_finetune` directly still carries the deprecated supervised point-loss weights (`point.weight=1.0`), not the active intensity pipeline.
- `default.yaml` / `default_dataset.yaml` — templates inherited via `defaults:`.
- Legacy variants (`mri_finetune_*`, `mri_p001_overfit`, `mri_volume_overfit`) and their sbatch scripts now live under `_archive/legacy_configs/` and `_archive/legacy_sbatch/`.

**Key knobs:**
- `max_img_per_gpu: 12` → one slice per slot at S=12. Reduce on OOM.
- `t_target_fixed: null` (default → multi-phase, uniform per train call) | `0` (reproduces ED-only behavior) | any int K (force `t_target=K`).
- `t_target_phases: null` (default → all T phases) | list e.g. `[0,7]` → restrict the multi-phase target pool to that subset (train samples uniformly, val cycles it deterministically). **Mutually exclusive with `t_target_fixed`** (single-phase wins if both set). Used by `sbatch/train_run{6,7,8}_*.sh`.
- `optim.frozen_module_names` — **two regimes.** (1) **Head-only** (`mri_finetune.yaml`, legacy target_t path): `["*patch_embed*", "*camera_token*", "*aggregator*"]` freezes the **entire** aggregator (incl. `z_embedder`/`t_embedder`); only `point_head` trains (~32.65M / 941M) — embedders frozen on purpose (trainable forces backward through all 48 frozen blocks, ~2.5× slower for no proven gain). (2) **aggft** (`mri_volume.yaml` reference default + `mri_volume_bspline/diffusion`): `["*patch_embed*"]` — the attention blocks, `z_embedder`, **`camera_token` (the reference anchor)**, and `point_head` all train (~2.8× slower, ~27 GB/A40). **aggft requires `distributed.find_unused_parameters=true`** (even single-GPU — trainer always DDP-wraps), else DDP crashes (`Expected to have finished reduction in the prior iteration…`) on params with no gradient in the point-only forward (register token, disabled depth/track/camera heads). `tests/test_freeze_pattern.py` guards the `mri_volume` (aggft) contract.
- `model.train_on_residual_dvf: true` → point head outputs Δ; `world_points = scanner_coords + Δ`.
- `logging.filmstrip_every_n_val_epochs: 5` → cadence for the multi-phase cardiac-cycle visualization.
- `data.augmentation.enable: false` (default) | `true` → opt into GPU augmentation. `data.augmentation.tier: conservative|moderate|aggressive`. See "Augmentation" below.

## Volume pipeline (one forward pass)

0. **Preprocess (cached, one-time per subject).** monai `PersistentDataset` resamples all 12 phase NIfTIs to `(1.4, 1.4, 12.0)` mm (Z=12 = CMRx **true slice pitch** = 8 mm thickness + 4 mm gap; the source affine Z=8 mm was the *thickness*, relabeled 8→12 on disk via `tools/relabel_slice_spacing.py` so the resample is a Z-identity — see `docs/18`), crop/zero-pads to `(256, 256, 12)` (geometric center), normalizes intensity against phase_00's 0.5/99.9 percentiles (computed over non-zero FOV voxels, excluding zero-padding), and stacks into one `(T=12, 256, 256, 12)` float16 tensor + a `(256,256,12)` content mask (1=native FOV, 0=zero-pad). Cached on `/tmp/vggt-mri_${USER}_monai_cache/`. Pipeline + custom transforms live in `training/data/preprocess.py`.
1. **Sample.** S ≤ 12 slices. **z is sampled only from within the geometric anatomy bbox** (in-FOV canonical planes) so small-Z subjects don't waste slots on zero-padded planes.
   - **Train:** slot 0 = `(t_target, random in-bbox z)` where `t_target = random.randrange(T)`; slots 1..S-1 = `(random t ≠ t_target, random in-bbox z)`. If `bbox_z_size < S`, z is sampled with replacement.
   - **Val:** `t_target = seq_index % T_total` (stratified). Slots use cyclic-within-bbox diagonal: slot i = `((t_target + i) mod T, bbox_z0 + (i mod bbox_z_size))`. Deterministic across runs (crc32-seeded).
   - **Fixed-phase fallback:** `t_target_fixed=K` overrides → every sample at phase K.
2. **Aggregator.** DINOv2 patch_embed + 24× alternating frame/global attention. The special (camera) token per slot is built from sinusoidal embeddings: `z_embedder(z_norm)` (linear, `z_norm = z/(D-1)*2-1`, D=12) is always on. **Reference default (`mri_volume`):** add the native two-token `camera_token` (slot 0 = anchor, rest = shared) so slot 0 is the target-phase reference; `t_embedder`/`target_t_embedder` OFF. **Legacy path (`mri_finetune`):** add `t_embedder(t_norm)` (cyclic) + `target_t_embedder` instead. Frozen vs aggft per the freeze-pattern note in Key knobs (reference default = aggft, so the aggregator + camera_token TRAIN).
3. **Point head (trainable, DPT).** Outputs per-pixel residual Δ (3 channels) + confidence (1, unused). `world_points = scanner_coords + Δ`, all in normalized [-1, 1].
4. **Splat.** `splat_to_volume(world_points, intensity, (12,256,256))` → `V_canon`. Differentiable trilinear scatter; divides by accumulated coverage (`vggt/utils/splat.py`). **`splat_weight = intensity > 1e-3` is kept** — padded-Z slots are all-zero, and the gate prevents their zero-intensity pixels from diluting V_canon if the model's Δ ever moves them into content planes.
5. **Loss.** `loss_volume = (V_canon - V_gt).abs().mean()` + `0.1 * TV(pos_pred)` — **full-volume L1**, no anatomy mask.

**Input slices:** each canonical `(256, 256)` slice is bilinear-resized to `518×518` for DINOv2 — **no letterbox, no padding** (the canonical slice is already square). `scanner_coords[py, px] = (px/517·2−1, py/517·2−1, z_i/11·2−1)` — a pure geometric mapping, identical for every subject. There is no `-2.0` invalid sentinel anymore; every pixel has a valid canonical coord.

**`V_gt`** = `phases[t_target]` from the cache (canonical frame, batch key `gt_target_volume`). **`anatomy_bbox`** = `(z0,z1,y0,y1,x0,x1)` geometric bbox of the content mask (used to restrict z sampling AND for the bbox metric). Both produced by `MRIDataset.get_data`.

## CMR data notes

Native cine shapes vary: W=256 fixed, H∈{162,204,246}, Z∈{6..14}, T=12. Spacing is **not** uniform: X median 1.3438 (range 1.3438–1.5781), Y median 1.3984 (range 1.3174–1.6423), Z always 8.0 — **but that 8.0 is slice THICKNESS, not center-to-center pitch**: the true pitch is 8 mm + 4 mm gap = **12 mm** (CMRxRecon2024 protocol; `info.csv` has no gap/position field). NIfTI affines were relabeled 8→12 on disk (`docs/18`); the canonical cube uses Z=12 mm. 28 unique spacing tuples across 301 subjects. Native FOV spans X 344–404, Y 215–404, Z 48–112 mm.

`MRIDataset` (`training/data/datasets/mri_dataset.py`) maps every subject onto **one fixed canonical cube**: `(1.4, 1.4, 12.0)` mm spacing, `(256, 256, 12)` voxels, 358.4 × 358.4 × 144 mm extent, `half_extent = (179.2, 179.2, 72.0)` mm. So canonical voxels have the **same physical size for every subject** (was per-subject before the refactor). Subjects with FOV < cube get zero-padded; subjects with FOV > cube get center-cropped (the heart is always near the acquisition center, so cropping loses only periphery). The model only ever sees normalized `[-1,+1]` coordinates.

**Axis-order gotcha:** monai/nibabel store volumes `(X, Y, Z)`; the splat consumes `(D, H, W) = (Z, Y, X)`. The single conversion site is the `permute(0, 3, 2, 1)` in `MRIDataset.get_data` right after the cache lookup — everything downstream is splat-order. Easy to break silently; tests in `test_canonical_invariants.py` guard it.

`mri_mode: "axial"` means **native SAX z-slicing** — not anatomical axial. The slices are short-axis views.

## Augmentation

GPU augmentation via `batchaug` (`training/data/gpu_aug.py`), **off by default** (`data.augmentation.enable: false`), train-only (val never augments). One affine per subject, applied across all 12 T-phases + content mask so cardiac motion stays phase-consistent; the trainer then re-derives `gt_target_volume`, re-extracts input slices at the original (t,z) pairs, and recomputes `anatomy_bbox` (`scanner_coords` need no update — pure geometry). Tiers (in-plane only — no through-plane rotation, no elastic, since Z is 8 mm anisotropic): `conservative` / `moderate` / `aggressive`, escalating affine + photometric (flip, rotate, translate/scale, noise, gamma, bias field). Visual proof: `tools/render_augmentation_examples.py` → `result/augmentation_examples/`.

**Respiratory-motion sim** (`training/data/respiratory.py`) is a SEPARATE toggle (`data.augmentation.respiratory.enable`; **ON by default in the active `mri_volume` config** — the proven resp/z-only recipe, docs/05 — and inherited by `mri_volume_diffusion`/`mri_volume_bspline`; the `mri_finetune` base still defaults it off) and runs *independently* of affine — a per-input-slice deform-then-reslice SI+AP shift (Lujan `sin^{2n}` waveform), applied **after** affine and overwriting **only the input slices**: target / `scanner_coords` / `gt_target_volume` / `anatomy_bbox` / `phases` stay at the unshifted end-expiration reference, so the model learns to **correct** breathing (blind to `r`). Unlike affine, it applies in **both train AND val** — train draws iid per epoch from a private generator (never perturbs the global RNG batchaug/dropout use), val draws **deterministically per `seq_index`** (so val measures the real corrupted→clean task, reproducibly). Per-slot iid, like z/t sampling. `direction_jitter_deg` (default 30°) randomizes the SI direction since SAX stacks are tilted ~20–45° off true SI (`direction_jitter_deg=0` → pure SI+AP). Disabling it ⇒ bit-identical to pre-respiratory. The new `seq_index` batch key (emitted in `mri_dataset.py`, converted in `composed_dataset.py`) carries the val seed. Visual proof: `tools/render_respiratory_examples.py` → `_html/06_respiratory_motion_simulation_examples.html`. Design + literature: `docs/01_respiratory_motion_simulation.md`.

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
Trainable total = ~32.65M / 941M (point_head only) in the legacy head-only freeze; the reference default (`mri_volume`, aggft) also trains the 24×24 attention blocks + z_embedder + camera_token.
```

Checkpoints save the **full 941M state dict** (~3.8 GB each), not just the trainable head. Optimizer + scaler state included.

## Inference / inspection

```python
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
preds = model(images, batch=batch)  # batch needs: z_indices, t_indices, scanner_coords
# To use compute_volume_intensity_loss: batch must also include gt_target_volume (already the t_target phase; t_target itself is only used for per-phase logging, not the loss).
```

Tools:
- `tools/preview_canonical_preprocess.py` — sanity-check the canonical resample on shape-extreme subjects (min/max Z, min/max H, typical); native vs canonical mid-z slice + content-mask + bbox overlay → `result/canonical_preview/`.
- `tools/render_augmentation_examples.py` — per-op + combined aug variant PNGs and a cardiac-cycle GIF → `result/augmentation_examples/`.
- `tools/render_volume_example.py` — random val sample, per-z V_gt/V_canon/diff panel.
- `tools/test_sequential_sampling.py` — diagonal `(t=k+offset, z=k)` for one subject; PNGs to `result/`.
- `baselines/eval_*.py` — identity-Δ / elastix-Δ / carmen-Δ PSNR sweeps over the val set.

## Logging (wandb, project `vggt-mri`)

Metrics carry a `_full` / `_bbox` suffix: `_full` = whole 12×256×256 cube, `_bbox` = subject's geometric content region. Equal for full-FOV subjects; for small-FOV subjects `_full` is inflated by padded zeros — **prefer `metric_psnr_3d_bbox` as the honest number** (SSIM is `_full` only). **Don't compare PSNR across the canonical-grid refactor** — V_gt frame, normalization, and metric defs all changed; treat as a fresh series.

- **Startup:** identity-Δ baseline (Δ=0 splat over val) for full+bbox → `${log_dir}/baseline_identity.json`; baked into the `val_psnr_{full,bbox}/` metric names.
- **Per train visual step (every 100):** `Train_Visuals_Volume` (input/V_gt/V_canon/diff per z) + `Train_Visuals_DVF` (input + per-slot Δx/Δy/Δz).
- **Per val epoch:** `Val_Loss/*`; `Val_Visuals_subj{0,7}_{Volume,DVF}` (subj0 t=0=ED, subj7 t=7≈ES); per-phase `val_psnr_{full,bbox}/t{k}_n{n}_base{b}` + `/mean_n{n_total}` (multi-phase only; `n_total = limit_val_batches`=200 default → ~16-17/phase, since val revisits the 30 subjects at different target phases). `save_val_volumes` (default true) dumps per-subject pred+GT NIfTIs to `${log_dir}/val_volumes/` (~360 MB, overwritten each epoch; rank-0 only under DDP).
- **Every N val epochs (`filmstrip_every_n_val_epochs`, default 5):** `Val_Visuals_cardiac_cycle` (2×12 V_gt/V_canon grid, subj0) + `_gif` (12-frame beating-heart GIF).
- **Tags:** config name + phase mode (`multiphase` / `t{K}`). **Gating:** fixed-phase (`t_target_fixed=K`) skips the per-phase `val_psnr` panels and auto-caps `limit_val_batches` to one deterministic pass over val (more iters = redundant); multi-phase unaffected. All diagnostic logging is `try/except`-wrapped — never raises into training.

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.
- **Monai cache is node-local `/tmp`, rebuilt per job** (`/tmp/vggt-mri_${USER}_monai_cache/`, ~55 MB/subject). Lazy first-epoch rebuild ~3–10 min for ~270 subjects, overlaps GPU compute. Intentionally not on GPFS — cached GPFS reads are ~18–20× slower than /tmp, so persisting would slow every epoch to save one rebuild.

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. The cluster script `sbatch/train_mri_volume.sh` sets `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`, `phase_mode:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`; `OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))`; `OmegaConf.register_new_resolver('phase_mode', lambda t: 'multiphase' if t is None else f't{int(t)}')`.

## Testing

```bash
micromamba run -n svr python -m pytest tests/
```
Synthetic in-memory CMR dataset (`tests/conftest.py`, native W=64, H=60, Z=8, **T=12**) — no real data needed. T=12 matches the canonical pipeline's phase count; each test session gets an isolated monai cache dir (`monai_cache_dir` fixture) so the shared `/tmp` cache isn't polluted. Test files: `test_mri_dataset.py` (dataset contract), `test_preprocess.py` (canonical transforms + geometric bbox), `test_canonical_invariants.py` (cross-subject coord consistency, V_gt zero outside bbox, axis-order), `test_gpu_aug.py` (aug identity passthrough + shape preservation), `test_loss_bbox.py` (bbox vs full metrics), `test_splat.py`, `test_freeze_pattern.py`, `test_trainer_diagnostics.py`, `test_loss.py`, `test_resume.py` (requeue/resume).

## Docs

Research findings, design decisions, and experiment write-ups live in **`docs/`** (numbered,
e.g. `docs/01_respiratory_motion_simulation.md`) — separate from per-version implementation logs
in `version_history/`. **When you make a non-trivial design choice, run a research/literature
sweep, or design/run an experiment, record the choice AND the reasoning (why, sources, rejected
alternatives) as a numbered `docs/NN_*.md`** so future agents can understand *why*, not just
*what*. Keep CLAUDE.md pointers short and link out to the doc for detail.

**Every `docs/NN_*.md` MUST open with a `> **TL;DR & takeaway**` blockquote** before any other
content — a plain-language summary of the conclusion, key decision, and status. This top block is
**human-facing** (the reader skims it and stops); everything below it is the **agent-facing**
detailed record (process, numbers, sources, open questions). Write the TL;DR for someone who will
read *only* it.

**The doc index lives in `docs/README.md`** (one line per doc). Read it to see which doc to open;
add a pointer there when you create a new `docs/NN_*.md`. Don't list individual docs here.

## Future enhancements (not implemented)

Notes for follow-up work. None of these are in the current pipeline.

- **Realistic real-time acquisition simulation (headline direction).** Current "simulation" = scattered single-frame sampling + in-plane aug on clean gated cine. Real transfer needs the physics real-time acquisition imposes: bSSFP transient/contrast, single-shot undersampling artifacts, respiratory + through-plane motion. `SPINER/` + `lixuan_simulation/` (untracked) are starting points.

  - **Respiratory motion simulation** — research + design scoped in `docs/01_respiratory_motion_simulation.md` (literature-validated; not implemented). Gist: per-slice respiratory phase sampled independently of cardiac phase (XCAT two-clocks); rigid SI translation ~10–15 mm along canonical Z + deform-then-reslice the cached `phases` bundle. See the doc for numbers, sources, the correct-vs-condition fork, and reference code (NeSVoR/SVRTK). **MRXCAT/XCAT was evaluated and dropped** (doc §6): MRXCAT only renders MR physics, the motion engine is XCAT's (closed binary; XCAT 3.0's public release is segmentation-only), so there's no portable sim code — we reimplement the simple rigid 6-DOF model ourselves on real cine (path A). The inspection did independently confirm our model + give reference amplitudes.

- **Option B — continuous-phase query.** Add a `target_t_embedder` alongside `TIndexEmbedder` so the model decodes any `t_target ∈ [0,1)`, not just the 12 discrete phases (new embedder + `target_t` batch field + light `point_head`/`register_token` fine-tune).

- **Inference information contract — blind input phase (design stance, `docs/04`).** Target the extreme: at inference the model knows only `z` per input slice; input cardiac `t` and respiratory `r` are assumed **unavailable** (one-frame-per-slice ⇒ no temporal stream ⇒ no self-gating; **no ECG, no respiratory device** assumed, for zero-auxiliary-hardware generality). *Input* phase is independent of *target* phase — `target_t`/`target_r` stay free (chosen queries, sim GT). Cardiac is content-inferable from a slice; respiratory is **not** (cropped SAX hides it) → **pin `target_r` to the reference (4D, correct-not-resolve)**, don't query it. Not yet implemented — current pipeline still conditions on input `t`. See `docs/04_inference_information_contract.md`.

- **Free-breathing extension.** Add a second cyclic Fourier embedder for respiratory phase (~14K params). Blockers: motion-resolved 5D reference (or self-supervised loss), per-slice cardiac+respiratory tags, gated→real-time domain shift.

- **Phase-recovery FALLBACK (if blind underperforms).** If content-inference of input phase isn't enough, recover and feed it: ECG-label cardiac `t` (decoupled clock — works at any sparsity; mild label noise for in-bore MHD), self-gate `t`/`r` from a temporal stream (k-space self-gating / image PCA / manifold), or add a respiratory navigator/bellows for true 5D. Implement the bridge as **input-phase dropout (+ noise)** so one model uses phase when present and content-infers when absent.

- **Fully unsupervised — drop `gt_target_volume`.** Current loss still uses the per-phase NIfTI (itself a derived recon). True self-supervision: sample `V_canon` at input world coords, compare against the input slice intensities. Risk: degenerate `V_canon≈0` — needs a coverage/completeness or stronger smoothness term.

- **UNet refiner on splat.** Small 3D UNet after the splat to inpaint low-coverage voxels + smooth seam artifacts. Drop-in; doubles as an ablation for splat-artifact vs. motion-prediction error.

- **UNet ablation — replace the splat.** Regress `V_canon` directly from features (no splat/coverage division). Loses splat interpretability for more capacity; run head-to-head to test whether explicit splatting helps.

- **Tagging data as in-plane motion validation.** `ChallengeData/Tagging` and `ChallengeData_AfterCompetition/Tagging` contain SAX tagging k-space for ~143/194 training subjects — same patients, same SAX slice geometry, same ECG-gated session as Cine. Tagging embeds a grid pattern in the tissue; tracking grid points across the 26 cardiac phases yields a dense 2D displacement field `(dx, dy)` per SAX slice — explicit GT for in-plane myocardial motion. Limitation: tagging is fundamentally 2D (grid is in-plane only), so it gives zero supervision on `dz` (through-plane), which is the hardest and most project-critical component. Pipeline: reconstruct tagging images from k-space (same `batch_reconstruct_cmrxrecon2024.py` approach), then run a grid-tracking method (e.g. HARP, SinMod, or optical flow) to extract `(dx,dy,t)` per slice. Use as a validation metric for the point head's in-plane `Δ` components on matched subjects.
