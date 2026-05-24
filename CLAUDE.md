# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** learn an unsupervised model that takes S=12 scattered 2D cine slices at arbitrary (cardiac phase t, z-position) pairs and reconstructs the full 3D volume at any chosen target phase. Long-term direction: extend to free-breathing / ungated real-time MR (see Future enhancements).

**Active pipeline: unsupervised intensity-based, multi-phase** (`mri_volume*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`). The model can query any of the 12 discrete training phases — discrete-only; continuous-phase query (Option B) is not implemented (see Future enhancements).

The "**4-day baseline**" referenced below is the prior production run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` (199 epochs over ~4 days, ED-only `t_target=0` always, achieved PSNR 31+ dB at ED). **Archive-only after the canonical-grid refactor** — it can't be resumed from because the input normalization (scanner_coords scale) and V_gt frame changed; the point head's memorized codes don't transfer. Treat the canonical-grid pipeline as a fresh-retrain series.

**Canonical-grid refactor (2026-05-24).** Every subject is now resampled to a fixed `(1.4, 1.4, 8.0)` mm spacing and cropped/zero-padded to `(256, 256, 12)` voxels (geometric-center aligned), so the canonical `[-1,+1]³` cube means the *same physical thing* for every subject (358.4 × 358.4 × 96 mm). Preprocessing runs once per subject via a monai `PersistentDataset` cached on `/tmp`; input slices and V_gt both live in this canonical frame. This replaced the old per-subject per-axis normalization (where canonical voxels had different physical sizes per subject). See "CMR data notes".

The old supervised-DVF pipeline is **fully removed** from the live data/loss path: `MRIDataset` no longer reads `dvf_elastix/dvf_frame_*.nii.gz` or `mask_frame_00.nii.gz`, and `gt_dvfs` / `scale_factors` are no longer in the batch. The DVF NIfTIs still sit on disk for reproducibility and `compute_cine_dvf_elastix.py` is kept around to regenerate them, but nothing in training consumes them. Legacy configs/sbatch scripts that used them live under `_archive/`.

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

# Resume from the 4-day baseline checkpoint (multi-phase fine-tune)
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
- `CKPT_ONLY=<ckpt_path>` → load model weights only into a **fresh** exp dir + new wandb run. Bumps `max_epochs=500` and sets `limit_val_batches=30` automatically. Default points at the 4-day baseline ckpt for multi-phase fine-tune.

**Configs** (`training/config/`):
- `mri_volume.yaml` — **active** unsupervised intensity pipeline. Inherits `mri_finetune.yaml` via `defaults:` and disables the deprecated DVF loss. Sets `config_name: "mri_volume"` (used as one of the wandb tags).
- `mri_finetune.yaml` — base config (shared optimizer / data / freeze pattern); used as parent by `mri_volume.yaml`. Not runnable standalone — the supervised pipeline that did so is fully removed.
- `default.yaml` / `default_dataset.yaml` — templates inherited via `defaults:`.
- Legacy variants (`mri_finetune_*`, `mri_p001_overfit`, `mri_volume_overfit`) and their sbatch scripts now live under `_archive/legacy_configs/` and `_archive/legacy_sbatch/`.

**Key knobs:**
- `max_img_per_gpu: 12` → one slice per slot at S=12. Reduce on OOM.
- `t_target_fixed: null` (default → multi-phase, uniform per train call) | `0` (reproduces ED-only behavior) | any int K (force `t_target=K`).
- `optim.frozen_module_names: ["*patch_embed*", "*camera_token*", "*aggregator*"]` — wildcard freezes the **entire** aggregator subtree, **including `z_embedder` and `t_embedder`**. Only `point_head` trains (~32.65M / 941M params). The z/t Fourier projections stay at their (random-init or resumed) values; point_head memorizes the codes. **This is intentional** — making the embedders trainable while keeping the rest of the aggregator frozen makes backward 2.5× slower (gradient must traverse all 48 frozen attention blocks via gradient checkpointing recomputation to reach the embedders; A/B-measured 1.32 → 3.25 sec/step). The 4-day baseline ran with frozen-random embedders and reached 31+ dB PSNR — trainable embedders aren't proven to help quality. If you ever unfreeze the aggregator's attention blocks (Option B, free-breathing, etc.), the backward already traverses them, so the embedders become free to unfreeze too — at that point, enumerate the subparts explicitly and let them learn. `tests/test_freeze_pattern.py` guards the current contract. **Gotcha if you unfreeze:** an aggregator-finetune (e.g. `optim.frozen_module_names='[*patch_embed*]'`) **crashes under DDP** with `Expected to have finished reduction in the prior iteration…` because unfreezing exposes params that get no gradient in the point-only forward (camera_token, register_token, disabled depth/track heads). Set **`distributed.find_unused_parameters=true`** to fix it. This is required even on a **single GPU** — the trainer always wraps the model in DDP, so the unused-param check fires regardless of world size. Verified: with the flag, 637.4M params train (Grad/aggregator nonzero), ~27 GB on an A40, ~2.8× slower than head-only (~4.4 vs ~1.6 sec/step).
- `model.train_on_residual_dvf: true` → point head outputs Δ; `world_points = scanner_coords + Δ`.
- `logging.filmstrip_every_n_val_epochs: 5` → cadence for the multi-phase cardiac-cycle visualization.
- `data.augmentation.enable: false` (default) | `true` → opt into GPU augmentation. `data.augmentation.tier: "conservative"` (only tier implemented). See "Augmentation" below.

## Volume pipeline (one forward pass)

0. **Preprocess (cached, one-time per subject).** monai `PersistentDataset` resamples all 12 phase NIfTIs to `(1.4, 1.4, 8.0)` mm, crop/zero-pads to `(256, 256, 12)` (geometric center), normalizes intensity against phase_00's 1/99.5 percentiles, and stacks into one `(T=12, 256, 256, 12)` float16 tensor + a `(256,256,12)` content mask (1=native FOV, 0=zero-pad). Cached on `/tmp/vggt-mri_${USER}_monai_cache/`. Pipeline + custom transforms live in `training/data/preprocess.py`.
1. **Sample.** S ≤ 12 slices. **z is sampled only from within the geometric anatomy bbox** (in-FOV canonical planes) so small-Z subjects don't waste slots on zero-padded planes.
   - **Train:** slot 0 = `(t_target, random in-bbox z)` where `t_target = random.randrange(T)`; slots 1..S-1 = `(random t ≠ t_target, random in-bbox z)`. If `bbox_z_size < S`, z is sampled with replacement.
   - **Val:** `t_target = seq_index % T_total` (stratified). Slots use cyclic-within-bbox diagonal: slot i = `((t_target + i) mod T, bbox_z0 + (i mod bbox_z_size))`. Deterministic across runs (crc32-seeded).
   - **Fixed-phase fallback:** `t_target_fixed=K` overrides → every sample at phase K.
2. **Aggregator (frozen).** DINOv2 patch_embed + 24× alternating frame/global attention. Replaces the camera token with `z_embedder(z_norm) + t_embedder(t_norm)` — sinusoidal Fourier. `t_embedder` is **cyclic** (`t_norm = t/T * 2 - 1`); `z_embedder` is linear (`z_norm = z/(D-1) * 2 - 1`, D=12 canonical). **Everything in the aggregator is frozen** — see freeze-pattern note in Key knobs.
3. **Point head (trainable, DPT).** Outputs per-pixel residual Δ (3 channels) + confidence (1, unused). `world_points = scanner_coords + Δ`, all in normalized [-1, 1].
4. **Splat.** `splat_to_volume(world_points, intensity, (12,256,256))` → `V_canon`. Differentiable trilinear scatter; divides by accumulated coverage (`vggt/utils/splat.py`). **`splat_weight = intensity > 1e-3` is kept** — padded-Z slots are all-zero, and the gate prevents their zero-intensity pixels from diluting V_canon if the model's Δ ever moves them into content planes.
5. **Loss.** `loss_volume = (V_canon - V_gt).abs().mean()` + `0.1 * TV(pos_pred)` — **full-volume L1**, no anatomy mask.

**Input slices:** each canonical `(256, 256)` slice is bilinear-resized to `518×518` for DINOv2 — **no letterbox, no padding** (the canonical slice is already square). `scanner_coords[py, px] = (px/517·2−1, py/517·2−1, z_i/11·2−1)` — a pure geometric mapping, identical for every subject. There is no `-2.0` invalid sentinel anymore; every pixel has a valid canonical coord.

**`V_gt`** = `phases[t_target]` from the cache (canonical frame, batch key `gt_target_volume`). **`anatomy_bbox`** = `(z0,z1,y0,y1,x0,x1)` geometric bbox of the content mask (used to restrict z sampling AND for the bbox metric). Both produced by `MRIDataset.get_data`.

## CMR data notes

Native cine shapes vary: W=256 fixed, H∈{162,204,246}, Z∈{6..14}, T=12. Spacing is **not** uniform: X median 1.3438 (range 1.3438–1.5781), Y median 1.3984 (range 1.3174–1.6423), Z always 8.0 (28 unique spacing tuples across 301 subjects). Native FOV spans X 344–404, Y 215–404, Z 48–112 mm.

`MRIDataset` (`training/data/datasets/mri_dataset.py`) maps every subject onto **one fixed canonical cube**: `(1.4, 1.4, 8.0)` mm spacing, `(256, 256, 12)` voxels, 358.4 × 358.4 × 96 mm extent, `half_extent = (179.2, 179.2, 48.0)` mm. So canonical voxels have the **same physical size for every subject** (was per-subject before the refactor). Subjects with FOV < cube get zero-padded; subjects with FOV > cube get center-cropped (the heart is always near the acquisition center, so cropping loses only periphery). The model only ever sees normalized `[-1,+1]` coordinates.

**Axis-order gotcha:** monai/nibabel store volumes `(X, Y, Z)`; the splat consumes `(D, H, W) = (Z, Y, X)`. The single conversion site is the `permute(0, 3, 2, 1)` in `MRIDataset.get_data` right after the cache lookup — everything downstream is splat-order. Easy to break silently; tests in `test_canonical_invariants.py` guard it.

`mri_mode: "axial"` means **native SAX z-slicing** — not anatomical axial. The slices are short-axis views.

## Augmentation

GPU augmentation via `batchaug` (`training/data/gpu_aug.py`), **off by default** (`data.augmentation.enable: false`). When enabled, applied in the trainer between `copy_data_to_device` and the model forward, train-only (val never augments). One affine is sampled per subject and applied across all 12 T-phases (T as channel) AND the content mask, so cardiac motion stays consistent across phases. After aug the trainer re-derives `gt_target_volume = phases_aug[t_target]`, re-extracts input slices at the original (t,z) pairs, and recomputes `anatomy_bbox` from the augmented mask. `scanner_coords` need no update (pure geometry). Conservative tier: in-plane H-flip, ±5° in-plane rotate, small translate/scale, gaussian noise, gamma, mild bias field — no through-plane rotation, no elastic. Visual proof: `tools/render_augmentation_examples.py` → `result/augmentation_examples/` (per-op PNGs + a cardiac-cycle GIF confirming cross-phase consistency).

## Architecture

```
VGGT (vggt/models/vggt.py) — ~941M total, base weights at ./scratch/torch_cache/model.pt
├── Aggregator
│   ├── DINOv2 patch_embed (518² inputs, patch=14 → 37² tokens)    [FROZEN, ~304M]
│   ├── 24× frame_blocks + 24× global_blocks (alternating attn)    [FROZEN, ~605M]
│   ├── rope / camera_token / register_token                        [FROZEN, ~10K]
│   └── ZIndexEmbedder, TIndexEmbedder (sinusoidal Fourier)         [FROZEN, ~28K — see Key knobs]
└── point_head — DPT upsampler → 4-channel (Δ, conf)                [TRAINABLE, ~32.65M]

Camera / depth / track heads disabled in mri_volume config.
Trainable total = ~32.65M / 941M (only point_head).
```

Checkpoints save the **full 941M state dict** (~3.8 GB each), not just the trainable head. Optimizer + scaler state included.

## Inference / inspection

```python
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained("facebook/VGGT-1B").cuda().eval()
preds = model(images, batch=batch)  # batch needs: z_indices, t_indices, scanner_coords
# To use compute_volume_intensity_loss: batch must also include gt_target_volume + t_target.
```

Tools:
- `tools/preview_canonical_preprocess.py` — sanity-check the canonical resample on shape-extreme subjects (min/max Z, min/max H, typical); native vs canonical mid-z slice + content-mask + bbox overlay → `result/canonical_preview/`.
- `tools/render_augmentation_examples.py` — per-op + combined aug variant PNGs and a cardiac-cycle GIF → `result/augmentation_examples/`.
- `tools/render_volume_example.py` — random val sample, per-z V_gt/V_canon/diff panel.
- `tools/test_sequential_sampling.py` — diagonal `(t=k+offset, z=k)` for one subject; PNGs to `result/`.
- `baselines/eval_*.py` — identity-Δ / elastix-Δ / carmen-Δ PSNR sweeps over the val set.

## Logging (wandb, project `vggt-mri`)

**At trainer startup:** identity-Δ baseline (splat `scanner_coords` directly with Δ=0 across the val set) is computed for BOTH the full-volume and bbox PSNR and persisted to `${log_dir}/baseline_identity.json` (`{full: {...}, bbox: {...}}`). The per-phase values are baked into the `val_psnr_full/` and `val_psnr_bbox/` metric names (see below).

**Every train visual step (`log_visual_frequency.train = 100`):**
- `Train_Visuals_Volume` — 4-row × (max S,D)+1 col grid: input slices, V_gt, V_canon, signed diff (per z, ±0.05).
- `Train_Visuals_DVF` — 4-row × S+1 col grid: input intensity + per-slot Δx/Δy/Δz (±0.05).

**Per val epoch:**
- `Val_Loss/*` — averaged val metrics. Metric names are now `_full` / `_bbox` suffixed: `metric_psnr_3d_full` (whole cube), `metric_psnr_3d_bbox` (subject's geometric content region only), likewise `mae`/`mse`. `metric_ssim_3d_full` only (bbox SSIM deferred — variable per-sample shape). The full and bbox metrics are **equal for full-FOV subjects** (bbox = cube) and **diverge for small-FOV subjects** (bbox excludes the padded zeros that inflate full PSNR). Both logged so they cross-check.
- `Val_Visuals_subj{0,7}_Volume` and `_DVF` — two fixed val subjects at diagnostic phases (subj0 t=0 = ED, subj7 t=7 ≈ ES population median). The cardiac-cycle filmstrip shows all 12 phases for subj0.
- `val_psnr_full/t{k}_n{n}_base{b:.1f}` and `val_psnr_bbox/t{k}_n{n}_base{b:.1f}` — **two parallel namespaces**, per-phase val PSNR (multi-phase mode only). `n` = sample count for phase k, `b` = identity-Δ baseline for that phase/metric. Deterministic stratified val → n constant per phase across runs.
- `val_psnr_full/mean_n{n_total}_base{b:.1f}` and `val_psnr_bbox/mean_n{n_total}_base{b:.1f}` — aggregate across the val loop. **`n_total = limit_val_batches`** (200 by default), NOT 30: `MRIDataset.__len__` returns 1000 for val too, so the loop runs `limit_val_batches` iterations with `subj_idx = seq_index % 30` and `t_target = seq_index % 12` — each of the 30 subjects is revisited ~6–7× but at *different* target phases each time (lcm(12,30)=60 period). So per-phase `n ≈ limit_val_batches/12 ≈ 16–17`. The identity baseline, by contrast, iterates each subject once (`len(subjects)=30`).
- **Migration note:** the old un-suffixed `val_psnr/*` and `metric_psnr_3d` keyspaces are gone — wandb dashboards built on them break across the canonical-grid refactor.

**Every N val epochs** (`logging.filmstrip_every_n_val_epochs`, default 5):
- `Val_Visuals_cardiac_cycle` — 2×12 grid: V_gt (top) and V_canon (bot) reconstructed at all 12 cardiac phases for val subject 0, mid-z slice. The qualitative proof of multi-phase reconstruction.
- `Val_Visuals_cardiac_cycle_gif` — same content as the still strip, but as a 12-frame animated GIF (4 fps) so the heart actually beats. Each frame is V_gt | V_canon side-by-side (horizontal), shared intensity scale. Small (~50–200 KB per gif, ~10 MB total per run).

**Every val epoch** (`logging.save_val_volumes`, default `true`):
- `${log_dir}/val_volumes/subj{idx:02d}_t{t_target:02d}_{subject}_{pred|gt}.nii.gz` — per-subject predicted V_canon + GT V_gt as NIfTI, identity affine in canonical voxel space (12×256×256), overwritten in place every val epoch. ~360 MB constant footprint for 30 val subjects. Single-GPU only — under DDP, only rank 0's subjects are saved. Set `save_val_volumes: false` to disable.

**Wandb tags.** Two tags applied per run via `logging.wandb_writer.tags`: the active config name (`mri_volume`) and the phase mode (`multiphase` when `t_target_fixed` is null, else `t{K}` for fixed phase K). The `phase_mode` resolver is registered in `training/launch.py`.

**Gating.** In `t_target_fixed=K` mode (single-phase), `val_psnr_full/*` and `val_psnr_bbox/*` panels are skipped (meaningless when all samples share one phase). The cardiac-cycle filmstrip + GIF run in both modes. All diagnostic logging is wrapped in `try/except` — a failure logs a warning but never raises into training.

**Fixed-phase val auto-caps `limit_val_batches`.** In `t_target_fixed=K` mode, `val_epoch` caps `limit_val_batches` to the number of val subjects (one deterministic pass over the set). Beyond one pass the val loop only re-evaluates byte-identical `(subject, phase)` samples (val is deterministic), so iterating to the default 200 would be pure redundant compute. No need to pass `limit_val_batches` for fixed-phase runs — it self-sizes. **Multi-phase runs are unaffected** (there, each iter hits a different target phase, so more iters = genuine per-phase coverage, not redundancy — CKPT_ONLY in `sbatch/train_mri_volume.sh` uses the config default `limit_val_batches=200` → ~16-17 val samples per phase). Val sampling (subject, `t_target=seq%T`, z/t slot diagonals) is deterministic across epochs **and** runs (`shuffle=False` + zero random calls in the val path); empirically verified identical across separate processes.

**PSNR caveat.** Don't compare across the canonical-grid refactor: the V_gt frame, normalization, and metric definitions all changed. Treat it as a fresh series. Within the new series, prefer `metric_psnr_3d_bbox` as the honest number (full-volume PSNR is inflated by padded zeros for small-FOV subjects).

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.
- **Monai cache is node-local `/tmp` and rebuilt per job.** The canonical preprocessing cache (step 0; ~55 MB/subject) lives on `/tmp/vggt-mri_${USER}_monai_cache/`, which is wiped per node. The dataloader rebuilds it lazily on the first epoch — a one-time cost of ~3–10 min for the run's ~270 subjects (4 workers; longer under GPFS contention), which overlaps GPU compute. This is intentionally not persisted to GPFS: cached reads off GPFS are ~18–20× slower than /tmp, so a persistent cache would slow every epoch to save a one-time few-minute rebuild — not worth it.

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. The cluster script `sbatch/train_mri_volume.sh` sets `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`, `phase_mode:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`; `OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))`; `OmegaConf.register_new_resolver('phase_mode', lambda t: 'multiphase' if t is None else f't{int(t)}')`.

## Testing

```bash
micromamba run -n svr python -m pytest tests/
```
Synthetic in-memory CMR dataset (`tests/conftest.py`, native W=64, H=60, Z=8, **T=12**) — no real data needed. T=12 matches the canonical pipeline's phase count; each test session gets an isolated monai cache dir (`monai_cache_dir` fixture) so the shared `/tmp` cache isn't polluted. Test files: `test_mri_dataset.py` (dataset contract), `test_preprocess.py` (canonical transforms + geometric bbox), `test_canonical_invariants.py` (cross-subject coord consistency, V_gt zero outside bbox, axis-order), `test_gpu_aug.py` (aug identity passthrough + shape preservation), `test_loss_bbox.py` (bbox vs full metrics), `test_splat.py`, `test_freeze_pattern.py`, `test_trainer_diagnostics.py`.

## Future enhancements (not implemented)

Notes for follow-up work. None of these are in the current pipeline.

- **Option B — continuous-phase query.** Add an explicit `target_t_embedder` alongside `TIndexEmbedder` in `vggt/models/aggregator.py` to let the model decode any `t_target ∈ [0, 1)`, not just the discrete training phases. Requires a new sinusoidal embedder, `target_t` broadcast as a batch field consumed in `aggregator.forward`, and a light fine-tune of `point_head` + `register_token`. Option A (current pipeline) only covers discrete-phase queries.

- **Free-breathing extension.** Add a second cyclic Fourier embedder for respiratory phase (analogous to `TIndexEmbedder`). The architecture scales trivially (~14K extra params per added phase). Bigger blockers: (1) need a motion-resolved 5D reference for supervision or move to self-supervised reconstruction loss; (2) need cardiac + respiratory phase tags per slice; (3) domain shift from breath-hold gated cines to real-time acquisitions.

- **Ungated extension.** Phase must be recovered, not measured. Standard solution: k-space center self-gating, image-based PCA, or manifold learning preprocessing producing `(cardiac_phase, respiratory_phase)` per slice. Pipeline downstream is unchanged — phases plug in exactly like gated data. To harden against estimation noise, add `c_norm += noise` augmentation during training. A research-level extension would replace preprocessing with a learned auxiliary phase head.

- **Fully unsupervised — drop `gt_target_volume`.** Current "unsupervised" pipeline still depends on the per-phase NIfTI as `V_gt`, which is itself a derived reconstruction. True self-supervision: sample `V_canon` at the input slices' world coordinates and compare against the **input slice intensities themselves** — input slices become their own supervision. Removes one layer of indirection and aligns with the free-breathing/ungated regime (where no canonical GT exists). Risk: degenerate solutions (`V_canon = 0` everywhere except at sampled pixels) — needs a coverage-based completeness term or a stronger TV/smoothness prior. Loss becomes `|sample(V_canon, input_world_coords) - input_intensities|`.

- **UNet refiner on top of splat.** Add a small 3D UNet after the splat: `V_canon (12×256×256) → UNet → V_canon' (12×256×256)`. Learns to inpaint low-coverage voxels and smooth out splat artifacts (especially the seams between slices at native Z resolution). Drop-in addition — doesn't touch the splat or the point head. Useful as an ablation to quantify how much of the loss is due to splat artifacts vs. genuine motion-prediction error.

- **UNet ablation — replace the splat entirely.** Skip the differentiable splat + coverage division; regress `V_canon` directly from input features with a learned 3D head. Loses the physical interpretability of the splat ("this voxel is the trilinear average of these N pixels") in exchange for more model capacity. Worth running as a head-to-head against the current splat-based pipeline to see whether the inductive bias of explicit splatting is helping or hurting.
