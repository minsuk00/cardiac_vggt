# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** learn an unsupervised model that takes S=12 scattered 2D cine slices at arbitrary (cardiac phase t, z-position) pairs and reconstructs the full 3D volume at any chosen target phase. Long-term direction: extend to free-breathing / ungated real-time MR (see Future enhancements).

**Active pipeline: unsupervised intensity-based, multi-phase** (`mri_volume*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`). The model can query any of the 12 discrete training phases — discrete-only; continuous-phase query (Option B) is not implemented (see Future enhancements).

The "**4-day baseline**" referenced below is the prior production run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` (199 epochs over ~4 days, ED-only `t_target=0` always, achieved PSNR 31+ dB at ED). All multi-phase fine-tuning currently warm-starts from its `ckpts/checkpoint_last.pt`.

The old supervised-DVF pipeline (`compute_cine_dvf_elastix.py`, cardiac-mask logic) is **deprecated** — kept for reproducibility, don't extend.

- MRI data: `/scratch/data/CMRxRecon2024/` (symlinked, GPFS)
- Env: `micromamba activate svr`
- SLURM: `spgpu` partition for training (A40 GPUs), `standard` for CPU jobs, account `jjparkcv98`

## Setup

```bash
micromamba activate svr
pip install -e .
pip install -r requirements.txt
pip install -r requirements_demo.txt  # demos only
```

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
- `mri_volume.yaml` — **active** unsupervised intensity pipeline. Inherits `mri_finetune.yaml` via `defaults:` and disables the deprecated DVF loss.
- `mri_finetune.yaml` — base config (shared optimizer / data / freeze pattern); used as parent by `mri_volume.yaml`. The legacy supervised pipeline that ran on this config standalone is **deprecated** — don't run `mri_finetune` directly.
- `mri_volume_overfit.yaml` — single-subject sanity check.
- `mri_finetune*.yaml` (other variants) — old supervised DVF experiments (deprecated).
- `default.yaml` — original Co3D.

**Key knobs:**
- `max_img_per_gpu: 12` → one slice per slot at S=12. Reduce on OOM.
- `t_target_fixed: null` (default → multi-phase, uniform per train call) | `0` (reproduces ED-only behavior) | any int K (force `t_target=K`).
- `optim.frozen_module_names: ["*patch_embed*", "*camera_token*", "*aggregator*"]` — wildcard freezes the **entire** aggregator subtree, **including `z_embedder` and `t_embedder`**. Only `point_head` trains (~32.65M / 941M params). The z/t Fourier projections stay at their (random-init or resumed) values; point_head memorizes the codes. **This is intentional** — making the embedders trainable while keeping the rest of the aggregator frozen makes backward 2.5× slower (gradient must traverse all 48 frozen attention blocks via gradient checkpointing recomputation to reach the embedders; A/B-measured 1.32 → 3.25 sec/step). The 4-day baseline ran with frozen-random embedders and reached 31+ dB PSNR — trainable embedders aren't proven to help quality. If you ever unfreeze the aggregator's attention blocks (Option B, free-breathing, etc.), the backward already traverses them, so the embedders become free to unfreeze too — at that point, enumerate the subparts explicitly and let them learn. `tests/test_freeze_pattern.py` guards the current contract.
- `model.train_on_residual_dvf: true` → point head outputs Δ; `world_points = scanner_coords + Δ`.
- `logging.filmstrip_every_n_val_epochs: 5` → cadence for the multi-phase cardiac-cycle visualization.

## Volume pipeline (one forward pass)

1. **Sample.** S ≤ 12 slices.
   - **Train:** slot 0 = `(t_target, random z)` where `t_target = random.randrange(T)`; slots 1..S-1 = `(random t ≠ t_target, distinct random z)`.
   - **Val:** `t_target = seq_index % T_total` (stratified — ~2–3 val subjects per phase across 30 subjects). Slots use **diagonal** acquisition: slot i = `((t_target + i) mod T, z=i)` — mimics real-life sequential slice acquisition. Deterministic across runs (crc32-seeded; no `PYTHONHASHSEED` dependency).
   - **Fixed-phase fallback:** `t_target_fixed=K` overrides both → every sample at phase K.
2. **Aggregator (frozen).** DINOv2 patch_embed + 24× alternating frame/global attention. Replaces the camera token with `z_embedder(z_norm) + t_embedder(t_norm)` — sinusoidal Fourier on per-axis normalized indices. `t_embedder` is **cyclic** (`t_norm = t/T * 2 - 1` — divides by T, not T-1, so wrap point sits at +1 outside the data); `z_embedder` is linear. **Everything in the aggregator (incl. z/t Fourier projections) is frozen** — see freeze-pattern note in Key knobs for why.
3. **Point head (trainable, DPT).** Outputs per-pixel residual Δ (3 channels) + confidence (1, unused). `world_points = scanner_coords + Δ`, all in normalized [-1, 1].
4. **Splat.** `splat_to_volume(world_points, intensity, (12,256,256))` → `V_canon`. Differentiable trilinear scatter; divides by accumulated coverage (`vggt/utils/splat.py`). Mask B: zero-intensity (padding) pixels get `splat_weight=0` so they don't dilute V_canon.
5. **Loss.** `loss_volume = (V_canon - V_gt).abs().mean()` + `0.1 * TV(pos_pred)` — **full-volume L1**, no anatomy mask. Anatomy-masked variant is commented in `loss.py` (kept giving free-pass over-prediction outside the heart; switched on user request).

`V_gt` is the on-disk NIfTI at `t_target` resampled to (12, 256, 256) once per sample via `scipy.ndimage.map_coordinates(order=1, mode="nearest")` in the same per-axis normalized frame as `V_canon`. Batch key: `gt_target_volume` (was `gt_phase0_volume` in the ED-only pipeline; renamed).

## CMR data notes

Native cine shapes vary: W=256 fixed, H∈{162,204,246}, Z∈{8..12}, T=12, spacing ≈ (1.34, 1.34, 8.0) mm.

`MRIDataset.get_data` (in `training/data/datasets/mri_dataset.py`) does **per-axis normalization** — each subject's physical FOV maps independently onto its own `[-1,1]³` cube. The canonical 12×256×256 grid has the same shape for every subject but different physical voxel sizes. The model only ever sees normalized coordinates.

`mri_mode: "axial"` means **native SAX z-slicing** (`vol[:, :, idx]`) — not anatomical axial. The slices are short-axis views.

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
- `tools/render_volume_example.py` — random val sample, per-z V_gt/V_canon/diff panel.
- `tools/test_sequential_sampling.py` — diagonal `(t=k+offset, z=k)` for one subject; PNGs to `result/`.
- `baselines/eval_*.py` — identity-Δ / elastix-Δ / carmen-Δ PSNR sweeps over the val set.

## Logging (wandb, project `vggt-mri`)

**At trainer startup (logged once at step 0):**
- `baseline/identity_PSNR_mean` — splat scanner_coords directly (Δ=0) across the val set; gives a "no motion correction" reference.
- `baseline/identity_PSNR_t0 .. _t11` — per-phase identity baseline (multi-phase mode only).
- Persisted to `${log_dir}/baseline_identity.json`.

**Every train visual step (`log_visual_frequency.train = 100`):**
- `Train_Visuals_Volume` — 4-row × (max S,D)+1 col grid: input slices, V_gt, V_canon, signed diff (per z, ±0.05).
- `Train_Visuals_DVF` — 4-row × S+1 col grid: input intensity + per-slot Δx/Δy/Δz (±0.05).

**Per val epoch:**
- `Val_Loss/*` — averaged val metrics (loss_volume, psnr_3d, ssim_3d, mae, …).
- `Val_Visuals_subj{0,1,2}_Volume` and `_DVF` — three fixed val subjects (subjects 0, 1, 2 deterministically). Distinct wandb keys so they don't overwrite.
- `per_phase/PSNR_t{0..11}` and `per_phase/n_t{0..11}` — per-phase val PSNR + sample counts (multi-phase mode only). With stratified sampling, n_t = 3 for t=0..5 and 2 for t=6..11.

**Every N val epochs** (`logging.filmstrip_every_n_val_epochs`, default 5):
- `Val_Visuals_cardiac_cycle` — 2×12 grid: V_gt (top) and V_canon (bot) reconstructed at all 12 cardiac phases for val subject 0. The qualitative proof of multi-phase reconstruction.

**Gating.** In `t_target_fixed=K` mode (single-phase), `per_phase/*` and `Val_Visuals_cardiac_cycle` are skipped (meaningless when all samples share one phase); the mean baseline is still logged. All diagnostic logging is wrapped in `try/except` — a failure logs a warning but never raises into training.

**PSNR caveat.** Pre-loss-switch logs reported anatomy-masked PSNR (over `V_gt > 1e-3`); current `compute_volume_intensity_loss` returns full-volume PSNR. Same model, ~7–9 dB lower number — don't compare across the switchover. Likewise, multi-phase val PSNR averages across all 12 phases and will be ~1–3 dB lower than the old ED-only val PSNR.

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. Cluster scripts in `sbatch/train_mri_volume*.sh` set `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`.

## Testing

```bash
micromamba run -n svr python -m pytest tests/
```
Synthetic in-memory CMR dataset (`tests/conftest.py`, W=32, H=30, Z=4) — no real data needed.

## Future enhancements (not implemented)

Notes for follow-up work. None of these are in the current pipeline.

- **Option B — continuous-phase query.** Add an explicit `target_t_embedder` alongside `TIndexEmbedder` in `vggt/models/aggregator.py` to let the model decode any `t_target ∈ [0, 1)`, not just the discrete training phases. Requires a new sinusoidal embedder, `target_t` broadcast as a batch field consumed in `aggregator.forward`, and a light fine-tune of `point_head` + `register_token`. Option A (current pipeline) only covers discrete-phase queries.

- **Free-breathing extension.** Add a second cyclic Fourier embedder for respiratory phase (analogous to `TIndexEmbedder`). The architecture scales trivially (~14K extra params per added phase). Bigger blockers: (1) need a motion-resolved 5D reference for supervision or move to self-supervised reconstruction loss; (2) need cardiac + respiratory phase tags per slice; (3) domain shift from breath-hold gated cines to real-time acquisitions.

- **Ungated extension.** Phase must be recovered, not measured. Standard solution: k-space center self-gating, image-based PCA, or manifold learning preprocessing producing `(cardiac_phase, respiratory_phase)` per slice. Pipeline downstream is unchanged — phases plug in exactly like gated data. To harden against estimation noise, add `c_norm += noise` augmentation during training. A research-level extension would replace preprocessing with a learned auxiliary phase head.
