# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

VGGT (Visual Geometry Grounded Transformer, CVPR 2025) adapted for **cardiac 4D MRI slice-to-volume reconstruction** on CMRxRecon2024 (`Cine_combined`, 301 subjects split 240/30/31 train/val/test via `training/splits/random_8_1_1.txt`).

**Research goal:** learn an unsupervised model that takes S=12 scattered 2D cine slices at arbitrary (cardiac phase t, z-position) pairs and reconstructs the full 3D volume at any chosen target phase. Long-term direction: extend to free-breathing / ungated real-time MR (see Future enhancements).

**Active pipeline: unsupervised intensity-based, multi-phase** (`mri_volume*` configs). No GT DVF. Each sample picks a target cardiac phase `t_target ∈ {0..T-1}`; loss compares splatted predicted volume `V_canon` against the on-disk NIfTI at that target phase (`V_gt`). The model can query any of the 12 discrete training phases — discrete-only; continuous-phase query (Option B) is not implemented (see Future enhancements).

The "**4-day baseline**" referenced below is the prior production run at `./scratch/logs/221086300_mri_volume_dynamic_axial_Cine_combined/` (199 epochs over ~4 days, ED-only `t_target=0` always, achieved PSNR 31+ dB at ED). All multi-phase fine-tuning currently warm-starts from its `ckpts/checkpoint_last.pt`.

The old supervised-DVF pipeline is **fully removed** from the live data/loss path: `MRIDataset` no longer reads `dvf_elastix/dvf_frame_*.nii.gz` or `mask_frame_00.nii.gz`, and `gt_dvfs` / `scale_factors` are no longer in the batch. The DVF NIfTIs still sit on disk for reproducibility and `compute_cine_dvf_elastix.py` is kept around to regenerate them, but nothing in training consumes them. Legacy configs/sbatch scripts that used them live under `_archive/`.

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
- `mri_volume.yaml` — **active** unsupervised intensity pipeline. Inherits `mri_finetune.yaml` via `defaults:` and disables the deprecated DVF loss. Sets `config_name: "mri_volume"` (used as one of the wandb tags).
- `mri_finetune.yaml` — base config (shared optimizer / data / freeze pattern); used as parent by `mri_volume.yaml`. Not runnable standalone — the supervised pipeline that did so is fully removed.
- `default.yaml` / `default_dataset.yaml` — templates inherited via `defaults:`.
- Legacy variants (`mri_finetune_*`, `mri_p001_overfit`, `mri_volume_overfit`) and their sbatch scripts now live under `_archive/legacy_configs/` and `_archive/legacy_sbatch/`.

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

**At trainer startup:** identity-Δ baseline (splat `scanner_coords` directly with Δ=0 across the val set) is computed and persisted to `${log_dir}/baseline_identity.json`. The per-phase value is **not** logged as its own wandb scalar — it's baked into the `val_psnr/` metric name (see below).

**Every train visual step (`log_visual_frequency.train = 100`):**
- `Train_Visuals_Volume` — 4-row × (max S,D)+1 col grid: input slices, V_gt, V_canon, signed diff (per z, ±0.05).
- `Train_Visuals_DVF` — 4-row × S+1 col grid: input intensity + per-slot Δx/Δy/Δz (±0.05).

**Per val epoch:**
- `Val_Loss/*` — averaged val metrics (loss_volume, psnr_3d, ssim_3d, mae, …).
- `Val_Visuals_subj{0,7}_Volume` and `_DVF` — two fixed val subjects at diagnostic phases. With stratified val sampling, val_step k → subject k → t_target = k % T_total, so the subject index doubles as the target phase. Subj0 at t=0 = ED. Subj7 at t=7 ≈ ES (empirically measured: ES across the 30 val subjects via LV-cavity bright-pixel count peaks at t=7-8, median=7). Note ES varies per subject — t=7 is the population median, not exact ES for every subject. The cardiac-cycle filmstrip (every N val epochs) shows all 12 phases for subj0, so you can verify the actual ES frame visually. Change `VAL_VISUAL_SUBJECT_INDICES` in `trainer.py` to log different subjects/phases.
- `val_psnr/t{k}_n{n}_base{b:.1f}` — per-phase val PSNR (multi-phase mode only). `n` is the sample count for phase k, `b` is the identity-Δ baseline PSNR for that phase. With deterministic stratified val, n is constant (3 for t=0..5, 2 for t=6..11) and the baseline is computed once at startup, so each phase produces exactly one panel for the lifetime of the run. If val ever loses determinism (e.g. someone re-enables `inside_random` on val), n drifts epoch-to-epoch and new panels appear under different names — that drift is the smoke alarm.
- `val_psnr/mean_n{n_total}_base{b:.1f}` — aggregate val PSNR across all val subjects (multi-phase mode only). Same panel group as the per-phase ones so you can compare model mean vs identity baseline at a glance. `n_total = 30` for the full val set.

**Every N val epochs** (`logging.filmstrip_every_n_val_epochs`, default 5):
- `Val_Visuals_cardiac_cycle` — 2×12 grid: V_gt (top) and V_canon (bot) reconstructed at all 12 cardiac phases for val subject 0, mid-z slice. The qualitative proof of multi-phase reconstruction.
- `Val_Visuals_cardiac_cycle_gif` — same content as the still strip, but as a 12-frame animated GIF (4 fps) so the heart actually beats. Each frame is V_gt | V_canon side-by-side (horizontal), shared intensity scale. Small (~50–200 KB per gif, ~10 MB total per run).

**Every val epoch** (`logging.save_val_volumes`, default `true`):
- `${log_dir}/val_volumes/subj{idx:02d}_t{t_target:02d}_{subject}_{pred|gt}.nii.gz` — per-subject predicted V_canon + GT V_gt as NIfTI, identity affine in canonical voxel space (12×256×256), overwritten in place every val epoch. ~360 MB constant footprint for 30 val subjects. Single-GPU only — under DDP, only rank 0's subjects are saved. Set `save_val_volumes: false` to disable.

**Wandb tags.** Two tags applied per run via `logging.wandb_writer.tags`: the active config name (`mri_volume`) and the phase mode (`multiphase` when `t_target_fixed` is null, else `t{K}` for fixed phase K). The `phase_mode` resolver is registered in `training/launch.py`.

**Gating.** In `t_target_fixed=K` mode (single-phase), `val_psnr/*` panels are skipped (meaningless when all samples share one phase). The cardiac-cycle filmstrip + GIF run in both modes — in fixed-K mode they show what the model does at phases it wasn't trained on (useful diagnostic). All diagnostic logging is wrapped in `try/except` — a failure logs a warning but never raises into training.

**PSNR caveat.** Pre-loss-switch logs reported anatomy-masked PSNR (over `V_gt > 1e-3`); current `compute_volume_intensity_loss` returns full-volume PSNR. Same model, ~7–9 dB lower number — don't compare across the switchover. Likewise, multi-phase val PSNR averages across all 12 phases and will be ~1–3 dB lower than the old ED-only val PSNR.

## SLURM

- Stagger mamba activations in array jobs: `sleep $((SLURM_ARRAY_TASK_ID * 15))`.
- Logs: `/home/minsukc/vggt/slurm_logs/`.

## Local gotchas

- Don't pipe `torchrun` through `| tail -N` in background — buffering. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- Initial VGGT-1B load takes ~9 min cold, ~1 min cached.
- Local pilots: `WANDB_MODE=offline`. The cluster script `sbatch/train_mri_volume.sh` sets `WANDB_MODE=online`.
- Hydra custom resolvers (`rev_ts:`, `basename:`, `phase_mode:`) are registered in `training/launch.py`. For standalone `compose()`: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`; `OmegaConf.register_new_resolver('basename', lambda p: os.path.basename(p))`; `OmegaConf.register_new_resolver('phase_mode', lambda t: 'multiphase' if t is None else f't{int(t)}')`.

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

- **Fully unsupervised — drop `gt_target_volume`.** Current "unsupervised" pipeline still depends on the per-phase NIfTI as `V_gt`, which is itself a derived reconstruction. True self-supervision: sample `V_canon` at the input slices' world coordinates and compare against the **input slice intensities themselves** — input slices become their own supervision. Removes one layer of indirection and aligns with the free-breathing/ungated regime (where no canonical GT exists). Risk: degenerate solutions (`V_canon = 0` everywhere except at sampled pixels) — needs a coverage-based completeness term or a stronger TV/smoothness prior. Loss becomes `|sample(V_canon, input_world_coords) - input_intensities|`.

- **UNet refiner on top of splat.** Add a small 3D UNet after the splat: `V_canon (12×256×256) → UNet → V_canon' (12×256×256)`. Learns to inpaint low-coverage voxels and smooth out splat artifacts (especially the seams between slices at native Z resolution). Drop-in addition — doesn't touch the splat or the point head. Useful as an ablation to quantify how much of the loss is due to splat artifacts vs. genuine motion-prediction error.

- **UNet ablation — replace the splat entirely.** Skip the differentiable splat + coverage division; regress `V_canon` directly from input features with a learned 3D head. Loses the physical interpretability of the splat ("this voxel is the trilinear average of these N pixels") in exchange for more model capacity. Worth running as a head-to-head against the current splat-based pipeline to see whether the inductive bias of explicit splatting is helping or hurting.
