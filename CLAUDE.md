# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VGGT (Visual Geometry Grounded Transformer) — CVPR 2025 Best Paper. This repo adapts VGGT for **cardiac 4D MRI reconstruction** using the CMRxRecon2024 dataset. The model predicts 3D geometry (camera pose, depth, 3D world coordinates, DVF) from multi-frame images/MRI slices.

This is adapting the vggt framework for cardiac mri slice to volume reconstruction/registration.

- MRI data: `/scratch/data/CMRxRecon2024/` (symlinked, GPFS)
- Micromamba env: `svr`
- SLURM cluster: `standard` partition (CPU), `gpu` partition (training), account `jjparkcv98`

---

## Environment

```bash
micromamba activate svr
pip install -e .                    # install vggt package
pip install -r requirements.txt
pip install -r requirements_demo.txt  # for demo scripts (gradio, viser, pycolmap)
```

---

## Training

Entry point: `training/launch.py` (Hydra-based config).

```bash
# MRI fine-tuning (main use case)
PYTHONPATH=training:. torchrun --nproc_per_node=1 --master_port=29507 \
    training/launch.py --config mri_finetune

# Multi-GPU
PYTHONPATH=training:. torchrun --nproc_per_node=4 \
    training/launch.py --config mri_finetune

# With overrides
PYTHONPATH=training:. torchrun --nproc_per_node=1 \
    training/launch.py --config mri_finetune optim.base_lr=1e-4
```

**Config files** (`training/config/`):
- `mri_finetune.yaml` — MRI dynamic (DVF prediction, residual mode) — OLD supervised pipeline
- `mri_finetune_general.yaml` — multi-subject general MRI
- `mri_p001_overfit.yaml` — overfit sanity for OLD supervised DVF pipeline
- `mri_volume.yaml` — NEW unsupervised volume pipeline (V_canon vs disk V_gt)
- `mri_volume_overfit.yaml` — single-subject overfit for new volume pipeline
- `default.yaml` — original Co3D training

**Key config parameters:**
- `max_img_per_gpu`: controls memory (default 48); reduce if OOM
- `model.enable_camera/depth/point/track`: toggle which heads are active
- `model.train_on_residual_dvf`: predict DVF residual from internal reconstruction
- `optim.frozen_module_names`: glob patterns to freeze (e.g., `["*aggregator*"]`)

---

## Inference

```python
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B").cuda()
images = load_and_preprocess_images(["img1.png", "img2.png"]).cuda()
with torch.no_grad():
    predictions = model(images)
# Keys: pose_enc, depth, depth_conf, world_points, world_points_conf, track (optional)
```

**Demo scripts:**
- `demo_viser.py` — interactive 3D point cloud viewer
- `demo_colmap.py` — export to COLMAP format
- `demo_gradio.py` — Gradio web UI

---

## CMR Data Shape Notes

Native CMRxRecon2024 subject shapes vary: W=256 (fixed), H ∈ {162, 204, 246}, Z ∈ {8..12}, T=12 phases. Don't assume fixed (D, H, W) across subjects. `mri_dataset.py` uses **per-axis normalization** (each axis's `[-1, 1]` spans that axis's own physical extent), not cubic max-axis.

---

## MRI Pipeline Scripts

```bash
# GPU-accelerated MRI reconstruction (sigpy + cupy)
micromamba run -n svr python batch_reconstruct_cmrxrecon2024.py

# Elastix-based DVF computation (groupwise 4D BSpline registration)
micromamba run -n svr python compute_cine_dvf_elastix.py
# SLURM array job (10 parallel workers):
sbatch sbatch/compute_dvf_elastix.sh

# Analyze DVF statistics across dataset
micromamba run -n svr python analyze_dvf_stats.py
```

DVFs are stored per-subject at `<subject>/sax/dvf_elastix/`:
- `dvf.npy` — shape `(T, D, H, W, 3)`, convention **T→0** (displacement at frame_T pointing to frame_0)
- `stats.json` — per-frame and summary Jacobian/magnitude statistics

---

## Architecture

### Model (`vggt/`)

```
VGGT (vggt/models/vggt.py)
├── Aggregator (vggt/models/aggregator.py)
│   └── Alternating attention: per-frame ViT + cross-frame global attention
│   └── Optional z-index embedding for slice ordering (use_z_pose_embedding)
├── CameraHead (vggt/heads/camera_head.py) — 9-dim pose encoding (quat + translation)
├── DPTHead for depth (vggt/heads/dpt_head.py) — depth + confidence
├── DPTHead for world points — 3D coords + confidence (used for DVF in MRI)
└── TrackHead (vggt/heads/track_head.py) — optical flow / point tracking
```

Camera convention: OpenCV (camera-from-world). Pose encoding decoded via `vggt/utils/pose_enc.py`.

### Training pipeline (`training/`)

```
launch.py → Trainer (trainer.py)
         → ComposedDataset (data/composed_dataset.py)
              → MRIDataset / Co3dDataset / VKittiDataset
              → DynamicTorchDataset (variable seq length, memory-aware batching)
         → MultitaskLoss (loss.py)
              → camera L1 (weight 5.0), depth conf+grad (1.0), point/DVF (1.0)
         → AdamW + linear warmup + cosine decay
```

**Batch sizing**: `batch_size = max_img_per_gpu / seq_len` — auto-adjusts per sequence length to prevent OOM.

### Datasets (`training/data/datasets/`)

- `mri_dataset.py` — dynamic MRI with pre-computed DVFs; modes: `axial`, `oblique`, `mixed`
- `mri_dataset_static.py` — static MRI (no DVF), for phase 1/2 training
- `co3d.py` — CO3D (images + COLMAP cameras + depth)
- `vkitti.py` — Virtual KITTI synthetic scenes

### Volume utilities (`vggt/utils/splat.py`)

- `splat_to_volume(pos, intensity, grid_shape, weight=None)` — differentiable trilinear scatter into a 3D grid.
- `sample_volume(volume, pos)` — thin wrapper around 3D `F.grid_sample`.
- Used by `mri_volume` pipeline to assemble per-pixel predictions into a canonical volume.

---

## Unsupervised Volume Pipeline (`mri_volume*` configs)

**Goal:** reconstruct the phase-0 (ED) volume from a stack of slices acquired at mixed cardiac phases, without using GT DVF.

**Pipeline (single forward pass):**
1. `S` slices in `(B, S, 3, H, W)` enter the aggregator (frozen).
2. Point head (DPTHead, ~32M trainable params) outputs a per-pixel canonical 3D position `(B, S, H, W, 3)` in normalized [-1, 1].
3. `vggt/utils/splat.py:splat_to_volume` scatters per-pixel `(position, intensity)` pairs into `V_canon` of shape `(B, 12, 256, 256)`.
4. Loss: `|V_canon - V_gt|` masked to voxels where the loaded phase-0 NIfTI has anatomy (`V_gt > 0`), plus a small TV regularizer on `pos_pred`. See `compute_volume_intensity_loss` in `training/loss.py`.

**Why this works:** V_gt is the actual phase-0 volume loaded from disk and resampled by the dataset to the canonical grid. The model can't trivially collapse (which it could with a self-consistency loss) — V_gt is fixed, so the only way to lower loss is to predict canonical positions where the input pixel's intensity matches the GT anatomy.

**GT volume in the batch:** `batch["gt_phase0_volume"]` shape `(B, 12, 256, 256)`. Loaded by `MRIDataset` and resampled at load time using the subject's own `center_mm` and per-axis `half_extent`.

**Sampling:** random `(z, t)` pairs per batch (no repeats within a batch). Slot 0 always has `t=0` so GT volume loading happens at the first slot of the loop.

---

## Logging & Checkpoints

- W&B project: `vggt-mri`. Logs DVF visualizations, per-frame metrics, config backup.
- Checkpoints saved every 5 epochs under `checkpoints/<run_name>/`.

---

## Known Limitation: Cardiac Mask Logic (OLD SUPERVISED DVF PIPELINE ONLY)

The section below applies to the `mri_finetune*` configs (supervised DVF regression). The new `mri_volume*` pipeline does NOT use the cardiac mask in its loss — `loss_volume` is masked only by `V_gt > 0` (where the loaded phase-0 anatomy exists).

**`mask_frame_00.nii.gz`** is an Otsu body mask computed from frame 0 (the reference). It is used in two places:

1. **During DVF generation** (`compute_cine_dvf_elastix.py`): elastix registration zeroes the DVF outside this mask, so background pixels have DVF = 0.0 mm exactly.
2. **During training** (`mri_dataset.py`): the mask is loaded and AND-ed into `point_masks` to restrict loss/MAE computation to cardiac tissue pixels only (~8–24% of each slice).

**Why the mask is necessary**: Without it, ~80–90% of each axial slice is background with DVF = 0. The model can achieve a very low MAE (e.g. 0.18mm) and low loss by simply predicting DVF = 0 everywhere, exploiting the confidence weighting to assign low confidence to cardiac pixels. The mask forces all loss/metric signal to come from cardiac pixels, preventing this trivial solution.

**The fundamental mismatch (known limitation)**: The mask is computed from frame-0 anatomy but is applied at frame-T voxel positions. Boundary cardiac voxels that moved *outside* the frame-0 body mask boundary during systole lose their DVF supervision. This is self-consistent — those voxels also have zeroed DVF (from registration) so no wrong gradients — but it means boundary cardiac motion is not supervised.

**`vol_mask`** is a separate, purely geometric bounds check: after applying the DVF to frame-T positions, does the resulting position still fall within `[0,W)×[0,H)×[0,Z)` in voxel space? With only 11 z-slices × 8mm = 88mm coverage, through-plane cardiac motion (~5mm) can push edge-slice voxels outside the scanned stack. These pixels are excluded from training but not wrong — they just don't exist in frame 0.

**To fix properly**: Re-run DVF generation with a union-of-all-frames mask (or dilated mask) so boundary cardiac voxels always have valid DVF values and supervision. Not yet done — accepted as a known limitation.

---

## SLURM Notes

- Always stagger mamba activations in array jobs to avoid lock contention:
  ```bash
  sleep $((SLURM_ARRAY_TASK_ID * 15))
  ```
- Logs: `/home/minsukc/vggt/slurm_logs/`

## Local Training Gotchas

- Don't pipe `torchrun` through `| tail -N` in background bash — output is buffered until the process exits. Redirect to file: `... > /tmp/run.log 2>&1 &`, then `tail -F /tmp/run.log`.
- Initial VGGT-1B checkpoint load (`./scratch/torch_cache/model.pt`, 941M params) takes ~9 min cold, ~1 min cached.
- Local pilots use `WANDB_MODE=offline`; cluster scripts (`sbatch/train_mri_volume*.sh`) export `WANDB_MODE=online`.
- Hydra configs use custom resolvers (`rev_ts:`, `basename:`) registered in `training/launch.py`. For `compose()` testing without launch.py, register manually: `OmegaConf.register_new_resolver('rev_ts', lambda: '0')`.

## Testing

`tests/` uses a synthetic in-memory CMR dataset (`tests/conftest.py`, W=32, H=30, Z=4) — no real data needed.

```bash
micromamba run -n svr python -m pytest tests/
```
