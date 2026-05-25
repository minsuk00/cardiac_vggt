"""batchaug GPU augmentation pipeline for VGGT-MRI.

Operates on the cached canonical `(B, T=12, D=12, H=256, W=256)` phases tensor
that `MRIDataset.get_data` puts in the batch under the `phases` key. One spatial
affine is sampled per subject (B-dim); the same affine is applied to all 12
T-phases (T as channel) AND to the `content_mask`, so:

    * augmented phases stay consistent across cardiac phases (no rotation jitter
      between t=0 and t=11)
    * scanner_coords don't need updating — they're a pure geometric mapping from
      pixel-index to canonical-cube-coord, decoupled from image content
    * `anatomy_bbox` is recomputed from the augmented mask (anatomy has moved)

After aug, the trainer:
    1. Re-derives `V_gt` = `phases_aug[b, t_target[b]]`
    2. Re-extracts `S` input slices from `phases_aug` at the original (t, z) pairs
    3. Recomputes `anatomy_bbox` from the augmented content_mask

batchaug backend is forced to `"pytorch"` at import. See `requirements.txt` —
our env has triton 2.3.1 (bundled with torch 2.3.1), and triton 3.x has API
breakage that risks torch.compile / inductor paths. The PyTorch backend is
slower for spatial ops but plenty fast for our workload (1 affine per subject,
small tensors). Revisit triton-3 only if augmentation becomes a bottleneck.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

try:
    import batchaug as _B
    _B.set_backend("pytorch")
except ImportError:  # pragma: no cover — batchaug is a hard dep for aug
    _B = None

# Local constants (kept in sync with preprocess.py / mri_dataset.py).
INPUT_IMG_SIZE = 518   # DINOv2 input — must match MRIDataset.target_size
CANON_HW = 256         # canonical slice height/width
CANON_D = 12           # canonical depth
NUM_PHASES = 12        # cardiac phases per subject


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build the batchaug Compose from config
# ──────────────────────────────────────────────────────────────────────────────
def build_gpu_transforms(aug_cfg=None):
    """Build a `batchaug.Compose` from the config, or return `None` if disabled.

    The trainer treats `None` as identity and skips the aug step entirely.

    Args:
        aug_cfg: object/dict with fields:
            enable (bool, default False)
            tier   ("conservative" | "moderate", default "conservative")

    Returns:
        `batchaug.Compose` or `None`.
    """
    enable = bool(getattr(aug_cfg, "enable", False)) if aug_cfg is not None else False
    if not enable:
        logging.info("GPU augmentation disabled (data.augmentation.enable=False)")
        return None
    if _B is None:
        raise RuntimeError(
            "batchaug is not importable but aug is enabled. "
            "Install via: pip install --no-deps -e /home/minsukc/MRI2CT/batchaug/"
        )

    tier = getattr(aug_cfg, "tier", "conservative")
    logging.info(f"GPU augmentation enabled: tier={tier}")
    keys = ["phases", "content_mask"]
    mode_dict = {"phases": "bilinear", "content_mask": "nearest"}

    if tier == "conservative":
        # Conservative tier: in-plane H-flip, ±5° in-plane rotation, small
        # translate/scale, light photometric. NO through-plane rotation
        # (slices are physically anisotropic 8 mm Z vs 1.4 mm X/Y), NO elastic
        # (defer to moderate tier once conservative is stable).
        # batchaug is POSITIONAL, not semantic: each tuple slot i maps to tensor
        # spatial dim i+2 (our dims are D=0, H=1, W=2 after the channel). BUT
        # `rotate_range` is special — its slots are PLANES of rotation, not axes:
        #   slot 0 → rotation in the H-W plane (about D)  = IN-PLANE  ← what we want
        #   slot 1 → rotation in the D-W plane            = through-plane
        #   slot 2 → rotation in the D-H plane            = through-plane
        # So the in-plane ±5° goes in rotate_range slot 0, NOT slot 2.
        # (translate_range / scale_range ARE per-axis (D, H, W): slot 0 = D, so
        #  freezing slot 0 there correctly disables through-plane shift/scale.)
        # RandFlipd spatial_axis=[2] flips dim W (in-plane left-right).
        transforms = [
            _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=(float(np.deg2rad(5)), 0.0, 0.0),   # in-plane (H-W) only
                translate_range=(0.0, 4.0, 4.0),                 # H, W only (D frozen)
                scale_range=(0.0, 0.05, 0.05),                   # H, W only (D frozen)
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask.
            _B.RandGaussianNoised(keys=["phases"], prob=0.3, std=(0.0, 0.02)),
            _B.RandAdjustContrastd(keys=["phases"], prob=0.3, gamma=(0.8, 1.25)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.3, degree=3, coeff_range=(-0.2, 0.2)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    if tier == "moderate":
        # Moderate tier: same IN-PLANE-ONLY discipline as conservative (NO
        # through-plane rotation, NO elastic — Z is 8 mm anisotropic), but with
        # stronger ranges and higher fire-probabilities so the effect is clearly
        # visible and provides real regularization rather than near-identity draws.
        # Same positional/plane semantics as the conservative block above:
        #   rotate_range slot 0 = in-plane (H-W); translate/scale slots are (D,H,W)
        #   with D frozen; RandFlipd spatial_axis=[2] = in-plane left-right (W).
        transforms = [
            _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.9,
                rotate_range=(float(np.deg2rad(12)), 0.0, 0.0),  # in-plane (H-W) only
                translate_range=(0.0, 8.0, 8.0),                 # H, W only (D frozen)
                scale_range=(0.0, 0.10, 0.10),                   # H, W only (D frozen)
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask.
            _B.RandGaussianNoised(keys=["phases"], prob=0.5, std=(0.0, 0.03)),
            _B.RandAdjustContrastd(keys=["phases"], prob=0.5, gamma=(0.7, 1.4)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.5, degree=3, coeff_range=(-0.3, 0.3)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    if tier == "aggressive":
        # Aggressive tier: still IN-PLANE ONLY (no through-plane rotation, no
        # elastic — Z is 8 mm anisotropic), but large, clearly-visible ranges and
        # high fire-probabilities for strong regularization. Same plane/axis
        # semantics as the blocks above (rotate slot 0 = in-plane H-W;
        # translate/scale slots = (D,H,W) with D frozen; flip W = in-plane L-R).
        transforms = [
            _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.95,
                rotate_range=(float(np.deg2rad(25)), 0.0, 0.0),  # in-plane (H-W) only
                translate_range=(0.0, 16.0, 16.0),               # H, W only (D frozen)
                scale_range=(0.0, 0.20, 0.20),                   # H, W only (D frozen)
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask.
            _B.RandGaussianNoised(keys=["phases"], prob=0.6, std=(0.0, 0.05)),
            _B.RandAdjustContrastd(keys=["phases"], prob=0.6, gamma=(0.6, 1.7)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.6, degree=3, coeff_range=(-0.5, 0.5)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    raise ValueError(f"unknown aug tier: {tier!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: bbox of a 3D mask (GPU-friendly, no python loop over voxels)
# ──────────────────────────────────────────────────────────────────────────────
def recompute_bbox_gpu(content_mask) -> torch.Tensor:
    """Geometric bbox of `content_mask > 0` in splat order.

    Args:
        content_mask: `(D, H, W)` tensor. Any dtype; tested as `> 0`.

    Returns:
        `(z0, z1, y0, y1, x0, x1)` int64 tensor on the same device. Falls back
        to the full cube if the mask is empty (post-aug edge case).
    """
    if content_mask.ndim != 3:
        raise ValueError(f"expected (D, H, W), got {tuple(content_mask.shape)}")
    D, H, W = content_mask.shape
    mask = content_mask > 0
    if not mask.any():
        return torch.tensor([0, D, 0, H, 0, W], dtype=torch.int64, device=content_mask.device)
    nz = mask.nonzero()
    z0, y0, x0 = nz.min(dim=0).values.tolist()
    z_max, y_max, x_max = nz.max(dim=0).values.tolist()
    return torch.tensor(
        [z0, z_max + 1, y0, y_max + 1, x0, x_max + 1],
        dtype=torch.int64, device=content_mask.device,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: re-extract S slices from an augmented (B, T, D, H, W) tensor
# ──────────────────────────────────────────────────────────────────────────────
def extract_slices_from_phases(phases, t_seq, z_seq):
    """Pull S slices per batch element from an augmented phases tensor and
    bilinear-upsample each to `(INPUT_IMG_SIZE, INPUT_IMG_SIZE)` for DINOv2.

    Args:
        phases: `(B, T, D, H=256, W=256)` float.
        t_seq:  `(B, S)` int64 — t index per slot.
        z_seq:  `(B, S)` int64 — z index per slot.

    Returns:
        `(B, S, 518, 518, 3)` float in `[0, 255]` — RGB-replicated, ready to
        replace `batch["images"]` after a `permute(0, 1, 4, 2, 3) / 255` in
        the trainer (matches the ComposedDataset contract).
    """
    Bsize, T, D, H, W = phases.shape
    S = t_seq.shape[1]
    b_idx = torch.arange(Bsize, device=phases.device).view(Bsize, 1).expand(Bsize, S)
    # Fancy indexing: pick `phases[b, t_seq[b, s], z_seq[b, s], :, :]` for all (b, s)
    slices_canon = phases[b_idx, t_seq, z_seq]  # (B, S, H, W)
    slices_canon = slices_canon.reshape(Bsize * S, 1, H, W)
    upsampled = F.interpolate(
        slices_canon,
        size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
        mode="bilinear",
        align_corners=True,
    )                                                    # (B*S, 1, 518, 518)
    upsampled = upsampled.view(Bsize, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE)
    upsampled = (upsampled * 255.0).clamp(0.0, 255.0)
    # RGB-replicate to (B, S, 518, 518, 3).
    return upsampled.unsqueeze(-1).expand(Bsize, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry: apply aug to the batch in place
# ──────────────────────────────────────────────────────────────────────────────
def gpu_augment_batch(batch, transforms, device):
    """Apply GPU augmentations to a batch and re-derive the dependent fields.

    If `transforms is None`, return the batch unchanged (identity — used in val
    and in train-with-aug-off).

    Required batch keys (when transforms is not None):
        phases           (B, T, D, H, W) float16/float32
        content_mask     (B, D, H, W)    uint8
        t_target         (B, 1)          int64
        timesteps        (B, S)          int64 — original t per slot
        slice_indices    (B, S)          int64 — original z per slot

    Updates (in place; returns the same dict):
        phases               replaced with augmented float
        content_mask         replaced with augmented uint8
        gt_target_volume     re-derived from augmented phases
        anatomy_bbox         recomputed from augmented mask
        images               re-extracted from augmented phases (B, S, 3, H, W) in [0, 1]
    """
    if transforms is None:
        return batch

    phases = batch["phases"]                 # (B, T, D, H, W) any float
    mask = batch["content_mask"]             # (B, D, H, W) uint8

    Bsize = phases.shape[0]
    phases_f = phases.to(device=device, dtype=torch.float32, non_blocking=True)
    # batchaug grid_sample needs float; mask must keep its semantics (0/1) under
    # nearest-neighbor interp.
    mask_f = mask.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
    # mask is (B, D, H, W); add a channel dim for batchaug → (B, 1, D, H, W).

    aug_dict = {"phases": phases_f, "content_mask": mask_f}
    try:
        aug_dict = transforms(aug_dict)
    except Exception as e:
        # Aug must never crash training; log and fall through with identity.
        logging.warning(f"gpu_augment_batch: aug pipeline failed (ignored): {e}")
        return batch

    phases_aug = aug_dict["phases"]                       # (B, T, D, H, W) float32
    mask_aug = aug_dict["content_mask"].squeeze(1)         # (B, D, H, W) float (0 or 1)
    mask_aug_u8 = (mask_aug > 0.5).to(torch.uint8)

    # Re-derive V_gt = phases_aug[b, t_target[b]]
    t_target = batch["t_target"]
    if t_target.ndim > 1:
        t_target = t_target.squeeze(-1)  # (B,)
    gt_target_volume = phases_aug[torch.arange(Bsize, device=device), t_target]  # (B, D, H, W)

    # Recompute bbox per sample (variable-shape result per sample → loop).
    bboxes = torch.stack([recompute_bbox_gpu(mask_aug_u8[b]) for b in range(Bsize)])

    # Re-extract input slices from augmented phases.
    images = extract_slices_from_phases(
        phases_aug,
        batch["timesteps"],
        batch["slice_indices"],
    )                                                       # (B, S, 518, 518, 3) in [0, 255]
    images = images.permute(0, 1, 4, 2, 3).contiguous() / 255.0  # (B, S, 3, 518, 518) in [0, 1]

    batch["phases"] = phases_aug.to(phases.dtype)
    batch["content_mask"] = mask_aug_u8
    batch["gt_target_volume"] = gt_target_volume
    batch["anatomy_bbox"] = bboxes
    batch["images"] = images
    return batch
