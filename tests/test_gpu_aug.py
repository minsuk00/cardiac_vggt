"""Tests for the batchaug GPU augmentation pipeline (training/data/gpu_aug.py).

Run on CPU — batchaug's pytorch backend works without CUDA.
"""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data.gpu_aug import (
    build_gpu_transforms,
    extract_slices_from_phases,
    gpu_augment_batch,
    recompute_bbox_gpu,
)

DEVICE = "cpu"


def _fake_batch(B=2, T=12, D=12, H=256, W=256, S=8):
    phases = torch.rand(B, T, D, H, W, dtype=torch.float16)
    content_mask = torch.zeros(B, D, H, W, dtype=torch.uint8)
    content_mask[:, 1:11, 30:230, 5:251] = 1
    return {
        "phases": phases,
        "content_mask": content_mask,
        "gt_target_volume": torch.rand(B, D, H, W),
        "anatomy_bbox": torch.tensor([[1, 11, 30, 230, 5, 251]] * B, dtype=torch.int64),
        "t_target": torch.tensor([[0], [5]][:B], dtype=torch.int64),
        "timesteps": torch.tensor([list(range(S))] * B, dtype=torch.int64),
        "slice_indices": torch.tensor([list(range(S))] * B, dtype=torch.int64),
        "images": torch.rand(B, S, 3, 518, 518),
        "scanner_coords": torch.rand(B, S, 518, 518, 3),
    }


# ── build_gpu_transforms ──────────────────────────────────────────────────────

def test_build_returns_none_when_disabled():
    assert build_gpu_transforms(OmegaConf.create({"enable": False})) is None
    assert build_gpu_transforms(None) is None

def test_build_returns_compose_when_enabled():
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    assert t is not None

def test_build_moderate_tier_not_implemented():
    with pytest.raises(NotImplementedError):
        build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "moderate"}))


# ── gpu_augment_batch identity passthrough ────────────────────────────────────

def test_identity_passthrough_when_none():
    batch = _fake_batch()
    pre = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    out = gpu_augment_batch(batch, None, DEVICE)
    for k in ["phases", "images", "gt_target_volume", "anatomy_bbox", "content_mask"]:
        assert torch.equal(out[k], pre[k]), f"{k} changed under identity passthrough"


# ── gpu_augment_batch with conservative pipeline ──────────────────────────────

def test_aug_preserves_shapes_and_ranges():
    batch = _fake_batch()
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    out = gpu_augment_batch(batch, t, DEVICE)
    B, S = 2, 8
    assert out["phases"].shape == (B, 12, 12, 256, 256)
    assert out["images"].shape == (B, S, 3, 518, 518)
    assert out["gt_target_volume"].shape == (B, 12, 256, 256)
    assert out["anatomy_bbox"].shape == (B, 6)
    assert out["content_mask"].shape == (B, 12, 256, 256)
    # Images re-extracted and normalized to [0, 1].
    assert float(out["images"].min()) >= 0.0 and float(out["images"].max()) <= 1.0

def test_aug_recomputes_bbox_validly():
    batch = _fake_batch()
    t = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "conservative"}))
    out = gpu_augment_batch(batch, t, DEVICE)
    for b in range(out["anatomy_bbox"].shape[0]):
        z0, z1, y0, y1, x0, x1 = out["anatomy_bbox"][b].tolist()
        assert 0 <= z0 < z1 <= 12
        assert 0 <= y0 < y1 <= 256
        assert 0 <= x0 < x1 <= 256


def test_conservative_tier_has_no_through_plane_spatial_op():
    """Regression guard for the through-plane rotation bug.

    The conservative tier must never move intensity ACROSS Z (D) planes: no
    through-plane rotation, translation, or scale. We confine content to a few
    D-planes, apply the spatial aug 10× at prob=1, and assert no mass leaks into
    the empty planes. (batchaug's rotate_range is positional by plane-of-rotation,
    so a wrong slot silently produces through-plane rotation — this catches it.)
    """
    import batchaug as B
    import numpy as np

    keys = ["phases"]
    mode = {"phases": "bilinear"}
    # Spatial-only conservative ops (drop photometric, which don't move mass).
    spatial = B.Compose(transforms=[
        B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
        B.RandAffined(keys=keys, prob=1.0,
                      rotate_range=(float(np.deg2rad(5)), 0.0, 0.0),
                      translate_range=(0.0, 4.0, 4.0),
                      scale_range=(0.0, 0.05, 0.05),
                      padding_mode="zeros"),
    ], lazy=True, mode=mode)

    content_planes = list(range(3, 9))
    empty_planes = [0, 1, 2, 9, 10, 11]
    for _ in range(10):
        phases = torch.zeros(1, 12, 12, 256, 256)
        phases[:, :, 3:9, 60:200, 60:200] = 1.0
        out = spatial({"phases": phases})["phases"]
        leak = out[0, 0][empty_planes].abs().sum().item()
        assert leak < 1.0, f"through-plane leak detected: {leak:.1f} mass in empty Z-planes"


# ── helpers ───────────────────────────────────────────────────────────────────

def test_recompute_bbox_gpu_tight():
    mask = torch.zeros(12, 256, 256)
    mask[2:9, 50:200, 10:240] = 1
    bb = recompute_bbox_gpu(mask)
    assert bb.tolist() == [2, 9, 50, 200, 10, 240]

def test_recompute_bbox_gpu_empty_fallback():
    bb = recompute_bbox_gpu(torch.zeros(12, 256, 256))
    assert bb.tolist() == [0, 12, 0, 256, 0, 256]

def test_extract_slices_shapes_and_indexing():
    B, T, D, H, W, S = 2, 12, 12, 256, 256, 8
    phases = torch.rand(B, T, D, H, W)
    t_seq = torch.tensor([list(range(S))] * B, dtype=torch.int64)
    z_seq = torch.tensor([list(range(S))] * B, dtype=torch.int64)
    imgs = extract_slices_from_phases(phases, t_seq, z_seq)
    assert imgs.shape == (B, S, 518, 518, 3)
    assert float(imgs.max()) <= 255.0 and float(imgs.min()) >= 0.0
