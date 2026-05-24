"""Tests for the canonical preprocess pipeline (training/data/preprocess.py)."""

import numpy as np
import torch

from data.preprocess import (
    TARGET_SHAPE,
    TARGET_SPACING,
    ScaleIntensityByT0PercentilesD,
    compute_geometric_bbox,
    get_canonical_transforms,
    build_data_dicts,
)


# ── compute_geometric_bbox ────────────────────────────────────────────────────

def test_bbox_full_cube_when_all_content():
    mask = torch.ones(12, 256, 256)
    bb = compute_geometric_bbox(mask)
    assert bb.tolist() == [0, 12, 0, 256, 0, 256]

def test_bbox_tight_around_content_block():
    mask = torch.zeros(12, 256, 256)
    mask[3:9, 40:200, 10:240] = 1
    bb = compute_geometric_bbox(mask)
    assert bb.tolist() == [3, 9, 40, 200, 10, 240]

def test_bbox_empty_mask_falls_back_to_full_cube():
    mask = torch.zeros(8, 64, 64)
    bb = compute_geometric_bbox(mask)
    assert bb.tolist() == [0, 8, 0, 64, 0, 64]

def test_bbox_padding_arg_expands_and_clamps():
    mask = torch.zeros(12, 256, 256)
    mask[5:7, 100:120, 100:120] = 1
    bb = compute_geometric_bbox(mask, padding=3)
    # padded by 3 each side, clamped to bounds
    assert bb.tolist() == [2, 10, 97, 123, 97, 123]

def test_bbox_rejects_non_3d():
    with np.testing.assert_raises(ValueError):
        compute_geometric_bbox(torch.ones(1, 12, 256, 256))


# ── ScaleIntensityByT0PercentilesD ────────────────────────────────────────────

def test_t0_percentile_norm_uses_phase0_stats_for_all():
    """All phases must be normalized against phase_00's percentiles, not their own."""
    t = ScaleIntensityByT0PercentilesD(keys=["phase_00", "phase_01"], ref_key="phase_00",
                                       lower=1.0, upper=99.5)
    # phase_00 in [0, 100]; phase_01 in [0, 200]. If each were normalized by its own
    # percentiles, phase_01's max would clip to 1.0. With shared phase_00 stats,
    # phase_01 exceeds 1.0 before clamp → clamps to 1.0 but the SCALING is phase_00's.
    p0 = torch.linspace(0, 100, 1000).reshape(1, 10, 10, 10)
    p1 = torch.linspace(0, 200, 1000).reshape(1, 10, 10, 10)
    out = t({"phase_00": p0.clone(), "phase_01": p1.clone()})
    # phase_00 spans ~[0, 1] after norm.
    assert abs(float(out["phase_00"].min())) < 0.05
    assert abs(float(out["phase_00"].max()) - 1.0) < 0.05
    # phase_01 (2× the intensity) saturates at 1.0 across most of its range.
    assert float(out["phase_01"].max()) == 1.0
    assert (out["phase_01"] >= 0.99).float().mean() > 0.4


# ── get_canonical_transforms (end-to-end on synthetic NIfTIs) ─────────────────

def test_canonical_transform_output_shape(synthetic_root):
    sax_dir = f"{synthetic_root}/Train_P001/sax"
    data_dict = build_data_dicts([sax_dir])[0]
    out = get_canonical_transforms()(data_dict)
    # ConcatItemsd(dim=0) absorbs the per-phase channel dim → (T, X, Y, Z) in monai order.
    assert out["phases"].shape == (12, TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2])
    assert out["content_mask"].shape == (1, TARGET_SHAPE[0], TARGET_SHAPE[1], TARGET_SHAPE[2])
    assert out["phases"].dtype == torch.float16
    assert out["content_mask"].dtype == torch.uint8

def test_canonical_content_mask_is_binary_and_nonempty(synthetic_root):
    sax_dir = f"{synthetic_root}/Train_P001/sax"
    data_dict = build_data_dicts([sax_dir])[0]
    out = get_canonical_transforms()(data_dict)
    m = out["content_mask"]
    uniq = set(torch.unique(m).tolist())
    assert uniq.issubset({0, 1}), f"mask must be binary, got values {uniq}"
    assert int(m.sum()) > 0, "content mask must mark the subject's FOV"
    # Synthetic native FOV (64, 60, 8 @ 1.4,1.4,8.0) < canonical cube → some zero-pad exists.
    assert int(m.sum()) < m.numel(), "small-FOV synthetic subject must have some zero-pad"
