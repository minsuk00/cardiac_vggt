"""Numerical-stability tests for training/loss.py."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_tv_term_dtype_invariant():
    """The TV regularizer is wrapped in `autocast(enabled=False)` + `.float()` so its
    ~786k-element mask reduction never runs in bf16. bf16 and fp32 pos_pred must
    produce numerically identical TV values; otherwise the bf16 path drifts ~1%
    due to mantissa rounding in mask_dh.sum() / mask_dw.sum().
    """
    from loss import compute_volume_intensity_loss

    torch.manual_seed(0)
    B, S, H, W, D = 1, 4, 32, 32, 8
    pos_pred = torch.rand(B, S, H, W, 3) * 0.5 - 0.25
    images = torch.rand(B, S, 3, H, W)
    V_gt = torch.rand(B, D, H, W)
    scanner = pos_pred.clone()
    batch = {"images": images, "gt_target_volume": V_gt, "scanner_coords": scanner}

    out_fp32 = compute_volume_intensity_loss(
        {"world_points": pos_pred}, batch, grid_shape=(D, H, W), tv_weight=0.1
    )
    out_bf16 = compute_volume_intensity_loss(
        {"world_points": pos_pred.bfloat16()}, batch, grid_shape=(D, H, W), tv_weight=0.1
    )

    diff = (out_fp32["loss_pos_tv"] - out_bf16["loss_pos_tv"]).abs().item()
    assert diff < 1e-5, (
        f"TV term diverges between bf16 and fp32 pos_pred (diff={diff:.6e}); "
        "internal fp32 promotion may have regressed."
    )
