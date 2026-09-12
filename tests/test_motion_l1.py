"""Tests for the motion-weighted L1 arm (docs/92, `loss.volume.motion_l1_weight`).

Two contracts guarded here:
  1. Knob OFF (motion_l1_weight=0) must be bit-identical to the shipped pipeline — the
     loss keeps the exact `(V_canon - V_gt).abs().mean()`.
  2. Knob ON must actually bite: the weight map matches the hand formula.

The ED/ES-biased target sampler that shipped alongside this on `arm/es-weight` was NOT
migrated to main (real-EF null/negative, docs/93); only the loss term lives here.
"""

import pytest
import torch

from loss import compute_volume_intensity_loss


# ── Synthetic loss batch (same pattern as test_loss_masked_metrics_golden) ──
B, S, D, H, W = 1, 4, 6, 16, 16
T = 12


def _batch(seed=0, constant_phases=False):
    g = torch.Generator().manual_seed(seed)
    static = torch.rand(B, 1, D, H, W, generator=g)
    phases = static.expand(B, T, D, H, W).clone()
    if not constant_phases:
        phases[:, :, 2:4, 4:12, 4:12] += torch.linspace(0, 1, T).view(1, T, 1, 1, 1)
    phases = phases.clamp(0, 1)
    gt = phases[:, 0].clone()

    z = torch.linspace(-1, 1, S).view(1, S, 1, 1).expand(B, S, H, W)
    y = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, S, H, W)
    x = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, S, H, W)
    scanner = torch.stack([x, y, z], dim=-1)

    return {
        "gt_target_volume": gt,
        "phases": phases,
        "scanner_coords": scanner,
        "images": torch.rand(B, S, 3, H, W, generator=g),
        "z_scale": torch.tensor([90.0 / 12.0]),
        "t_target": torch.tensor([[0]], dtype=torch.int64),
    }


def _run(batch, **kw):
    pos = batch["scanner_coords"].clone()
    return compute_volume_intensity_loss({"world_points": pos}, batch, **kw)


def test_lambda_zero_is_bit_identical():
    base = _run(_batch())["loss_volume"]
    off = _run(_batch(), motion_l1_weight=0.0)["loss_volume"]
    assert torch.equal(base, off)


def test_lambda_positive_changes_the_loss():
    base = _run(_batch())["loss_volume"]
    on = _run(_batch(), motion_l1_weight=10.0)["loss_volume"]
    assert not torch.equal(base, on)


def test_constant_phases_give_uniform_weights():
    # swing == 0 everywhere -> swing_norm == 0 -> w == 1 -> weighted mean == plain mean.
    base = _run(_batch(constant_phases=True))["loss_volume"]
    on = _run(_batch(constant_phases=True), motion_l1_weight=10.0)["loss_volume"]
    assert torch.allclose(base, on)


def test_weighted_loss_matches_hand_formula():
    lam = 10.0
    batch = _batch()
    out = _run(batch, motion_l1_weight=lam)
    # Recompute by hand with the documented formula.
    ref = _run(batch)  # for V_canon (same positions -> same splat)
    err = (ref["V_canon"].float() - batch["gt_target_volume"].float()).abs()
    swing = (batch["phases"].amax(dim=1) - batch["phases"].amin(dim=1)).float()
    swing_norm = (swing / torch.quantile(swing, 0.995).clamp(min=1e-6)).clamp(0, 1)
    w = 1.0 + lam * swing_norm
    expected = (w * err).sum() / w.sum()
    assert torch.allclose(out["loss_volume"], expected)
    # The map is graded and bounded: 1 <= w <= 1 + lam, and the moving block is upweighted.
    assert w.min() >= 1.0 and w.max() <= 1.0 + lam
    assert w[:, 2:4, 4:12, 4:12].mean() > w[:, 0].mean()


def test_missing_phases_raises():
    batch = _batch()
    del batch["phases"]
    with pytest.raises(RuntimeError, match="phases"):
        _run(batch, motion_l1_weight=10.0)
    _run(batch, motion_l1_weight=0.0)  # lambda=0 never touches phases
