"""Tests for the L2 diffusion regularizer ‖∇u‖² on the DVF in compute_volume_intensity_loss
(config: loss.volume.diffusion_weight; default 0.0 ⇒ no-op)."""
import numpy as np
import pytest
import torch

from loss import compute_volume_intensity_loss, diffusion_loss_l2


def test_diffusion_helper_zero_on_constant():
    f = torch.full((1, 2, 16, 16, 3), 0.37)
    assert diffusion_loss_l2(f).item() == pytest.approx(0.0, abs=1e-7)


def test_diffusion_helper_penalizes_high_freq_more_than_smooth():
    """Same amplitude, different frequency: a 2-px ripple must score far higher than a
    smooth low-freq field — the whole point of the L2 diffusion penalty."""
    x = torch.linspace(0, 1, 64)
    smooth = torch.sin(2 * np.pi * x / 50)[None, None, None, :, None].expand(1, 2, 64, 64, 3)
    ripple = torch.sin(2 * np.pi * x / 2)[None, None, None, :, None].expand(1, 2, 64, 64, 3)
    assert diffusion_loss_l2(ripple) > 50 * diffusion_loss_l2(smooth)


def _loss_inputs(B=2, S=4, H=32, W=32, D=12, Hv=256, Wv=256):
    torch.manual_seed(0)
    pos = torch.rand(B, S, H, W, 3) * 2 - 1
    images = torch.rand(B, S, 3, H, W) * 255
    V_gt = torch.rand(B, D, Hv, Wv) * 0.5
    z_scale = torch.full((B,), (D - 1) / 2.0)
    batch = {"images": images, "gt_target_volume": V_gt, "scanner_coords": pos.clone(), "z_scale": z_scale}
    return pos, batch, (D, Hv, Wv)


def test_diffusion_weight_zero_is_noop():
    """diffusion_weight=0 ⇒ loss_diffusion is exactly 0 and the other terms are untouched."""
    pos, batch, grid = _loss_inputs()
    base = compute_volume_intensity_loss({"world_points": pos}, batch, tv_weight=0.1)
    withd = compute_volume_intensity_loss({"world_points": pos}, batch,
                                          tv_weight=0.1, diffusion_weight=0.0)
    assert withd["loss_diffusion"].item() == 0.0
    assert torch.allclose(base["loss_volume"], withd["loss_volume"])
    assert torch.allclose(base["loss_pos_tv"], withd["loss_pos_tv"])


def test_diffusion_weight_positive_adds_positive_term():
    pos, batch, grid = _loss_inputs()
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        tv_weight=0.0, diffusion_weight=10.0)
    assert out["loss_diffusion"].item() > 0.0


def test_diffusion_uses_dvfs_not_world_points_when_present():
    """The regularizer must act on the displacement (predictions['dvfs']) when available,
    falling back to world_points otherwise."""
    pos, batch, grid = _loss_inputs()
    smooth = torch.zeros_like(pos)                       # smooth dvf ⇒ ~0 diffusion
    preds = {"world_points": pos, "dvfs": smooth}
    out = compute_volume_intensity_loss(preds, batch, tv_weight=0.0, diffusion_weight=10.0)
    assert out["loss_diffusion"].item() == pytest.approx(0.0, abs=1e-6)  # used dvfs (smooth), not pos
