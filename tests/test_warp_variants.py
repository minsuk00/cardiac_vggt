"""Tests for the two additive warp-smoothness variants:

  (A) L2 diffusion regularizer ‖∇u‖² on the DVF in compute_volume_intensity_loss
      (config: loss.volume.diffusion_weight; default 0.0 ⇒ no-op).
  (B) BSplineWarpHead (config: model.warp_head_type="bspline") — a smooth-by-construction
      drop-in for the DPT point head.

Both must be purely additive: defaults reproduce the existing pipeline exactly.
"""
import numpy as np
import pytest
import torch

from loss import compute_volume_intensity_loss, diffusion_loss_l2
from vggt.heads.bspline_head import BSplineWarpHead
from vggt.heads.dpt_head import DPTHead


# ───────────────────────── (A) L2 diffusion regularizer ─────────────────────────

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
    batch = {"images": images, "gt_target_volume": V_gt, "scanner_coords": pos.clone()}
    return pos, batch, (D, Hv, Wv)


def test_diffusion_weight_zero_is_noop():
    """diffusion_weight=0 ⇒ loss_diffusion is exactly 0 and the other terms are untouched."""
    pos, batch, grid = _loss_inputs()
    base = compute_volume_intensity_loss({"world_points": pos}, batch, grid_shape=grid, tv_weight=0.1)
    withd = compute_volume_intensity_loss({"world_points": pos}, batch, grid_shape=grid,
                                          tv_weight=0.1, diffusion_weight=0.0)
    assert withd["loss_diffusion"].item() == 0.0
    assert torch.allclose(base["loss_volume"], withd["loss_volume"])
    assert torch.allclose(base["loss_pos_tv"], withd["loss_pos_tv"])


def test_diffusion_weight_positive_adds_positive_term():
    pos, batch, grid = _loss_inputs()
    out = compute_volume_intensity_loss({"world_points": pos}, batch, grid_shape=grid,
                                        tv_weight=0.0, diffusion_weight=10.0)
    assert out["loss_diffusion"].item() > 0.0


def test_diffusion_uses_dvfs_not_world_points_when_present():
    """The regularizer must act on the displacement (predictions['dvfs']) when available,
    falling back to world_points otherwise."""
    pos, batch, grid = _loss_inputs()
    smooth = torch.zeros_like(pos)                       # smooth dvf ⇒ ~0 diffusion
    preds = {"world_points": pos, "dvfs": smooth}
    out = compute_volume_intensity_loss(preds, batch, grid_shape=grid, tv_weight=0.0, diffusion_weight=10.0)
    assert out["loss_diffusion"].item() == pytest.approx(0.0, abs=1e-6)  # used dvfs (smooth), not pos


# ───────────────────────── (B) BSplineWarpHead ─────────────────────────

def _fake_tokens(B=1, S=3, dim_in=2048, patch_start_idx=5, patch_hw=37):
    N = patch_start_idx + patch_hw * patch_hw
    return [torch.randn(B, S, N, dim_in, requires_grad=True)]


def test_bspline_head_output_contract_matches_dpt():
    B, S, H, W = 1, 3, 518, 518
    head = BSplineWarpHead(dim_in=2048, grid_size=32, activation="linear")
    toks = _fake_tokens(B, S)
    preds, conf = head(toks, images=torch.zeros(B, S, 3, H, W), patch_start_idx=5)
    assert preds.shape == (B, S, H, W, 3)
    assert conf.shape == (B, S, H, W)


def test_bspline_head_is_differentiable():
    head = BSplineWarpHead(dim_in=2048, grid_size=32)
    toks = _fake_tokens()
    preds, _ = head(toks, images=torch.zeros(1, 3, 3, 518, 518), patch_start_idx=5)
    preds.sum().backward()
    assert toks[0].grad is not None
    assert all(p.grad is not None for p in head.parameters())


def test_bspline_head_output_is_smooth():
    """The dense Δ must carry almost no energy at the 14-px patch period (smooth by
    construction) — the defect a DPT decoder would imprint."""
    head = BSplineWarpHead(dim_in=2048, grid_size=32)
    toks = _fake_tokens(B=1, S=3)
    preds, _ = head(toks, images=torch.zeros(1, 3, 3, 518, 518), patch_start_idx=5)
    d = preds.detach()[0, 0, ..., 0].numpy()
    spec = np.abs(np.fft.rfft(d - d.mean(), axis=1)).mean(0)
    k = 518 // 14
    band_frac = spec[k - 1:k + 2].sum() / spec.sum()
    assert band_frac < 0.02, f"14px-band energy {band_frac:.4f} too high — not smooth"


def test_bspline_head_is_smaller_than_dpt():
    bsp = sum(p.numel() for p in BSplineWarpHead(dim_in=2048, grid_size=32).parameters())
    dpt = sum(p.numel() for p in DPTHead(dim_in=2048, output_dim=4).parameters())
    assert bsp < dpt / 10, f"bspline {bsp} should be ≪ dpt {dpt}"


# ───────────────────────── (B) model construction gate ─────────────────────────

@pytest.fixture(scope="module")
def _vggt():
    from vggt.models.vggt import VGGT
    return VGGT  # class, built per-test (construction exercises the gate)


def test_warp_head_gate_selects_bspline(_vggt):
    m = _vggt(enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
              train_on_residual_dvf=True, warp_head_type="bspline", bspline_grid_size=32)
    assert isinstance(m.point_head, BSplineWarpHead)


def test_warp_head_gate_default_is_dpt(_vggt):
    m = _vggt(enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
              train_on_residual_dvf=True)
    assert isinstance(m.point_head, DPTHead)


def test_warp_head_gate_rejects_unknown(_vggt):
    with pytest.raises(ValueError):
        _vggt(enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
              warp_head_type="nope")
