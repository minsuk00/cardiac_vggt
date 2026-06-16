"""Tests for the optional 3D UNet refiner.

Most tests use tiny hand-built tensors (no heavy aggregator forward) to exercise the
splat helper + two-term loss + the refiner module directly. One freeze test instantiates
the full VGGT (construction only, no forward) to confirm the freeze patterns isolate the
refiner. The headline guarantee: with the refiner OFF, the loss path is unchanged.
"""
import os

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from vggt.utils.splat import splat_predictions
from vggt.models.refiner import VolumeRefiner
from loss import compute_volume_intensity_loss, MultitaskLoss

GRID = (4, 8, 8)


def _toy(B=1, S=2, H=8, W=8, seed=0):
    """A tiny predictions/batch pair on CPU (grid 4x8x8) — fast, no model forward."""
    g = torch.Generator().manual_seed(seed)
    D, Hv, Wv = GRID
    world_points = (torch.rand(B, S, H, W, 3, generator=g) * 2 - 1)  # [-1,1]
    images = torch.rand(B, S, 3, H, W, generator=g)                  # [0,1]
    scanner = (torch.rand(B, S, H, W, 3, generator=g) * 2 - 1)
    gt = torch.rand(B, D, Hv, Wv, generator=g)
    bbox = torch.tensor([[0, D, 0, Hv, 0, Wv]], dtype=torch.int64).repeat(B, 1)
    phases = torch.rand(B, 6, D, Hv, Wv, generator=g)
    preds = {"world_points": world_points}
    batch = {"images": images, "gt_target_volume": gt, "scanner_coords": scanner,
             "anatomy_bbox": bbox, "phases": phases}
    return preds, batch


# ─────────────────────────────────────────────────────────────────────────────
# Refiner module
# ─────────────────────────────────────────────────────────────────────────────
def test_refiner_residual_identity_at_init():
    r = VolumeRefiner(in_channels=2, base_channels=16, levels=2, use_coverage=True)
    Vc = torch.randn(1, *GRID)
    cov = torch.rand(1, *GRID)
    out = r(Vc, cov)
    assert out.shape == Vc.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out, Vc, atol=1e-6), "zero-init output conv ⇒ V_refined == V_canon at init"
    # last conv truly zero-initialized
    assert torch.count_nonzero(r.out_conv.weight) == 0
    assert torch.count_nonzero(r.out_conv.bias) == 0


def test_refiner_one_channel_runs():
    r = VolumeRefiner(in_channels=1, base_channels=8, levels=2, use_coverage=False)
    Vc = torch.randn(2, *GRID)
    out = r(Vc)
    assert out.shape == Vc.shape and torch.allclose(out, Vc, atol=1e-6)


def test_refiner_gradients_flow():
    r = VolumeRefiner(in_channels=1, base_channels=8, levels=2, use_coverage=False)
    Vc = torch.randn(1, *GRID, requires_grad=True)
    gt = torch.rand(1, *GRID)
    # break the zero-init so the output conv gets gradient
    with torch.no_grad():
        r.out_conv.weight.add_(0.01)
    loss = (r(Vc) - gt).abs().mean()
    loss.backward()
    grads = [p.grad for p in r.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(grads) > 0, "refiner params should receive gradient"


# ─────────────────────────────────────────────────────────────────────────────
# Splat helper ↔ loss equivalence (the bitwise proof)
# ─────────────────────────────────────────────────────────────────────────────
def test_loss_uses_same_splat_helper():
    """compute_volume_intensity_loss (OFF path) must produce V_canon byte-identical to
    splat_predictions — the same helper VGGT.forward uses when the refiner is ON."""
    preds, batch = _toy()
    V_helper, _ = splat_predictions(preds, batch, GRID)
    out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID, tv_weight=0.1)
    assert torch.equal(out["V_canon"], V_helper)


def test_off_path_has_no_refiner_keys():
    """No V_canon in predictions ⇒ no V_refined / loss_refiner ⇒ objective unchanged."""
    preds, batch = _toy()
    out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID, tv_weight=0.1)
    assert "loss_refiner" not in out
    assert "V_refined" not in out
    assert "metric_psnr_3d_full_refined" not in out


def test_consume_model_provided_volumes():
    """When predictions carries V_canon/coverage (refiner path), the loss reuses them
    rather than re-splatting."""
    preds, batch = _toy()
    Vc, cov = splat_predictions(preds, batch, GRID)
    preds2 = dict(preds, V_canon=Vc, coverage=cov)
    out = compute_volume_intensity_loss(preds2, batch, grid_shape=GRID, tv_weight=0.1)
    assert out["V_canon"] is Vc  # reused, not recomputed


# ─────────────────────────────────────────────────────────────────────────────
# Two-term deep-supervised loss + λ
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("lam", [0.0, 1.0, 2.0])
def test_two_term_loss_lambda(lam):
    preds, batch = _toy()
    Vc, cov = splat_predictions(preds, batch, GRID)
    V_refined = Vc + 0.1  # a deterministic "refined" volume
    preds2 = dict(preds, V_canon=Vc, coverage=cov, V_refined=V_refined)
    out = compute_volume_intensity_loss(preds2, batch, grid_shape=GRID, tv_weight=0.1,
                                        refiner_lambda=lam)
    expected = lam * (V_refined - batch["gt_target_volume"]).abs().mean()
    assert torch.allclose(out["loss_refiner"], expected)
    # refined metrics present
    assert "metric_psnr_3d_full_refined" in out
    assert "metric_psnr_3d_bbox_refined" in out
    assert "metric_psnr_3d_motion_refined" in out


def test_multitask_objective_includes_refiner():
    preds, batch = _toy()
    Vc, cov = splat_predictions(preds, batch, GRID)
    V_refined = Vc + 0.1
    preds2 = dict(preds, V_canon=Vc, coverage=cov, V_refined=V_refined)
    volcfg = {"weight": 1.0, "grid_shape": GRID, "tv_weight": 0.1, "refiner_lambda": 1.0}
    mtl = MultitaskLoss(camera={"weight": 0.0}, depth={"weight": 0.0},
                        point={"weight": 0.0}, volume=volcfg)
    ld = mtl(preds2, batch)
    expected = (ld["loss_volume"] + ld["loss_pos_tv"] + ld["loss_refiner"]) * 1.0
    assert torch.allclose(ld["objective"], expected)

    # And OFF (no V_refined): objective excludes the refiner term.
    out_off = compute_volume_intensity_loss(preds, batch, grid_shape=GRID, tv_weight=0.1)
    ld_off = mtl(preds, batch)
    expected_off = (out_off["loss_volume"] + out_off["loss_pos_tv"]) * 1.0
    assert "loss_refiner" not in ld_off
    assert torch.allclose(ld_off["objective"], expected_off)


# ─────────────────────────────────────────────────────────────────────────────
# Freeze isolation (full VGGT, construction only — no forward)
# ─────────────────────────────────────────────────────────────────────────────
def _build_vggt(enable_refiner):
    from vggt.models.vggt import VGGT
    return VGGT(img_size=518, patch_size=14, embed_dim=1024,
                enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                use_z_pose_embedding=True, use_t_pose_embedding=False,
                use_target_t_pose_embedding=True, train_on_residual_dvf=True,
                enable_refiner=enable_refiner, grid_shape=(12, 256, 256),
                refiner_base_channels=16, refiner_levels=2, refiner_use_coverage=True)


def test_refiner_off_has_no_refiner_submodule():
    model = _build_vggt(enable_refiner=False)
    assert model.refiner is None
    assert not any("refiner" in n for n, _ in model.named_parameters())


@pytest.mark.slow
def test_freeze_only_refiner_trainable():
    """Mode A: freeze everything except the refiner ⇒ only refiner.* trains."""
    from train_utils.freeze import freeze_modules
    model = _build_vggt(enable_refiner=True)
    freeze_modules(model, patterns=["*patch_embed*", "*camera_token*", "*aggregator*", "*point_head*"])
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, "expected some trainable params"
    assert all(n.startswith("refiner.") for n in trainable), \
        f"non-refiner params still trainable: {sorted(n for n in trainable if not n.startswith('refiner.'))[:5]}"
    # and the refiner has > 0 trainable params
    n_ref = sum(p.numel() for n, p in model.named_parameters() if n.startswith("refiner.") and p.requires_grad)
    assert n_ref > 0


@pytest.mark.slow
def test_freeze_joint_trains_aggregator_point_and_refiner():
    """Mode B: freeze only patch_embed ⇒ aggregator + point_head + refiner train."""
    from train_utils.freeze import freeze_modules
    model = _build_vggt(enable_refiner=True)
    freeze_modules(model, patterns=["*patch_embed*"])
    def trainable(prefix):
        return sum(p.numel() for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad)
    assert trainable("refiner.") > 0
    assert trainable("point_head.") > 0
    assert trainable("aggregator.") > 0
    # patch_embed frozen
    nf = sum(p.numel() for n, p in model.named_parameters()
             if "patch_embed" in n and not p.requires_grad)
    assert nf > 0
