"""ARM twophase-diff: two-phase difference loss (docs/9x).

Guards the two contracts the arm depends on:
  1. `build_twophase_inputs` produces a render-2 input set that is bit-identical to
     render 1 on slots 1..S-1 (shared tensors) and differs ONLY in slot 0 (the
     reference slice, re-extracted at t2 = (t1 + T//2) % T with the SAME respiratory
     displacement), plus the matching `gt_target_volume_2`.
  2. `MultitaskLoss` with `predictions_2` adds exactly one heart-ROI-weighted L1 term
     on the between-phase difference, and with `predictions_2=None` (the val path and
     every diff_weight=0 run) is bit-identical to the pre-arm behavior.
"""

import pytest
import torch

from data.gpu_aug import (
    build_twophase_inputs,
    extract_slices_from_phases,
    gpu_augment_batch,
)
from data.respiratory import extract_slices_with_respiratory_vec
from loss import MultitaskLoss


T, D, H = 12, 4, 16   # tiny synthetic cine: T phases, D planes, H=W in-plane
R = 24                # model-input resolution (distinct from H to exercise the resize)
S = D                 # one-frame-per-slice: S == D


def _make_batch(device="cpu"):
    """Post-dataset, pre-gpu_aug batch (defer_input_images style: no images yet)."""
    g = torch.Generator().manual_seed(0)
    # Phase-dependent content: every (t, z) plane is distinct so a wrong-phase or
    # wrong-plane extraction cannot silently pass the equality asserts.
    phases = torch.rand((1, T, D, H, H), generator=g)
    t1 = 3
    timesteps = torch.tensor([[t1, 1, 5, 9]], dtype=torch.int64)      # slot 0 = reference at t1
    slice_indices = torch.tensor([[2.0, 0.0, 1.0, 3.0]])              # slot 0 = mid plane
    batch = {
        "phases": phases,
        "timesteps": timesteps,
        "slice_indices": slice_indices,
        "t_target": torch.tensor([[t1]], dtype=torch.int64),
        "scanner_coords": torch.zeros((1, S, R, R, 3)),
        "dz_mm": torch.tensor([12.0]),
        "z_scale": torch.tensor([90.0 / 12.0]),
        "seq_index": torch.tensor([[0]], dtype=torch.int64),
    }
    return batch, t1


def _finalize(batch):
    """Run the real no-aug gpu_augment_batch path to build images/images_splat."""
    return gpu_augment_batch(batch, None, torch.device("cpu"),
                             respiratory_cfg=None, train=True)


class TestBuildTwophaseInputs:
    def test_shared_slots_bit_identical(self):
        batch, t1 = _make_batch()
        batch = _finalize(batch)
        batch = build_twophase_inputs(batch, torch.device("cpu"))
        assert torch.equal(batch["images_2"][:, 1:], batch["images"][:, 1:])
        assert torch.equal(batch["images_splat_2"][:, 1:], batch["images_splat"][:, 1:])

    def test_slot0_is_t2_content(self):
        batch, t1 = _make_batch()
        batch = _finalize(batch)
        batch = build_twophase_inputs(batch, torch.device("cpu"))
        t2 = (t1 + T // 2) % T
        assert int(batch["t_target_2"].reshape(-1)[0]) == t2
        # Slot 0 must differ from render 1 (phases are phase-dependent by construction)...
        assert not torch.equal(batch["images_splat_2"][:, 0], batch["images_splat"][:, 0])
        # ...and match a direct extraction at (t2, z0).
        direct = extract_slices_from_phases(
            batch["phases"].float(),
            torch.tensor([[t2]], dtype=torch.int64),
            batch["slice_indices"][:, :1],
            out_size=H)[..., 0] / 255.0
        assert torch.equal(batch["images_splat_2"][:, :1], direct)

    def test_gt_target_volume_2(self):
        batch, t1 = _make_batch()
        batch = _finalize(batch)
        batch = build_twophase_inputs(batch, torch.device("cpu"))
        t2 = (t1 + T // 2) % T
        assert torch.equal(batch["gt_target_volume_2"], batch["phases"].float()[:, t2])

    def test_respiratory_disp_shared(self):
        """When resp is on, render 2's slot 0 must use the SAME displacement."""
        batch, t1 = _make_batch()
        batch = _finalize(batch)
        # Simulate the resp branch's output: shift every slot by a known disp and
        # surface it exactly the way gpu_augment_batch does (resp_disp_mm).
        disp = torch.tensor([[[3.0, 2.0, -1.5]] * S])          # (B, S, 3) mm
        native = extract_slices_with_respiratory_vec(
            batch["phases"].float(), batch["timesteps"], batch["slice_indices"],
            disp, spacing=(12.0, 1.4, 1.4), out_size=H)[..., 0] / 255.0
        batch["images_splat"] = native
        batch["resp_disp_mm"] = disp
        batch = build_twophase_inputs(batch, torch.device("cpu"))
        t2 = (t1 + T // 2) % T
        direct = extract_slices_with_respiratory_vec(
            batch["phases"].float(),
            torch.tensor([[t2]], dtype=torch.int64),
            batch["slice_indices"][:, :1],
            disp[:, :1], spacing=(12.0, 1.4, 1.4), out_size=H)[..., 0] / 255.0
        assert torch.equal(batch["images_splat_2"][:, :1], direct)
        assert torch.equal(batch["images_splat_2"][:, 1:], native[:, 1:])


def _loss_batch_and_preds():
    """Minimal batch + predictions for the loss: identity-Δ predictions whose splat
    is deterministic, with render 2 sharing the SAME predictions/intensities so
    V_canon(t1) == V_canon(t2) and the analytic loss_diff reduces to the GT swing."""
    batch, t1 = _make_batch()
    batch = _finalize(batch)
    batch = build_twophase_inputs(batch, torch.device("cpu"))
    roi = torch.zeros((1, D, H, H), dtype=torch.uint8)
    roi[:, 1:3, 4:12, 4:12] = 1
    batch["heart_roi_canonical"] = roi
    batch["gt_target_volume"] = batch["phases"].float()[:, t1]
    # scanner-coordinate field at model res R (the world_points contract): exact
    # plane placement, Δ = 0.
    ys, xs = torch.meshgrid(torch.arange(R), torch.arange(R), indexing="ij")
    xy = torch.stack([xs, ys], -1).float() / (R - 1) * 2 - 1
    wp = torch.zeros((1, S, R, R, 3))
    wp[..., :2] = xy
    z_scale = float(batch["z_scale"][0])
    for s in range(S):
        z_idx = float(batch["slice_indices"][0, s])
        wp[0, s, :, :, 2] = (z_idx - (D - 1) / 2) / z_scale
    preds = {"world_points": wp.clone().requires_grad_(True)}
    return batch, preds


VOLUME_CFG = dict(weight=1.0, tv_weight=0.0, diffusion_weight=0.0,
                  gather_weight=0.0, heart_weight=0.0, corseg_weight=0.0)


class TestDiffLoss:
    def test_predictions_2_none_bit_identical(self):
        batch, preds = _loss_batch_and_preds()
        base = MultitaskLoss(volume=dict(VOLUME_CFG))(preds, batch)
        armd = MultitaskLoss(volume=dict(VOLUME_CFG, diff_weight=0.7))(preds, batch)
        assert torch.equal(base["objective"], armd["objective"])
        assert float(armd.get("loss_diff", 0.0)) == 0.0

    def test_analytic_value(self):
        batch, preds = _loss_batch_and_preds()
        w = 0.7
        # Render 2 with IDENTICAL predictions and intensities → V_canon_2 == V_canon
        # → dC = 0 → loss_diff = w * mean_roi |dG|.
        batch["images_splat_2"] = batch["images_splat"]
        preds2 = {"world_points": preds["world_points"]}
        out = MultitaskLoss(volume=dict(VOLUME_CFG, diff_weight=w))(
            preds, batch, predictions_2=preds2)
        roi = batch["heart_roi_canonical"].bool()
        dG = batch["gt_target_volume"].float() - batch["gt_target_volume_2"].float()
        expected = (dG.abs() * roi).sum() / roi.sum() * w
        assert torch.allclose(out["loss_diff"], expected, atol=1e-6)
        assert torch.allclose(out["objective"],
                              out["loss_volume"] + out["loss_pos_tv"] + out["loss_diff"],
                              atol=1e-6)

    def test_gradient_flows_through_both_renders(self):
        batch, preds = _loss_batch_and_preds()
        wp2 = preds["world_points"].detach().clone().requires_grad_(True)
        preds2 = {"world_points": wp2}
        out = MultitaskLoss(volume=dict(VOLUME_CFG, diff_weight=1.0))(
            preds, batch, predictions_2=preds2)
        out["objective"].backward()
        assert preds["world_points"].grad is not None
        assert wp2.grad is not None
        assert torch.isfinite(wp2.grad).all()

    def test_missing_images_splat_2_raises(self):
        batch, preds = _loss_batch_and_preds()
        del batch["images_splat_2"]
        with pytest.raises(RuntimeError, match="images_splat_2"):
            MultitaskLoss(volume=dict(VOLUME_CFG, diff_weight=1.0))(
                preds, batch, predictions_2={"world_points": preds["world_points"]})
