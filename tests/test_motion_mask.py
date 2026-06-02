"""Tests for the cardiac-motion mask + motion-masked PSNR metric (val_motion panel)."""
import torch

from loss import compute_motion_mask, MOTION_MASK_TAU, compute_volume_intensity_loss


def test_motion_mask_selects_only_moving_voxels():
    # (B=1, T=4, D=2, H=3, W=3). One voxel ramps across phases (moving); one is bright
    # but constant across phases (static — must NOT be selected); rest are zero.
    phases = torch.zeros(1, 4, 2, 3, 3)
    phases[0, :, 0, 0, 0] = 0.9                                    # bright but static
    phases[0, :, 1, 1, 1] = torch.tensor([0.0, 0.1, 0.2, 0.3])     # swing 0.3 > tau
    mask = compute_motion_mask(phases, tau=0.05)
    assert mask.shape == (1, 2, 3, 3)
    assert mask.dtype == torch.bool
    assert bool(mask[0, 1, 1, 1])         # moving voxel selected
    assert not bool(mask[0, 0, 0, 0])     # bright-but-static voxel rejected
    assert int(mask.sum()) == 1           # exactly the one moving voxel


def test_motion_mask_threshold_boundary():
    phases = torch.zeros(1, 2, 1, 1, 2)
    phases[0, :, 0, 0, 0] = torch.tensor([0.0, 0.04])   # swing 0.04 < 0.05 → out
    phases[0, :, 0, 0, 1] = torch.tensor([0.0, 0.06])   # swing 0.06 > 0.05 → in
    mask = compute_motion_mask(phases)                  # default tau = MOTION_MASK_TAU
    assert MOTION_MASK_TAU == 0.05
    assert not bool(mask[0, 0, 0, 0])
    assert bool(mask[0, 0, 0, 1])


def _toy_batch(B=1, S=3, H=8, W=8, D=2, T=4, seed=0):
    torch.manual_seed(seed)
    images = torch.rand(B, S, 3, H, W)
    world_points = torch.rand(B, S, H, W, 3) * 2 - 1
    gt = torch.rand(B, D, H, W)
    phases = torch.full((B, T, D, H, W), 0.2)            # static baseline
    phases[:, :, 0, 0:3, 0:3] = torch.rand(B, T, 3, 3)   # a moving sub-region
    batch = {
        "images": images,
        "gt_target_volume": gt,
        "phases": phases,
        "scanner_coords": world_points,
    }
    return batch, world_points, (D, H, W)


def test_loss_emits_motion_metric_matching_manual():
    batch, wp, grid = _toy_batch()
    out = compute_volume_intensity_loss({"world_points": wp}, batch, grid_shape=grid, tv_weight=0.0)
    assert "metric_psnr_3d_motion" in out
    assert "metric_motion_frac" in out
    assert torch.isfinite(out["metric_psnr_3d_motion"])

    # Manually recompute masked PSNR over V_canon/V_gt and confirm the loss matches.
    mask = compute_motion_mask(batch["phases"])[0]
    Vc = out["V_canon"][0][mask]
    Vg = out["V_gt"][0][mask]
    mse = ((Vc - Vg) ** 2).mean().clamp(min=1e-10)
    psnr_manual = (10.0 * torch.log10(1.0 / mse)).item()
    assert abs(out["metric_psnr_3d_motion"].item() - psnr_manual) < 1e-3

    # The motion mask is a strict subset of the cube → motion frac < 1.
    assert 0.0 < out["metric_motion_frac"].item() < 1.0


def test_loss_skips_motion_metric_without_phases():
    batch, wp, grid = _toy_batch()
    del batch["phases"]
    out = compute_volume_intensity_loss({"world_points": wp}, batch, grid_shape=grid, tv_weight=0.0)
    assert "metric_psnr_3d_motion" not in out
    # Full/bbox path unaffected.
    assert "metric_psnr_3d_full" in out


def test_motion_metric_isolates_moving_region():
    # Build V_canon that is PERFECT inside the motion region and wrong outside.
    # motion PSNR should be ~inf-large, while a full-cube PSNR over the same volumes
    # is finite — proving the mask restricts to the moving voxels.
    B, D, H, W, T = 1, 2, 6, 6, 4
    phases = torch.full((B, T, D, H, W), 0.2)
    phases[:, :, 0, 0:2, 0:2] = torch.rand(B, T, 2, 2)   # moving region
    mask = compute_motion_mask(phases)[0]                # (D,H,W)
    V_gt = torch.rand(B, D, H, W)
    V_canon = V_gt.clone()
    V_canon[0][~mask] += 0.5                             # corrupt only static voxels

    Vc_m = V_canon[0][mask]; Vg_m = V_gt[0][mask]
    mse_m = ((Vc_m - Vg_m) ** 2).mean().clamp(min=1e-10)
    psnr_motion = (10.0 * torch.log10(1.0 / mse_m)).item()
    mse_full = ((V_canon - V_gt) ** 2).mean().clamp(min=1e-10)
    psnr_full = (10.0 * torch.log10(1.0 / mse_full)).item()
    assert psnr_motion > psnr_full + 30.0   # motion region clean, full cube corrupted
