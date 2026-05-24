"""Differentiable trilinear splat and sample for slice-to-volume reconstruction."""

import torch
import torch.nn.functional as F


def splat_to_volume(pos, intensity, grid_shape, weight=None):
    """Trilinear scatter of (position, intensity) pairs into a 3D grid.

    Args:
        pos: (B, N, 3) normalized to [-1, 1] in (x, y, z) order (grid_sample convention).
        intensity: (B, N) scalar per point.
        grid_shape: (D, H, W) target voxel grid.
        weight: (B, N) optional per-point gate ∈ [0, 1]. Points with weight=0 contribute
                to neither the intensity numerator nor the coverage denominator.

    Returns:
        volume: (B, D, H, W) accumulated intensity divided by accumulated weight.
        coverage: (B, D, H, W) accumulated trilinear weight per voxel.
    """
    # Force fp32 regardless of outer autocast — bf16 (7-bit mantissa) loses precision
    # after thousands of scatter_add contributions per voxel, capping achievable PSNR.
    pos = pos.float()
    intensity = intensity.float()
    if weight is not None:
        weight = weight.float()

    B, N, _ = pos.shape
    D, H, W = grid_shape
    device = pos.device
    dtype = intensity.dtype

    # Normalized [-1, 1] → continuous voxel coords [0, W-1] etc.
    px = (pos[..., 0] + 1) * 0.5 * (W - 1)
    py = (pos[..., 1] + 1) * 0.5 * (H - 1)
    pz = (pos[..., 2] + 1) * 0.5 * (D - 1)

    # Floor for indices; keep raw floats for weight computation.
    x0f = torch.floor(px)
    y0f = torch.floor(py)
    z0f = torch.floor(pz)

    wx1 = px - x0f
    wy1 = py - y0f
    wz1 = pz - z0f
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    # In-bounds mask: zero out any point whose 8-neighborhood reaches outside the volume.
    in_bounds = (
        (x0f >= 0) & (x0f <= W - 2)
        & (y0f >= 0) & (y0f <= H - 2)
        & (z0f >= 0) & (z0f <= D - 2)
    ).to(dtype)
    if weight is not None:
        in_bounds = in_bounds * weight.to(dtype)

    # Clamp indices for safe scatter; weights are gated by in_bounds below.
    x0 = x0f.long().clamp(0, W - 2); x1 = x0 + 1
    y0 = y0f.long().clamp(0, H - 2); y1 = y0 + 1
    z0 = z0f.long().clamp(0, D - 2); z1 = z0 + 1

    volume = torch.zeros(B, D, H, W, device=device, dtype=dtype)
    coverage = torch.zeros_like(volume)

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, N)

    corners = [
        (z0, y0, x0, in_bounds * wz0 * wy0 * wx0),
        (z0, y0, x1, in_bounds * wz0 * wy0 * wx1),
        (z0, y1, x0, in_bounds * wz0 * wy1 * wx0),
        (z0, y1, x1, in_bounds * wz0 * wy1 * wx1),
        (z1, y0, x0, in_bounds * wz1 * wy0 * wx0),
        (z1, y0, x1, in_bounds * wz1 * wy0 * wx1),
        (z1, y1, x0, in_bounds * wz1 * wy1 * wx0),
        (z1, y1, x1, in_bounds * wz1 * wy1 * wx1),
    ]

    vol_flat = volume.view(-1)
    cov_flat = coverage.view(-1)
    for (z, y, x, w) in corners:
        flat_idx = ((b_idx * D + z) * H + y) * W + x
        vol_flat.scatter_add_(0, flat_idx.reshape(-1), (w * intensity).reshape(-1))
        cov_flat.scatter_add_(0, flat_idx.reshape(-1), w.reshape(-1))

    # Use additive epsilon (not clamp) so the gradient w.r.t. coverage stays smooth
    # at very low-coverage voxels; clamp would zero the gradient and produce a
    # discontinuous loss landscape at the coverage threshold.
    volume = volume / (coverage + 1e-6)
    return volume, coverage


def sample_volume(volume, pos):
    """Trilinear sample of a 3D volume at given normalized positions.

    Args:
        volume: (B, D, H, W) scalar volume.
        pos: (B, N, 3) in [-1, 1], (x, y, z) order.

    Returns:
        sampled: (B, N) interpolated intensities.
    """
    B, D, H, W = volume.shape
    N = pos.shape[1]
    v = volume.unsqueeze(1)  # (B, 1, D, H, W)
    grid = pos.view(B, N, 1, 1, 3)
    sampled = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.view(B, N)
