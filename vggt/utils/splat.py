"""Differentiable trilinear splat and sample for slice-to-volume reconstruction."""

import torch
import torch.nn.functional as F


def splat_to_volume(pos, intensity, grid_shape, z_scale, weight=None):
    """Trilinear scatter of (position, intensity) pairs into a 3D grid.

    Args:
        pos: (B, N, 3) normalized in (x, y, z) order (grid_sample convention). x/y are
             [-1, 1] over the fixed in-plane extent (same for every subject). z is
             PHYSICAL (z_mm / Z_HALF_MM) — NOT index-normalized, since D (this call's
             own grid_shape[0]) varies per subject under native-z (docs/58).
        intensity: (B, N) scalar per point.
        grid_shape: (D, H, W) target voxel grid — D is THIS SUBJECT's own native slice count.
        z_scale: REQUIRED, no default. Converts the physical z coordinate to a voxel-index
            delta: `z_scale = Z_HALF_MM / dz` (dz = this subject's own slice pitch, mm).
            A silent/wrong default would compress or stretch the volume with no error —
            see docs/58 §6.2. Plain python float (batch_size is always 1 in this pipeline).
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
    pz = pos[..., 2] * z_scale + (D - 1) * 0.5

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

    # In-bounds mask: check the CONTINUOUS position against the true valid domain
    # [0, size-1] (NOT the floored index against [0, size-2]). A point sitting EXACTLY
    # on the last voxel (e.g. px == W-1) is fully valid and needs no "next" neighbor to
    # interpolate — but the old floor-based check excluded it (no room for x0f+1 <= W-1),
    # silently dropping the boundary plane/row/column of every splat. Usually invisible
    # in x/y (518->256 oversampling covers for it) and harmless under the old fixed-12
    # z-grid (the top plane was often zero-padding), but under native-z D is each
    # subject's own real slice count, so the top z-plane is real anatomy for every
    # subject. Points genuinely beyond the domain are still correctly dropped.
    in_bounds = (
        (px >= 0) & (px <= W - 1)
        & (py >= 0) & (py <= H - 1)
        & (pz >= 0) & (pz <= D - 1)
    ).to(dtype)
    if weight is not None:
        in_bounds = in_bounds * weight.to(dtype)

    # Clamp BOTH corner indices into range. At an exact boundary x0 == x1, but the
    # weight on the "x1" corner (wx1 = px - x0f) is exactly 0 there, so this never
    # double-counts — all weight still lands on the single true plane.
    x0 = x0f.long().clamp(0, W - 1); x1 = (x0 + 1).clamp(0, W - 1)
    y0 = y0f.long().clamp(0, H - 1); y1 = (y0 + 1).clamp(0, H - 1)
    z0 = z0f.long().clamp(0, D - 1); z1 = (z0 + 1).clamp(0, D - 1)

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


def splat_predictions(predictions, batch, grid_shape, z_scale):
    """Splat per-pixel predicted positions + image intensities into V_canon.

    Pure function of `predictions["world_points"]` (B, S, H, W, 3) and `batch["images"]`
    (B, S, 3, H, W). Returns `(V_canon, coverage)` each (B, D, H, W). `splat_to_volume`
    forces fp32 internally.

    z_scale: required, see `splat_to_volume`.
    """
    pos_pred = predictions["world_points"]
    images = batch["images"]

    B, S, H, W, _ = pos_pred.shape
    # Rescale uint8-range [0, 255] inputs to [0, 1] (a no-op if already normalized), branchlessly:
    # a 0-D scalar factor scales ALL pixels uniformly, so contrast is preserved exactly and there is
    # no `if intensity.max() > 2.0` host-device sync (which also graph-broke every torch.compile).
    # MULTIPLY by the reciprocal rather than divide by 255.0: eager lowers `x / 255.0` (a Python
    # scalar) to `x * (1/255.0)`, but `x / tensor(255.0)` does a true division, which differs from it
    # by 1 ULP on ~74% of elements. Multiplying keeps this bit-identical to the pre-refactor pipeline.
    inv_scale = torch.where((images > 2.0).any(), 1.0 / 255.0, 1.0)
    intensity = images.float().mean(dim=2) * inv_scale

    pos_flat = pos_pred.reshape(B, S * H * W, 3)
    int_flat = intensity.reshape(B, S * H * W)

    # Intensity gate: exclude zero-intensity input pixels from BOTH the numerator and the
    # coverage denominator (padded-Z / off-FOV slots are all-zero; gating stops them diluting
    # V_canon once the model's Δ crosses Z planes).
    splat_weight = (int_flat > 1e-3).to(int_flat.dtype)

    grid_shape = tuple(grid_shape)
    V_canon, coverage = splat_to_volume(pos_flat, int_flat, grid_shape, z_scale, weight=splat_weight)
    return V_canon, coverage


def sample_volume(volume, pos, z_scale):
    """Trilinear sample of a 3D volume at given normalized positions.

    Args:
        volume: (B, D, H, W) scalar volume.
        pos: (B, N, 3), (x, y, z) order. x/y in [-1, 1] over the fixed in-plane extent;
             z is PHYSICAL (z_mm / Z_HALF_MM), same convention as `splat_to_volume`.
        z_scale: REQUIRED, no default — see `splat_to_volume`. `F.grid_sample` assumes
            [-1, 1] spans exactly this volume's own D planes (align_corners=True), which
            physical z does NOT satisfy directly, so z is first converted to a voxel index
            via z_scale (same formula as the push side) and then re-normalized to
            grid_sample's own per-call convention.

    Returns:
        sampled: (B, N) interpolated intensities.
    """
    B, D, H, W = volume.shape
    N = pos.shape[1]
    v = volume.unsqueeze(1)  # (B, 1, D, H, W)
    pz = pos[..., 2] * z_scale + (D - 1) * 0.5             # physical z_norm → voxel index
    gz = (pz / max(D - 1, 1)) * 2.0 - 1.0                  # voxel index → grid_sample's own [-1,1] for THIS D
    pos = torch.stack([pos[..., 0], pos[..., 1], gz], dim=-1)
    grid = pos.view(B, N, 1, 1, 3)
    sampled = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return sampled.view(B, N)
