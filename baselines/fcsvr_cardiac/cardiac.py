"""Cardiac data/geometry adapter for the released FC-SVR Stage-1 model.

The released network uses external flow channels (x, y, z), whereas VGGT's
physical geometry is expressed as (D, H, W).  Conversions live here so the
published architecture, loss, compensation, and upsampling code stay intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vggt.utils.splat import splat_to_volume

Z_HALF_MM = 90.0
SLAB_SPACING_MM = (3.0, 3.0, 3.0)


@dataclass(frozen=True)
class Stage1Meta:
    native_depth: int
    stage1_depth: int
    pad_before: int
    pad_after: int
    native_hw: tuple[int, int]
    slab_spacing_mm: tuple[float, float, float] = SLAB_SPACING_MM


def augment_reference_inplane(
    phases_tdhw: torch.Tensor,
    content_mask_dhw: torch.Tensor,
    heart_mask_dhw: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Applicable released FC-SVR reference augmentations on native CMR.

    One transform is shared across all phases and z planes: horizontal flip,
    ±10% zoom, in-plane rotation ±180 degrees, and translation ±13 pixels.
    Through-plane rotation is intentionally excluded because 12-mm native CMR
    cannot support the paper's isotropic 3D resampling without inventing data.
    """
    if phases_tdhw.ndim != 4 or content_mask_dhw.shape != phases_tdhw.shape[1:]:
        raise ValueError("expected phases (T,D,H,W) and matching masks (D,H,W)")
    if heart_mask_dhw.shape != content_mask_dhw.shape:
        raise ValueError("heart mask must match content mask")
    device = phases_tdhw.device
    dtype = torch.float32

    def uniform(low, high):
        return low + (high - low) * torch.rand((), device=device, generator=generator, dtype=dtype)

    angle = uniform(-torch.pi, torch.pi)
    scale = uniform(0.9, 1.1)
    tx_px = uniform(-13.0, 13.0)
    ty_px = uniform(-13.0, 13.0)
    flip = torch.rand((), device=device, generator=generator) < 0.5
    _, _, H, W = phases_tdhw.shape
    cosine, sine = torch.cos(angle) / scale, torch.sin(angle) / scale
    flip_sign = -1.0 if bool(flip) else 1.0
    theta = torch.stack([
        torch.stack([flip_sign * cosine, -sine, tx_px * 2 / (W - 1)]),
        torch.stack([flip_sign * sine, cosine, ty_px * 2 / (H - 1)]),
    ]).unsqueeze(0)

    def resample(x, mode):
        flat = x.reshape(-1, 1, H, W).float()
        grid = F.affine_grid(theta.expand(flat.shape[0], -1, -1), flat.shape, align_corners=True)
        return F.grid_sample(
            flat, grid, mode=mode, padding_mode="zeros", align_corners=True
        ).reshape(x.shape)

    phases_aug = resample(phases_tdhw, "bilinear").clamp(0, 1)
    content_aug = resample(content_mask_dhw, "nearest").gt(0.5).float()
    heart_aug = resample(heart_mask_dhw, "nearest").gt(0.5).float() * content_aug
    return phases_aug, content_aug, heart_aug


def augment_slice_intensity(slices_dhw: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Paper Appendix A.1 slice intensity augmentation: gamma plus σ=0.01 noise."""
    gamma = 0.9 + 0.1 * torch.rand(
        (), device=slices_dhw.device, generator=generator, dtype=torch.float32
    )
    noise = torch.randn(
        slices_dhw.shape, device=slices_dhw.device, generator=generator, dtype=torch.float32
    ) * 0.01
    return (slices_dhw.float().clamp(0, 1).pow(gamma) + noise).clamp(0, 1)


def native_to_stage1(
    slices_dhw: torch.Tensor,
    mask_dhw: torch.Tensor,
    displacement_dhw_mm: torch.Tensor,
    loss_mask_dhw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Stage1Meta]:
    """Build the paper-style coarse Stage-1 pair from a native CMR24 stack.

    Input slices are resampled in-plane 256-ish -> 120, repeated four times in
    depth, then subsampled by two in every dimension.  The resulting grid is
    `(2D, 60, 60)` at approximately 6-mm isotropic spacing.  Depths below 16
    are padded only in this temporary representation.
    """
    if slices_dhw.ndim != 3 or mask_dhw.shape != slices_dhw.shape:
        raise ValueError("slices_dhw and mask_dhw must have identical (D,H,W) shapes")
    depth = slices_dhw.shape[0]
    if displacement_dhw_mm.shape != (depth, 3):
        raise ValueError(f"displacement_dhw_mm must be ({depth},3) in (D,H,W) order")
    if loss_mask_dhw is None:
        loss_mask_dhw = mask_dhw
    if loss_mask_dhw.shape != slices_dhw.shape:
        raise ValueError("loss_mask_dhw must match slices_dhw")

    slab = F.interpolate(
        slices_dhw[None, None].float(), size=(depth, 120, 120), mode="trilinear", align_corners=True
    )[0, 0].repeat_interleave(4, dim=0)
    slab_mask = F.interpolate(
        mask_dhw[None, None].float(), size=(depth, 120, 120), mode="nearest"
    )[0, 0].repeat_interleave(4, dim=0)
    slab_loss_mask = F.interpolate(
        loss_mask_dhw[None, None].float(), size=(depth, 120, 120), mode="nearest"
    )[0, 0].repeat_interleave(4, dim=0)
    coarse = slab[::2, ::2, ::2]
    coarse_mask = slab_mask[::2, ::2, ::2]
    coarse_loss_mask = slab_loss_mask[::2, ::2, ::2]
    # Released real-stack inference multiplies the resized intensity by the
    # resized foreground. Apply after interpolation so bilinear edge bleed
    # cannot reintroduce non-foreground intensity.
    coarse = coarse * coarse_mask

    # FC-SVR's public tensor convention is (x,y,z); the loss flips it to match
    # the internal (D,H,W) grid.  At the coarse scale one voxel is ~6 mm.
    disp_xyz_vox = displacement_dhw_mm[:, [2, 1, 0]].float() / 6.0
    target_flow = disp_xyz_vox.repeat_interleave(2, dim=0)
    target_flow = target_flow.T[:, :, None, None].expand(3, 2 * depth, 60, 60).clone()

    pad_total = max(0, 16 - 2 * depth)
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    if pad_total:
        coarse = F.pad(coarse, (0, 0, 0, 0, pad_before, pad_after))
        coarse_mask = F.pad(coarse_mask, (0, 0, 0, 0, pad_before, pad_after))
        coarse_loss_mask = F.pad(coarse_loss_mask, (0, 0, 0, 0, pad_before, pad_after))
        target_flow = F.pad(target_flow, (0, 0, 0, 0, pad_before, pad_after))

    inputs = torch.stack([coarse, coarse_mask], dim=0)
    target = torch.cat([target_flow, coarse_loss_mask.unsqueeze(0)], dim=0)
    meta = Stage1Meta(depth, 2 * depth, pad_before, pad_after, tuple(slices_dhw.shape[-2:]))
    return inputs, target, meta


def extract_respiratory_slices_256(
    phases: torch.Tensor,
    t_seq: torch.Tensor,
    z_seq: torch.Tensor,
    disp_dhw_mm: torch.Tensor,
    spacing_mm: tuple[float, float, float],
) -> torch.Tensor:
    """VGGT respiratory translation/reslice geometry without its DINO output tail.

    Returns grayscale `(B,S,H,W)` at the source grid resolution. `disp_dhw_mm`
    contains one three-component translation per acquired slice; it is expanded
    over the dense sampling grid only because `grid_sample` needs one source
    coordinate per output pixel. It is not a predicted dense deformation field.
    """
    if phases.ndim != 5:
        raise ValueError("phases must have shape (B,T,D,H,W)")
    B, _, D, H, W = phases.shape
    if t_seq.shape != z_seq.shape or t_seq.ndim != 2:
        raise ValueError("t_seq and z_seq must have identical (B,S) shapes")
    S = t_seq.shape[1]
    if t_seq.shape[0] != B or disp_dhw_mm.shape != (B, S, 3):
        raise ValueError("disp_dhw_mm must have shape (B,S,3)")
    if D < 2 or H < 2 or W < 2:
        raise ValueError("respiratory reslicing requires every spatial dimension >= 2")

    device = phases.device
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, S)
    volumes = phases.float()[b_idx, t_seq].reshape(B * S, 1, D, H, W)
    disp = disp_dhw_mm.to(device=device, dtype=torch.float32).reshape(B * S, 3)

    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    y_base = (ys / (H - 1) * 2 - 1).view(1, H, 1).expand(1, H, W)
    x_base = (xs / (W - 1) * 2 - 1).view(1, 1, W).expand(1, H, W)
    z_base = (z_seq.float() / (D - 1) * 2 - 1).reshape(B * S)

    dz = disp[:, 0] / spacing_mm[0] * (2 / (D - 1))
    dy = disp[:, 1] / spacing_mm[1] * (2 / (H - 1))
    dx = disp[:, 2] / spacing_mm[2] * (2 / (W - 1))
    z_coord = (z_base + dz).view(B * S, 1, 1, 1).expand(B * S, 1, H, W)
    y_coord = y_base.unsqueeze(0).expand(B * S, 1, H, W) + dy.view(-1, 1, 1, 1)
    x_coord = x_base.unsqueeze(0).expand(B * S, 1, H, W) + dx.view(-1, 1, 1, 1)
    grid = torch.stack([x_coord, y_coord, z_coord], dim=-1)
    return F.grid_sample(
        volumes, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    ).reshape(B, S, H, W)


def remove_stage1_padding(flow_xyz: torch.Tensor, meta: Stage1Meta) -> torch.Tensor:
    """Remove temporary coarse-grid depth padding before released upsampling."""
    if flow_xyz.ndim != 5 or flow_xyz.shape[1] != 3:
        raise ValueError("flow_xyz must have shape (B,3,D,H,W)")
    start = meta.pad_before
    return flow_xyz[:, :, start : start + meta.stage1_depth]


def stage1_flow_to_native_mm(flow_xyz: torch.Tensor, meta: Stage1Meta) -> torch.Tensor:
    """Convert released slab flow to dense native-grid physical motion.

    `flow_xyz` is the output of the released `upsample_flow()` and is therefore
    in 3-mm slab voxels. Four repeated slab planes are averaged at each in-plane
    location; H/W are never averaged. Channels change from (x,y,z) to (D,H,W),
    voxel displacements become millimetres, and the dense field is interpolated
    from the 120-grid back to the original native in-plane grid.
    """
    if flow_xyz.ndim != 5 or flow_xyz.shape[1] != 3:
        raise ValueError("flow_xyz must have shape (B,3,4D,H,W)")
    if flow_xyz.shape[2] != 4 * meta.native_depth:
        raise ValueError(f"expected slab depth {4 * meta.native_depth}, got {flow_xyz.shape[2]}")
    B, _, _, H, W = flow_xyz.shape
    per_plane_xyz = flow_xyz.reshape(B, 3, meta.native_depth, 4, H, W).mean(dim=3)
    spacing_xyz = flow_xyz.new_tensor(meta.slab_spacing_mm[::-1]).view(1, 3, 1, 1, 1)
    physical_dhw = (per_plane_xyz * spacing_xyz)[:, [2, 1, 0]]
    dense = F.interpolate(
        physical_dhw.permute(0, 2, 1, 3, 4).reshape(B * meta.native_depth, 3, H, W),
        size=meta.native_hw, mode="bilinear", align_corners=True,
    )
    return dense.reshape(B, meta.native_depth, 3, *meta.native_hw).permute(0, 1, 3, 4, 2)


def motion_metrics(
    predicted_dhw_mm: torch.Tensor,
    target_dhw_mm: torch.Tensor,
    foreground_mask: torch.Tensor,
    *,
    slab_spacing_mm: float,
) -> dict[str, float]:
    """Adapted native-grid motion errors on foreground in mm and slab voxels."""
    if predicted_dhw_mm.ndim != 5 or predicted_dhw_mm.shape[-1] != 3:
        raise ValueError("predicted_dhw_mm must have shape (B,D,H,W,3)")
    B, D, H, W, _ = predicted_dhw_mm.shape
    if target_dhw_mm.shape != (B, D, 3) or foreground_mask.shape != (B, D, H, W):
        raise ValueError("target/mask shapes do not match predicted motion")
    valid = foreground_mask.bool().unsqueeze(-1)
    if not valid.any():
        raise ValueError("motion metrics require non-empty foreground")
    error_mm = predicted_dhw_mm - target_dhw_mm[:, :, None, None, :]
    selected_mm = error_mm.masked_select(valid.expand_as(error_mm)).reshape(-1, 3)
    error_vox = selected_mm / slab_spacing_mm
    return {
        "metric_motion_component_mse_mm2": selected_mm.square().mean().item(),
        "metric_motion_paper_mse_mm2": selected_mm.square().sum(-1).mean().item(),
        "metric_motion_epe_mm": selected_mm.square().sum(-1).sqrt().mean().item(),
        "metric_motion_component_mse_slab_vox2": error_vox.square().mean().item(),
        "metric_motion_paper_mse_slab_vox2": error_vox.square().sum(-1).mean().item(),
        "metric_motion_epe_slab_vox": error_vox.square().sum(-1).sqrt().mean().item(),
    }


def released_coarse_motion_metrics(
    predicted_xyz_vox: torch.Tensor,
    target_xyz_mask: torch.Tensor,
) -> dict[str, float]:
    try:
        from .models.losses import l21_loss_affine_invariant, l22_loss_affine_invariant
    except ImportError:  # Script entrypoints put this fork directly on PYTHONPATH.
        from models.losses import l21_loss_affine_invariant, l22_loss_affine_invariant

    return {
        "metric_released_coarse_l22_component_vox2": l22_loss_affine_invariant(
            predicted_xyz_vox, target_xyz_mask, eps=0
        ).item(),
        "metric_released_coarse_l21_epe_vox": l21_loss_affine_invariant(
            predicted_xyz_vox, target_xyz_mask, eps=0
        ).item(),
    }


def compensate_motion(out_xyz: torch.Tensor, target_xyz_mask: torch.Tensor) -> torch.Tensor:
    """Released paper compensation, factored out for testable cardiac use."""
    batch, chans, *size = out_xyz.shape
    if chans != 3 or target_xyz_mask.shape[:2] != (batch, 4):
        raise ValueError("expected out (B,3,D,H,W) and target (B,4,D,H,W)")
    if batch != 1:
        raise ValueError("released compensation is only valid for the Stage-1 batch size of 1")
    grid = torch.stack(
        torch.meshgrid(
            [torch.arange(1.0, s + 1, device=out_xyz.device) for s in size], indexing="ij"
        )
    ).to(out_xyz.dtype)
    mask = target_xyz_mask[:, chans:]
    if mask.count_nonzero() < 4:
        raise ValueError("compensation needs at least four foreground voxels for a rigid fit")
    B = (out_xyz.flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1)
    A = (target_xyz_mask[:, :chans].flip(1) + grid).masked_select(mask.bool()).reshape(batch, chans, -1)
    mean_B = B.mean(-1, keepdim=True).detach()
    mean_A = A.mean(-1, keepdim=True).detach()
    svd = torch.linalg.svd(
        torch.linalg.lstsq((B - mean_B).transpose(1, 2), (A - mean_A).transpose(1, 2)).solution.detach()
    )
    rotation = (svd.U @ svd.S.sign().diag_embed() @ svd.Vh).transpose(1, 2)
    aligned = out_xyz.flip(1) + grid - mean_B.view(batch, chans, 1, 1, 1)
    aligned = (rotation @ aligned.flatten(2)).unflatten(2, out_xyz.shape[2:])
    return (aligned - grid + mean_A.view(batch, chans, 1, 1, 1)).flip(1)


def reconstruct_native(
    slices: torch.Tensor,
    displacement_dhw_mm: torch.Tensor,
    *,
    spacing_mm: tuple[float, float, float] = (12.0, 1.4, 1.4),
    foreground_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splat each original corrupted slice once onto its native `(D,H,W)` grid."""
    if slices.ndim == 3:
        slices = slices.unsqueeze(0)
    if slices.ndim != 4:
        raise ValueError("slices must have shape (D,H,W) or (B,D,H,W)")
    B, D, H, W = slices.shape
    if displacement_dhw_mm.shape == (B, D, 3):
        displacement_dhw_mm = displacement_dhw_mm[:, :, None, None, :].expand(B, D, H, W, 3)
    elif displacement_dhw_mm.shape != (B, D, H, W, 3):
        raise ValueError(
            f"displacement_dhw_mm must have shape ({B},{D},3) or ({B},{D},{H},{W},3)"
        )

    z = torch.arange(D, device=slices.device, dtype=torch.float32)
    y = torch.linspace(-1, 1, H, device=slices.device)
    x = torch.linspace(-1, 1, W, device=slices.device)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    base_z = (zz - (D - 1) / 2) * spacing_mm[0] / Z_HALF_MM
    base = torch.stack([xx, yy, base_z], dim=-1).unsqueeze(0).expand(B, -1, -1, -1, -1)
    disp = displacement_dhw_mm
    delta = torch.stack(
        [
            disp[..., 2] * 2 / (spacing_mm[2] * (W - 1)),
            disp[..., 1] * 2 / (spacing_mm[1] * (H - 1)),
            disp[..., 0] / Z_HALF_MM,
        ],
        dim=-1,
    )
    pos = (base + delta).reshape(B, -1, 3)
    intensity = slices.float().reshape(B, -1)
    if foreground_mask is None:
        weight = (intensity > 1e-3).float()
    else:
        if foreground_mask.shape != slices.shape:
            raise ValueError("foreground_mask must match slices")
        weight = foreground_mask.float().reshape(B, -1)
    return splat_to_volume(pos, intensity, (D, H, W), Z_HALF_MM / spacing_mm[0], weight=weight)
