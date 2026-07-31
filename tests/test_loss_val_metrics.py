"""Tests for the val-only diagnostic metrics in compute_volume_intensity_loss.

Two regressions motivated these:

1. The docs/38 ship-decision block (`recov_frac_heart` / `psnr_3d_static` /
   `hole_frac_heart` / `mse_heart_*`) is wrapped in `try/except Exception ->
   logging.warning` so it can never raise into the training loop. That also means a
   plain NameError inside it disables every one of those metrics *silently* — which is
   exactly what happened (an edit dropped the `intensity = ...` definition while keeping
   its consumer). Nothing else in the suite touches this block, so the whole suite stayed
   green. These tests assert both that the keys appear AND that the block logged no
   warning, so any future exception in there fails loudly.

2. `metric_psnr_3d_motion` averages over samples. A sample with zero moving voxels has
   mse == 0, which the `clamp(min=1e-10)` floor turns into 100 dB — silently inflating
   the batch mean. The pre-vectorization loop skipped such samples; the vectorized
   version must too.
"""

import torch

from loss import compute_volume_intensity_loss

D, HV, WV = 12, 64, 64
S, H, W = 4, 32, 32


def _val_batch(B=1, static_samples=(), device="cpu"):
    """A val-shaped batch: no grad on world_points, `phases` + `scanner_coords` present.

    `phases` mimics real cardiac data: a static tissue background everywhere plus a small
    time-varying "heart" box. That matters — with uniform-random phases every voxel clears
    MOTION_MASK_TAU, so the static control region (`psnr_3d_static`) would be empty and the
    metric legitimately absent.

    `static_samples` lists batch indices with no moving box at all (zero motion voxels).
    """
    torch.manual_seed(0)
    # scanner_coords: the pure geometric canonical mapping (a distinct tensor from
    # world_points, so the val block's `pos_pred is not scanner_coords` guard passes).
    zs = torch.linspace(-1, 1, S, device=device)
    ys = torch.linspace(-1, 1, H, device=device)
    xs = torch.linspace(-1, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([gx[None].expand(S, -1, -1),
                          gy[None].expand(S, -1, -1),
                          zs[:, None, None].expand(-1, H, W)], dim=-1)
    scanner_coords = coords[None].expand(B, -1, -1, -1, -1).contiguous()

    # Static tissue background, identical across all 12 phases (swing 0 < tau).
    tissue = torch.rand(B, 1, D, HV, WV, device=device) * 0.5 + 0.2   # all > 1e-3
    phases = tissue.repeat(1, 12, 1, 1, 1)
    # A moving "heart" box: sinusoidal over t with amplitude well above MOTION_MASK_TAU.
    beat = 0.3 * torch.sin(torch.arange(12, device=device, dtype=phases.dtype)
                           * (2 * torch.pi / 12))
    for b in range(B):
        if b in static_samples:
            continue
        phases[b, :, 3:8, 24:40, 24:40] += beat[:, None, None, None]

    return {
        "images": torch.rand(B, S, 3, H, W, device=device) * 255,
        "gt_target_volume": phases[:, 0].contiguous(),
        "scanner_coords": scanner_coords,
        "phases": phases,
        # z_scale = (D-1)/2 reproduces the same index-normalized z the linspace(-1,1,S)
        # scanner_coords above already assumes — see test_splat.py's convention note.
        "z_scale": torch.full((B,), (D - 1) / 2.0, device=device),
    }


def _run(batch):
    pos = batch["scanner_coords"] + 0.01          # a small Δ, and NOT the same object
    assert not pos.requires_grad                  # val gate: block only runs without grad
    return compute_volume_intensity_loss({"world_points": pos}, batch, tv_weight=0.1)


DOCS38_KEYS = [
    "metric_mse_heart_identity",
    "metric_mse_heart_model",
    "metric_mse_heart_oracle",
    "metric_hole_frac_heart",
    "metric_psnr_3d_static",
]


def test_docs38_ship_decision_metrics_present(caplog):
    """The docs/38 val metrics must be produced, and the block must not swallow an error."""
    with caplog.at_level("WARNING"):
        out = _run(_val_batch())

    swallowed = [r.message for r in caplog.records if "docs/38" in r.message]
    assert not swallowed, f"docs/38 val-metric block raised and was swallowed: {swallowed}"

    for k in DOCS38_KEYS:
        assert k in out, f"{k} missing — the docs/38 ship-decision block did not run"
        assert torch.isfinite(out[k]).all(), f"{k} is not finite"


def test_docs38_block_skipped_when_grad_enabled():
    """Train path (world_points requires grad) must skip the extra splats entirely."""
    batch = _val_batch()
    pos = (batch["scanner_coords"] + 0.01).requires_grad_(True)
    out = compute_volume_intensity_loss({"world_points": pos}, batch, tv_weight=0.1)
    for k in DOCS38_KEYS:
        assert k not in out, f"{k} was computed on the train path (should be val-only)"


def test_zero_motion_sample_excluded_from_motion_psnr():
    """A zero-motion sample must not contribute a 100 dB floor value to the mean."""
    # Sample 0 static (no moving voxels), sample 1 moving.
    two = _val_batch(B=2, static_samples=(0,))
    out_two = _run(two)

    # The same moving sample, alone. Splatting + metrics are per-sample independent,
    # so excluding sample 0 must reproduce this exactly.
    one = {k: v[1:2].contiguous() for k, v in two.items()}
    out_one = _run(one)

    assert torch.allclose(out_two["metric_psnr_3d_motion"], out_one["metric_psnr_3d_motion"],
                          atol=1e-4), (
        f"zero-motion sample leaked into the mean: B=2 gives "
        f"{out_two['metric_psnr_3d_motion'].item():.3f} dB vs "
        f"{out_one['metric_psnr_3d_motion'].item():.3f} dB for the moving sample alone")
    assert torch.allclose(out_two["metric_mae_3d_motion"], out_one["metric_mae_3d_motion"],
                          atol=1e-6)


def test_all_samples_static_does_not_fabricate_high_psnr():
    """With no motion anywhere, the metric must not report the 100 dB clamp floor."""
    out = _run(_val_batch(B=2, static_samples=(0, 1)))
    assert out["metric_psnr_3d_motion"] < 50.0, (
        f"fabricated {out['metric_psnr_3d_motion'].item():.1f} dB motion PSNR "
        "from a batch with zero moving voxels")
