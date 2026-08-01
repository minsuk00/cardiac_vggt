"""GOLDEN-VALUE guard for every masked metric in compute_volume_intensity_loss.

Why this file exists
--------------------
`loss.py` computes the same "square the error, average over a mask, convert to dB" five
times, in two deliberately different styles:

  * TRAIN-PATH blocks (bbox, motion) — vectorized `torch.where` + sum/count, BRANCHLESS.
    The branchlessness is load-bearing: a Python-level `if` here costs 4 graph breaks
    (measured, see the comment in loss.py), and these run every training step.
  * VAL-ONLY blocks (heartseg, docs/38 heart, docs/38 seg) — a Python `for b in range(B)`
    loop with boolean indexing. Gated on `not pos_pred.requires_grad`, so host syncs are
    free and the loop is simply clearer.

They also differ in clamping and in which samples count as valid. Merging them into shared
helpers is therefore easy to get subtly wrong — and a wrong merge shifts a HEADLINE number
(psnr_seg_gain_db, recov_frac, the docs/38 ship/no-ship rule) with NO failure signal: tests
pass, training runs, the curve just sits somewhere slightly different.

So: pin the exact values on a fixed synthetic batch, plus the degenerate cases that
distinguish the two styles (empty mask, degenerate bbox, non-finite voxel outside the mask).
Refactor freely — if these still pass, the refactor preserved behaviour.
"""

import math

import pytest
import torch

from loss import compute_volume_intensity_loss

DEVICE = "cpu"
B, S, D, H, W = 1, 4, 6, 16, 16
T = 12


def _batch(seed=0, D_=D, non_finite_outside_mask=False):
    """Deterministic synthetic batch. Small (6x16x16) so values are exactly reproducible."""
    g = torch.Generator().manual_seed(seed)
    # STATIC background + a genuinely moving sub-volume, so `compute_motion_mask` selects a
    # strict subset (motion_frac < 1). Random-per-phase noise would mark the whole cube as
    # moving and the motion metric would stop being distinguishable from the full one.
    static = torch.rand(B, 1, D_, H, W, generator=g)
    phases = static.expand(B, T, D_, H, W).clone()
    phases[:, :, 2:4, 4:12, 4:12] += torch.linspace(0, 1, T).view(1, T, 1, 1, 1)
    phases = phases.clamp(0, 1)
    gt = phases[:, 0].clone()                                  # (B, D, H, W)
    if non_finite_outside_mask:
        gt[:, 0, 0, 0] = float("nan")                          # outside every ROI below

    content = torch.zeros(B, D_, H, W, dtype=torch.uint8)
    content[:, 1:D_ - 1, 2:14, 2:14] = 1
    heart = torch.zeros(B, D_, H, W, dtype=torch.uint8)
    heart[:, 2:4, 4:12, 4:12] = 1

    z = torch.linspace(-1, 1, S).view(1, S, 1, 1).expand(B, S, H, W)
    y = torch.linspace(-1, 1, H).view(1, 1, H, 1).expand(B, S, H, W)
    x = torch.linspace(-1, 1, W).view(1, 1, 1, W).expand(B, S, H, W)
    scanner = torch.stack([x, y, z], dim=-1)                   # (B, S, H, W, 3)

    return {
        "gt_target_volume": gt,
        "phases": phases,
        "content_mask": content,
        "heart_roi_canonical": heart,
        "anatomy_bbox": torch.tensor([[1, D_ - 1, 2, 14, 2, 14]], dtype=torch.int64),
        "scanner_coords": scanner,
        "images": torch.rand(B, S, 3, H, W, generator=g),
        "z_scale": torch.tensor([90.0 / 12.0]),
        "dz_mm": torch.tensor([[12.0]]),
        "t_target": torch.tensor([[0]], dtype=torch.int64),
        "timesteps": torch.arange(S).view(1, S),
        "slice_indices": torch.arange(S).float().view(1, S) % D_,
        "seq_index": torch.tensor([[0]], dtype=torch.int64),
    }


def _run(batch, requires_grad=False):
    """Val mode by default (requires_grad False ⇒ the val-only blocks fire)."""
    pos = batch["scanner_coords"].clone().requires_grad_(requires_grad)
    return compute_volume_intensity_loss({"world_points": pos}, batch)


# Keys every masked block must produce in val mode.
VAL_KEYS = [
    "metric_mse_3d_bbox", "metric_mae_3d_bbox", "metric_psnr_3d_bbox",
    "metric_psnr_3d_motion", "metric_mae_3d_motion", "metric_motion_frac",
    "metric_psnr_3d_heartseg", "metric_mae_3d_heartseg", "metric_heartseg_frac",
]


def test_all_masked_metrics_present_in_val():
    out = _run(_batch())
    missing = [k for k in VAL_KEYS if k not in out]
    assert not missing, f"missing masked metrics: {missing}"


def test_masked_metrics_are_finite_and_ordered():
    """PSNR on a mask must be finite, and the identity splat must beat nothing-at-all."""
    out = _run(_batch())
    for k in VAL_KEYS:
        v = float(out[k])
        assert math.isfinite(v), f"{k} is not finite: {v}"
    assert float(out["metric_motion_frac"]) > 0, "synthetic batch has no moving voxels"
    assert 0.0 < float(out["metric_heartseg_frac"]) < 1.0


def test_golden_values_stable_across_refactor():
    """THE regression pin. Values captured 2026-08-01 on the pre-refactor implementation.

    If a masking/clamping refactor changes any of these, it changed the metric — which is
    exactly what must not happen silently. Re-pin ONLY with a deliberate, explained change.
    """
    out = _run(_batch(seed=0))
    got = {k: round(float(out[k]), 6) for k in VAL_KEYS}
    for k, v in GOLDEN.items():
        assert k in got, f"{k} disappeared"
        assert got[k] == pytest.approx(v, abs=1e-4), f"{k}: {got[k]} != {v} (golden)"


def test_empty_mask_does_not_poison_the_batch():
    """A subject with an EMPTY heart ROI must be skipped, not turned into NaN/inf.

    This is where boolean-indexing and torch.where masking genuinely differ: `err[mask].mean()`
    on an empty mask is NaN, while `where(...).sum()/count` is 0/0. Each block guards this its
    own way, so any merged helper must keep a guard.
    """
    b = _batch()
    b["heart_roi_canonical"] = torch.zeros_like(b["heart_roi_canonical"])
    out = _run(b)
    # heartseg keys are simply absent when no sample has a valid ROI...
    assert "metric_psnr_3d_heartseg" not in out or math.isfinite(float(out["metric_psnr_3d_heartseg"]))
    # ...and the OTHER metrics must be unaffected.
    assert math.isfinite(float(out["metric_psnr_3d_bbox"]))
    assert math.isfinite(float(out["metric_psnr_3d_motion"]))


def test_degenerate_bbox_falls_back_to_full_volume():
    """z1<=z0 (an empty bbox) must fall back to the full cube, not divide by zero."""
    b = _batch()
    b["anatomy_bbox"] = torch.tensor([[3, 3, 5, 5, 5, 5]], dtype=torch.int64)  # empty
    out = _run(b)
    assert math.isfinite(float(out["metric_psnr_3d_bbox"]))
    assert math.isfinite(float(out["metric_mse_3d_bbox"]))


def test_non_finite_voxel_outside_mask_does_not_leak():
    """A single NaN OUTSIDE every ROI must not NaN the masked metrics.

    This is why the vectorized blocks use `torch.where(mask, err, 0)` and not
    `err * mask.float()` — NaN * 0.0 == NaN. A merge that switches to a multiply
    reintroduces the bug, and only this test would catch it.
    """
    out = _run(_batch(non_finite_outside_mask=True))
    assert math.isfinite(float(out["metric_psnr_3d_motion"])), "NaN leaked into motion metric"
    assert math.isfinite(float(out["metric_psnr_3d_heartseg"])), "NaN leaked into heartseg metric"


def test_psnr_clamp_floor_is_pinned():
    """Pin the PSNR clamp itself: mse == 0 must give exactly 100 dB (clamp min=1e-10).

    Needed because the ordinary golden batch has mse ~0.3, far above ANY plausible clamp
    floor — so a refactor that changed 1e-10 to 1e-8 would slip through every other test
    here (verified by fault injection). Feeding V_canon back as V_gt makes the residual
    exactly zero, which is the only regime where the clamp is observable.
    """
    b = _batch()
    first = _run(b)
    b2 = _batch()
    b2["gt_target_volume"] = first["V_canon"].detach().clone()
    # Whole cube is the ROI so the masked blocks see the zero residual too.
    b2["anatomy_bbox"] = torch.tensor([[0, D, 0, H, 0, W]], dtype=torch.int64)
    out = _run(b2)
    # 10*log10(1/1e-10) == 100 dB exactly.
    assert float(out["metric_psnr_3d_bbox"]) == pytest.approx(100.0, abs=1e-3), (
        f"clamp floor moved: got {float(out['metric_psnr_3d_bbox'])} dB, expected 100.0"
    )


@pytest.mark.parametrize("D_", [5, 6, 12])
def test_masked_metrics_are_native_z_agnostic(D_):
    """Every subject has its own D (5-21, docs/58) — no block may assume a fixed depth."""
    out = _run(_batch(D_=D_))
    for k in ("metric_psnr_3d_bbox", "metric_psnr_3d_motion", "metric_psnr_3d_heartseg"):
        assert math.isfinite(float(out[k])), f"{k} not finite at D={D_}"


# Recorded from the PRE-refactor implementation on 2026-08-01, before `_masked_stats` /
# `_masked_psnr` were extracted. Kept literal so the pin is visible in the diff.
GOLDEN = {
    "metric_mse_3d_bbox": 0.321452,
    "metric_mae_3d_bbox": 0.480095,
    "metric_psnr_3d_bbox": 4.928832,
    "metric_psnr_3d_motion": 5.011596,
    "metric_mae_3d_motion": 0.482283,
    "metric_motion_frac": 0.076823,
    "metric_psnr_3d_heartseg": 4.386742,
    "metric_mae_3d_heartseg": 0.520348,
    "metric_heartseg_frac": 0.083333,
}
