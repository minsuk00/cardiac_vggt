"""Tests for the bbox-cropped metrics in compute_volume_intensity_loss."""

import torch

from loss import compute_volume_intensity_loss


def _fake_loss_inputs(B=2, S=4, H=64, W=64, D=12, Hv=256, Wv=256, device="cpu"):
    torch.manual_seed(0)
    pos = torch.rand(B, S, H, W, 3, device=device) * 2 - 1
    images = torch.rand(B, S, 3, H, W, device=device) * 255
    V_gt = torch.rand(B, D, Hv, Wv, device=device) * 0.5
    return pos, images, V_gt


def test_bbox_metric_equals_full_when_bbox_is_full_cube():
    pos, images, V_gt = _fake_loss_inputs()
    B, D, Hv, Wv = V_gt.shape
    bboxes = torch.tensor([[0, D, 0, Hv, 0, Wv]] * B, dtype=torch.int64)
    batch = {"images": images, "gt_target_volume": V_gt,
             "scanner_coords": pos.clone(), "anatomy_bbox": bboxes}
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        grid_shape=(D, Hv, Wv), tv_weight=0.1)
    # With bbox = full cube, bbox metrics must equal full metrics.
    assert torch.allclose(out["metric_psnr_3d_bbox"], out["metric_psnr_3d_full"], atol=1e-4)
    assert torch.allclose(out["metric_mae_3d_bbox"], out["metric_mae_3d_full"], atol=1e-4)


def test_bbox_metric_present_and_finite():
    pos, images, V_gt = _fake_loss_inputs()
    B, D, Hv, Wv = V_gt.shape
    bboxes = torch.tensor([[1, 11, 30, 230, 5, 251]] * B, dtype=torch.int64)
    batch = {"images": images, "gt_target_volume": V_gt,
             "scanner_coords": pos.clone(), "anatomy_bbox": bboxes}
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        grid_shape=(D, Hv, Wv), tv_weight=0.1)
    for k in ["metric_psnr_3d_bbox", "metric_mae_3d_bbox", "metric_mse_3d_bbox"]:
        assert k in out
        assert torch.isfinite(out[k]).all()


def test_full_metrics_renamed_present():
    """The legacy `metric_psnr_3d` names must be replaced by `_full` suffixed names."""
    pos, images, V_gt = _fake_loss_inputs()
    B, D, Hv, Wv = V_gt.shape
    batch = {"images": images, "gt_target_volume": V_gt, "scanner_coords": pos.clone()}
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        grid_shape=(D, Hv, Wv), tv_weight=0.1)
    assert "metric_psnr_3d_full" in out and "metric_mae_3d_full" in out
    assert "metric_psnr_3d" not in out, "legacy un-suffixed metric name must be gone"


def test_bbox_metrics_absent_without_anatomy_bbox():
    """If the batch has no anatomy_bbox, only full metrics are produced (no crash)."""
    pos, images, V_gt = _fake_loss_inputs()
    B, D, Hv, Wv = V_gt.shape
    batch = {"images": images, "gt_target_volume": V_gt, "scanner_coords": pos.clone()}
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        grid_shape=(D, Hv, Wv), tv_weight=0.1)
    assert "metric_psnr_3d_bbox" not in out
    assert "metric_psnr_3d_full" in out


def test_empty_bbox_falls_back_to_full():
    """A degenerate (empty) bbox for one sample must not produce NaN — falls back to full."""
    pos, images, V_gt = _fake_loss_inputs()
    B, D, Hv, Wv = V_gt.shape
    # Subject 0: empty bbox (z1<=z0). Subject 1: valid.
    bboxes = torch.tensor([[5, 5, 0, Hv, 0, Wv], [0, D, 0, Hv, 0, Wv]], dtype=torch.int64)
    batch = {"images": images, "gt_target_volume": V_gt,
             "scanner_coords": pos.clone(), "anatomy_bbox": bboxes}
    out = compute_volume_intensity_loss({"world_points": pos}, batch,
                                        grid_shape=(D, Hv, Wv), tv_weight=0.1)
    assert torch.isfinite(out["metric_psnr_3d_bbox"]).all()
