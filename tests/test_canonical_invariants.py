"""Cross-cutting invariants of the canonical-grid pipeline.

These guard the load-bearing properties that the resample refactor depends on:
  - scanner_coords are a pure geometric function of pixel index — identical for
    every subject (no per-subject spacing leaks in).
  - V_gt is zero outside the subject's content region (the padded zone).
  - the full `phases` bundle is internally consistent with the per-slot inputs.
"""

import numpy as np
import pytest


def _two_subject_dataset(synthetic_root, common_conf, monai_cache_dir, tmp_path):
    """Train split with BOTH synthetic subjects, so we can compare across subjects."""
    from data.datasets.mri_dataset import MRIDataset
    sf = tmp_path / "both.txt"
    # Dataset is built with split="val", so list both subjects under [val].
    sf.write_text("[val]\nTrain_P001\nVal_P001\n")
    return MRIDataset(common_conf, synthetic_root, split="val", split_file=str(sf),
                      mode="dynamic", mri_mode="axial", num_slices=8, cache_dir=monai_cache_dir)


def test_scanner_coords_identical_across_subjects(synthetic_root, common_conf, monai_cache_dir, tmp_path):
    """The (x, y) part of scanner_coords depends only on pixel index, not subject —
    same canonical geometry for everyone. (z depends on the chosen slice index.)"""
    ds = _two_subject_dataset(synthetic_root, common_conf, monai_cache_dir, tmp_path)
    s0 = ds.get_data(0, img_per_seq=8)       # subject 0
    s_other = ds.get_data(1, img_per_seq=8)  # subject 1
    # Compare the x,y planes of slot 0 between the two subjects.
    xy0 = s0["scanner_coords"][0][..., :2]
    xy_other = s_other["scanner_coords"][0][..., :2]
    np.testing.assert_allclose(xy0, xy_other, atol=1e-5,
                               err_msg="scanner_coords x/y must be subject-independent geometry")


def test_vgt_zero_outside_bbox(synthetic_root, common_conf, monai_cache_dir, tmp_path):
    """V_gt must be ~zero in the zero-padded region (outside the content bbox)."""
    ds = _two_subject_dataset(synthetic_root, common_conf, monai_cache_dir, tmp_path)
    s = ds.get_data(0, img_per_seq=8)
    V_gt = s["gt_target_volume"]            # (D, H, W)
    z0, z1, y0, y1, x0, x1 = [int(v) for v in s["anatomy_bbox"].tolist()]
    # Build a boolean "outside bbox" mask and check V_gt is ~0 there.
    outside = np.ones_like(V_gt, dtype=bool)
    outside[z0:z1, y0:y1, x0:x1] = False
    if outside.any():
        assert float(np.abs(V_gt[outside]).max()) < 1e-3, \
            "V_gt must be zero in the padded region outside the content bbox"


def test_phases_bundle_matches_vgt_at_t_target(synthetic_root, common_conf, monai_cache_dir, tmp_path):
    """gt_target_volume must equal phases[t_target] (consistency of the cached bundle)."""
    ds = _two_subject_dataset(synthetic_root, common_conf, monai_cache_dir, tmp_path)
    s = ds.get_data(3, img_per_seq=8)
    t_target = int(np.asarray(s["t_target"]).item())
    phases = s["phases"]                    # (T, D, H, W) float16
    V_gt = s["gt_target_volume"]            # (D, H, W) float32
    np.testing.assert_allclose(
        phases[t_target].astype(np.float32), V_gt, atol=1e-2,
        err_msg="gt_target_volume must equal phases[t_target]",
    )


def test_content_mask_matches_bbox(synthetic_root, common_conf, monai_cache_dir, tmp_path):
    """anatomy_bbox must be the tight bbox of content_mask > 0."""
    ds = _two_subject_dataset(synthetic_root, common_conf, monai_cache_dir, tmp_path)
    s = ds.get_data(0, img_per_seq=8)
    mask = s["content_mask"]                # (D, H, W) uint8
    z0, z1, y0, y1, x0, x1 = [int(v) for v in s["anatomy_bbox"].tolist()]
    nz = np.argwhere(mask > 0)
    assert nz[:, 0].min() == z0 and nz[:, 0].max() + 1 == z1
    assert nz[:, 1].min() == y0 and nz[:, 1].max() + 1 == y1
    assert nz[:, 2].min() == x0 and nz[:, 2].max() + 1 == x1
