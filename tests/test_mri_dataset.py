"""
Comprehensive tests for MRIDataset (CMRxRecon2024 / Cine_combined format).

All tests use a synthetic in-memory dataset — no real data required.
Target runtime: < 10 seconds total.

Synthetic setup (from conftest.py):
  W=32, H=30, Z=4, T=3 frames, spacing=(1.344, 1.398, 8.0) mm
  DVF = constant [2.0, 1.5, 4.0] mm for all dynamic frames
  EXPECTED_SCALE = max(32*1.344, 30*1.398, 4*8.0) / 2 = max(43.0, 41.9, 32.0) / 2 = 21.5 mm
"""

import glob
import os

import numpy as np
import pytest

# Synthetic dataset constants — must match conftest.py
SPACING = (1.344, 1.398, 8.0)
W, H, Z, T = 32, 30, 4, 3
KNOWN_DVF_MM = [2.0, 1.5, 4.0]
# Per-axis half_extent in mm — each axis's [-1, 1] spans that axis's full physical extent.
EXPECTED_HALF_EXTENT = (W * SPACING[0] / 2, H * SPACING[1] / 2, Z * SPACING[2] / 2)  # (21.5, 20.97, 16.0)
EXPECTED_SCALE = float(np.mean(EXPECTED_HALF_EXTENT))  # backward-compat scalar
TARGET_SIZE = 518


# ── 1. Subject discovery via split file ──────────────────────────────────────

def test_find_subjects_train_count(train_ds):
    assert len(train_ds.subjects) == 1

def test_find_subjects_train_path(train_ds):
    assert train_ds.subjects[0].endswith("Train_P001/sax")

def test_find_subjects_val(synthetic_root, split_file, common_conf):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial")
    assert len(ds.subjects) == 1
    assert "Val_P001" in ds.subjects[0]

def test_find_subjects_empty_section(synthetic_root, split_file, common_conf):
    """[test] section is empty in the fixture split file."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="test", split_file=split_file,
                    mode="dynamic", mri_mode="axial")
    assert len(ds.subjects) == 0

def test_find_subjects_no_split_file(synthetic_root, common_conf):
    """Missing split_file → empty subject list with a warning (no crash)."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=None,
                    mode="dynamic", mri_mode="axial")
    assert len(ds.subjects) == 0

def test_overfit_split_same_subject_train_val(synthetic_root, overfit_split_file, common_conf):
    """Overfit split: same subject in both train and val."""
    from data.datasets.mri_dataset import MRIDataset
    train = MRIDataset(common_conf, synthetic_root, split="train",
                       split_file=overfit_split_file, mode="dynamic", mri_mode="axial")
    val   = MRIDataset(common_conf, synthetic_root, split="val",
                       split_file=overfit_split_file, mode="dynamic", mri_mode="axial")
    assert train.subjects == val.subjects

def test_split_file_preserves_order(synthetic_root, common_conf, tmp_path):
    """Subjects are returned in file order, not sorted."""
    from data.datasets.mri_dataset import MRIDataset
    sf = tmp_path / "order_test.txt"
    sf.write_text("[train]\nVal_P001\nTrain_P001\n")
    ds = MRIDataset(common_conf, synthetic_root, split="train",
                    split_file=str(sf), mode="dynamic", mri_mode="axial")
    assert len(ds.subjects) == 2
    assert "Val_P001" in ds.subjects[0]
    assert "Train_P001" in ds.subjects[1]

def test_comments_and_blank_lines_ignored(synthetic_root, common_conf, tmp_path):
    """# comments and blank lines in split file are ignored."""
    from data.datasets.mri_dataset import MRIDataset
    sf = tmp_path / "comments.txt"
    sf.write_text("# header\n\n[train]\n# skip this\nTrain_P001\n\n")
    ds = MRIDataset(common_conf, synthetic_root, split="train",
                    split_file=str(sf), mode="dynamic", mri_mode="axial")
    assert len(ds.subjects) == 1


# ── 2. Frame file discovery ───────────────────────────────────────────────────

def test_t_total_correct(train_ds):
    nii = glob.glob(os.path.join(train_ds.subjects[0], "3d_recon", "sax_frame_*.nii.gz"))
    assert len(nii) == T  # 3 frames (00, 01, 02)


# ── 3. Output shapes ──────────────────────────────────────────────────────────

def test_output_shapes(train_ds):
    s = train_ds.get_data(0, img_per_seq=3)
    for key in ["images", "world_points", "scanner_coords", "gt_dvfs"]:
        for item in s[key]:
            assert item.shape == (TARGET_SIZE, TARGET_SIZE, 3), \
                f"{key}: expected ({TARGET_SIZE},{TARGET_SIZE},3), got {item.shape}"
    for item in s["point_masks"]:
        assert item.shape == (TARGET_SIZE, TARGET_SIZE)
    assert len(s["scale_factors"]) == 3
    assert len(s["timesteps"]) == 3
    assert len(s["z_indices"]) == 3

def test_seq_name_is_string(train_ds):
    s = train_ds.get_data(0, img_per_seq=2)
    assert isinstance(s["seq_name"], str) and "mri_axial" in s["seq_name"]

def test_images_in_valid_range(train_ds):
    s = train_ds.get_data(0, img_per_seq=2)
    for img in s["images"]:
        assert img.min() >= 0.0 and img.max() <= 255.0


# ── 4. DVF values (mm units) ──────────────────────────────────────────────────

def test_dvf_matches_slot_timestep(train_ds):
    """DVF for any slot is zero iff its t_idx == 0 (DVF files only exist for t > 0).
    With t_target sampling, slot 0 may be at any t; the invariant is per-slot, not per-slot-index.
    """
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=3)
        for i, (t_idx, dvf) in enumerate(zip(s["timesteps"], s["gt_dvfs"])):
            if t_idx == 0:
                assert np.abs(dvf).max() == 0.0, f"slot {i} t={t_idx}: expected zero DVF"
            else:
                assert np.abs(dvf).max() > 0.0, f"slot {i} t={t_idx}: expected nonzero DVF"

def test_dvf_values_match_known_mm(train_ds):
    """
    Synthetic DVF = constant [2.0, 1.5, 4.0] mm.
    In voxels this would be [2.0/1.344, 1.5/1.398, 4.0/8.0] = [1.49, 1.07, 0.5].
    We check the stored mm values match, distinguishing mm from voxel representation.
    """
    # Find a slot with t > 0 (DVF file exists). With t_target sampling, slot 0 may be t=0 or not.
    s = train_ds.get_data(0, img_per_seq=3)
    nonzero_slots = [i for i, t in enumerate(s["timesteps"]) if t > 0]
    assert nonzero_slots, "Need at least one slot with t>0 to check DVF values"
    i = nonzero_slots[0]
    dvf = s["gt_dvfs"][i]   # dynamic frame, shape (518, 518, 3)
    mask = s["point_masks"][i]
    assert mask.sum() > 0, "No valid pixels in mask"

    np.testing.assert_allclose(dvf[mask, 0].mean(), KNOWN_DVF_MM[0], atol=0.3,
                               err_msg="DVF x-component should be ~2.0 mm")
    np.testing.assert_allclose(dvf[mask, 1].mean(), KNOWN_DVF_MM[1], atol=0.3,
                               err_msg="DVF y-component should be ~1.5 mm")
    np.testing.assert_allclose(dvf[mask, 2].mean(), KNOWN_DVF_MM[2], atol=0.5,
                               err_msg="DVF z-component should be ~4.0 mm")

def test_dvf_not_divided_by_spacing(train_ds):
    """If DVF were wrongly converted to voxels, z-component would be 4.0/8.0=0.5, not ~4.0."""
    s = train_ds.get_data(0, img_per_seq=3)
    nonzero_slots = [i for i, t in enumerate(s["timesteps"]) if t > 0]
    assert nonzero_slots, "Need at least one slot with t>0 to check DVF mm scale"
    i = nonzero_slots[0]
    dvf = s["gt_dvfs"][i]
    mask = s["point_masks"][i]
    z_mean = dvf[mask, 2].mean()
    assert z_mean > 1.0, \
        f"DVF z should be in mm (~4.0), got {z_mean:.3f} — looks like it was divided by spacing"


# ── 5. Coordinate space (physical mm normalization) ───────────────────────────

def test_scale_factor_is_mean_half_extent(train_ds):
    """scale_factor = mean of per-axis half_extents (backward-compat scalar for legacy viz)."""
    s = train_ds.get_data(0, img_per_seq=2)
    sf = s["scale_factors"][0][0]
    assert sf == pytest.approx(EXPECTED_SCALE, rel=0.01), \
        f"Expected scale_factor ≈ {EXPECTED_SCALE:.2f} mm, got {sf:.2f}"

def test_world_points_scanner_coords_identity(train_ds):
    """
    Per-axis normalization invariant for a dynamic frame:
        (world_points[i] - scanner_coords[i]) * half_extent[i] ≈ gt_dvf[i]  for axis i ∈ {0,1,2}
    because per-axis: pos_norm[i] = (pos_mm[i] - center[i]) / half_extent[i]
    """
    s = train_ds.get_data(0, img_per_seq=3)
    for i, t in enumerate(s["timesteps"]):
        if t > 0:  # dynamic frame
            wp   = s["world_points"][i]
            sc   = s["scanner_coords"][i]
            dvf  = s["gt_dvfs"][i]
            mask = s["point_masks"][i]
            for ax in range(3):
                diff_mm = (wp[mask][:, ax] - sc[mask][:, ax]) * EXPECTED_HALF_EXTENT[ax]
                np.testing.assert_allclose(diff_mm, dvf[mask][:, ax], atol=0.15,
                                           err_msg=f"(wp - sc) * half_extent should equal gt_dvf in mm along axis {ax}")
            break

def test_z_axis_not_degenerate(train_ds):
    """
    With per-axis mm normalization, z spans a meaningful range across slots.
    Random sampling covers all z's when img_per_seq = Z_total.
    """
    s = train_ds.get_data(0, img_per_seq=Z)  # Z=4 in synthetic data → all z's sampled
    all_z = []
    for slot in range(len(s["world_points"])):
        wp = s["world_points"][slot]
        mask = s["point_masks"][slot]
        all_z.extend(wp[mask, 2].tolist())
    z_range = max(all_z) - min(all_z)
    assert z_range > 0.3, \
        f"z range {z_range:.3f} is too small — mm normalization likely not applied"

def test_world_points_normalized_range(train_ds):
    """world_points should be in roughly [-1.1, 1.1] (may slightly exceed due to DVF)."""
    s = train_ds.get_data(0, img_per_seq=3)
    for wp in s["world_points"]:
        mask_flat = wp[..., 0] != -2.0  # -2.0 is the invalid sentinel
        valid = wp[mask_flat]
        assert valid.min() > -2.5 and valid.max() < 2.5, \
            f"world_points out of expected range: [{valid.min():.2f}, {valid.max():.2f}]"


# ── 6. Volume mask ────────────────────────────────────────────────────────────

def test_vol_mask_has_valid_pixels(train_ds):
    """After padding, most pixels in the active slice area should be valid."""
    s = train_ds.get_data(0, img_per_seq=2)
    for mask in s["point_masks"]:
        assert mask.sum() > 200, \
            f"Expected > 200 valid pixels, got {mask.sum()}"

def test_vol_mask_invalid_pixels_have_sentinel(train_ds):
    """Pixels outside volume get world_points = -2.0."""
    s = train_ds.get_data(0, img_per_seq=2)
    for i, (wp, mask) in enumerate(zip(s["world_points"], s["point_masks"])):
        invalid = ~mask
        if invalid.any():
            assert np.all(wp[invalid] == -2.0), \
                "Invalid pixels should have world_points = -2.0"


# ── 7. Timesteps and frame indexing ──────────────────────────────────────────

def test_slot0_anchored_to_t_target(train_ds):
    """Slot 0 is anchored to the per-call t_target (was always t=0 in the old pipeline)."""
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=4)
        t_target = int(np.asarray(s["t_target"]).item())
        assert 0 <= t_target < T, f"t_target {t_target} out of [0, {T})"
        assert s["timesteps"][0] == t_target, \
            f"Slot 0 must equal t_target ({t_target}), got {s['timesteps'][0]}"

def test_dynamic_timesteps_in_valid_range(train_ds):
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=4)
        t_target = int(np.asarray(s["t_target"]).item())
        for t in s["timesteps"][1:]:
            assert 0 <= t <= T - 1, f"Dynamic timestep {t} out of range [0, {T-1}]"
            assert t != t_target, \
                f"Dynamic slot has t={t}, equal to t_target={t_target} (should differ)"


def test_val_t_target_is_stratified(synthetic_root, split_file, common_conf):
    """Val sampling: t_target = seq_index % T_total. Deterministic + balanced across phases.
    Ensures per-phase val metrics get reproducible sample counts.
    """
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=4)
    # Across multiple seq_index values, t_target must equal seq_index % T_total exactly.
    for seq_index in range(15):
        s = ds.get_data(seq_index, img_per_seq=4)
        t_target = int(np.asarray(s["t_target"]).item())
        assert t_target == seq_index % T, \
            f"Val seq_index={seq_index} expected t_target={seq_index % T}, got {t_target}"
        # And slot 0 must follow this assignment.
        assert s["timesteps"][0] == t_target, \
            f"Val slot 0 (={s['timesteps'][0]}) should equal t_target={t_target}"

def test_z_indices_in_range(train_ds):
    s = train_ds.get_data(0, img_per_seq=4)
    for z_idx in s["z_indices"]:
        assert -1.0 <= z_idx[0] <= 1.0, f"z_index {z_idx[0]} out of [-1, 1]"


# ── 8. Static mode ────────────────────────────────────────────────────────────

def test_static_mode_all_same_timestep(synthetic_root, split_file, common_conf):
    """In static mode every slot uses t=t_target → all timesteps identical.
    Static mode never loads DVFs (legacy behavior), so gt_dvfs is always zero.
    """
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="static", mri_mode="axial", num_slices=4)
    s = ds.get_data(0, img_per_seq=4)
    t_target = int(np.asarray(s["t_target"]).item())
    for t in s["timesteps"]:
        assert t == t_target, f"Static mode: slot t={t} != t_target={t_target}"
    for dvf in s["gt_dvfs"]:
        assert np.abs(dvf).max() == 0.0, "Static mode should always have zero DVF regardless of t_target"
