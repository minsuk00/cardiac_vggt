"""
Comprehensive tests for MRIDataset (CMRxRecon2024 / Cine_combined format).

All tests use a synthetic in-memory dataset — no real data required.
Target runtime: < 10 seconds total.

Synthetic setup (from conftest.py):
  W=32, H=30, Z=4, T=3 frames, spacing=(1.344, 1.398, 8.0) mm
"""

import glob
import os

import numpy as np
import pytest

# Synthetic dataset constants — must match conftest.py
SPACING = (1.344, 1.398, 8.0)
W, H, Z, T = 32, 30, 4, 3
# Per-axis half_extent in mm — each axis's [-1, 1] spans that axis's full physical extent.
EXPECTED_HALF_EXTENT = (W * SPACING[0] / 2, H * SPACING[1] / 2, Z * SPACING[2] / 2)  # (21.5, 20.97, 16.0)
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
    for key in ["images", "world_points", "scanner_coords"]:
        for item in s[key]:
            assert item.shape == (TARGET_SIZE, TARGET_SIZE, 3), \
                f"{key}: expected ({TARGET_SIZE},{TARGET_SIZE},3), got {item.shape}"
    for item in s["point_masks"]:
        assert item.shape == (TARGET_SIZE, TARGET_SIZE)
    assert "gt_dvfs" not in s, "gt_dvfs should be removed from the batch (supervised DVF pipeline deprecated)"
    assert "scale_factors" not in s, "scale_factors should be removed from the batch (supervised DVF pipeline deprecated)"
    assert len(s["timesteps"]) == 3
    assert len(s["z_indices"]) == 3

def test_seq_name_is_string(train_ds):
    s = train_ds.get_data(0, img_per_seq=2)
    assert isinstance(s["seq_name"], str) and "mri_axial" in s["seq_name"]

def test_images_in_valid_range(train_ds):
    s = train_ds.get_data(0, img_per_seq=2)
    for img in s["images"]:
        assert img.min() >= 0.0 and img.max() <= 255.0


# ── 4. Coordinate space (physical mm normalization) ──────────────────────────

def test_world_points_equals_scanner_coords(train_ds):
    """
    After the supervised-DVF pipeline was removed, the dataset no longer applies any
    motion to coordinates: world_points and scanner_coords are identical (the model
    learns the displacement Δ itself via the unsupervised intensity loss).
    """
    s = train_ds.get_data(0, img_per_seq=3)
    for i in range(len(s["world_points"])):
        wp = s["world_points"][i]
        sc = s["scanner_coords"][i]
        mask = s["point_masks"][i]
        np.testing.assert_array_equal(
            wp[mask], sc[mask],
            err_msg=f"slot {i}: world_points must equal scanner_coords after DVF removal",
        )

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

def test_padded_pixels_zero_position_and_zero_intensity(train_ds):
    """Padded pixels (outside the resized native FOV) get position 0 and intensity 0.
    The splat's intensity gate (Mask B) drops them, so the position value is inert
    — this test pins the contract that they're 0, not the legacy -2.0 sentinel."""
    s = train_ds.get_data(0, img_per_seq=2)
    for wp, img, mask in zip(s["world_points"], s["images"], s["point_masks"]):
        invalid = ~mask
        if invalid.any():
            assert np.all(wp[invalid] == 0.0), "Padded pixels should have world_points = 0"
            assert np.all(img[invalid] == 0), "Padded pixels should have intensity = 0 (Mask B gate)"


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
    """In static mode every slot uses t=t_target → all timesteps identical."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="static", mri_mode="axial", num_slices=4)
    s = ds.get_data(0, img_per_seq=4)
    t_target = int(np.asarray(s["t_target"]).item())
    for t in s["timesteps"]:
        assert t == t_target, f"Static mode: slot t={t} != t_target={t_target}"
