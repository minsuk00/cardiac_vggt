"""
Tests for MRIDataset — canonical-grid edition (CMRxRecon2024 / Cine_combined).

All tests use a synthetic in-memory dataset (conftest.py) — no real data needed.

Canonical contract (post-resample refactor):
  - Every subject resampled to (1.4, 1.4, 8.0) mm, cropped/padded to (256, 256, 12).
  - Input slices: canonical 256×256 → bilinear-resize to 518×518 (NO letterbox/pad).
  - scanner_coords: purely geometric, in [-1, +1], same formula for every subject.
  - world_points == scanner_coords (DVF supervision removed).
  - z sampled from within the geometric anatomy bbox (in-FOV planes only).
"""

import glob
import os

import numpy as np
import pytest

from conftest import SYN_T  # 12 phases

TARGET_SIZE = 518
CANON_D = 12  # canonical depth


# ── 1. Subject discovery via split file ──────────────────────────────────────

def test_find_subjects_train_count(train_ds):
    assert len(train_ds.subjects) == 1

def test_find_subjects_train_path(train_ds):
    assert train_ds.subjects[0].endswith("Train_P001/sax")

def test_find_subjects_val(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir)
    assert len(ds.subjects) == 1
    assert "Val_P001" in ds.subjects[0]

def test_find_subjects_empty_section(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="test", split_file=split_file,
                    mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir)
    assert len(ds.subjects) == 0

def test_find_subjects_no_split_file(synthetic_root, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=None,
                    mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir)
    assert len(ds.subjects) == 0

def test_split_file_preserves_order(synthetic_root, common_conf, tmp_path, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    sf = tmp_path / "order_test.txt"
    sf.write_text("[train]\nVal_P001\nTrain_P001\n")
    ds = MRIDataset(common_conf, synthetic_root, split="train",
                    split_file=str(sf), mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir)
    assert len(ds.subjects) == 2
    assert "Val_P001" in ds.subjects[0]
    assert "Train_P001" in ds.subjects[1]


# ── 2. Frame file discovery ───────────────────────────────────────────────────

def test_t_total_correct(train_ds):
    nii = glob.glob(os.path.join(train_ds.subjects[0], "3d_recon", "sax_frame_*.nii.gz"))
    assert len(nii) == SYN_T  # 12 phases


# ── 3. Output shapes + new batch fields ───────────────────────────────────────

def test_output_shapes(train_ds):
    s = train_ds.get_data(0, img_per_seq=8)
    for key in ["images", "world_points", "scanner_coords"]:
        for item in s[key]:
            assert item.shape == (TARGET_SIZE, TARGET_SIZE, 3), \
                f"{key}: expected ({TARGET_SIZE},{TARGET_SIZE},3), got {item.shape}"
    for item in s["point_masks"]:
        assert item.shape == (TARGET_SIZE, TARGET_SIZE)
    assert len(s["timesteps"]) == 8
    assert len(s["z_indices"]) == 8

def test_legacy_dvf_fields_absent(train_ds):
    s = train_ds.get_data(0, img_per_seq=4)
    assert "gt_dvfs" not in s, "gt_dvfs should be gone (supervised DVF pipeline deprecated)"
    assert "scale_factors" not in s, "scale_factors should be gone"

def test_new_canonical_batch_fields_present(train_ds):
    """Canonical-grid pipeline adds gt_target_volume, anatomy_bbox, content_mask, phases."""
    s = train_ds.get_data(0, img_per_seq=8)
    assert s["gt_target_volume"].shape == (CANON_D, 256, 256)
    assert s["anatomy_bbox"].shape == (6,)
    assert s["content_mask"].shape == (CANON_D, 256, 256)
    assert s["phases"].shape == (SYN_T, CANON_D, 256, 256)
    # gt_target_volume is V_gt at t_target — must be in [0, 1] after percentile norm.
    assert 0.0 <= float(s["gt_target_volume"].min()) and float(s["gt_target_volume"].max()) <= 1.0

def test_seq_name_is_string(train_ds):
    s = train_ds.get_data(0, img_per_seq=2)
    assert isinstance(s["seq_name"], str) and "mri_axial" in s["seq_name"]

def test_images_in_valid_range(train_ds):
    s = train_ds.get_data(0, img_per_seq=4)
    for img in s["images"]:
        assert img.min() >= 0.0 and img.max() <= 255.0


# ── 4. Coordinate space (canonical, geometric) ────────────────────────────────

def test_world_points_equals_scanner_coords(train_ds):
    """DVF supervision removed → world_points and scanner_coords are identical."""
    s = train_ds.get_data(0, img_per_seq=8)
    for i in range(len(s["world_points"])):
        np.testing.assert_array_equal(
            s["world_points"][i], s["scanner_coords"][i],
            err_msg=f"slot {i}: world_points must equal scanner_coords after DVF removal",
        )

def test_scanner_coords_normalized_range(train_ds):
    """Every pixel has a valid canonical coord in [-1, 1] — no -2.0 sentinel, no padding."""
    s = train_ds.get_data(0, img_per_seq=8)
    for sc in s["scanner_coords"]:
        assert sc.min() >= -1.0001 and sc.max() <= 1.0001, \
            f"scanner_coords out of [-1, 1]: [{sc.min():.3f}, {sc.max():.3f}]"

def test_scanner_coords_corners_are_geometric(train_ds):
    """Corners of the 518×518 grid map to fixed canonical (∓1, ∓1, z) — same for every subject."""
    s = train_ds.get_data(0, img_per_seq=4)
    sc = s["scanner_coords"][1]  # any slot
    # (x, y, z): top-left = (-1, -1, z); bottom-right = (+1, +1, z).
    np.testing.assert_allclose(sc[0, 0, :2], [-1.0, -1.0], atol=1e-4)
    np.testing.assert_allclose(sc[-1, -1, :2], [1.0, 1.0], atol=1e-4)
    # z is constant across the whole slice.
    assert np.allclose(sc[..., 2], sc[0, 0, 2])

def test_z_axis_spans_range(train_ds):
    """z across slots spans a meaningful range (canonical D=12 planes, normalized)."""
    s = train_ds.get_data(0, img_per_seq=8)
    z_vals = [float(z[0]) for z in s["z_indices"]]
    assert max(z_vals) - min(z_vals) > 0.2, f"z range too small: {z_vals}"


# ── 5. Masks (all-True; no letterbox padding region) ──────────────────────────

def test_point_masks_all_true(train_ds):
    """No letterbox padding under the canonical pipeline → every pixel is valid."""
    s = train_ds.get_data(0, img_per_seq=4)
    for mask in s["point_masks"]:
        assert mask.all(), "point_masks must be all-True (no padding region)"


# ── 6. z sampled within the geometric bbox ────────────────────────────────────

def test_z_sampled_within_bbox(train_ds):
    """All sampled z slots must lie inside [bbox_z0, bbox_z1) — no padded-Z planes."""
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=8)
        z0, z1 = int(s["anatomy_bbox"][0]), int(s["anatomy_bbox"][1])
        for z in s["slice_indices"]:
            assert z0 <= z < z1, f"slot z={z} outside bbox z[{z0}:{z1}]"


# ── 6b. S capped to in-bbox z extent — no z wrap / no duplicate z ──────────────

def test_S_capped_to_bbox_no_wrap_val(synthetic_root, split_file, common_conf, monai_cache_dir):
    """Requesting MORE slices than the subject's in-FOV z extent must shrink S to
    bbox_z_size (fewer than 12 input slices) instead of wrapping z back to bbox_z0.
    Regression for the canonical-padding bug: S was pinned to the padded D=12, so the
    val diagonal wrapped — e.g. slot (t=10,z=10) → (t=11,z=0) — re-sampling z planes."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=12, cache_dir=monai_cache_dir)
    s = ds.get_data(0, img_per_seq=12)  # ask for 12 — more than the synthetic's z extent
    z0, z1 = int(s["anatomy_bbox"][0]), int(s["anatomy_bbox"][1])
    bbox_z_size = z1 - z0
    zs = [int(z) for z in s["slice_indices"]]
    assert bbox_z_size < 12, "fixture should have a sub-12 z extent to exercise the cap"
    assert len(zs) == bbox_z_size, f"S must cap to bbox_z_size={bbox_z_size}, got {len(zs)} slices"
    assert len(set(zs)) == len(zs), f"val z must not repeat (no wrap), got {zs}"
    assert zs == list(range(z0, z0 + len(zs))), f"val z must be the monotonic in-bbox diagonal, got {zs}"


def test_S_capped_to_bbox_no_dup_train(train_ds):
    """Train: asking for more slices than bbox_z_size shrinks S and samples z WITHOUT
    replacement — no duplicate z planes."""
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=12)
        z0, z1 = int(s["anatomy_bbox"][0]), int(s["anatomy_bbox"][1])
        bbox_z_size = z1 - z0
        zs = [int(z) for z in s["slice_indices"]]
        assert len(zs) == bbox_z_size, f"S must cap to bbox_z_size={bbox_z_size}, got {len(zs)}"
        assert len(set(zs)) == len(zs), f"train z must be distinct (no replacement), got {zs}"


# ── 7. Timesteps and frame indexing ──────────────────────────────────────────

def test_slot0_anchored_to_t_target(train_ds):
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=8)
        t_target = int(np.asarray(s["t_target"]).item())
        assert 0 <= t_target < SYN_T
        assert s["timesteps"][0] == t_target, \
            f"Slot 0 must equal t_target ({t_target}), got {s['timesteps'][0]}"

def test_dynamic_timesteps_distinct_from_target(train_ds):
    for _ in range(5):
        s = train_ds.get_data(0, img_per_seq=8)
        t_target = int(np.asarray(s["t_target"]).item())
        for t in s["timesteps"][1:]:
            assert 0 <= t < SYN_T
            assert t != t_target, f"Dynamic slot t={t} equals t_target={t_target}"

def test_val_t_target_is_stratified(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8, cache_dir=monai_cache_dir)
    for seq_index in range(15):
        s = ds.get_data(seq_index, img_per_seq=8)
        t_target = int(np.asarray(s["t_target"]).item())
        assert t_target == seq_index % SYN_T, \
            f"Val seq_index={seq_index} expected t_target={seq_index % SYN_T}, got {t_target}"
        assert s["timesteps"][0] == t_target

def test_t_target_phases_val_cycles_pool(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    pool = [0, 7]
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8,
                    t_target_phases=pool, cache_dir=monai_cache_dir)
    for seq_index in range(12):
        s = ds.get_data(seq_index, img_per_seq=8)
        t_target = int(np.asarray(s["t_target"]).item())
        assert t_target == pool[seq_index % len(pool)], \
            f"Val seq_index={seq_index} expected t_target={pool[seq_index % len(pool)]}, got {t_target}"
        assert s["timesteps"][0] == t_target

def test_t_target_phases_train_only_from_pool(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    pool = [0, 7]
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8,
                    t_target_phases=pool, cache_dir=monai_cache_dir)
    seen = set()
    for seq_index in range(40):
        s = ds.get_data(seq_index, img_per_seq=8)
        t_target = int(np.asarray(s["t_target"]).item())
        assert t_target in pool, f"Train t_target={t_target} not in pool {pool}"
        assert s["timesteps"][0] == t_target
        seen.add(t_target)
    assert seen == set(pool), f"Train never sampled the full pool {pool}; saw {seen}"

def test_t_target_phases_inputs_span_beyond_pool(synthetic_root, split_file, common_conf, monai_cache_dir):
    """TARGET-ONLY guarantee: restricting the pool restricts only t_target (slot 0);
    the other input slots must still draw from ALL phases, including ones outside the pool."""
    from data.datasets.mri_dataset import MRIDataset
    pool = [0, 7]
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=12,
                    t_target_phases=pool, cache_dir=monai_cache_dir)
    input_phases = set()
    for seq_index in range(30):
        s = ds.get_data(seq_index, img_per_seq=8)
        input_phases.update(int(t) for t in s["timesteps"])
    outside = input_phases - set(pool)
    assert len(outside) >= 3, \
        f"Inputs should span phases beyond the target pool {pool}; only saw {sorted(input_phases)}"

def test_t_target_phases_val_deterministic_across_instances(synthetic_root, split_file, common_conf, monai_cache_dir):
    """Val phase selection must be identical across two independent dataset objects
    (the basis for stable per-phase val metrics across runs)."""
    from data.datasets.mri_dataset import MRIDataset
    pool = [0, 7]
    def seq(ds):
        return [int(np.asarray(ds.get_data(i, img_per_seq=8)["t_target"]).item()) for i in range(10)]
    ds_a = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                      mode="dynamic", mri_mode="axial", num_slices=8,
                      t_target_phases=pool, cache_dir=monai_cache_dir)
    ds_b = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                      mode="dynamic", mri_mode="axial", num_slices=8,
                      t_target_phases=pool, cache_dir=monai_cache_dir)
    assert seq(ds_a) == seq(ds_b) == [pool[i % 2] for i in range(10)]

def test_t_target_phases_single_element(synthetic_root, split_file, common_conf, monai_cache_dir):
    """A 1-element pool pins every sample to that phase, in both splits."""
    from data.datasets.mri_dataset import MRIDataset
    for split in ("train", "val"):
        ds = MRIDataset(common_conf, synthetic_root, split=split, split_file=split_file,
                        mode="dynamic", mri_mode="axial", num_slices=8,
                        t_target_phases=[5], cache_dir=monai_cache_dir)
        for seq_index in range(6):
            t = int(np.asarray(ds.get_data(seq_index, img_per_seq=8)["t_target"]).item())
            assert t == 5, f"{split}: 1-element pool should pin t_target=5, got {t}"

def test_t_target_phases_out_of_range_wraps(synthetic_root, split_file, common_conf, monai_cache_dir):
    """Pool entries are taken mod T (defensive), so [12, 19] behaves like [0, 7] for T=12."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8,
                    t_target_phases=[12, 19], cache_dir=monai_cache_dir)  # 12%12=0, 19%12=7
    ts = [int(np.asarray(ds.get_data(i, img_per_seq=8)["t_target"]).item()) for i in range(6)]
    assert ts == [0, 7, 0, 7, 0, 7], f"out-of-range pool should wrap mod {SYN_T}; got {ts}"

def test_t_target_phases_empty_raises(synthetic_root, split_file, common_conf, monai_cache_dir):
    """An empty pool is a misconfig and must fail loudly at construction, not mid-training."""
    from data.datasets.mri_dataset import MRIDataset
    with pytest.raises(ValueError, match="non-empty"):
        MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                   mode="dynamic", mri_mode="axial", num_slices=8,
                   t_target_phases=[], cache_dir=monai_cache_dir)

# ── 7b. Original t_target behavior must be preserved (regression) ──────────────

def test_t_target_fixed_forces_phase(synthetic_root, split_file, common_conf, monai_cache_dir):
    """REGRESSION: t_target_fixed=K forces every sample (train AND val) to phase K."""
    from data.datasets.mri_dataset import MRIDataset
    for split in ("train", "val"):
        ds = MRIDataset(common_conf, synthetic_root, split=split, split_file=split_file,
                        mode="dynamic", mri_mode="axial", num_slices=8,
                        t_target_fixed=3, cache_dir=monai_cache_dir)
        for seq_index in range(8):
            s = ds.get_data(seq_index, img_per_seq=8)
            t = int(np.asarray(s["t_target"]).item())
            assert t == 3, f"{split}: t_target_fixed=3 expected 3, got {t}"
            assert s["timesteps"][0] == 3

def test_t_target_fixed_overrides_phases(synthetic_root, split_file, common_conf, monai_cache_dir):
    """PRIORITY: if both t_target_fixed and t_target_phases are set, fixed wins."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8,
                    t_target_fixed=3, t_target_phases=[0, 7], cache_dir=monai_cache_dir)
    for seq_index in range(12):
        t = int(np.asarray(ds.get_data(seq_index, img_per_seq=8)["t_target"]).item())
        assert t == 3, f"t_target_fixed must override t_target_phases; got {t}"

def test_all_phases_train_spans_many(synthetic_root, split_file, common_conf, monai_cache_dir):
    """REGRESSION: with neither knob set, train samples t_target uniformly over all T phases."""
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="dynamic", mri_mode="axial", num_slices=8, cache_dir=monai_cache_dir)
    seen = {int(np.asarray(ds.get_data(i, img_per_seq=8)["t_target"]).item()) for i in range(60)}
    assert seen <= set(range(SYN_T))
    assert len(seen) >= 6, f"all-phases train should spread across phases; saw only {sorted(seen)}"

def test_z_indices_in_range(train_ds):
    s = train_ds.get_data(0, img_per_seq=8)
    for z_idx in s["z_indices"]:
        assert -1.0 <= z_idx[0] <= 1.0, f"z_index {z_idx[0]} out of [-1, 1]"


# ── 8. Static mode ────────────────────────────────────────────────────────────

def test_static_mode_all_same_timestep(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    ds = MRIDataset(common_conf, synthetic_root, split="train", split_file=split_file,
                    mode="static", mri_mode="axial", num_slices=8, cache_dir=monai_cache_dir)
    s = ds.get_data(0, img_per_seq=8)
    t_target = int(np.asarray(s["t_target"]).item())
    for t in s["timesteps"]:
        assert t == t_target, f"Static mode: slot t={t} != t_target={t_target}"
