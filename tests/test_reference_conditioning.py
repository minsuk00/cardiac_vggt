"""Tests for target-phase reference-slice conditioning (dataset half, `reference_slot`).

Slot 0 = (t_target, mid-ventricular z); the remaining slots provide full z-coverage +
LV-weighted extra frames (multi-frame; planes may repeat). The model half (the camera_token
anchor on slot 0) is always on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ──────────────────────────────────────────────────────────────────────────────
# Dataset: reference_slot
# ──────────────────────────────────────────────────────────────────────────────
def _make_ds(synthetic_root, split_file, common_conf, monai_cache_dir, split, reference_slot):
    from data.datasets.mri_dataset import MRIDataset
    return MRIDataset(
        common_conf, synthetic_root,
        split=split, split_file=split_file,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
        reference_slot=reference_slot,
        cache_dir=monai_cache_dir,
    )


@pytest.fixture(scope="module")
def ref_train_ds(synthetic_root, split_file, common_conf, monai_cache_dir):
    return _make_ds(synthetic_root, split_file, common_conf, monai_cache_dir, "train", True)


def _zmid(data):
    bb = np.asarray(data["anatomy_bbox"]).astype(np.int64)
    return (int(bb[0]) + int(bb[1])) // 2


def _ttarget(data):
    return int(np.asarray(data["t_target"]).reshape(-1)[0])


def test_default_reference_slot_off(train_ds):
    """The default dataset (conftest train_ds) keeps reference_slot OFF (legacy decoupled)."""
    assert train_ds.reference_slot is False


def test_reference_slot0_observes_target_phase_at_midz(ref_train_ds):
    """Slot 0 = (t_target, mid-ventricular z) on every draw."""
    for seq in range(8):
        d = ref_train_ds.get_data(seq_index=seq, img_per_seq=12)
        tt, z_mid = _ttarget(d), _zmid(d)
        assert d["timesteps"][0] == tt, f"slot 0 phase {d['timesteps'][0]} != t_target {tt}"
        assert d["slice_indices"][0] == z_mid, f"slot 0 z {d['slice_indices'][0]} != z_mid {z_mid}"


def test_reference_multiframe_full_coverage_with_repeats(ref_train_ds):
    """Multi-frame reference contract: slot 0 = z_mid; the remaining slots provide FULL
    z-coverage (every in-bbox plane ≥once) plus LV-weighted extra frames — so planes MAY
    repeat (including z_mid, the LV plane where multi-frame redundancy is desired)."""
    for seq in range(5):
        d = ref_train_ds.get_data(seq_index=seq, img_per_seq=12)
        bb = np.asarray(d["anatomy_bbox"]).astype(np.int64)
        z0, z1 = int(bb[0]), int(bb[1])
        zs = [int(z) for z in d["slice_indices"]]
        assert len(zs) == 12, f"S must equal the budget (12), got {len(zs)}"
        assert zs[0] == _zmid(d), "slot 0 must be the z_mid reference plane"
        assert set(zs) >= set(range(z0, z1)), f"every in-bbox plane must be covered, got {sorted(set(zs))}"
        assert len(zs) > len(set(zs)), "extras must repeat planes (multi-frame)"


def test_reference_other_slots_are_scattered_in_phase(ref_train_ds):
    """Slots 1..S-1 sample scattered phases (not all pinned to t_target)."""
    seen = set()
    for _ in range(25):
        d = ref_train_ds.get_data(seq_index=0, img_per_seq=12)
        seen.update(d["timesteps"][1:])
    assert len(seen) >= 3, f"scattered input phases not diverse: {seen}"


def test_reference_val_deterministic_across_instances(
    synthetic_root, split_file, common_conf, monai_cache_dir
):
    """Two independent val reference datasets → identical (t, z) sequences at the same seq_index."""
    a = _make_ds(synthetic_root, split_file, common_conf, monai_cache_dir, "val", True)
    b = _make_ds(synthetic_root, split_file, common_conf, monai_cache_dir, "val", True)
    for seq in (0, 1, 5):
        da = a.get_data(seq_index=seq, img_per_seq=12)
        db = b.get_data(seq_index=seq, img_per_seq=12)
        assert da["timesteps"] == db["timesteps"], f"val timesteps not reproducible @ {seq}"
        assert da["slice_indices"] == db["slice_indices"], f"val slice_indices not reproducible @ {seq}"
        # slot 0 still observes the (deterministic) target phase at z_mid.
        assert da["timesteps"][0] == _ttarget(da)
        assert da["slice_indices"][0] == _zmid(da)
