"""Tests for target-phase reference-slice conditioning (docs/24, docs/25).

Two flag-gated halves replace the content-free target_t index:
  - DATASET (`reference_slot`): slot 0 = (t_target, mid-ventricular z); other slots scattered
    with the reference plane excluded.
  - MODEL (`use_reference_token`): the native two-token camera_token marks slot 0 as the anchor
    (index 0 = first frame, index 1 = the rest), added onto the per-slot z embedding.

These guard: the dataset slot-0 contract + determinism, the model anchor actually contributes,
default-off is a no-op, and warm-start tolerance (old target_t ckpt → reference model).
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vggt.models.aggregator import Aggregator


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


def test_reference_other_slots_exclude_ref_plane_and_are_distinct(ref_train_ds):
    """Slots 1..S-1 never re-observe the reference plane; all z planes are distinct."""
    for seq in range(5):
        d = ref_train_ds.get_data(seq_index=seq, img_per_seq=12)
        z_mid = _zmid(d)
        rest_z = d["slice_indices"][1:]
        assert z_mid not in rest_z, "a scattered slot duplicated the reference plane"
        assert len(set(d["slice_indices"])) == len(d["slice_indices"]), "z planes not distinct"


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


# ──────────────────────────────────────────────────────────────────────────────
# Model: use_reference_token (tiny conv-patch aggregator, CPU-fast)
# ──────────────────────────────────────────────────────────────────────────────
def _tiny(**kw):
    return Aggregator(
        img_size=28, patch_size=14, embed_dim=64, depth=2, num_heads=4,
        patch_embed="conv", use_z_pose_embedding=True, use_t_pose_embedding=False,
        **kw,
    ).eval()


def test_reference_token_contributes_to_output():
    """Toggling use_reference_token on the SAME model (all other weights identical) changes the
    aggregated tokens → the camera_token anchor actually reaches the output (not a silent no-op).
    """
    agg = _tiny(use_reference_token=True)
    agg.camera_token.data.normal_()  # ensure index-0 (anchor) != index-1 (rest), non-trivial
    torch.manual_seed(0)
    images = torch.rand(1, 3, 3, 28, 28)
    z = torch.randn(1, 3, 1)
    with torch.no_grad():
        out_on, _ = agg(images, z_indices=z)
        agg.use_reference_token = False
        out_off, _ = agg(images, z_indices=z)
    assert not torch.allclose(out_on[-1], out_off[-1], atol=1e-5), \
        "use_reference_token has no effect on the output — anchor token dropped"


def test_reference_token_works_without_other_embeddings():
    """use_reference_token alone (no z/t/target_t) must still enter the conditioning branch."""
    agg = Aggregator(
        img_size=28, patch_size=14, embed_dim=64, depth=2, num_heads=4,
        patch_embed="conv", use_z_pose_embedding=False, use_t_pose_embedding=False,
        use_reference_token=True,
    ).eval()
    images = torch.rand(1, 2, 3, 28, 28)
    with torch.no_grad():
        out, _ = agg(images)  # no z/t/target_t provided — must not raise
    assert out[-1].shape[0] == 1


def test_reference_token_off_by_default():
    """Flag defaults off; no anchor is added and the module attribute reflects that."""
    agg = _tiny()
    assert getattr(agg, "use_reference_token", False) is False


def test_no_target_t_embedder_in_reference_model():
    """Reference model uses the camera_token anchor, not a target_t index → no target_t_embedder."""
    agg = _tiny(use_reference_token=True)
    assert not hasattr(agg, "target_t_embedder")


# ──────────────────────────────────────────────────────────────────────────────
# Warm-start tolerance
# ──────────────────────────────────────────────────────────────────────────────
def test_warmstart_old_target_t_ckpt_into_reference_model():
    """An OLD target_t checkpoint loads into the reference model under strict=False:
    target_t_embedder.* is flagged unexpected (dropped); the shared camera_token/z_embedder load."""
    ref = _tiny(use_reference_token=True)                       # no target_t_embedder
    old = _tiny(use_target_t_pose_embedding=True)               # HAS target_t_embedder
    result = ref.load_state_dict(old.state_dict(), strict=False)
    assert any("target_t_embedder" in k for k in result.unexpected_keys), \
        f"expected target_t_embedder.* in unexpected_keys; got {result.unexpected_keys}"
    assert not any("camera_token" in k for k in result.missing_keys)
    assert not any("camera_token" in k for k in result.unexpected_keys)


def test_warmstart_reference_to_reference_clean():
    """Reference → reference loads with no missing/unexpected keys (no accidental new params)."""
    a = _tiny(use_reference_token=True)
    b = _tiny(use_reference_token=True)
    result = b.load_state_dict(a.state_dict(), strict=False)
    assert not result.missing_keys, f"unexpected missing keys: {result.missing_keys}"
    assert not result.unexpected_keys, f"unexpected extra keys: {result.unexpected_keys}"
