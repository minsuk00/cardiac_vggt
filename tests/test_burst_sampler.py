"""Burst sampling (burst_k > 1): k sequential frames per z-plane, ref slot unchanged.

Synthetic in-memory dataset (conftest.py). Also guards that burst_k=1 (the default)
leaves the one_frame_per_slice sampler bit-identical (RNG-stream equivalence on val).
"""

from collections import Counter

import numpy as np
import pytest

from conftest import SYN_T, SYN_Z

K = 3


@pytest.fixture
def burst_ds(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    return MRIDataset(
        common_conf, synthetic_root, split="val", split_file=split_file,
        mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir,
        reference_slot=True, one_frame_per_slice=True, burst_k=K,
    )


def test_burst_slot_structure(burst_ds):
    s = burst_ds.get_data(seq_index=2, img_per_seq=1 + K * SYN_Z)
    zs, ts = list(s["slice_indices"]), list(s["timesteps"])
    bz0, bz1 = int(s["anatomy_bbox"][0]), int(s["anatomy_bbox"][1])
    n_planes = bz1 - bz0
    z_mid = (bz0 + bz1) // 2
    S = len(zs)
    assert S == 1 + K * n_planes
    # slot 0 = target-phase reference at z_mid
    assert zs[0] == z_mid and ts[0] == int(s["t_target"][0])
    # bursts: K consecutive slots share one z; t sequential mod T
    for b in range(n_planes):
        seg_z = zs[1 + b * K: 1 + (b + 1) * K]
        seg_t = ts[1 + b * K: 1 + (b + 1) * K]
        assert len(set(seg_z)) == 1
        for j in range(1, K):
            assert seg_t[j] == (seg_t[j - 1] + 1) % SYN_T
    # every in-bbox plane exactly K times (+1 ref at z_mid)
    c = Counter(zs)
    for z in range(bz0, bz1):
        assert c[z] == K + (1 if z == z_mid else 0)
    # per-slot keys consistent
    for key in ("scanner_coords", "z_indices", "t_indices", "target_t_indices", "images"):
        assert len(s[key]) == S


def test_burst_budget_guard(burst_ds):
    with pytest.raises(ValueError, match="burst_k"):
        burst_ds.get_data(seq_index=0, img_per_seq=K * SYN_Z)  # one short of 1 + K*Z


def test_burst_val_deterministic(burst_ds):
    a = burst_ds.get_data(seq_index=5, img_per_seq=1 + K * SYN_Z)
    b = burst_ds.get_data(seq_index=5, img_per_seq=1 + K * SYN_Z)
    assert list(a["slice_indices"]) == list(b["slice_indices"])
    assert list(a["timesteps"]) == list(b["timesteps"])


def test_burst_image_content_matches_slot(burst_ds):
    import torch
    import torch.nn.functional as F
    s = burst_ds.get_data(seq_index=1, img_per_seq=1 + K * SYN_Z)
    ph = torch.from_numpy(s["phases"]).float()
    i = 4
    R = s["images"][i].shape[0]
    canon = ph[s["timesteps"][i], s["slice_indices"][i]].unsqueeze(0).unsqueeze(0)
    up = F.interpolate(canon, size=(R, R), mode="bilinear", align_corners=True)
    ref = (up.squeeze() * 255.0).clamp(0, 255).numpy()
    assert np.allclose(s["images"][i][..., 0], ref, atol=1e-3)


def test_burst_k1_bit_identical_to_one_frame(synthetic_root, split_file, common_conf,
                                             monai_cache_dir):
    """burst_k=1 must not perturb the existing sampler (same RNG stream on val)."""
    from data.datasets.mri_dataset import MRIDataset
    kw = dict(mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir,
              reference_slot=True, one_frame_per_slice=True)
    a = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file, **kw)
    b = MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                   burst_k=1, **kw)
    for i in (0, 3, 7):
        sa = a.get_data(seq_index=i, img_per_seq=20)
        sb = b.get_data(seq_index=i, img_per_seq=20)
        assert list(sa["slice_indices"]) == list(sb["slice_indices"])
        assert list(sa["timesteps"]) == list(sb["timesteps"])
        assert int(sa["t_target"][0]) == int(sb["t_target"][0])


def test_burst_rejects_continuous_z(synthetic_root, split_file, common_conf, monai_cache_dir):
    from data.datasets.mri_dataset import MRIDataset
    with pytest.raises(ValueError, match="continuous_z"):
        MRIDataset(common_conf, synthetic_root, split="val", split_file=split_file,
                   mode="dynamic", mri_mode="axial", cache_dir=monai_cache_dir,
                   reference_slot=True, one_frame_per_slice=True,
                   burst_k=K, continuous_z=True)
