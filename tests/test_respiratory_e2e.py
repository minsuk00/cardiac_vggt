"""End-to-end tests for respiratory wiring: dataset → batch → gpu_augment_batch.

Uses the synthetic CMR dataset (conftest.py) so no real data is needed. Proves the
`seq_index` key flows, respiratory changes ONLY input slices on real-shaped data, and
val breathing is deterministic per seq_index.

Run on CPU.
"""

import time

import numpy as np
import pytest
import torch

from data.gpu_aug import extract_slices_from_phases, gpu_augment_batch
from data.respiratory import RespiratoryConfig, extract_slices_with_respiratory_vec

DEVICE = "cpu"


def _batch_from_sample(s):
    """Assemble a (B=1) batch dict (the fields gpu_augment_batch touches) from a
    raw MRIDataset.get_data() sample, mirroring composed_dataset's conversions."""
    return {
        "phases": torch.from_numpy(np.asarray(s["phases"])).unsqueeze(0),               # (1,T,D,H,W) fp16
        "timesteps": torch.from_numpy(np.stack(s["timesteps"]).astype(np.int64)).unsqueeze(0),     # (1,S)
        "slice_indices": torch.from_numpy(np.stack(s["slice_indices"]).astype(np.int64)).unsqueeze(0),
        "t_target": torch.from_numpy(s["t_target"].astype(np.int64)).unsqueeze(0),       # (1,1)
        "seq_index": torch.from_numpy(s["seq_index"].astype(np.int64)).unsqueeze(0),      # (1,1)
        "gt_target_volume": torch.from_numpy(s["gt_target_volume"].astype(np.float32)).unsqueeze(0),
        "content_mask": torch.from_numpy(s["content_mask"].astype(np.uint8)).unsqueeze(0),
        "scanner_coords": torch.from_numpy(np.stack(s["scanner_coords"]).astype(np.float32)).unsqueeze(0),
    }


# ── seq_index emission + collate contract ─────────────────────────────────────
def test_get_data_emits_seq_index(train_ds):
    for k in [0, 1]:
        s = train_ds.get_data(k, img_per_seq=8)
        assert "seq_index" in s
        assert s["seq_index"].tolist() == [k]
        assert s["seq_index"].dtype == np.int64


def test_seq_index_collates_to_b1(train_ds):
    from torch.utils.data._utils.collate import default_collate
    a = torch.from_numpy(train_ds.get_data(0, img_per_seq=8)["seq_index"].astype(np.int64))
    b = torch.from_numpy(train_ds.get_data(1, img_per_seq=8)["seq_index"].astype(np.int64))
    out = default_collate([{"seq_index": a}, {"seq_index": b}])["seq_index"]
    assert out.shape == (2, 1) and out.tolist() == [[0], [1]]


# ── respiratory changes ONLY images, on real-shaped data ──────────────────────
def test_e2e_resp_changes_only_images(train_ds):
    s = train_ds.get_data(0, img_per_seq=8)
    batch = _batch_from_sample(s)
    ref = {k: batch[k].clone() for k in ["phases", "gt_target_volume", "content_mask", "scanner_coords"]}
    g = torch.Generator(device=DEVICE).manual_seed(0)
    cfg = RespiratoryConfig(enable=True)

    # Baseline (no-aug) images = the plain extractor.
    base_images = extract_slices_from_phases(
        batch["phases"].float(), batch["timesteps"], batch["slice_indices"])
    base_images = base_images.permute(0, 1, 4, 2, 3).contiguous() / 255.0

    out = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=True, resp_generator=g)
    for k, v in ref.items():
        assert torch.equal(out[k], v), f"{k} must stay at the reference"
    assert not torch.equal(out["images"], base_images)            # breathing moved the inputs
    assert out["images"].shape == base_images.shape
    assert torch.isfinite(out["images"]).all()
    assert float(out["images"].min()) >= 0.0 and float(out["images"].max()) <= 1.0


def test_e2e_resp_off_leaves_images(train_ds):
    """affine off + respiratory disabled → batch is returned untouched."""
    s = train_ds.get_data(0, img_per_seq=8)
    batch = _batch_from_sample(s)
    batch["images"] = torch.rand(1, 8, 3, 518, 518)
    pre = batch["images"].clone()
    out = gpu_augment_batch(batch, None, DEVICE,
                            respiratory_cfg=RespiratoryConfig(enable=False), train=True)
    assert torch.equal(out["images"], pre)


# ── deterministic val breathing per seq_index ─────────────────────────────────
def test_e2e_val_breathing_deterministic(train_ds):
    s = train_ds.get_data(0, img_per_seq=8)
    cfg = RespiratoryConfig(enable=True)

    batch = _batch_from_sample(s)
    a = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=False)["images"].clone()
    b = gpu_augment_batch(batch, None, DEVICE, respiratory_cfg=cfg, train=False)["images"].clone()
    assert torch.equal(a, b)                                      # same seq_index → identical

    # Different seq_index → different breathing (same underlying volume).
    batch2 = _batch_from_sample(s)
    batch2["seq_index"] = torch.tensor([[123]], dtype=torch.int64)
    c = gpu_augment_batch(batch2, None, DEVICE, respiratory_cfg=cfg, train=False)["images"].clone()
    assert not torch.equal(a, c)


# ── perf sanity (lenient — not a hard CI gate) ────────────────────────────────
def test_resp_extraction_perf_is_reasonable():
    B, T, D, H, W, S = 4, 12, 12, 256, 256, 12
    phases = torch.rand(B, T, D, H, W, dtype=torch.float32)
    t_seq = torch.randint(0, T, (B, S))
    z_seq = torch.randint(0, D, (B, S))
    disp = torch.rand(B, S, 3) * 16 - 8

    # Warmup.
    extract_slices_from_phases(phases, t_seq, z_seq)
    extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp)

    t0 = time.perf_counter()
    for _ in range(3):
        extract_slices_from_phases(phases, t_seq, z_seq)
    t_base = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(3):
        extract_slices_with_respiratory_vec(phases, t_seq, z_seq, disp)
    t_resp = time.perf_counter() - t0

    # The respiratory path adds one grid_sample over D; should stay within a small
    # multiple of the fancy-index baseline (generous bound to avoid CI flakiness).
    assert t_resp < max(0.05, t_base * 10.0)
