"""Tests for the val-only diagnostics added to Trainer:
  - per-phase PSNR accumulator behavior (in `_update_and_log_scalars`)
  - cardiac-cycle filmstrip restores dataset state
  - gating: diagnostics are skipped when `t_target_fixed` is not None
  - training-time scalar logging is unchanged (train path doesn't touch the new code)

These tests don't instantiate a full Trainer (heavy due to DDP + dataloaders).
Instead they bind the methods to a stub object and verify behavior in isolation.
"""
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_data(t_targets, B=None, V_shape=(2, 4, 4)):
    """Build a fake `log_data` dict with V_canon, V_gt, t_target tensors.
    V_gt is identical to V_canon → PSNR is infinite, so we add a fixed offset
    so PSNR per sample varies meaningfully with t_target value (just for testing).
    """
    if B is None:
        B = len(t_targets)
    # V_canon = ones; V_gt = zeros + small noise scaled by t_target.
    V_canon = torch.ones(B, *V_shape, dtype=torch.float32)
    V_gt = torch.zeros(B, *V_shape, dtype=torch.float32)
    for b, tt in enumerate(t_targets):
        # Different error per t_target so per-phase PSNR is distinguishable.
        V_gt[b] = (tt / 12.0)  # constant per slot; MSE = (1 - tt/12)^2.
    t_target = torch.tensor(t_targets, dtype=torch.int64).reshape(B, 1)
    extrinsics = torch.eye(3, 4).unsqueeze(0).expand(B, -1, -1).contiguous()  # required by _update_and_log_scalars
    return {
        "V_canon": V_canon, "V_gt": V_gt, "t_target": t_target,
        "extrinsics": extrinsics,
    }


def _make_stub_trainer(t_target_fixed=None, rank=0):
    """Stub Trainer-like object that has just enough state for the val-only methods we test."""
    from trainer import Trainer
    stub = SimpleNamespace()
    stub.t_target_fixed = t_target_fixed
    stub._per_phase_val_psnr = defaultdict(list)
    stub.rank = rank
    stub.logging_conf = SimpleNamespace(log_freq=1)
    # Mock scalar log to a list we can assert against.
    stub._logged = []
    stub._log_scalar = lambda key, val, step: stub._logged.append((key, float(val), step))
    # Bind only the methods we test.
    stub._update_and_log_scalars = Trainer._update_and_log_scalars.__get__(stub)
    stub._get_scalar_log_keys = lambda phase: []  # no scalar meters needed for these tests
    return stub


# ── 1. Per-phase accumulator behavior ─────────────────────────────────────────

def test_per_phase_accumulator_buckets_correctly():
    """Val batches at different t_targets fill the right buckets."""
    stub = _make_stub_trainer(t_target_fixed=None)
    # Three val batches, each B=2, varying t_targets.
    for t_targets in [[0, 1], [0, 5], [11, 3]]:
        data = _make_data(t_targets)
        stub._update_and_log_scalars(data, phase="val", step=0, loss_meters={})
    # Expected: bucket 0 has 2 entries; bucket 1 has 1; bucket 3 has 1; bucket 5 has 1; bucket 11 has 1.
    counts = {t: len(v) for t, v in stub._per_phase_val_psnr.items()}
    assert counts == {0: 2, 1: 1, 5: 1, 3: 1, 11: 1}, f"unexpected per-phase counts: {counts}"


def test_per_phase_accumulator_skipped_for_train_phase():
    """Train-phase calls must not populate the val accumulator."""
    stub = _make_stub_trainer(t_target_fixed=None)
    data = _make_data([3, 5])
    stub._update_and_log_scalars(data, phase="train", step=0, loss_meters={})
    assert len(stub._per_phase_val_psnr) == 0, \
        f"train phase incorrectly populated val accumulator: {dict(stub._per_phase_val_psnr)}"


def test_per_phase_accumulator_skipped_when_t_target_fixed():
    """Val calls with t_target_fixed set must NOT accumulate per-phase PSNR."""
    stub = _make_stub_trainer(t_target_fixed=0)
    data = _make_data([0, 0, 0])
    stub._update_and_log_scalars(data, phase="val", step=0, loss_meters={})
    assert len(stub._per_phase_val_psnr) == 0, \
        f"fixed-target mode populated accumulator: {dict(stub._per_phase_val_psnr)}"


def test_per_phase_accumulator_handles_missing_keys():
    """If the batch is missing V_canon/V_gt/t_target (legacy supervised pipeline?),
    the diagnostic should silently skip — never raise."""
    stub = _make_stub_trainer(t_target_fixed=None)
    data = {"extrinsics": torch.eye(3, 4).unsqueeze(0)}  # nothing else
    stub._update_and_log_scalars(data, phase="val", step=0, loss_meters={})
    assert len(stub._per_phase_val_psnr) == 0


# ── 2. Filmstrip state restoration ────────────────────────────────────────────

def test_filmstrip_restores_dataset_state():
    """`_log_cardiac_cycle_filmstrip` must restore `t_target_fixed` even if it errors."""
    from trainer import Trainer

    class MockMRIDataset:
        gt_grid_shape = (12, 256, 256)
        num_slices = 12
        t_target_fixed = 7  # the original value

        def get_data(self, seq_index, img_per_seq):
            # Force the filmstrip loop to raise mid-flight (no `images` key etc.)
            return {}

    mock_ds = MockMRIDataset()
    stub = SimpleNamespace()
    stub.wandb_writer = SimpleNamespace(log=lambda *a, **kw: None)  # dummy logger
    stub.model = SimpleNamespace()
    stub.device = "cpu"
    stub.t_target_fixed = None
    stub._get_mri_dataset = lambda: mock_ds
    stub._log_cardiac_cycle_filmstrip = Trainer._log_cardiac_cycle_filmstrip.__get__(stub)

    orig = mock_ds.t_target_fixed
    stub._log_cardiac_cycle_filmstrip(log_step=42)  # should error internally and recover
    assert mock_ds.t_target_fixed == orig, \
        f"t_target_fixed not restored: was {orig}, now {mock_ds.t_target_fixed}"


def test_filmstrip_skipped_without_wandb():
    """No wandb_writer → method returns silently, no error."""
    from trainer import Trainer
    stub = SimpleNamespace()
    stub.wandb_writer = None
    stub._get_mri_dataset = lambda: None
    stub._log_cardiac_cycle_filmstrip = Trainer._log_cardiac_cycle_filmstrip.__get__(stub)
    stub._log_cardiac_cycle_filmstrip(log_step=0)  # must not raise


# ── 3. Identity baseline gating ───────────────────────────────────────────────

def test_baseline_skipped_when_no_dataset():
    """If `_get_mri_dataset()` returns None, baseline computation must skip gracefully."""
    from trainer import Trainer
    stub = SimpleNamespace()
    stub.t_target_fixed = None
    stub.device = "cpu"
    stub.rank = 0
    stub.logging_conf = SimpleNamespace(log_dir="/tmp")
    stub._get_mri_dataset = lambda: None
    stub._log_scalar = lambda *a, **kw: None
    stub._compute_identity_baseline = Trainer._compute_identity_baseline.__get__(stub)
    stub._compute_identity_baseline()  # must not raise
