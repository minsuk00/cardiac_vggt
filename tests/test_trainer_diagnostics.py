"""Tests for the val-only diagnostics in `Monitor`:
  - per-phase PSNR accumulator behavior (in `_update_and_log_scalars`)
  - gating: diagnostics are skipped when `t_target_fixed` is not None
  - training-time scalar logging is unchanged (train path doesn't touch the new code)

A `Monitor` is cheap to build (no model, no dataloaders), so these tests construct a real
one with stub logging collaborators.
"""
from types import SimpleNamespace

import pytest
import torch


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
    images = torch.zeros(B, 1)  # only .shape[0] (batch size) is read by _update_and_log_scalars
    return {
        "V_canon": V_canon, "V_gt": V_gt, "t_target": t_target,
        "images": images,
    }


def _make_monitor(t_target_fixed=None, val_ds=None, wandb_writer=None, log_dir="/tmp"):
    """A real Monitor with stub collaborators; scalars are captured in `m._logged`."""
    from training.monitor import Monitor
    log_conf = SimpleNamespace(log_freq=1, log_dir=log_dir, scalar_keys_to_log=None)
    # The per-subject val record writes a row per sample; a missing run_log would raise
    # inside the per-sample loop, which the outer try/except swallows — silently
    # truncating the per-phase accumulation for the rest of the batch.
    run_log = SimpleNamespace(subject_row=lambda row: None)
    m = Monitor(log_conf, run_log, wandb_writer, val_ds, "cpu")
    m.t_target_fixed = t_target_fixed
    m._logged = []
    m.scalar = lambda key, val, step: m._logged.append((key, float(val), step))
    return m


def _accumulate(m, data, phase="val"):
    m._update_and_log_scalars(data, phase=phase, step=0, train_step=0, meters={})


# ── 1. Per-phase accumulator behavior ─────────────────────────────────────────

def test_per_phase_accumulator_buckets_correctly():
    """Val batches at different t_targets fill the right buckets."""
    m = _make_monitor(t_target_fixed=None)
    # Three val batches, each B=2, varying t_targets.
    for t_targets in [[0, 1], [0, 5], [11, 3]]:
        _accumulate(m, _make_data(t_targets))
    # Expected: bucket 0 has 2 entries; bucket 1 has 1; bucket 3 has 1; bucket 5 has 1; bucket 11 has 1.
    counts = {t: len(v) for t, v in m.val_psnr["full"].items()}
    assert counts == {0: 2, 1: 1, 5: 1, 3: 1, 11: 1}, f"unexpected per-phase counts: {counts}"


def _make_data_with_phases(t_targets, V_shape=(2, 4, 4), T=4):
    """`_make_data` plus a `phases` (B, T, D, H, W) bundle with a deterministic
    moving 2×2 sub-region in z-plane 0 (swing 0.9 > tau), the rest static."""
    data = _make_data(t_targets, V_shape=V_shape)
    B = data["V_canon"].shape[0]
    D, H, W = V_shape
    phases = torch.full((B, T, D, H, W), 0.2)
    ramp = torch.linspace(0.0, 0.9, T).view(1, T, 1, 1)  # swing 0.9 across phases
    phases[:, :, 0, 0:2, 0:2] = ramp                     # 4 moving voxels in plane 0
    data["phases"] = phases
    return data


def test_per_phase_motion_accumulator_buckets_correctly():
    """When `phases` is present, motion PSNR buckets by t_target like full/bbox."""
    m = _make_monitor(t_target_fixed=None)
    for t_targets in [[0, 1], [0, 5]]:
        _accumulate(m, _make_data_with_phases(t_targets))
    counts = {t: len(v) for t, v in m.val_psnr["motion"].items()}
    assert counts == {0: 2, 1: 1, 5: 1}, f"unexpected motion counts: {counts}"
    # Every accumulated value is a finite PSNR.
    assert all(torch.isfinite(torch.tensor(v)).all() for v in m.val_psnr["motion"].values())


def test_motion_accumulator_empty_without_phases():
    """No `phases` in the batch → motion accumulator stays empty, full still fills."""
    m = _make_monitor(t_target_fixed=None)
    _accumulate(m, _make_data([0, 1]))
    assert len(m.val_psnr["motion"]) == 0
    assert sum(len(v) for v in m.val_psnr["full"].values()) == 2  # full path unaffected


def test_per_phase_accumulator_skipped_for_train_phase():
    """Train-phase calls must not populate the val accumulator."""
    m = _make_monitor(t_target_fixed=None)
    _accumulate(m, _make_data([3, 5]), phase="train")
    assert len(m.val_psnr["full"]) == 0, \
        f"train phase incorrectly populated val accumulator: {dict(m.val_psnr['full'])}"


def test_per_phase_accumulator_runs_when_t_target_fixed():
    """Fixed-target val NOW accumulates (bucketed under the single fixed phase) so the motion
    scalar can still be logged. Previously the whole block was skipped, which dropped motion —
    the headline metric — for single-phase runs. The per-phase full/bbox panels are filtered
    out at LOG time (end_val), not here: this path just fills the accumulators.
    """
    m = _make_monitor(t_target_fixed=0)
    # Two val batches (B=2 each), phases present so motion is computed; all t_target=0.
    for _ in range(2):
        _accumulate(m, _make_data_with_phases([0, 0]))
    # Motion + full bucket under the one fixed phase 0 (4 samples total).
    assert {t: len(v) for t, v in m.val_psnr["motion"].items()} == {0: 4}, \
        f"fixed-target motion not accumulated under phase 0: {dict(m.val_psnr['motion'])}"
    assert {t: len(v) for t, v in m.val_psnr["full"].items()} == {0: 4}


def test_fixed_phase_logs_only_motion_panels():
    """The log-time half of the gate above: single-phase val emits val/psnr/motion only."""
    m = _make_monitor(t_target_fixed=0)
    _accumulate(m, _make_data_with_phases([0, 0]))
    m._log_per_phase_psnr(step=5)
    names = [k for k, _, _ in m._logged]
    assert names and all(k.startswith("val/psnr/motion/") for k in names), names


def test_per_phase_accumulator_handles_missing_keys():
    """If the batch is missing V_canon/V_gt/t_target (legacy supervised pipeline?),
    the diagnostic should silently skip — never raise."""
    m = _make_monitor(t_target_fixed=None)
    _accumulate(m, {"images": torch.zeros(1, 1)})  # only the batch-size key, nothing else
    assert len(m.val_psnr["full"]) == 0


# ── 2. Visual diagnostics ─────────────────────────────────────────────────────

def test_motion_mask_example_logs_under_val_motion():
    """`log_motion_mask_example` renders an image and logs it under `media_others/val_motion_mask_example`;
    and returns silently when there's no wandb writer."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    from training.monitor_panels import log_motion_mask_example

    # No wandb → silent no-op.
    log_motion_mask_example(None, None, (0,), 0)  # must not raise

    # With wandb + a mock dataset → logs exactly the media_others/val_motion_mask_example key.
    class MockDS:
        num_slices = 12
        # Distinct source prefixes so the per-source subject picker (docs/60) has
        # something to spread over, instead of the old hardcoded 0/7/15.
        # NOTE the trailing "/sax": _find_subjects builds `<root>/<line>/sax`, so the id
        # is the PARENT dir. Getting this wrong made every subject read as "sax".
        subjects = [f"/data/{src}_sax/{src}_subj{i:02d}/sax"
                    for src in ("ACDC", "MNMs", "CMRx24") for i in range(6)]

        def get_data(self, seq_index, img_per_seq):
            T, D, H, W = 4, 3, 16, 16
            phases = np.full((T, D, H, W), 0.2, dtype=np.float32)
            phases[:, 1, 4:8, 4:8] = np.linspace(0.0, 0.9, T)[:, None, None]  # moving region
            return {"phases": phases, "anatomy_bbox": np.array([0, D, 0, H, 0, W])}

    logged = []
    writer = SimpleNamespace(log=lambda key, val, step: logged.append((key, step)))
    # One index per source, matching what Monitor.visual_subjects would compute.
    log_motion_mask_example(writer, MockDS(), (0, 6, 12), 99)
    assert ("media_others/val_motion_mask_example", 99) in logged, f"image not logged: {logged}"


def test_filmstrip_skipped_without_wandb():
    """No wandb_writer → method returns silently, no error."""
    m = _make_monitor()
    m._log_cardiac_cycle_filmstrip(model=None, step=0)  # must not raise


# ── 3. Identity baseline gating ───────────────────────────────────────────────

def test_baseline_skipped_when_no_dataset():
    """With no val dataset, baseline computation must skip gracefully."""
    m = _make_monitor()
    m._compute_identity_baseline()  # must not raise
    assert m.identity_baseline == {}
