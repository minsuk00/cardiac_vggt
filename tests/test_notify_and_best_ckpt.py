"""Tests for the docs/64 safeguards: the gradient-collapse alarm and best-val checkpointing.

The alarm is only worth having if it FIRES on the real failure and stays silent otherwise,
so both directions are tested against the actual measured numbers from docs/64.
"""
import os

import pytest

from training.utils.notify import GradientCollapseAlarm, send_email


class _FakeSMTP:
    """Stand-in transport. Collects messages instead of talking to an MTA."""
    sent = []

    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def send_message(self, m): _FakeSMTP.sent.append(m); return {}


@pytest.fixture(autouse=True)
def _no_real_mail(monkeypatch):
    """Never touch a real MTA from the test suite.

    Stub the TRANSPORT, not `send_email` — patching `send_email` itself would make the
    tests that exercise `send_email` assert against their own stub.
    """
    _FakeSMTP.sent = []
    monkeypatch.setattr("smtplib.SMTP", _FakeSMTP)
    import training.utils.notify as notify
    monkeypatch.setattr(notify, "_SENT_KEYS", set())
    monkeypatch.setattr(notify, "DEFAULT_TO", "test@example.invalid")
    return _FakeSMTP.sent


def test_alarm_fires_on_the_docs64_signature():
    """grad_aggregator pinned < 1e-6 — the measured collapsed-run value was ~1e-10."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=200)
    fired = [a.update(1.56e-10, step=i) for i in range(250)]
    assert any(fired), "alarm must fire on a sustained sub-threshold gradient"
    assert fired.index(True) == 199, "must fire exactly at `patience` consecutive steps"


def test_alarm_fires_only_once():
    a = GradientCollapseAlarm(threshold=1e-6, patience=10)
    fired = [a.update(0.0, step=i) for i in range(100)]
    assert sum(fired) == 1, "a per-step tripwire must not emit repeatedly"


def test_alarm_silent_on_healthy_run():
    """Measured healthy median is 1e-2..8e-2 (docs/64)."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=200)
    assert not any(a.update(0.02, step=i) for i in range(5000))


def test_alarm_silent_on_degraded_but_alive_run():
    """The 3e-4 arm sat at ~6e-5 — bad, but NOT the dead-ReLU signature. No false alarm."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=200)
    assert not any(a.update(5.93e-5, step=i) for i in range(5000))


def test_alarm_counter_resets_on_recovery():
    """A transient dip must not accumulate toward the patience threshold."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=100)
    for i in range(99):
        assert not a.update(0.0, step=i)
    assert not a.update(0.5, step=99), "healthy step must reset the run"
    assert a.run == 0
    for i in range(99):
        assert not a.update(0.0, step=100 + i)


def test_alarm_handles_none_and_disabled():
    a = GradientCollapseAlarm(threshold=1e-6, patience=1)
    assert not a.update(None, step=0)
    b = GradientCollapseAlarm(threshold=1e-6, patience=1, enabled=False)
    assert not b.update(0.0, step=0)


def test_send_email_never_raises(monkeypatch):
    """Notification is best-effort; a broken MTA must not kill a 14-day run."""
    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("smtplib.SMTP", boom)
    import training.utils.notify as notify
    assert notify.send_email("subject", "body") is False


def test_send_email_once_key_dedupes(_no_real_mail):
    import training.utils.notify as notify
    assert notify.send_email("s", "b", once_key="k") is True
    assert notify.send_email("s", "b", once_key="k") is False
    assert len(_no_real_mail) == 1


def test_send_email_without_recipient_only_logs(_no_real_mail, monkeypatch):
    """No VGGT_NOTIFY_EMAIL -> nothing is sent; the alarm still fires (it logs first)."""
    import training.utils.notify as notify
    monkeypatch.setattr(notify, "DEFAULT_TO", None)
    assert notify.send_email("s", "b") is False
    a = GradientCollapseAlarm(threshold=1e-6, patience=1)
    assert a.update(0.0, step=0) is True
    assert _no_real_mail == []


def test_alarm_actually_sends_one_email(_no_real_mail):
    """End-to-end: the alarm must reach the transport exactly once."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=5)
    for i in range(200):
        a.update(0.0, step=i, epoch=3)
    assert len(_no_real_mail) == 1
    assert "GRADIENT COLLAPSE" in _no_real_mail[0]["Subject"]


def _bare_trainer(save_dir, mode="train"):
    """A Trainer with only the state the best-checkpoint code touches."""
    import torch
    from omegaconf import OmegaConf
    from training.trainer import Trainer
    t = Trainer.__new__(Trainer)
    t.checkpoint_conf = OmegaConf.create({"save_dir": str(save_dir)})
    t.model = torch.nn.Linear(2, 2)
    t.mode = mode
    t.epoch = 0
    return t


def _best_value(save_dir):
    import torch
    return torch.load(os.path.join(save_dir, "checkpoint_best.pt"), weights_only=False)["best_metric_value"]


def test_best_survives_requeue(tmp_path):
    """A requeued process must not overwrite a better checkpoint_best.pt with a worse epoch."""
    t = _bare_trainer(tmp_path)
    t._restore_best_val_metric()
    t._last_val_metric = 30.0
    t._maybe_save_best_checkpoint()

    t2 = _bare_trainer(tmp_path)          # fresh process after the requeue
    t2._restore_best_val_metric()
    t2._last_val_metric = 25.0
    t2._maybe_save_best_checkpoint()
    assert _best_value(tmp_path) == 30.0


def test_final_val_can_become_best(tmp_path, monkeypatch):
    t = _bare_trainer(tmp_path)
    t._restore_best_val_metric()
    monkeypatch.setattr(t, "run_train", lambda: None)
    monkeypatch.setattr(t, "_log_exit", lambda *a, **k: None)
    monkeypatch.setattr(t, "run_val", lambda: setattr(t, "_last_val_metric", 31.0))
    t.run()
    assert _best_value(tmp_path) == 31.0
