"""Tests for the docs/64 safeguards: the gradient-collapse alarm and best-val checkpointing.

The alarm is only worth having if it FIRES on the real failure and stays silent otherwise,
so both directions are tested against the actual measured numbers from docs/64.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from train_utils.notify import GradientCollapseAlarm, send_email  # noqa: E402


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
    import train_utils.notify as notify
    monkeypatch.setattr(notify, "_SENT_KEYS", set())
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
    import train_utils.notify as notify
    assert notify.send_email("subject", "body") is False


def test_send_email_once_key_dedupes(_no_real_mail):
    import train_utils.notify as notify
    assert notify.send_email("s", "b", once_key="k") is True
    assert notify.send_email("s", "b", once_key="k") is False
    assert len(_no_real_mail) == 1


def test_alarm_actually_sends_one_email(_no_real_mail):
    """End-to-end: the alarm must reach the transport exactly once."""
    a = GradientCollapseAlarm(threshold=1e-6, patience=5)
    for i in range(200):
        a.update(0.0, step=i, epoch=3)
    assert len(_no_real_mail) == 1
    assert "GRADIENT COLLAPSE" in _no_real_mail[0]["Subject"]
