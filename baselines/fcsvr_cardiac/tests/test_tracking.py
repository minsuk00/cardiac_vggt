import sys
from types import SimpleNamespace

from baselines.fcsvr_cardiac.tracking import get_or_create_wandb_id, init_wandb


def test_wandb_id_is_persisted_and_reused(tmp_path):
    generated = []

    first = get_or_create_wandb_id(tmp_path, lambda: generated.append("run123") or "run123")
    second = get_or_create_wandb_id(tmp_path, lambda: "wrong")

    assert first == second == "run123"
    assert generated == ["run123"]
    assert (tmp_path / "wandb_id.txt").read_text() == "run123"


def test_wandb_resume_rewinds_history_to_checkpoint_step(tmp_path, monkeypatch):
    captured = {}
    fake = SimpleNamespace(
        util=SimpleNamespace(generate_id=lambda length: "run123"),
        init=lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)

    init_wandb(tmp_path, {"seed": 42}, resume_step=5000)

    assert captured["id"] == "run123"
    assert captured["resume_from"] == "run123?_step=5000"
    assert "resume" not in captured


def test_fresh_wandb_run_uses_allow_resume_for_precheckpoint_restart(tmp_path, monkeypatch):
    captured = {}
    fake = SimpleNamespace(
        util=SimpleNamespace(generate_id=lambda length: "run123"),
        init=lambda **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake)

    init_wandb(tmp_path, {"seed": 42}, resume_step=None)

    assert captured["resume"] == "allow"
    assert "resume_from" not in captured
