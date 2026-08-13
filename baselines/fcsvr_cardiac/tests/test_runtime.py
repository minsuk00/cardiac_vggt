import os

import argparse
import pytest
import torch

from baselines.fcsvr_cardiac.runtime import (
    EpochShuffle,
    clip_gradients_like_author,
    configure_reproducibility,
    deterministic_validation,
    ensure_fresh_output,
    positive_int,
    truncate_jsonl_after,
)


def test_configure_reproducibility_enables_deterministic_cuda_algorithms(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    configure_reproducibility(17, strict=False)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    assert not torch.are_deterministic_algorithms_enabled()
    with deterministic_validation():
        assert torch.are_deterministic_algorithms_enabled()
    assert not torch.are_deterministic_algorithms_enabled()


def test_epoch_shuffle_visits_every_subject_once_and_resumes_deterministically():
    sampler = EpochShuffle(size=7, seed=23)
    first_epoch = [sampler.index(step) for step in range(7)]
    second_epoch = [sampler.index(step) for step in range(7, 14)]
    resumed = EpochShuffle(size=7, seed=23)

    assert sorted(first_epoch) == list(range(7))
    assert sorted(second_epoch) == list(range(7))
    assert first_epoch != second_epoch
    assert resumed.index(11) == second_epoch[4]


def test_gradient_clipping_matches_author_value_clip_point_five():
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 3.0)

    clip_gradients_like_author(model)

    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 0.5))


def test_resume_drops_a_torn_final_jsonl_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1, "loss": 2.0}\n{"step": 2, "loss":', encoding="utf-8")

    truncate_jsonl_after(path, step=1)

    assert path.read_text(encoding="utf-8") == '{"step": 1, "loss": 2.0}\n'


def test_resume_rejects_a_malformed_interior_jsonl_record(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 1}\n{"step":\n{"step": 3}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        truncate_jsonl_after(path, step=1)


def test_positive_int_rejects_zero_and_negative_values():
    assert positive_int("3") == 3
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-2")


def test_fresh_output_rejects_any_existing_run_state(tmp_path):
    ensure_fresh_output(tmp_path)
    (tmp_path / "wandb_id.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        ensure_fresh_output(tmp_path)
