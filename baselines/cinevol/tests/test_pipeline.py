import json
from pathlib import Path
import numpy as np
import nibabel as nib
import pytest
import torch
from cinevol.demo import generate
from cinevol.data import Observations
from cinevol.prepare import prepare
from cinevol.fit import fit, load_checkpoint, model_from_checkpoint
from cinevol.reconstruct import query
from cinevol.config import configuration


@pytest.fixture
def subject(tmp_path):
    return generate(tmp_path / "demo", size=8, frames=3)


def test_target_not_required_by_fitting_loader(subject):
    path = Path(subject)
    meta = json.loads(path.read_text())
    meta["reference"] = "/a/nonexistent/evaluation/target.nii.gz"
    path.write_text(json.dumps(meta))
    data = Observations(path)
    assert data.n > 0
    assert data.n_slices == 12
    assert data.arrays["spacing"].shape == (12,3)


def test_single_stack_and_missing_states_rejected(subject, tmp_path):
    spec = Path(subject).parent.parent / "acquisitions.json"
    obj = json.loads(spec.read_text())
    obj["stacks"] = obj["stacks"][:1]
    p = tmp_path / "single.json"
    p.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="multiplanar"):
        prepare(p, tmp_path / "prepared")
    meta = json.loads(Path(subject).read_text())
    meta["minimum_stack_coverage"] = 1
    Path(subject).write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="coverage"):
        Observations(subject)


def test_checkpoint_reload_and_resume(subject, tmp_path, monkeypatch):
    import importlib
    fitting = importlib.import_module("cinevol.fit")
    def tiny_config(profile, smoke):
        c = configuration(profile, True)
        c["optimization"].update(steps_per_subject=2, batch_observed_pixels=16)
        return c
    monkeypatch.setattr(fitting, "configuration", tiny_config)
    run = tmp_path / "run"
    checkpoint = fit(subject, run, profile="phantom", backend="torch", device="cpu", smoke=True, microbatch=8, checkpoint_every=1)
    state = load_checkpoint(checkpoint)
    assert state["step"] == 2
    model, _ = model_from_checkpoint(checkpoint, "cpu")
    grid = state["subject_metadata"]["output_grid"]
    before = query(model, grid["shape"], np.array(grid["affine"]), .23, 0, chunk=64)
    model2, _ = model_from_checkpoint(checkpoint, "cpu")
    after = query(model2, grid["shape"], np.array(grid["affine"]), .23, 0, chunk=64)
    np.testing.assert_array_equal(before, after)
    with pytest.raises(FileExistsError):
        fit(subject, run, profile="phantom", backend="torch", device="cpu", smoke=True)
    fit(subject, run, profile="phantom", backend="torch", device="cpu", smoke=True, resume=True)
    assert len((run / "losses.jsonl").read_text().splitlines()) == 2
    # Simulate an interruption immediately after checkpoint 1, retaining its RNG
    # and optimizer. Resuming must reproduce step 2 from an uninterrupted run.
    reference_model = {k:v.clone() for k,v in state["model"].items()}
    class Interrupt(Exception):
        pass
    original_gradient = fitting.batch_gradient
    calls = 0
    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Interrupt()
        return original_gradient(*args, **kwargs)
    monkeypatch.setattr(fitting, "batch_gradient", interrupt_second)
    run2 = tmp_path / "interrupted"
    with pytest.raises(Interrupt):
        fit(subject, run2, profile="phantom", backend="torch", device="cpu", smoke=True, microbatch=8, checkpoint_every=1)
    monkeypatch.setattr(fitting, "batch_gradient", original_gradient)
    fit(subject, run2, profile="phantom", backend="torch", device="cpu", smoke=True, microbatch=8, resume=True)
    resumed = load_checkpoint(run2 / "last.pt")
    for k, v in reference_model.items():
        torch.testing.assert_close(v,resumed["model"][k],rtol=0,atol=0)
