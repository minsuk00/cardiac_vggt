"""Checkpoint save/restore and the train-step contract, on the REAL optimizer path.

A tiny CPU model stands in for VGGT, but the optimizer, its schedulers and the gradient
clipper are built from the composed `default.yaml` exactly as the trainer builds them, the
steps go through `trainer.train_step`, and checkpoints go through `make_checkpoint` +
`robust_torch_save` / `restore_checkpoint`.
"""
import math
import os

import pytest
import torch
import torch.nn as nn

from training.utils.checkpoint import make_checkpoint, restore_checkpoint, robust_torch_save
from training.utils.optimizer import construct_optimizer
from training.trainer import train_step


@pytest.fixture(scope="module")
def optim_conf():
    from hydra import compose, initialize_config_dir
    from training.utils.resolvers import register_all

    register_all()
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training", "config"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        return compose(config_name="default").optim


class TinyModel(nn.Module):
    """Parameter names cover both clipper groups of default.yaml ("aggregator", "point")."""

    def __init__(self):
        super().__init__()
        self.aggregator = nn.Linear(4, 4)
        self.point_head = nn.Linear(4, 3)

    def forward(self, images, batch):
        return {"world_points": self.point_head(torch.relu(self.aggregator(images)))}


def l1_loss(y_hat, batch, val_metrics=False):
    return {"objective": (y_hat["world_points"] - batch["target"]).abs().mean()}


def nan_loss(y_hat, batch, val_metrics=False):
    return {"objective": y_hat["world_points"].sum() * float("nan")}


def _parts(optim_conf, seed=0):
    from hydra.utils import instantiate

    torch.manual_seed(seed)
    model = TinyModel()
    optim = construct_optimizer(model, optim_conf)
    clipper = instantiate(optim_conf.gradient_clip)
    clipper.setup_clipping(model)
    return model, optim, clipper


def _batch(i):
    g = torch.Generator().manual_seed(100 + i)
    return {"images": torch.randn(2, 4, generator=g), "target": torch.randn(2, 3, generator=g)}


def _run(model, optim, clipper, steps, start=0, n_total=10):
    for i in range(start, start + steps):
        _, _, finite, _ = train_step(model, l1_loss, optim, clipper, _batch(i),
                                     where=i / n_total, amp=False)
        assert finite


def _assert_state_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
        else:
            assert a[k] == b[k], k


def test_roundtrip_restores_epoch_steps_optimizer_and_schedule(optim_conf, tmp_path):
    model, optim, clipper = _parts(optim_conf)
    _run(model, optim, clipper, steps=3)
    steps = {"train": 3, "val": 5}
    path = str(tmp_path / "checkpoint_last.pt")
    robust_torch_save(make_checkpoint(model, optim, epoch=4, steps=steps, time_elapsed=12.5), path)
    assert not os.path.exists(path + ".tmp")

    model2, optim2, clipper2 = _parts(optim_conf, seed=1)       # different init on purpose
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert list(ckpt) == ["prev_epoch", "steps", "time_elapsed", "optimizer", "model"]
    epoch, steps2, elapsed = restore_checkpoint(ckpt, model2, optim2, strict=True)

    assert epoch == 4 + 1
    assert steps2 == steps
    assert elapsed == 12.5
    _assert_state_equal(model.state_dict(), model2.state_dict())
    s1, s2 = optim.optimizer.state_dict(), optim2.optimizer.state_dict()
    assert s1["state"].keys() == s2["state"].keys() and len(s1["state"]) > 0
    for k in s1["state"]:
        _assert_state_equal(s1["state"][k], s2["state"][k])

    # The schedule is a pure function of `where`, so the next step's lr/wd must agree.
    where_next = 3 / 10
    optim.step_schedulers(where_next)
    optim2.step_schedulers(where_next)
    for opt in ("lr", "weight_decay"):
        assert optim.optimizer.param_groups[0][opt] == optim2.optimizer.param_groups[0][opt]

    # And training continues identically from the restored state.
    clipper2.setup_clipping(model2)
    _run(model, optim, clipper, steps=2, start=3)
    _run(model2, optim2, clipper2, steps=2, start=3)
    _assert_state_equal(model.state_dict(), model2.state_dict())


def test_old_format_checkpoint_still_loads(optim_conf, tmp_path):
    """Older checkpoints carry "scaler"/"ema_models" and a list-wrapped optimizer state."""
    model, optim, clipper = _parts(optim_conf)
    _run(model, optim, clipper, steps=2)
    ckpt = make_checkpoint(model, optim, epoch=7, steps={"train": 2, "val": 0}, time_elapsed=1.0)
    ckpt["optimizer"] = [ckpt["optimizer"]]
    ckpt["scaler"] = {"scale": 65536.0, "_growth_tracker": 0}
    ckpt["ema_models"] = None
    path = str(tmp_path / "old.pt")
    robust_torch_save(ckpt, path)

    model2, optim2, _ = _parts(optim_conf, seed=1)
    epoch, steps, _ = restore_checkpoint(torch.load(path, weights_only=False), model2, optim2,
                                         strict=True)
    assert (epoch, steps) == (8, {"train": 2, "val": 0})
    _assert_state_equal(model.state_dict(), model2.state_dict())
    assert len(optim2.optimizer.state_dict()["state"]) == len(optim.optimizer.state_dict()["state"])


def test_bare_state_dict_is_a_cold_start(optim_conf):
    """Base weights are a plain state dict: weights load, progress starts from zero, and a
    val-mode trainer (optim=None) loads weights only."""
    model, _, _ = _parts(optim_conf)
    model2, _, _ = _parts(optim_conf, seed=1)
    epoch, steps, elapsed = restore_checkpoint(model.state_dict(), model2, None, strict=True)
    assert (epoch, steps, elapsed) == (0, {"train": 0, "val": 0}, 0)
    _assert_state_equal(model.state_dict(), model2.state_dict())


def test_nonfinite_loss_skips_backward_and_leaves_params_unchanged(optim_conf):
    model, optim, clipper = _parts(optim_conf)
    _run(model, optim, clipper, steps=1)                     # optimizer state exists
    params = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state = optim.optimizer.state_dict()["state"]
    opt_state = {k: {n: t.clone() for n, t in s.items()} for k, s in opt_state.items()}

    _, loss_dict, finite, grad_norms = train_step(model, nan_loss, optim, clipper, _batch(1),
                                                  where=0.5, amp=False)

    assert finite is False and math.isnan(loss_dict["objective"].item())
    assert all(p.grad is None for p in model.parameters()), "backward must be skipped"
    _assert_state_equal(params, model.state_dict())
    after = optim.optimizer.state_dict()["state"]
    for k in opt_state:
        _assert_state_equal(opt_state[k], after[k])
    # The schedule still advances: it is a function of `where`, not of the step outcome.
    assert optim.optimizer.param_groups[0]["lr"] == optim.schedulers["lr"](0.5)
    assert all(v == 0.0 for v in grad_norms.values())


def masked_loss(y_hat, batch, val_metrics=False):
    """Finite objective even when world_points holds NaN (like the splat, which drops them)."""
    wp = y_hat["world_points"]
    return {"objective": torch.where(torch.isfinite(wp), wp, torch.zeros_like(wp)).abs().mean()}


def test_nonfinite_gradient_from_finite_loss_skips_the_update(optim_conf):
    """A NaN input gives a finite (masked) loss but NaN weight gradients: the step is skipped."""
    model, optim, clipper = _parts(optim_conf)
    _run(model, optim, clipper, steps=1)
    params = {k: v.clone() for k, v in model.state_dict().items()}
    opt_state = optim.optimizer.state_dict()["state"]
    opt_state = {k: {n: t.clone() for n, t in s.items()} for k, s in opt_state.items()}
    batch = _batch(1)
    batch["images"][0, 0] = float("nan")

    _, loss_dict, finite, grad_norms = train_step(model, masked_loss, optim, clipper, batch,
                                                  where=0.5, amp=False)

    assert math.isfinite(loss_dict["objective"].item())
    assert not all(math.isfinite(v) for v in grad_norms.values())
    assert finite is False
    _assert_state_equal(params, model.state_dict())
    after = optim.optimizer.state_dict()["state"]
    for k in opt_state:
        _assert_state_equal(opt_state[k], after[k])


def test_finite_loss_updates_params(optim_conf):
    model, optim, clipper = _parts(optim_conf)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    _, _, finite, grad_norms = train_step(model, l1_loss, optim, clipper, _batch(0),
                                          where=0.5, amp=False)
    assert finite
    assert set(grad_norms) == {"aggregator", "point"}
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
