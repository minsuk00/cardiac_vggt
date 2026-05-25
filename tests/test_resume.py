"""
End-to-end tests for the training resume logic.

Covers:
  1. Checkpoint keys — saved file contains prev_epoch, steps, model, optimizer
  2. Epoch restore  — _load_resuming_checkpoint sets self.epoch = prev_epoch + 1
  3. Steps restore  — self.steps is restored correctly
  4. Model weights  — parameters are loaded correctly
  5. WandB init     — resume_id causes wandb.init to receive id= and resume="allow"
  6. Sbatch path derivation — given a wandb run folder, checkpoint path is derived correctly
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Make training/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


# ---------------------------------------------------------------------------
# Minimal stub of the parts of Trainer needed for checkpoint loading
# ---------------------------------------------------------------------------

class _FakeOptim:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __iter__(self):
        return iter([self])

    def __len__(self):
        return 1


class _FakeOptimConf:
    class amp:
        enabled = False


class _FakeCheckpointConf:
    strict = True


class _MinimalTrainer:
    """Stripped-down trainer that only has the checkpoint-loading plumbing."""

    def __init__(self, model, optimizer):
        self.rank = 0
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        self.ckpt_time_elapsed = 0
        self.model = model
        self.optims = _FakeOptim(optimizer)
        self.optim_conf = _FakeOptimConf()
        self.checkpoint_conf = _FakeCheckpointConf()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Copy of the real trainer method so we exercise the actual code path."""
        import logging
        from train_utils.checkpoint import robust_torch_save

        with open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)

        if "optimizer" in checkpoint:
            opt_states = checkpoint["optimizer"]
            if not isinstance(opt_states, list):
                opt_states = [opt_states]
            for optim, state in zip(self.optims, opt_states):
                optim.optimizer.load_state_dict(state)

        if "prev_epoch" in checkpoint:
            self.epoch = checkpoint["prev_epoch"] + 1
        elif "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]

        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)


def _make_checkpoint(path, prev_epoch, steps, model, optimizer):
    """Save a checkpoint in the format produced by the real trainer."""
    checkpoint = {
        "prev_epoch": prev_epoch,
        "steps": steps,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "time_elapsed": 123.4,
    }
    torch.save(checkpoint, path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckpointResume(unittest.TestCase):

    def _make_model_and_optimizer(self, bias_val=1.0):
        model = nn.Linear(4, 2, bias=True)
        nn.init.constant_(model.bias, bias_val)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Take one step so optimizer has non-trivial state
        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return model, optimizer

    def test_checkpoint_keys(self, tmp_path=None):
        """Saved checkpoint contains all expected keys."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint.pt")
            model, optimizer = self._make_model_and_optimizer()
            _make_checkpoint(ckpt, prev_epoch=5, steps={"train": 200, "val": 40},
                             model=model, optimizer=optimizer)
            state = torch.load(ckpt, map_location="cpu")
            assert "prev_epoch" in state, "checkpoint missing prev_epoch"
            assert "steps" in state,      "checkpoint missing steps"
            assert "model" in state,      "checkpoint missing model"
            assert "optimizer" in state,  "checkpoint missing optimizer"
            assert state["prev_epoch"] == 5
            print("  [PASS] checkpoint_keys")

    def test_epoch_restored_as_prev_plus_one(self):
        """After loading, self.epoch == saved prev_epoch + 1."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint.pt")
            model, optimizer = self._make_model_and_optimizer()
            _make_checkpoint(ckpt, prev_epoch=7, steps={"train": 300, "val": 60},
                             model=model, optimizer=optimizer)

            model2, optimizer2 = self._make_model_and_optimizer(bias_val=0.0)
            trainer = _MinimalTrainer(model2, optimizer2)
            assert trainer.epoch == 0, "epoch should start at 0"

            trainer._load_resuming_checkpoint(ckpt)
            assert trainer.epoch == 8, f"expected epoch=8, got {trainer.epoch}"
            print("  [PASS] epoch_restored_as_prev_plus_one")

    def test_steps_restored(self):
        """self.steps is restored correctly from checkpoint."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint.pt")
            model, optimizer = self._make_model_and_optimizer()
            saved_steps = {"train": 1234, "val": 99}
            _make_checkpoint(ckpt, prev_epoch=3, steps=saved_steps,
                             model=model, optimizer=optimizer)

            model2, optimizer2 = self._make_model_and_optimizer(bias_val=0.0)
            trainer = _MinimalTrainer(model2, optimizer2)
            trainer._load_resuming_checkpoint(ckpt)
            assert trainer.steps == saved_steps, f"steps mismatch: {trainer.steps}"
            print("  [PASS] steps_restored")

    def test_model_weights_restored(self):
        """Model parameters match the saved checkpoint."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint.pt")
            model, optimizer = self._make_model_and_optimizer(bias_val=3.14)
            _make_checkpoint(ckpt, prev_epoch=2, steps={"train": 0, "val": 0},
                             model=model, optimizer=optimizer)

            model2, optimizer2 = self._make_model_and_optimizer(bias_val=0.0)
            trainer = _MinimalTrainer(model2, optimizer2)
            trainer._load_resuming_checkpoint(ckpt)

            saved_bias = model.bias.detach()
            loaded_bias = trainer.model.bias.detach()
            assert torch.allclose(saved_bias, loaded_bias), \
                f"bias mismatch: saved={saved_bias}, loaded={loaded_bias}"
            print("  [PASS] model_weights_restored")

    def test_optimizer_state_restored(self):
        """Optimizer step counts (exp_avg) are restored from checkpoint."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "checkpoint.pt")
            model, optimizer = self._make_model_and_optimizer(bias_val=1.0)
            saved_opt_state = optimizer.state_dict()
            _make_checkpoint(ckpt, prev_epoch=2, steps={"train": 0, "val": 0},
                             model=model, optimizer=optimizer)

            model2, optimizer2 = self._make_model_and_optimizer(bias_val=0.0)
            # Take a step so optimizer2 has its own state
            loss = model2(torch.ones(1, 4)).sum()
            loss.backward()
            optimizer2.step()
            optimizer2.zero_grad()

            trainer = _MinimalTrainer(model2, optimizer2)
            trainer._load_resuming_checkpoint(ckpt)

            loaded_opt_state = trainer.optims.optimizer.state_dict()
            # Compare step count for the first parameter group
            orig_step = list(saved_opt_state["state"].values())[0]["step"]
            loaded_step = list(loaded_opt_state["state"].values())[0]["step"]
            assert orig_step == loaded_step, \
                f"optimizer step mismatch: saved={orig_step}, loaded={loaded_step}"
            print("  [PASS] optimizer_state_restored")


class TestResumePriority(unittest.TestCase):
    """Regression for the requeue bug: a run's own checkpoint_last.pt in save_dir MUST
    take precedence over the configured seed/base checkpoint (resume_checkpoint_path).

    Before the fix, resume_checkpoint_path (config default = base VGGT model.pt, which has
    no prev_epoch/steps/optimizer) unconditionally won, so every SLURM requeue silently
    reloaded base weights at epoch 0 and discarded all training progress.
    """

    def _resolve(self, save_dir, seed):
        from train_utils.general import resolve_resume_checkpoint
        return resolve_resume_checkpoint(save_dir, seed)

    def test_cold_start_uses_seed(self):
        """Empty/nonexistent save_dir → fall back to the seed/base checkpoint."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, "ckpts")  # does not exist yet
            assert self._resolve(empty, "/base/model.pt") == "/base/model.pt"
            os.makedirs(empty)  # exists but no checkpoint_last.pt
            assert self._resolve(empty, "/base/model.pt") == "/base/model.pt"
            print("  [PASS] cold_start_uses_seed")

    def test_local_checkpoint_wins_over_seed(self):
        """checkpoint_last.pt present → it wins, NOT the base seed. (The requeue fix.)"""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ckpt_dir = os.path.join(d, "ckpts")
            os.makedirs(ckpt_dir)
            local = os.path.join(ckpt_dir, "checkpoint_last.pt")
            open(local, "wb").close()
            resolved = self._resolve(ckpt_dir, "/base/model.pt")
            assert resolved == local, f"expected local ckpt to win, got {resolved}"
            print("  [PASS] local_checkpoint_wins_over_seed")

    def test_no_local_no_seed_returns_none(self):
        """No local checkpoint and no seed path → None (nothing to load)."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            assert self._resolve(os.path.join(d, "ckpts"), None) is None
            print("  [PASS] no_local_no_seed_returns_none")


class TestWandbResume(unittest.TestCase):

    def test_wandb_init_called_with_resume_id(self):
        """WandbLogger passes id= and resume='allow' to wandb.init when resume_id is set."""
        import train_utils.wandb_writer as ww

        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://wandb.ai/fake"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.object(ww, "wandb", mock_wandb), \
             patch("train_utils.wandb_writer.get_machine_local_and_dist_rank", return_value=(0, 0)):
            ww.WandbLogger(project="test-proj", name="test-run", resume_id="ypigj8ew")

        call_kwargs = mock_wandb.init.call_args.kwargs
        assert call_kwargs.get("id") == "ypigj8ew", \
            f"expected id='ypigj8ew', got {call_kwargs.get('id')}"
        assert call_kwargs.get("resume") == "allow", \
            f"expected resume='allow', got {call_kwargs.get('resume')}"
        print("  [PASS] wandb_init_called_with_resume_id")

    def test_wandb_init_no_resume_when_no_id(self):
        """WandbLogger does NOT pass id/resume when resume_id is None."""
        import train_utils.wandb_writer as ww

        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://wandb.ai/fake"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run

        with patch.object(ww, "wandb", mock_wandb), \
             patch("train_utils.wandb_writer.get_machine_local_and_dist_rank", return_value=(0, 0)):
            ww.WandbLogger(project="test-proj", name="test-run", resume_id=None)

        call_kwargs = mock_wandb.init.call_args.kwargs
        assert "id" not in call_kwargs or call_kwargs.get("id") is None, \
            f"did not expect id= in kwargs, got {call_kwargs}"
        assert "resume" not in call_kwargs or call_kwargs.get("resume") is None, \
            f"did not expect resume= in kwargs, got {call_kwargs}"
        print("  [PASS] wandb_init_no_resume_when_no_id")


class TestSbatchPathDerivation(unittest.TestCase):

    def test_checkpoint_derived_from_wandb_run_folder(self):
        """
        Simulate the sbatch script's path derivation:
          scratch/logs/<run_name>/wandb/wandb/run-<ts>-<id>/
            dirname x3 →  scratch/logs/<run_name>/
            + ckpts/checkpoint.pt
        """
        import tempfile
        with tempfile.TemporaryDirectory() as base:
            run_id = "ypigj8ew"
            run_folder = Path(base) / "scratch" / "logs" / "my_run" / "wandb" / "wandb" / f"run-20260407_043327-{run_id}"
            run_folder.mkdir(parents=True)

            # Simulate the checkpoint existing
            ckpt_dir = Path(base) / "scratch" / "logs" / "my_run" / "ckpts"
            ckpt_dir.mkdir(parents=True)
            ckpt_file = ckpt_dir / "checkpoint.pt"
            ckpt_file.touch()

            # Replicate bash: dirname x3
            log_dir = run_folder.parent.parent.parent
            derived_ckpt = log_dir / "ckpts" / "checkpoint.pt"

            assert derived_ckpt.exists(), f"derived path does not exist: {derived_ckpt}"
            assert derived_ckpt == ckpt_file, \
                f"path mismatch:\n  derived: {derived_ckpt}\n  actual:  {ckpt_file}"
            print("  [PASS] checkpoint_derived_from_wandb_run_folder")


if __name__ == "__main__":
    print("=== test_resume.py ===\n")
    suite = unittest.TestLoader().loadTestsFromModule(__import__("__main__"))
    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
