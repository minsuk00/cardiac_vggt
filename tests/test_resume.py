"""Resume logic: which checkpoint a (re)started run loads, and wandb run reattachment."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))


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
        with tempfile.TemporaryDirectory() as d:
            empty = os.path.join(d, "ckpts")  # does not exist yet
            assert self._resolve(empty, "/base/model.pt") == "/base/model.pt"
            os.makedirs(empty)  # exists but no checkpoint_last.pt
            assert self._resolve(empty, "/base/model.pt") == "/base/model.pt"

    def test_local_checkpoint_wins_over_seed(self):
        """checkpoint_last.pt present → it wins, NOT the base seed. (The requeue fix.)"""
        with tempfile.TemporaryDirectory() as d:
            ckpt_dir = os.path.join(d, "ckpts")
            os.makedirs(ckpt_dir)
            local = os.path.join(ckpt_dir, "checkpoint_last.pt")
            open(local, "wb").close()
            assert self._resolve(ckpt_dir, "/base/model.pt") == local

    def test_no_local_no_seed_returns_none(self):
        """No local checkpoint and no seed path → None (nothing to load)."""
        with tempfile.TemporaryDirectory() as d:
            assert self._resolve(os.path.join(d, "ckpts"), None) is None


class TestWandbResume(unittest.TestCase):

    def _init_kwargs(self, resume_id):
        import train_utils.wandb_writer as ww
        mock_run = MagicMock()
        mock_run.get_url.return_value = "https://wandb.ai/fake"
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = mock_run
        with patch.object(ww, "wandb", mock_wandb):
            ww.WandbLogger(project="test-proj", name="test-run", resume_id=resume_id)
        return mock_wandb.init.call_args.kwargs

    def test_wandb_init_called_with_resume_id(self):
        """WandbLogger passes id= and resume='allow' to wandb.init when resume_id is set."""
        kw = self._init_kwargs("ypigj8ew")
        assert kw.get("id") == "ypigj8ew"
        assert kw.get("resume") == "allow"

    def test_wandb_init_no_resume_when_no_id(self):
        """WandbLogger does NOT pass id/resume when resume_id is None."""
        kw = self._init_kwargs(None)
        assert kw.get("id") is None
        assert kw.get("resume") is None
