"""Unit tests for stage_checkpoint_to_local (node-local /tmp checkpoint staging).

Base-weights-only scope: copy an immutable source to /tmp once, reuse thereafter.
No staleness check by design — the trainer only stages resume_checkpoint_path.

The real staging root is `tempfile.gettempdir()` (/tmp). Tests monkeypatch that so
the staged copies land in an isolated pytest tmp dir (and so the source, which on the
cluster is on GPFS, isn't mistaken for "already node-local").
"""
import getpass
import hashlib
import os
import tempfile

import train_utils.checkpoint as ckpt_mod
from train_utils.checkpoint import stage_checkpoint_to_local


def _staged_path_for(src_abspath, root):
    stage_dir = os.path.join(root, f"vggt-ckpt-stage_{getpass.getuser()}")
    return os.path.join(stage_dir, hashlib.sha1(src_abspath.encode()).hexdigest() + ".pt")


def test_stage_copies_then_reuses(tmp_path, monkeypatch):
    fakeroot = tmp_path / "tmproot"
    fakeroot.mkdir()
    monkeypatch.setattr(ckpt_mod.tempfile, "gettempdir", lambda: str(fakeroot))

    src = tmp_path / "src" / "vggt1b_base.pt"  # sibling of fakeroot, not "under /tmp root"
    src.parent.mkdir()
    src.write_bytes(b"ORIGINAL-WEIGHTS")
    staged_expected = _staged_path_for(str(src.resolve()), str(fakeroot))

    # First call: copies to the staging root and loads from there.
    staged = stage_checkpoint_to_local(str(src))
    assert staged == staged_expected
    assert os.path.isfile(staged)
    assert open(staged, "rb").read() == b"ORIGINAL-WEIGHTS"

    # Mutate the STAGED copy, then call again. A cache hit must reuse the staged file
    # as-is (no re-copy from source) → we still see the mutated bytes.
    with open(staged, "wb") as f:
        f.write(b"MUTATED-STAGE")
    staged2 = stage_checkpoint_to_local(str(src))
    assert staged2 == staged
    assert open(staged2, "rb").read() == b"MUTATED-STAGE", \
        "cache hit must NOT re-copy from source"


def test_already_in_tmp_returned_unchanged(tmp_path, monkeypatch):
    fakeroot = tmp_path / "tmproot"
    fakeroot.mkdir()
    monkeypatch.setattr(ckpt_mod.tempfile, "gettempdir", lambda: str(fakeroot))

    # A source already under the staging root is not worth staging → returned as-is.
    src = fakeroot / "already_local.pt"
    src.write_bytes(b"x")
    assert stage_checkpoint_to_local(str(src)) == str(src)


def test_missing_source_falls_back():
    # Nonexistent / remote path → return as-is so the caller's open() reports it.
    assert stage_checkpoint_to_local("/no/such/file.pt") == "/no/such/file.pt"
