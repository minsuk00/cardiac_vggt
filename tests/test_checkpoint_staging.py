"""Unit tests for stage_checkpoint_to_local (node-local /tmp checkpoint staging).

Path-keyed cache validated by (size, mtime): an unchanged source is a cache hit (no
re-copy); a source whose content changed (size or mtime) is re-staged, never served
stale. The real staging root is `tempfile.gettempdir()` (/tmp) — tests monkeypatch that
so staged copies land in an isolated pytest tmp dir (and so the source, which on the
cluster is on GPFS, isn't mistaken for "already node-local").
"""
import getpass
import hashlib
import os
import tempfile

import vggt.utils.checkpoint_stage as ckpt_mod
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local


def _staged_path_for(src_abspath, root):
    stage_dir = os.path.join(root, f"vggt-ckpt-stage_{getpass.getuser()}")
    return os.path.join(stage_dir, hashlib.sha1(src_abspath.encode()).hexdigest() + ".pt")


def _setup(tmp_path, monkeypatch):
    """Isolate the staging root and return a helper to make source files outside it."""
    fakeroot = tmp_path / "tmproot"
    fakeroot.mkdir()
    monkeypatch.setattr(ckpt_mod.tempfile, "gettempdir", lambda: str(fakeroot))
    srcdir = tmp_path / "src"
    srcdir.mkdir()
    return fakeroot, srcdir


def test_stage_copies_then_cache_hit(tmp_path, monkeypatch):
    fakeroot, srcdir = _setup(tmp_path, monkeypatch)
    src = srcdir / "vggt1b_base.pt"
    src.write_bytes(b"ORIGINAL-WEIGHTS")
    staged_expected = _staged_path_for(str(src.resolve()), str(fakeroot))

    # First call: copies to the staging root and loads from there.
    staged = stage_checkpoint_to_local(str(src))
    assert staged == staged_expected
    assert open(staged, "rb").read() == b"ORIGINAL-WEIGHTS"

    # Second call on an UNCHANGED source: cache hit → no re-copy. Prove it by asserting
    # the staged file's identity (inode) and mtime are untouched.
    st1 = os.stat(staged)
    staged2 = stage_checkpoint_to_local(str(src))
    st2 = os.stat(staged2)
    assert staged2 == staged
    assert (st2.st_ino, st2.st_mtime_ns) == (st1.st_ino, st1.st_mtime_ns), \
        "unchanged source must be a cache hit (no re-copy)"


def test_restage_when_source_newer_same_size(tmp_path, monkeypatch):
    fakeroot, srcdir = _setup(tmp_path, monkeypatch)
    src = srcdir / "checkpoint_last.pt"
    src.write_bytes(b"AAAAAAAA")  # 8 bytes
    staged = stage_checkpoint_to_local(str(src))
    assert open(staged, "rb").read() == b"AAAAAAAA"

    # Overwrite in place with SAME-size but different content, and a strictly newer mtime
    # (mimics a live checkpoint_last.pt being re-saved). Must re-stage, not serve stale.
    src.write_bytes(b"BBBBBBBB")
    newer = os.stat(staged).st_mtime + 10
    os.utime(src, (newer, newer))
    staged2 = stage_checkpoint_to_local(str(src))
    assert staged2 == staged
    assert open(staged2, "rb").read() == b"BBBBBBBB", "newer source must re-stage (not stale)"


def test_restage_when_size_differs(tmp_path, monkeypatch):
    fakeroot, srcdir = _setup(tmp_path, monkeypatch)
    src = srcdir / "ckpt.pt"
    src.write_bytes(b"SHORT")
    staged = stage_checkpoint_to_local(str(src))

    # Change size but pin mtime OLDER than the staged copy, so the mtime check alone would
    # pass — proving the SIZE guard independently triggers a re-stage.
    src.write_bytes(b"A-MUCH-LONGER-PAYLOAD")
    older = os.stat(staged).st_mtime - 10
    os.utime(src, (older, older))
    staged2 = stage_checkpoint_to_local(str(src))
    assert open(staged2, "rb").read() == b"A-MUCH-LONGER-PAYLOAD", "size change must re-stage"


def test_distinct_paths_get_distinct_stages(tmp_path, monkeypatch):
    fakeroot, srcdir = _setup(tmp_path, monkeypatch)
    # Two runs both named checkpoint_last.pt → keyed by ABSOLUTE path → no collision.
    a = srcdir / "runA" / "checkpoint_last.pt"
    b = srcdir / "runB" / "checkpoint_last.pt"
    a.parent.mkdir(); b.parent.mkdir()
    a.write_bytes(b"MODEL-A"); b.write_bytes(b"MODEL-B")
    sa = stage_checkpoint_to_local(str(a))
    sb = stage_checkpoint_to_local(str(b))
    assert sa != sb
    assert open(sa, "rb").read() == b"MODEL-A"
    assert open(sb, "rb").read() == b"MODEL-B"


def test_already_in_tmp_returned_unchanged(tmp_path, monkeypatch):
    fakeroot, _ = _setup(tmp_path, monkeypatch)
    # A source already under the staging root is not worth staging → returned as-is.
    src = fakeroot / "already_local.pt"
    src.write_bytes(b"x")
    assert stage_checkpoint_to_local(str(src)) == str(src)


def test_missing_source_falls_back():
    # Nonexistent / remote path → return as-is so the caller's open() reports it.
    assert stage_checkpoint_to_local("/no/such/file.pt") == "/no/such/file.pt"
