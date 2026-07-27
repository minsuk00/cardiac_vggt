"""Staleness guarantees for `stage_checkpoint_to_local` (see docs/50).

The staged copy must never be served after the source's content changes. The original
`staged.mtime >= src.mtime` test could not enforce that: `shutil.copyfile` does not
preserve mtime, so the copy is always newer than its source, making the test vacuously
true. Two realistic replacements then served stale weights silently, and both are covered
here: an mtime-preserving overwrite (`cp -p`/`rsync -a`/`tar -x`) at identical byte size,
and a source rewritten while the copy is in flight.
"""

import os
import shutil

import pytest

from vggt.utils.checkpoint_stage import stage_checkpoint_to_local


@pytest.fixture
def stage_env(tmp_path, monkeypatch):
    """Point staging at an isolated tmpdir so the real /tmp cache is untouched."""
    fake_tmp = tmp_path / "tmpdir"
    fake_tmp.mkdir()
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(fake_tmp))
    src = tmp_path / "src" / "checkpoint_last.pt"
    src.parent.mkdir()
    return src


def _write(path, payload, mtime_ns=None):
    path.write_bytes(payload)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))


def test_stages_then_hits_cache(stage_env):
    src = stage_env
    _write(src, b"A" * 4096)
    p1 = stage_checkpoint_to_local(str(src))
    assert p1 != str(src), "should have staged to the local dir"
    assert open(p1, "rb").read() == b"A" * 4096
    p2 = stage_checkpoint_to_local(str(src))
    assert p2 == p1, "second call must reuse the staged copy"


def test_mtime_preserving_replacement_is_not_served_stale(stage_env):
    """`cp -p` a DIFFERENT checkpoint of the SAME size over the source.

    This is the case the old `staged.mtime >= src.mtime` test got wrong: same size, and
    the source's mtime is preserved (so still older than the staged copy) => cache "hit"
    on stale bytes.
    """
    src = stage_env
    old_mtime = 1_600_000_000_000_000_000
    _write(src, b"A" * 4096, mtime_ns=old_mtime)
    staged = stage_checkpoint_to_local(str(src))
    assert open(staged, "rb").read() == b"A" * 4096

    # Replace content, same byte count, preserving the original (older) mtime.
    _write(src, b"B" * 4096, mtime_ns=old_mtime - 10_000_000_000)

    served = stage_checkpoint_to_local(str(src))
    assert open(served, "rb").read() == b"B" * 4096, (
        "served the PREVIOUS checkpoint after an mtime-preserving, same-size replacement")


def test_source_replaced_during_copy_is_not_published(stage_env, monkeypatch):
    """Trainer's atomic save lands mid-copy: the staged bytes may be the old epoch.

    Rather than publish them (stamped with a newer mtime, so every later load would reuse
    them), staging must fall back to the source path.
    """
    src = stage_env
    _write(src, b"OLD" * 2048)
    real_copyfile = shutil.copyfile

    def copy_then_replace_source(a, b, **kw):
        out = real_copyfile(a, b, **kw)
        # Simulate the trainer's atomic save landing right now: write a sibling temp and
        # os.replace it over the source, exactly as train_utils/checkpoint.py does. That
        # allocates a NEW inode, which is what makes the swap detectable even when the new
        # checkpoint has the same byte size and lands inside one filesystem timestamp tick.
        newer = src.parent / "checkpoint_last.pt.tmp"
        newer.write_bytes(b"NEW" * 2048)
        os.replace(newer, src)
        return out

    monkeypatch.setattr(shutil, "copyfile", copy_then_replace_source)
    served = stage_checkpoint_to_local(str(src))
    assert served == str(src), "must fall back to the source when it changed mid-copy"
    assert open(served, "rb").read() == b"NEW" * 2048

    monkeypatch.setattr(shutil, "copyfile", real_copyfile)
    # And the next call stages the new content cleanly rather than reusing a torn copy.
    served2 = stage_checkpoint_to_local(str(src))
    assert open(served2, "rb").read() == b"NEW" * 2048


def test_temp_name_is_process_unique(stage_env, monkeypatch):
    """A fixed `.tmp` name lets a concurrent stager truncate our in-flight copy."""
    src = stage_env
    _write(src, b"A" * 1024)
    seen = {}

    real_copyfile = shutil.copyfile

    def record(a, b, **kw):
        seen["tmp"] = b
        return real_copyfile(a, b, **kw)

    monkeypatch.setattr(shutil, "copyfile", record)
    stage_checkpoint_to_local(str(src))
    assert str(os.getpid()) in os.path.basename(seen["tmp"]), (
        f"temp name {seen['tmp']!r} is not process-unique")


def test_failure_falls_back_to_source(stage_env, monkeypatch):
    src = stage_env
    _write(src, b"A" * 1024)

    def boom(*a, **kw):
        raise OSError("no space left on device")

    monkeypatch.setattr(shutil, "copyfile", boom)
    assert stage_checkpoint_to_local(str(src)) == str(src)


def test_source_already_local_is_passed_through(stage_env):
    """A source already under the tmp dir needs no staging."""
    import tempfile
    inside = os.path.join(tempfile.gettempdir(), "already_here.pt")
    with open(inside, "wb") as f:
        f.write(b"A" * 64)
    assert stage_checkpoint_to_local(inside) == inside
