"""Stage a checkpoint onto node-local /tmp before loading, to dodge GPFS's slow reads.

`torch.load` straight off GPFS is pathologically slow because it reads the file
storage-by-storage (many small, seeky reads) — measured ~266 s for an ~8 GB ckpt vs
~5 s from /tmp; a *sequential* copy read of the same file is fast. So we copy once to
/tmp and load from there. The win compounds when the same ckpt is loaded more than once
per node (repeated smoke runs sharing base weights; an eval sweep revisiting a model).

Shared by both training (`trainer.py`) and inference (`inference/inference.py`); lives
under `vggt/utils/` because that is the one package both import (inference never adds
`training/` to its path). Pure filesystem code — no torch dependency.
"""

import getpass
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time


def _src_identity(stat_result) -> dict:
    """The source fingerprint a staged copy is validated against.

    `st_ino` is what makes this robust rather than merely plausible. Every atomic save in
    this repo publishes via `os.replace(tmp, final)` (see `train_utils/checkpoint.py`), and
    `rsync -a`/`tar -x` also write-then-rename — all of which allocate a NEW inode. Size and
    mtime alone can miss such a swap: consecutive checkpoints of one model have identical
    byte size, and if the swap lands within the filesystem's timestamp granularity the mtime
    is unchanged too. The inode changes regardless.

    Residual (accepted) hole: an in-place overwrite that keeps the same inode, the same size
    AND the same mtime_ns is indistinguishable by stat. Detecting that would require hashing
    the file, which costs as much as the copy this module exists to avoid.
    """
    return {
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "ino": stat_result.st_ino,
    }


def stage_checkpoint_to_local(ckpt_path: str) -> str:
    """Return a node-local (/tmp) path to load `ckpt_path` from, staging it if needed.

    The staged copy is keyed by the source's ABSOLUTE PATH (so two runs that both name
    their file `checkpoint_last.pt` never collide) and validated against a SIDECAR that
    records the source's exact `(size, mtime_ns)` as observed when the copy was made. A
    cached copy is reused only if the source still has that exact fingerprint.

    The sidecar is what makes staleness impossible; comparing the staged file's own mtime
    to the source's cannot work, because `shutil.copyfile` does not preserve mtime, so the
    copy is ALWAYS newer than its source. That made a `staged.mtime >= src.mtime` test
    vacuously true, and two realistic situations then served stale weights silently:
      - replacing the source with an mtime-preserving copy (`cp -p`, `rsync -a`, `tar -x`):
        successive checkpoints of one model have identical byte size, so nothing changed
        from the test's point of view — the OLD staged copy was returned forever;
      - staging a live run's `checkpoint_last.pt` while the trainer's atomic `os.replace`
        lands mid-copy: the copy reads the old inode start-to-finish, so the staged bytes
        are the previous epoch, stamped with a newer mtime.
    An exact-fingerprint match fixes the first; re-stat'ing the source after the copy and
    discarding the result if it moved fixes the second.

    Exactly one staged copy (plus sidecar) is kept per source path, so /tmp holds one copy
    per distinct checkpoint — no unbounded growth.

    Staging is a pure performance optimization: on any failure, or when the source is not
    a local file / already under /tmp, it returns the original path so a load can never
    be blocked by staging.
    """
    tmp = None
    try:
        if not os.path.isfile(ckpt_path):
            return ckpt_path  # remote URI / nonexistent → let the caller handle it
        src = os.path.abspath(ckpt_path)
        if src.startswith(tempfile.gettempdir() + os.sep):
            return ckpt_path  # already node-local; nothing to gain

        stage_dir = os.path.join(
            tempfile.gettempdir(), f"vggt-ckpt-stage_{getpass.getuser()}"
        )
        os.makedirs(stage_dir, exist_ok=True)
        staged = os.path.join(stage_dir, hashlib.sha1(src.encode()).hexdigest() + ".pt")
        sidecar = staged + ".json"

        want = _src_identity(os.stat(src))
        if os.path.isfile(staged) and os.path.isfile(sidecar):
            try:
                with open(sidecar) as f:
                    have = json.load(f)
            except (OSError, ValueError):
                have = None
            if have == want and os.path.getsize(staged) == want["size"]:
                logging.info(f"Using node-local checkpoint stage {staged} (source {src})")
                return staged

        logging.info(f"Staging checkpoint {src} → {staged} (copy to node-local /tmp)")
        t0 = time.time()
        # PID-unique temp name: a fixed one lets a second process truncate this copy
        # between our copyfile and our os.replace, publishing a half-written file.
        tmp = f"{staged}.{os.getpid()}.tmp"
        shutil.copyfile(src, tmp)
        after = _src_identity(os.stat(src))
        if after != want:
            # The source was rewritten while we were reading it, so `tmp` may hold the old
            # (or a torn) version. Don't publish it — load from the source instead.
            raise OSError(f"source changed during staging ({want} → {after})")
        os.replace(tmp, staged)  # atomic; an interrupted copy leaves no usable staged file
        tmp = None
        with open(sidecar, "w") as f:  # written only after the copy is published
            json.dump(want, f)
        logging.info(f"Staged checkpoint in {time.time() - t0:.1f}s")
        return staged
    except Exception as e:
        # Any real failure (unreadable source, /tmp full, permission error, source changed
        # mid-copy) → fall back to the original path so the load still runs. Catches
        # Exception (not BaseException) so a Ctrl-C mid-copy still propagates.
        logging.warning(f"Checkpoint staging failed ({e}); loading directly from {ckpt_path}")
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return ckpt_path
