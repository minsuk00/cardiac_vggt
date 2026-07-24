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
import logging
import os
import shutil
import tempfile
import time


def stage_checkpoint_to_local(ckpt_path: str) -> str:
    """Return a node-local (/tmp) path to load `ckpt_path` from, staging it if needed.

    The staged copy is keyed by the source's ABSOLUTE PATH (so two runs that both name
    their file `checkpoint_last.pt` never collide) and validated against the source's
    (size, mtime): a cached copy is reused only if it exists, matches the source size,
    AND is at least as new as the source. Consequences:
      - an immutable ckpt (base weights, a finished run) → copied once, then every later
        load on that node is an instant cache hit;
      - a ckpt overwritten in place (a live `checkpoint_last.pt`; each atomic save gives
        it a strictly newer mtime) → stat mismatch → re-staged, never silently stale.
    Exactly one staged copy is kept per source path (overwritten atomically in place), so
    /tmp holds one copy per distinct checkpoint — no unbounded growth.

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

        src_stat = os.stat(src)
        stage_dir = os.path.join(
            tempfile.gettempdir(), f"vggt-ckpt-stage_{getpass.getuser()}"
        )
        os.makedirs(stage_dir, exist_ok=True)
        staged = os.path.join(stage_dir, hashlib.sha1(src.encode()).hexdigest() + ".pt")

        if (
            os.path.isfile(staged)
            and os.path.getsize(staged) == src_stat.st_size
            and os.stat(staged).st_mtime >= src_stat.st_mtime
        ):
            logging.info(f"Using node-local checkpoint stage {staged} (source {src})")
            return staged

        logging.info(f"Staging checkpoint {src} → {staged} (copy to node-local /tmp)")
        t0 = time.time()
        tmp = staged + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, staged)  # atomic; an interrupted copy leaves no usable staged file
        tmp = None
        logging.info(f"Staged checkpoint in {time.time() - t0:.1f}s")
        return staged
    except Exception as e:
        # Any real failure (unreadable source, /tmp full, permission error) → fall back to
        # the original path so the load still runs. Catches Exception (not BaseException) so
        # a Ctrl-C mid-copy still propagates and lets the user abort.
        logging.warning(f"Checkpoint staging failed ({e}); loading directly from {ckpt_path}")
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return ckpt_path
