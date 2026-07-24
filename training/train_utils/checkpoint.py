# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import getpass
import hashlib
import logging
import shutil
import tempfile
import time
from typing import (
    Any,
    Dict,
    List,
)

import torch
import torch.nn as nn
import os
from iopath.common.file_io import g_pathmgr
from wcmatch import fnmatch





# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)




def stage_checkpoint_to_local(ckpt_path: str) -> str:
    """Stage an IMMUTABLE checkpoint onto node-local /tmp before loading, then return
    the path to load from.

    `torch.load` straight off GPFS is pathologically slow because it reads the file
    storage-by-storage (many small, seeky reads) — measured ~266 s for an ~8 GB ckpt
    vs ~5 s from /tmp; a *sequential* copy read is fine. So we copy once to /tmp keyed
    by the source's absolute path and reuse it thereafter — the win is loading the same
    ckpt more than once per node (e.g. repeated smoke runs sharing the base weights).

    IMPORTANT — there is NO staleness check by design. Only pass a checkpoint that is
    immutable for the life of the /tmp cache: the base/seed weights via
    `resume_checkpoint_path` (`vggt1b_base.pt` never changes). Do NOT use this for a
    file overwritten in place (e.g. `checkpoint_last.pt` across SLURM requeues) — it
    would reuse the stale cached copy. If you ever regenerate a staged source in place,
    clear `/tmp/vggt-ckpt-stage_<user>/`.

    Staging is a pure performance optimization: on any failure, or when the source is
    not a local file / already under /tmp, it returns the original path so a load can
    never be blocked by staging.
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

        if os.path.isfile(staged):
            logging.info(f"Using node-local checkpoint stage {staged} (source {src})")
            return staged

        logging.info(f"Staging checkpoint {src} → {staged} (one-time copy to node-local /tmp)")
        t0 = time.time()
        tmp = staged + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, staged)  # atomic; an interrupted copy leaves no usable staged file
        tmp = None
        logging.info(f"Staged checkpoint in {time.time() - t0:.1f}s")
        return staged
    except BaseException as e:
        logging.warning(f"Checkpoint staging failed ({e}); loading directly from {ckpt_path}")
        if tmp is not None:
            try:
                os.remove(tmp)
            except OSError:
                pass
        return ckpt_path


class CheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        for ckpt_name in self.checkpoint_names:
            checkpoint_path = os.path.join(
                self.checkpoint_folder, f"{ckpt_name}.pt"
            )
            logging.info(
                f"Saving checkpoint at epoch {self.epoch} to {checkpoint_path}"
            )
            robust_torch_save(checkpoint, checkpoint_path)



def robust_torch_save(checkpoint: Dict[str, Any], checkpoint_path: str) -> None:
    """
    A more robust version of torch.save that works better with preemptions
    and corruptions if a job is preempted during save.

    Writes to a temp file then atomically renames it into place (os.replace is
    atomic on POSIX, including a same-directory rename on GPFS). So an interrupted
    save — e.g. a SLURM auto-requeue exiting mid-write — leaves the previous
    checkpoint fully intact rather than a truncated, unloadable file. The atomic
    rename supersedes the older move-to-.bak scheme (which left a window where the
    live file was absent or partial).

    If the write fails (e.g. OSError errno 122, disk quota exceeded) the partial tmp
    is removed before re-raising: leaving it behind permanently consumes the very
    quota that just ran out, making the next save more likely to fail too.
    """
    tmp_checkpoint_path = checkpoint_path + ".tmp"
    try:
        with g_pathmgr.open(tmp_checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)
        os.replace(tmp_checkpoint_path, checkpoint_path)
    except BaseException:
        try:
            os.remove(tmp_checkpoint_path)
        except OSError:
            pass
        raise