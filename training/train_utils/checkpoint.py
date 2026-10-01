# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

import torch


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
        with open(tmp_checkpoint_path, "wb") as f:
            torch.save(checkpoint, f)
        os.replace(tmp_checkpoint_path, checkpoint_path)
    except BaseException:
        try:
            os.remove(tmp_checkpoint_path)
        except OSError:
            pass
        raise
