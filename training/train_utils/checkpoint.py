# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
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




class DDPCheckpointSaver:
    def __init__(
        self,
        checkpoint_folder: str,
        checkpoint_names: List[str],
        rank: int,
        epoch: int,
    ):
        super().__init__()
        self.checkpoint_folder = checkpoint_folder
        self.checkpoint_names = checkpoint_names
        self.worker_id = rank
        self.epoch = epoch

    def save_checkpoint(
        self,
        model: nn.Module,
        **kwargs: Any,
    ) -> None:
        checkpoint = dict(**kwargs)
        checkpoint["model"] = model.state_dict()

        if self.worker_id == 0:
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
    """
    tmp_checkpoint_path = checkpoint_path + ".tmp"
    with g_pathmgr.open(tmp_checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    os.replace(tmp_checkpoint_path, checkpoint_path)