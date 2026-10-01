# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Tuple

import torch


def make_checkpoint(model, optim, epoch: int, steps: Dict[str, int],
                    time_elapsed: float) -> Dict[str, Any]:
    """The resumable training checkpoint; `prev_epoch` is the epoch just completed."""
    return {
        "prev_epoch": epoch,
        "steps": steps,
        "time_elapsed": time_elapsed,
        "optimizer": optim.optimizer.state_dict(),
        "model": model.state_dict(),
    }


def restore_checkpoint(ckpt: Dict[str, Any], model, optim,
                       strict: bool) -> Tuple[int, Dict[str, int], float]:
    """Load `ckpt` into `model` (and into `optim` unless it is None); return
    `(epoch to run next, steps, time_elapsed)`.

    Accepts a bare state dict (base weights) as well as a full checkpoint. Keys of older
    checkpoints that nothing consumes any more ("scaler", "ema_models") are ignored.
    """
    model_state_dict = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(model_state_dict, strict=strict)
    logging.info(f"Model state loaded. Missing keys count: {len(missing) if missing else 0}. "
                 f"Unexpected keys count: {len(unexpected) if unexpected else 0}.")

    if "optimizer" in ckpt and optim is not None:
        logging.info("Loading optimizer state dict")
        opt_state = ckpt["optimizer"]
        if isinstance(opt_state, list):     # multi-optimizer format; only [0] ever existed
            opt_state = opt_state[0]
        optim.optimizer.load_state_dict(opt_state)

    if "prev_epoch" in ckpt:
        epoch = ckpt["prev_epoch"] + 1
    else:
        epoch = ckpt.get("epoch", 0)
    steps = ckpt["steps"] if "steps" in ckpt else {"train": 0, "val": 0}
    return epoch, steps, ckpt.get("time_elapsed", 0)


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
