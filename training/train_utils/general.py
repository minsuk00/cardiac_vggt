# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import os
import math
import random
import numpy as np
from typing import Union, Optional
import logging
from iopath.common.file_io import g_pathmgr
from pathlib import Path
from typing import Dict, Iterable, List



from typing import Any, Mapping




def get_resume_checkpoint(checkpoint_save_dir):
    if not g_pathmgr.isdir(checkpoint_save_dir):
        return None
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint_last.pt")
    if not g_pathmgr.isfile(ckpt_file):
        return None

    return ckpt_file


def resolve_resume_checkpoint(checkpoint_save_dir, seed_checkpoint_path):
    """Decide which checkpoint to load at trainer startup.

    A run's OWN latest checkpoint in `checkpoint_save_dir` (checkpoint_last.pt) always
    wins, so a SLURM auto-requeue or crash-restart resumes mid-run with epoch / steps /
    optimizer / scaler state intact. `seed_checkpoint_path` (the configured
    `resume_checkpoint_path` — e.g. the VGGT-1B base or a CKPT_ONLY seed) is used only on a
    COLD start, when save_dir has no checkpoint yet. Returns the path to load, or None.

    Before this, `resume_checkpoint_path` unconditionally won, so a requeue reloaded the
    base state_dict (which carries no prev_epoch/steps/optimizer keys) and silently
    restarted at epoch 0, discarding all training progress.
    """
    local_ckpt = get_resume_checkpoint(checkpoint_save_dir)
    if local_ckpt is not None:
        return local_ckpt
    return seed_checkpoint_path

class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"



class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def copy_data_to_device(data, device: torch.device, *args: Any, **kwargs: Any):
    """Recursively move a batch to `device`. Non-tensors pass through unchanged.

    Simplified 2026-08-01. The upstream VGGT version also handled named tuples,
    defaultdicts (preserving default_factory), dataclasses (with a second pass for
    non-`init` fields) and a `_CopyableData` Protocol — that generality existed for
    upstream's `FrameData` dataclass batches. Ours is a plain dict of tensors built by
    `MRIDataset.get_data` + `ComposedDataset`, so only the dict/tensor paths ever ran.
    """
    if torch.is_tensor(data):
        return data.to(device, *args, **kwargs)
    if isinstance(data, Mapping):
        return {k: copy_data_to_device(v, device, *args, **kwargs) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    return data



def safe_makedirs(path: str):
    if not path:
        logging.warning("safe_makedirs called with an empty path. No operation performed.")
        return False

    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logging.error(f"Failed to create directory '{path}'. Reason: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred while creating directory '{path}'. Reason: {e}")
        raise



def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"GPU SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU




class AverageMeter:
    """Computes and stores the average and current value.
    Args:
        name (str): Name of the metric being tracked
        device (torch.device, optional): Device for tensor operations. Defaults to None.
        fmt (str): Format string for displaying values. Defaults to ":f"
    """

    def __init__(self, name: str, device: Optional[torch.device] = None, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation showing current and average values."""
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

#################


_UNITS = ('', ' K', ' M', ' B', ' T')          # U+202F = thin-space for nicer look

def pretty_int(n: int) -> str:
    """Abbreviate a non-negative integer (0 → 0, 12_345 → '12.3 K')."""
    assert n >= 0, 'pretty_int() expects a non-negative int'
    if n < 1_000:
        return f'{n:,}'
    exp = int(math.log10(n) // 3)        # group of 3 digits
    exp = min(exp, len(_UNITS) - 1)      # cap at trillions
    value = n / 10 ** (3 * exp)
    return f'{value:.1f}'.rstrip('0').rstrip('.') + _UNITS[exp]


def model_summary(model: torch.nn.Module,
                  *,
                  log_file = None,
                  prefix: str = '',
                  logging_func = None) -> None:
    """
    Print / save a compact parameter summary.

    Args
    ----
    model      : The PyTorch nn.Module to inspect.
    log_file   : Optional path – if given, the full `str(model)` and per-parameter
                 lists are written there (three separate *.txt files).
    prefix     : Optional string printed at the beginning of every log line
                 (handy when several models share the same stdout).
    logging_func: Optional logging function (e.g. logging.info)
    """
    # --- counts -------------------------------------------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable

    summary = []
    summary.append('='*60)
    summary.append(f'Model type : {model.__class__.__name__}')
    summary.append(f'Total      : {pretty_int(total)} parameters')
    summary.append(f'  trainable: {pretty_int(trainable)}')
    summary.append(f'  frozen   : {pretty_int(frozen)}')
    summary.append('='*60)

    for line in summary:
        print(prefix + line)
        if logging_func:
            logging_func(line)

    # --- optional file dump -------------------------------------------------
    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.write_text(str(model))                      # full architecture

    # two extra detailed lists
    def _dump(names: Iterable[str], fname: str):
        """Write a formatted per-parameter list to *log_file.with_name(fname)*."""
        with open(log_file.with_name(fname), 'w') as f:
            for n in names:
                p = dict(model.named_parameters())[n]
                shape = str(tuple(p.shape))
                f.write(f'{n:<60s} {shape:<20} {p.numel()}\n')

    named = dict(model.named_parameters())
    _dump([n for n,p in named.items() if p.requires_grad],  'trainable.txt')
    _dump([n for n,p in named.items() if not p.requires_grad], 'frozen.txt')




