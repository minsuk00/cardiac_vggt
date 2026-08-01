# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic per-worker seeding for the dataloader."""

import torch
import random
import numpy as np

from functools import partial


def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    """Give each dataloader worker a distinct, reproducible seed.

    Single-GPU only, so the rank/world_size terms of the original (upstream VGGT,
    DDP-era) formula are constant and folded in: rank=0 contributed 0, world_size=1
    contributed +1, and `RANK` is unset under a 1-process launch. The `+ 1` below IS
    that world_size term — dropping it would silently reseed every worker and change
    the training data stream, so it is kept to stay bit-identical with prior runs.

    Args:
        worker_id (int): ID of the dataloader worker.
        num_workers (int): Total number of dataloader workers (unused; kept so the
            `partial(...)` call sites and any external callers keep working).
        epoch (int): Current training epoch.
        seed (int, optional): Base seed for randomization. Defaults to 0.
    """
    worker_seed = worker_id + seed + 1 + epoch * 12345

    print(f"Worker seed: {worker_seed}")

    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def get_worker_init_fn(seed, num_workers, epoch):
    """
    Get a worker initialization function for dataloaders.

    Args:
        seed (int): Base seed for randomization.
        num_workers (int): Number of dataloader workers.
        epoch (int): Current training epoch.

    Returns:
        callable: A worker initialization function to use with DataLoader.
    """
    return partial(
        default_worker_init_fn,
        num_workers=num_workers,
        epoch=epoch,
        seed=seed,
    )
