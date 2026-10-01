"""The train/val DataLoader: a plain torch DataLoader over an MRIDataset.

Batch size is pinned to 1: every subject has its own D and dz,
and two subjects with the same D but different pitch collate SILENTLY into one batch that
the loss/aug then splat at a single `z_scale`. To trade memory, change D or the model.

Single-GPU only. The order reproduces what `DistributedSampler(num_replicas=1, rank=0)`
produced (and what every past run trained on): a `torch.randperm` from a private generator
seeded `seed + epoch` when shuffling, else `range(N)`. No global RNG is touched.
"""

import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader


def epoch_order(n, *, seed, epoch, shuffle):
    """Subject order for one epoch: each index exactly once."""
    if not shuffle:
        return list(range(n))
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    return torch.randperm(n, generator=g).tolist()


def _seed_worker(worker_id, *, seed, epoch):
    """Distinct, reproducible seed per worker. The `+ 1` is the folded-in world_size term of
    the upstream DDP formula; dropping it would reseed every worker and change the data stream."""
    worker_seed = worker_id + seed + 1 + epoch * 12345
    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset, *, seed, epoch, shuffle, num_workers):
    """One epoch's DataLoader. Build (and iterate) it once per epoch: `iter()` draws the
    workers' base seed from the global torch RNG."""
    return DataLoader(
        dataset,
        batch_size=1,
        sampler=epoch_order(len(dataset), seed=seed, epoch=epoch, shuffle=shuffle),
        num_workers=num_workers,
        worker_init_fn=partial(_seed_worker, seed=seed, epoch=epoch),
    )
