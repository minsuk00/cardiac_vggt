"""data/loader.py reproduces the subject order of the old DistributedSampler-based stack."""
import torch
from torch.utils.data import DistributedSampler

from training.data.loader import epoch_order


def test_epoch_order_matches_single_replica_distributed_sampler():
    ds = list(range(37))
    for shuffle in (True, False):
        for epoch in (0, 1, 5):
            ref = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=shuffle, seed=42)
            ref.set_epoch(epoch)
            assert epoch_order(len(ds), seed=42, epoch=epoch, shuffle=shuffle) == list(ref)


def test_epoch_order_leaves_global_rng_alone():
    state = torch.random.get_rng_state()
    epoch_order(10, seed=1, epoch=2, shuffle=True)
    assert torch.equal(state, torch.random.get_rng_state())
