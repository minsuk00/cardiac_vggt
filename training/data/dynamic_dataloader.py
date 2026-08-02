# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from hydra.utils import instantiate
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
from abc import ABC, abstractmethod

from .worker_fn import get_worker_init_fn

class DynamicTorchDataset(ABC):
    def __init__(
        self,
        dataset: dict,
        common_config: dict,
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool = True,
        collate_fn: Optional[Callable] = None,
        persistent_workers: bool = False,
        seed: int = 42,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.persistent_workers = persistent_workers
        self.seed = seed        # property below: propagates to the sampler once it exists

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Per-sample slot BUDGET (`img_nums`). Under `one_frame_per_slice` (the default) the
        # dataset sets S = this subject's own in-FOV plane count == D, so this is no longer a
        # sample count — it is the CAP that `mri_dataset` enforces (docs/59 F19), which is why
        # it must still reach `get_data`. The upstream per-batch aspect-ratio draw was removed
        # (2026-08-01): it was fixed at [1.0, 1.0], and the value died unread in
        # `get_data(**kwargs)` — cardiac slices are square 256x256 by construction.
        self.image_num_range = common_config.img_nums    # e.g., [20, 20]
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers. Single-GPU only: pass num_replicas=1, rank=0 explicitly so the
        # sampler does NOT require an initialized torch process group (it otherwise falls back
        # to dist.get_world_size()/get_rank()). This yields the identical unsharded permutation
        # that the former 1-process DDP group produced — torch.randperm(N, seed+epoch).
        self.sampler = DynamicDistributedSampler(self.dataset, num_replicas=1, rank=0, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.image_num_range,
            seed=seed,
        )

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        """Propagate to the sampler, which is constructed BEFORE the trainer assigns the
        real `seed_value` (`trainer._setup_dataloaders`). A plain attribute reached only
        `worker_init_fn`, so the subject permutation stayed on the ctor default 42 no matter
        what `seed_value` said — two "different-seed" runs replayed the same subject order
        (docs/62 §5.5). Guarded with getattr because __init__ assigns this before `sampler`
        exists. Default 42 is unchanged, so existing runs are bit-identical.
        """
        self._seed = value
        if getattr(self, "sampler", None) is not None:
            self.sampler.seed = value

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 ):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
        """
        self.sampler = sampler
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Slot budget for this sample. Uniform over [lo, hi]; at the shipped
                # `img_nums: [20, 20]` this is the constant 20. (Was a weighted np.random
                # draw over a dict of all-1.0 weights — same result, more machinery, and it
                # advanced the global numpy RNG for nothing.)
                random_image_num = self.rng.randint(
                    int(self.image_num_range[0]), int(self.image_num_range[1]))
                self.sampler.update_parameters(image_num=random_image_num)

                # BATCH SIZE IS PINNED TO 1 (docs/59 F9/F19).
                #
                # Upstream VGGT scaled batch size inversely with the per-sample image count
                # (`floor(max_img_per_gpu / random_image_num)`) to hold GPU memory roughly
                # constant. Under native-z that scheme was both INERT and UNSAFE, so it was
                # removed outright (docs/59 F9) — `max_img_per_gpu` is gone from the configs
                # and the signatures:
                #   * INERT — with one_frame_per_slice the dataset overrides S to the subject's
                #     own D, so the memory a sample costs has nothing to do with
                #     `random_image_num` any more; and at the shipped config the formula already
                #     evaluated to floor(20/20) = 1.
                #   * UNSAFE — every subject now has its OWN D and dz. Two subjects with
                #     different D fail loudly in `default_collate` ("stack expects each tensor to
                #     be equal size"), but two with the SAME D and DIFFERENT pitch collate
                #     silently, and the loss/aug read a single scalar `z_scale`/`dz_mm` for the
                #     whole batch — so row 1 would be splatted and breathed at row 0's scale
                #     (verified: a (10, dz=7.5) + (10, dz=18.0) pair collates cleanly). The
                #     uniformity guards in loss.py / gpu_aug.py now catch that, but the only
                #     configuration that is safe by construction is B = 1.
                # To trade memory, change D or the model — not this knob.
                batch_size = 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to attach the per-sample slot budget
    (`image_num`) to each index, so it reaches the dataset's __getitem__.
    (The companion `aspect_ratio` was dropped 2026-08-01 — always 1.0, never read.)
    """
    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches the slot budget.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num)

    def update_parameters(self, image_num):
        """
        Updates the per-sample slot budget.

        Args:
            image_num: The number of images to set.
        """
        self.image_num = image_num
