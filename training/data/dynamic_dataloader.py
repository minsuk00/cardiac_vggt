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
        worker_init_fn: Optional[Callable] = None,
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
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed

        # Instantiate the dataset
        self.dataset = instantiate(dataset, common_config=common_config, _recursive_=False)

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums    # e.g., [2, 24]

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers. Single-GPU only: pass num_replicas=1, rank=0 explicitly so the
        # sampler does NOT require an initialized torch process group (it otherwise falls back
        # to dist.get_world_size()/get_rank()). This yields the identical unsharded permutation
        # that the former 1-process DDP group produced — torch.randperm(N, seed+epoch).
        self.sampler = DynamicDistributedSampler(self.dataset, num_replicas=1, rank=0, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
        )

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
                worker_init_fn=self.worker_init_fn,
            ),
        )
        

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """
    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 ):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1]+1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

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
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

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
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
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
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
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
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num
