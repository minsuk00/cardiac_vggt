# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import random
from abc import ABC

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset


class ComposedDataset(Dataset, ABC):
    """
    Composes base datasets and converts their raw output to batched tensors.

    Wraps the base dataset(s) and handles the numpy->tensor conversion for every
    batch key. (The original VGGT photometric augmentation was removed: it was
    disabled on the MRI pipeline and only ever operated on natural RGB photos.)
    """

    def __init__(self, dataset_configs: dict, common_config: dict, **kwargs):
        """
        Initializes the ComposedDataset.

        Args:
            dataset_configs (dict): List of Hydra configurations for base datasets.
            common_config (dict): Shared configurations (augs, tracks, mode, etc.).
            **kwargs: Additional arguments (unused).
        """
        base_dataset_list = []

        # Instantiate each base dataset with common configuration
        for baseset_dict in dataset_configs:
            baseset = instantiate(baseset_dict, common_conf=common_config)
            base_dataset_list.append(baseset)

        # Use custom concatenation class that supports tuple indexing
        self.base_dataset = TupleConcatDataset(base_dataset_list, common_config)

        # --- Optional Fixed Settings (useful for debugging) ---
        # Force each sequence to have exactly this many images (if > 0)
        self.fixed_num_images = common_config.fix_img_num
        # Force a specific aspect ratio for all images
        self.fixed_aspect_ratio = common_config.fix_aspect_ratio

        self.common_config = common_config

        self.total_samples = len(self.base_dataset)

    def __len__(self):
        """Returns the total number of sequences in the dataset."""
        return self.total_samples

    def __getitem__(self, idx_tuple):
        """
        Retrieves a data sample (sequence) from the dataset.

        Loads raw data, converts to PyTorch tensors, applies augmentations,
        and prepares tracks if enabled.

        Args:
            idx_tuple (tuple): a tuple of (seq_idx, num_images, aspect_ratio)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # If fixed settings are provided, override the tuple values
        if self.fixed_num_images > 0:
            seq_idx = idx_tuple[0] if isinstance(idx_tuple, tuple) else idx_tuple
            idx_tuple = (seq_idx, self.fixed_num_images, self.fixed_aspect_ratio)

        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        # --- Data Conversion and Preparation ---
        # Convert numpy arrays to tensors
        images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
        # Normalize images from [0, 255] to [0, 1]
        images = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        scanner_coords = torch.from_numpy(np.stack(batch["scanner_coords"]).astype(np.float32)) if "scanner_coords" in batch else None
        ids = torch.from_numpy(batch["ids"])  # Frame indices sampled from the original sequence

        # --- Prepare Final Sample Dictionary ---
        sample = {
            "seq_name": seq_name,
            "ids": ids,
            "images": images,
        }
        if scanner_coords is not None:
            sample["scanner_coords"] = scanner_coords

        if "z_indices" in batch:
            sample["z_indices"] = torch.from_numpy(np.stack(batch["z_indices"]).astype(np.float32))
        if "t_indices" in batch:
            sample["t_indices"] = torch.from_numpy(np.stack(batch["t_indices"]).astype(np.float32))
        if "target_t_indices" in batch:
            sample["target_t_indices"] = torch.from_numpy(np.stack(batch["target_t_indices"]).astype(np.float32))
        if "timesteps" in batch:
            sample["timesteps"] = torch.from_numpy(np.stack(batch["timesteps"]).astype(np.int64))
        if "slice_indices" in batch:
            # float32 (not int64): z may be CONTINUOUS (continuous_z). Re-extraction paths
            # (respiratory grid_sample, gpu_aug 2-plane blend) interpolate; integer-valued z
            # is exact, so the discrete-grid pipeline is numerically unchanged. timesteps stays
            # int64 — cardiac phase is always discrete.
            sample["slice_indices"] = torch.from_numpy(np.stack(batch["slice_indices"]).astype(np.float32))
        if "rotations" in batch:
            sample["rotations"] = torch.from_numpy(np.stack(batch["rotations"]).astype(np.float32))
        if "gt_target_volume" in batch:
            sample["gt_target_volume"] = torch.from_numpy(batch["gt_target_volume"].astype(np.float32))
        if "t_target" in batch:
            sample["t_target"] = torch.from_numpy(batch["t_target"].astype(np.int64))
        if "seq_index" in batch:
            sample["seq_index"] = torch.from_numpy(batch["seq_index"].astype(np.int64))
        if "dz_mm" in batch:
            sample["dz_mm"] = torch.from_numpy(batch["dz_mm"].astype(np.float32))
        if "z_scale" in batch:
            sample["z_scale"] = torch.from_numpy(batch["z_scale"].astype(np.float32))
        if "anatomy_bbox" in batch:
            sample["anatomy_bbox"] = torch.from_numpy(batch["anatomy_bbox"].astype(np.int64))
        if "content_mask" in batch:
            sample["content_mask"] = torch.from_numpy(batch["content_mask"].astype(np.uint8))
        if "heart_roi_canonical" in batch:
            sample["heart_roi_canonical"] = torch.from_numpy(batch["heart_roi_canonical"].astype(np.uint8))
        if "phases" in batch:
            # Full (T, D, H, W) canonical bundle in float16. Used by GPU aug
            # (Phase 4); inert under aug-off. Kept as float16 to keep batch
            # transfer cheap (~18 MB per sample at T=12, D=12, H=W=256).
            sample["phases"] = torch.from_numpy(np.asarray(batch["phases"]))

        return sample


class TupleConcatDataset(ConcatDataset):
    """
    A custom ConcatDataset that supports indexing with a tuple.

    Standard PyTorch ConcatDataset only accepts an integer index. This class extends
    that functionality to allow passing a tuple like (sample_idx, num_images, aspect_ratio),
    where the first element is used to determine which sample to fetch, and the full
    tuple is passed down to the selected dataset's __getitem__ method.

    It also supports an option to randomly sample across all datasets, ignoring the
    provided index. This is useful during training when shuffling the entire dataset
    might cause memory issues due to duplicating dictionaries. If doing this, you can
    set pytorch's dataloader shuffle to False.
    """

    def __init__(self, datasets, common_config):
        """
        Initialize the TupleConcatDataset.

        Args:
            datasets (iterable): An iterable of PyTorch Dataset objects to concatenate.
            common_config (dict): Common configuration dict, used to check for random sampling.
        """
        super().__init__(datasets)
        # If True, ignores the input index and samples randomly across all datasets
        # This provides an alternative to dataloader shuffling for large datasets
        self.inside_random = common_config.inside_random

    def __getitem__(self, idx):
        """
        Retrieves an item using either an integer index or a tuple index.

        Args:
            idx (int or tuple): The index. If tuple, the first element is the sequence
                               index across the concatenated datasets, and the rest are
                               passed down. If int, it's treated as the sequence index.

        Returns:
            The item returned by the underlying dataset's __getitem__ method.

        Raises:
            ValueError: If the index is out of range or the tuple doesn't have exactly 3 elements.
        """
        idx_tuple = None
        if isinstance(idx, tuple):
            idx_tuple = idx
            idx = idx_tuple[0]  # Extract the sequence index

        # Override index with random value if inside_random is enabled
        if self.inside_random:
            total_len = self.cumulative_sizes[-1]
            idx = random.randint(0, total_len - 1)

        # Handle negative indices
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        # Find which dataset the index belongs to
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        # Create the tuple to pass to the underlying dataset
        if len(idx_tuple) == 3:
            idx_tuple = (sample_idx,) + idx_tuple[1:]
        else:
            raise ValueError("Tuple index must have exactly three elements")

        # Pass the modified tuple to the appropriate dataset
        return self.datasets[dataset_idx][idx_tuple]
