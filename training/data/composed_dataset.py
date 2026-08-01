# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC

import numpy as np
import torch
from hydra.utils import instantiate
from torch.utils.data import Dataset


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

        self.base_dataset = TupleIndexedDataset(base_dataset_list, common_config)

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
            idx_tuple (tuple): a tuple of (seq_idx, num_images)

        Returns:
            dict: A dictionary containing the sequence data (images, poses, tracks, etc.).
        """
        # Retrieve the raw data batch from the appropriate base dataset
        batch = self.base_dataset[idx_tuple]
        seq_name = batch["seq_name"]

        # --- Data Conversion and Preparation ---
        # `images` is absent when the dataset ran with `defer_input_images` — the trainer's
        # gpu_augment_batch builds it on GPU instead (it re-extracts every slice anyway).
        # Pass the absence straight through; that missing key IS the signal.
        sample = {"seq_name": seq_name}
        if "images" in batch:
            images = torch.from_numpy(np.stack(batch["images"]).astype(np.float32)).contiguous()
            # Normalize images from [0, 255] to [0, 1]
            sample["images"] = images.permute(0, 3, 1, 2).to(torch.get_default_dtype()).div(255)

        # Convert other data to tensors with appropriate types
        scanner_coords = torch.from_numpy(np.stack(batch["scanner_coords"]).astype(np.float32)) if "scanner_coords" in batch else None

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


class TupleIndexedDataset:
    """Indexes the base dataset with a `(seq_idx, num_images)` tuple.

    Replaced `TupleConcatDataset(ConcatDataset)` on 2026-08-01. That class carried
    `cumulative_sizes` + a `bisect` lookup to route a global index to one of SEVERAL
    concatenated datasets (upstream VGGT trains on Co3D/ScanNet/MegaDepth together). This
    pipeline configures exactly ONE dataset, so every lookup ran the bisect and returned
    `datasets[0]`. `.datasets` is kept as a list because the trainer, inference and tools
    all reach the MRIDataset via `...base_dataset.datasets[0]`.
    """

    def __init__(self, datasets, common_config):
        datasets = list(datasets)
        if len(datasets) != 1:
            raise ValueError(
                f"Exactly one base dataset is supported, got {len(datasets)}. Multi-dataset "
                "concatenation was removed with TupleConcatDataset; the pooled cohort is a "
                "single MRIDataset driven by one split file."
            )
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        # The sampler's index is HONOURED. `inside_random` was removed 2026-08-01: it
        # discarded the index and drew a subject uniformly at random instead, so a train
        # "epoch" was `limit_train_batches` draws WITH REPLACEMENT — ~37% of subjects unseen
        # per epoch (a different 37% each time), which is not what the config's "one exact
        # pass per epoch" claimed. Training now does the ML default: DistributedSampler's
        # seeded permutation, reshuffled every epoch via set_epoch, each subject exactly once.
        # (Upstream VGGT used it to avoid materialising a shuffled index list for very large
        # multi-dataset training — irrelevant at 935 paths.)
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        idx = idx_tuple[0]

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx

        return self.datasets[0][(idx,) + tuple(idx_tuple[1:])]
