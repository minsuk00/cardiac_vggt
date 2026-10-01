# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, img_size=518, patch_size=14, embed_dim=1024, enable_point=True,
        gradient_checkpointing=True, backbone="dinov2_vitl14_reg",
    ):
        super().__init__()
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed=backbone, gradient_checkpointing=gradient_checkpointing)

        # The head predicts a residual DVF; world_points = scanner_coords + DVF.
        if not enable_point:
            self.point_head = None
        else:
            self.point_head = DPTHead(dim_in=2 * embed_dim, patch_size=patch_size, output_dim=4, activation="linear", conf_activation="expp1")

    def forward(self, images: torch.Tensor, batch: dict = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            batch (dict, optional): Batch dictionary with the extra inputs the point head needs —
                z_indices, scanner_coords.

        Returns:
            dict: A dictionary containing the following predictions:
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                  (scanner_coords + predicted DVF).
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W].
                - dvfs (torch.Tensor): The predicted normalized T→0 DVF [B, S, H, W, 3].
                - images (torch.Tensor): Original input images, preserved for visualization. Only when
                  not self.training (i.e. inference).
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        z_indices = batch.get("z_indices") if batch is not None else None
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, z_indices=z_indices)

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            if self.point_head is not None:
                head_output, head_conf = self.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)

                # Head predicted normalized T→0 DVF. world_points = scanner_coords + dvf.
                assert batch is not None and "scanner_coords" in batch, "scanner_coords required for residual DVF but not found in batch."
                scanner_coords = batch["scanner_coords"]  # voxel position at time T, normalized mm
                dvf = head_output  # predicted T→0 DVF, normalized
                assert scanner_coords.shape == dvf.shape, f"scanner_coords {scanner_coords.shape} and dvf {dvf.shape} must share shape and normalization"
                world_points = scanner_coords + dvf
                predictions["dvfs"] = dvf

                predictions["world_points"] = world_points
                predictions["world_points_conf"] = head_conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
