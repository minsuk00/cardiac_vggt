# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from vggt.heads.dpt_head import DPTHead
from vggt.models.aggregator import Aggregator
from vggt.models.backbones import backbone_patch_size


class VGGT(nn.Module):
    def __init__(
        self, img_size=518, patch_size=None, embed_dim=1024,
        gradient_checkpointing=True, backbone="dinov2_vitl14_reg",
    ):
        super().__init__()
        # patch_size is derived from the backbone; an explicit value (old configs pass it) must agree.
        derived = backbone_patch_size(backbone)
        if patch_size is None:
            patch_size = derived
        elif patch_size != derived:
            raise ValueError(f"{backbone} requires patch_size={derived}, got {patch_size}")
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, patch_embed=backbone, gradient_checkpointing=gradient_checkpointing)

        # The head predicts a residual displacement Δ; world_points = scanner_coords + Δ.
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
                - world_points (torch.Tensor): Points in the normalized canonical frame, shape [B, S, H, W, 3]
                  (scanner_coords + dvfs).
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W].
                - dvfs (torch.Tensor): The predicted residual displacement Δ, in the same normalized
                  units as scanner_coords, shape [B, S, H, W, 3].
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        z_indices = batch.get("z_indices") if batch is not None else None
        patch_tokens = batch.get("patch_tokens") if batch is not None else None
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, z_indices=z_indices,
                                                                  patch_tokens=patch_tokens)

        predictions = {}

        with torch.amp.autocast("cuda", enabled=False):
            head_output, head_conf = self.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)

            # world_points = scanner_coords + Δ, both in the same normalized units.
            assert batch is not None and "scanner_coords" in batch, "scanner_coords required for residual DVF but not found in batch."
            scanner_coords = batch["scanner_coords"]  # per-pixel scanner position, normalized
            dvf = head_output  # predicted residual displacement Δ, normalized
            assert scanner_coords.shape == dvf.shape, f"scanner_coords {scanner_coords.shape} and dvf {dvf.shape} must share shape and normalization"
            world_points = scanner_coords + dvf
            predictions["dvfs"] = dvf

            predictions["world_points"] = world_points
            predictions["world_points_conf"] = head_conf

        return predictions
