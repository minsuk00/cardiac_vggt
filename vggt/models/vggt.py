# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.models.aggregator import Aggregator
from vggt.models.refiner import VolumeRefiner
from vggt.utils.splat import splat_predictions


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self, img_size=518, patch_size=14, embed_dim=1024, enable_camera=True, enable_point=True, enable_depth=True, enable_track=True, use_z_pose_embedding=False, use_t_pose_embedding=False, use_target_t_pose_embedding=False, train_on_residual_dvf=False,
        enable_refiner=False, grid_shape=(12, 256, 256), refiner_base_channels=16, refiner_levels=2, refiner_use_coverage=False
    ):
        super().__init__()
        self.train_on_residual_dvf = train_on_residual_dvf

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, use_z_pose_embedding=use_z_pose_embedding, use_t_pose_embedding=use_t_pose_embedding, use_target_t_pose_embedding=use_target_t_pose_embedding)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None

        point_activation = "linear" if train_on_residual_dvf else "inv_log"
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation=point_activation, conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # Optional 3D UNet refiner on the splatted volume (default OFF → pipeline unchanged).
        # Runs INSIDE forward (so its params are used in the DDP-wrapped forward); the loss
        # then consumes predictions["V_canon"/"V_refined"]. See vggt/models/refiner.py.
        self.enable_refiner = enable_refiner
        self.grid_shape = tuple(grid_shape)
        self.refiner_use_coverage = refiner_use_coverage
        self.refiner = VolumeRefiner(
            in_channels=2 if refiner_use_coverage else 1,
            base_channels=refiner_base_channels, levels=refiner_levels,
            use_coverage=refiner_use_coverage,
        ) if enable_refiner else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None, batch: dict = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            batch (dict, optional): Batch dictionary containing extra inputs like z_indices.

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        z_indices = batch.get("z_indices") if batch is not None else None
        t_indices = batch.get("t_indices") if batch is not None else None
        target_t_indices = batch.get("target_t_indices") if batch is not None else None
        aggregated_tokens_list, patch_start_idx = self.aggregator(images, z_indices=z_indices, t_indices=t_indices, target_t_indices=target_t_indices)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                head_output, head_conf = self.point_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx)

                if self.train_on_residual_dvf:
                    # Head predicted normalized T→0 DVF. world_points = scanner_coords + dvf.
                    assert batch is not None and "scanner_coords" in batch, "scanner_coords required for residual DVF training but not found in batch."
                    scanner_coords = batch["scanner_coords"]  # voxel position at time T, normalized mm
                    dvf = head_output  # predicted T→0 DVF, normalized
                    assert scanner_coords.shape == dvf.shape, f"scanner_coords {scanner_coords.shape} and dvf {dvf.shape} must share shape and normalization"
                    world_points = scanner_coords + dvf
                    predictions["dvfs"] = dvf
                else:
                    world_points = head_output

                predictions["world_points"] = world_points
                predictions["world_points_conf"] = head_conf

                # Optional refiner: splat HERE (so refiner params are used inside the
                # DDP-wrapped forward) and refine. Loss consumes these keys. OFF ⇒ skipped
                # ⇒ no V_canon/V_refined keys ⇒ loss splats as before (bitwise identical).
                if self.refiner is not None:
                    assert batch is not None and "images" in batch, "refiner requires batch['images']"
                    V_canon, coverage = splat_predictions(predictions, batch, self.grid_shape)
                    V_refined = self.refiner(V_canon, coverage if self.refiner_use_coverage else None)
                    predictions["V_canon"] = V_canon
                    predictions["coverage"] = coverage
                    predictions["V_refined"] = V_refined

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points)
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        return predictions
