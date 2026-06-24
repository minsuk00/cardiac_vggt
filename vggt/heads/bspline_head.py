# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""B-spline control-grid warp head — a smooth-by-construction alternative to DPTHead.

Instead of decoding a free per-pixel displacement from the ViT patch tokens (which leaves
14-px patch-boundary seams in Δ → folds the splat → lattice artifact), this head predicts a
COARSE control grid of 3-vectors (one per `grid_size`-spaced node) and smoothly upsamples it
to the dense per-pixel Δ. A bicubic interpolation of a coarse grid is C¹-continuous and has
no energy at the patch period, so the warp is smooth by construction.

Drop-in for DPTHead: identical `forward(aggregated_tokens_list, images, patch_start_idx)`
signature and identical output shapes `(B, S, H, W, 3)`, `(B, S, H, W)`, so VGGT.forward,
the residual-DVF logic, the splat, and the loss are all unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.heads.head_act import activate_head


class BSplineWarpHead(nn.Module):
    def __init__(
        self,
        dim_in,
        patch_size=14,
        grid_size=32,
        output_dim=4,                 # 3 (Δx,Δy,Δz) + 1 (conf), to reuse activate_head
        activation="linear",
        conf_activation="expp1",
        hidden=256,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.activation = activation
        self.conf_activation = conf_activation

        self.norm = nn.LayerNorm(dim_in)
        # Small conv stack on the 37x37 token grid: cheap channel reduction + one 3x3 for
        # spatial mixing between neighboring tokens, then project to control values.
        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.to_ctrl = nn.Conv2d(hidden, output_dim, kernel_size=1)

    def forward(self, aggregated_tokens_list, images, patch_start_idx, frames_chunk_size=8):
        # frames_chunk_size is accepted for DPTHead interface parity; this head is cheap
        # (coarse grid) so no frame chunking is needed.
        B, S = images.shape[0], images.shape[1]
        H, W = images.shape[-2], images.shape[-1]
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Last-layer patch tokens → (B*S, C, patch_h, patch_w), same extraction as DPTHead.
        x = aggregated_tokens_list[-1][:, :, patch_start_idx:]      # (B, S, N_patch, dim_in)
        x = x.reshape(B * S, -1, x.shape[-1])                       # (B*S, N_patch, dim_in)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B * S, x.shape[-1], patch_h, patch_w)

        feat = self.proj(x)                                        # (B*S, hidden, patch_h, patch_w)
        feat = F.adaptive_avg_pool2d(feat, (self.grid_size, self.grid_size))
        ctrl = self.to_ctrl(feat)                                  # (B*S, output_dim, grid, grid)

        # B-spline / bicubic upsample of the control grid → dense per-pixel field.
        out = F.interpolate(ctrl, size=(H, W), mode="bicubic", align_corners=True)

        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)
        preds = preds.view(B, S, *preds.shape[1:])                 # (B, S, H, W, 3)
        conf = conf.view(B, S, *conf.shape[1:])                    # (B, S, H, W)
        return preds, conf
