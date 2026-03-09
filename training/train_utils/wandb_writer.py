# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from .distributed import get_machine_local_and_dist_rank


class WandbLogger:
    """A wrapper around Weights & Biases with distributed training support."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self._rank = 0
        _, self._rank = get_machine_local_and_dist_rank()

        if self._rank == 0 and wandb is not None:
            # Using print AND logging.info to ensure visibility everywhere
            msg = f"Initializing WandB: project={project}, name={name}"
            print(msg)
            logging.info(msg)
            run = wandb.init(project=project, name=name, config=wandb_config, dir=dir, **kwargs)
            if run is not None:
                logging.info(f"WandB Run URL: {run.get_url()}")
                # Exclude large directories from code logging

                run.log_code(".", exclude_fn=lambda path: "scratch" in path or "logs" in path or ".git" in path)
        elif wandb is None and self._rank == 0:
            logging.warning("WandB is not installed. Skipping initialization.")

    def log(self, name: str, data: Any, step: int) -> None:
        if self._rank == 0 and wandb is not None and wandb.run is not None:
            wandb.log({name: data}, step=step)

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        if self._rank == 0 and wandb is not None and wandb.run is not None:
            wandb.log(payload, step=step)

    def log_visuals(self, name: str, data: Union[torch.Tensor, np.ndarray], step: int, fps: int = 4) -> None:
        if self._rank == 0 and wandb is not None and wandb.run is not None:
            import matplotlib.pyplot as plt

            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()

            if data.ndim == 3:  # Image: (C, H, W)
                if data.shape[0] == 3:
                    data = data.transpose(1, 2, 0)

                # Create a figure to enforce consistent scaling via vmin/vmax
                h, w = data.shape[:2]
                fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])
                # We use vmin=0, vmax=1 for normalized floats, or 0-255 for ints.
                # Since the trainer/dataset might vary, we detect the range.
                v_max = 1.0 if data.max() <= 1.1 else 255.0
                ax.imshow(data, vmin=0, vmax=v_max)
                ax.axis("off")

                wandb.log({name: wandb.Image(fig)}, step=step)
                plt.close(fig)
            elif data.ndim == 5:  # Video: (B, T, C, H, W)
                wandb.log({name: wandb.Image(data)}, step=step)
            else:
                logging.warning(f"Unsupported visual dimensions for WandB: {data.ndim}")

    def log_3d_point_cloud(self, name: str, cloud_data: np.ndarray, step: int) -> None:
        """Logs an interactive 3D point cloud to WandB.
        cloud_data: (N, 3) or (N, 6) array. If (N, 6), format is [x, y, z, r, g, b].
        """
        if self._rank == 0 and wandb is not None and wandb.run is not None:
            wandb.log({name: wandb.Object3D.from_numpy(cloud_data)}, step=step)

    def close(self) -> None:
        if self._rank == 0 and wandb is not None and wandb.run is not None:
            wandb.finish()
