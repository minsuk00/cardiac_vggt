# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Any, Dict, Optional

try:
    import wandb
except ImportError:
    wandb = None

class WandbLogger:
    """A thin wrapper around Weights & Biases."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        dir: Optional[str] = None,
        resume_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # The live wandb run. Read back by the trainer (`run_meta.jsonl`'s wandb_id/url) —
        # keep it an attribute, not a local, or that link back to the dashboard is silently null.
        self.run = None
        if wandb is not None:
            # Using print AND logging.info to ensure visibility everywhere
            msg = f"Initializing WandB: project={project}, name={name}"
            print(msg)
            logging.info(msg)
            if resume_id is not None:
                kwargs["id"] = resume_id
                kwargs["resume"] = "allow"
            if "tags" in kwargs and kwargs["tags"] is not None:
                tags = kwargs["tags"]
                if isinstance(tags, str):
                    tags = [tags]
                kwargs["tags"] = [str(t) for t in tags]
            # Hydra's instantiate re-wraps a passed-in plain dict as a DictConfig, which
            # wandb's config sanitizer chokes on (asdict→defaultdict TypeError). Convert
            # any OmegaConf config back to a plain container before handing it to wandb.
            if wandb_config is not None:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(wandb_config):
                    # resolve=True can raise (e.g. an interpolation pointing at a MISSING value);
                    # config logging must never crash training startup, so fall back to the
                    # unresolved container. No current config triggers this — insurance only.
                    try:
                        wandb_config = OmegaConf.to_container(wandb_config, resolve=True)
                    except Exception as e:
                        logging.warning(f"wandb config resolve failed, logging unresolved (ignored): {e}")
                        wandb_config = OmegaConf.to_container(wandb_config, resolve=False)
            run = self.run = wandb.init(project=project, name=name, config=wandb_config, dir=dir, **kwargs)
            if run is not None:
                logging.info(f"WandB Run URL: {run.get_url()}")
                # Exclude large directories and specifically include .py and .yaml files
                run.log_code(".", 
                    include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
                    exclude_fn=lambda path: "scratch" in path or "logs" in path or ".git" in path
                )
        else:
            logging.warning("WandB is not installed. Skipping initialization.")

    def log(self, name: str, data: Any, step: int) -> None:
        if wandb is not None and wandb.run is not None:
            wandb.log({name: data}, step=step)

    def close(self) -> None:
        if wandb is not None and wandb.run is not None:
            wandb.finish()
