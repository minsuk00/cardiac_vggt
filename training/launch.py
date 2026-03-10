# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer
import time

# Custom resolver for reverse-chronological sorting
# Use a fixed value per-run so all config accesses match
REVERSE_TS = str(2000000000 - int(time.time()))
OmegaConf.register_new_resolver("rev_ts", lambda: REVERSE_TS)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument("--config", type=str, default="default", help="Name of the config file (without .yaml extension, default: default)")
    args, hydra_overrides = parser.parse_known_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=hydra_overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
