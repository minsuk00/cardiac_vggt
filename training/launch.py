# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import signal
import subprocess
import time

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


def _install_requeue_handler():
    """SLURM auto-requeue: catch SIGUSR1 (sent ~lead seconds before walltime by
    `#SBATCH --signal=B:USR1@<lead>` and forwarded to this worker by the sbatch
    trap), requeue the same job id, and exit cleanly. The job restarts with
    SLURM_RESTART_COUNT incremented and resumes from the per-epoch
    checkpoint_last.pt (the sbatch script pins a stable exp dir so it's found).
    No-op outside SLURM so local runs are unaffected."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_id is None:
        return

    def _handler(signum, frame):
        # Array-aware target; only rank 0 issues the requeue, every rank exits.
        array_job = os.environ.get("SLURM_ARRAY_JOB_ID")
        if array_job:
            target = f"{array_job}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
        else:
            target = job_id
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[requeue] SIGUSR1 caught — requeuing job {target}", flush=True)
            try:
                subprocess.run(["scontrol", "requeue", target], check=False)
            except Exception as e:  # noqa: BLE001
                print(f"[requeue] scontrol requeue failed: {e}", flush=True)
        os._exit(0)

    signal.signal(signal.SIGUSR1, _handler)

# Custom resolver for reverse-chronological sorting
# Use a fixed value per-run so all config accesses match
REVERSE_TS = str(2000000000 - int(time.time()))
OmegaConf.register_new_resolver("rev_ts", lambda: REVERSE_TS)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
# Wandb phase-mode tag: "multiphase" when t_target_fixed is null, else "tK".
OmegaConf.register_new_resolver(
    "phase_mode", lambda t: "multiphase" if t is None else f"t{int(t)}"
)


def main():
    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument("--config", type=str, default="default", help="Name of the config file (without .yaml extension, default: default)")
    args, hydra_overrides = parser.parse_known_args()

    _install_requeue_handler()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config, overrides=hydra_overrides)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
