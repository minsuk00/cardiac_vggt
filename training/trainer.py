# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import contextlib
import gc
import json
import logging
import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import DictConfig, ListConfig, OmegaConf
from train_utils.checkpoint import DDPCheckpointSaver
from train_utils.distributed import get_machine_local_and_dist_rank
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.normalization import normalize_camera_extrinsics_and_points_batch
from train_utils.optimizer import construct_optimizers


class Trainer:
    """
    A generic trainer for DDP training. This should naturally support multi-node training.

    This class orchestrates the entire training and validation process, including:
    - Setting up the distributed environment (DDP).
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to TensorBoard.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (TensorBoard, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
            accum_steps: Number of steps to accumulate gradients before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)

        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        # Initialize the DDP process group
        dist.init_process_group(backend=distributed_conf.backend, timeout=timedelta(minutes=distributed_conf.timeout_mins))
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys count: {len(missing) if missing else 0}. Unexpected keys count: {len(unexpected) if unexpected else 0}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint:
            logging.info(f"Loading optimizer state dict (rank {self.rank})")
            opt_states = checkpoint["optimizer"]
            if not isinstance(opt_states, list):
                opt_states = [opt_states]
            for optim, state in zip(self.optims, opt_states):
                optim.optimizer.load_state_dict(state)

        # Load training progress
        if "prev_epoch" in checkpoint:
            self.epoch = checkpoint["prev_epoch"] + 1
        elif "epoch" in checkpoint:
            self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"] if "steps" in checkpoint else {"train": 0, "val": 0}
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}
        self._point_cloud_logged_epoch = {"train": -1, "val": -1}

        # Instantiate components from configs
        self.tb_writer = instantiate(self.logging_conf.tensorboard_writer, _recursive_=False)
        self.wandb_writer = None
        if hasattr(self.logging_conf, "wandb_writer") and self.logging_conf.wandb_writer is not None:
            self.wandb_writer = instantiate(self.logging_conf.wandb_writer, _recursive_=False)

        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.optim_conf.amp.enabled)

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(f"[Start] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}")
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(f"[Done] Freezing modules: {self.optim_conf.frozen_module_names} on rank {self.distributed_rank}")

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path, logging_func=logging.info)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train and validation datasets and dataloaders."""
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get("val", None), _recursive_=False)
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """
        Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: A list of names for the checkpoint file (e.g., "checkpoint_latest").
                              If None, saves "checkpoint" and "checkpoint_{epoch}" on frequency.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint_last"]
            if self.checkpoint_conf.save_freq > 0 and int(epoch) % self.checkpoint_conf.save_freq == 0 and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "prev_epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }

        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save the checkpoint for DDP only
        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model = self.model.module

        saver.save_checkpoint(
            model=model,
            ema_models=None,
            skip_saving_parameters=[],
            **checkpoint_content,
        )

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log and phase in self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return ["loss_objective"] if phase == "val" else []

    def _log_scalar(self, name: str, value: Any, step: int):
        """Logs a scalar value to both TensorBoard and WandB."""
        if self.tb_writer:
            self.tb_writer.log(name, value, step)
        if self.wandb_writer:
            self.wandb_writer.log(name, value, step)

    def _log_visuals(self, name: str, data: Any, step: int, fps: int = 4, caption: Optional[str] = None):
        """Logs visual data to both TensorBoard and WandB."""
        if self.tb_writer:
            self.tb_writer.log_visuals(name, data, step, fps)
        if self.wandb_writer:
            self.wandb_writer.log_visuals(name, data, step, fps, caption=caption)

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Optionally run a final validation after all training is done
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
            self.train_epoch(dataloader)

            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency
            # Skips validation after the last training epoch, as it can be run separately.
            if self.epoch % self.val_epoch_freq == 0 and self.epoch < self.max_epochs - 1:
                self.run_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self):
        """Runs a full validation epoch if a validation dataset is available."""
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch + self.distributed_rank))
        self.val_epoch(dataloader)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "val"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names_prefixed = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names_prefixed}

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        limit_val_batches = iters_per_epoch if self.limit_val_batches is None else self.limit_val_batches

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            amp_type = self.optim_conf.amp.amp_dtype
            assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
            if amp_type == "bfloat16":
                amp_type = torch.bfloat16
            else:
                amp_type = torch.float16

            # compute output
            with torch.no_grad():
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    val_loss_dict = self._step(batch, self.model, phase, loss_meters)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                # Update progress display meters with current batch values
                for name, meter in loss_meters.items():
                    # Find the corresponding meter in progress for display
                    for p_meter in progress.meters:
                        if p_meter.name == f"Loss/{phase}_{name}":
                            p_meter.update(meter.val)
                progress.display(data_iter)

        # Log validation averages at the end of the epoch to WandB and TB
        # We log these at the current TRAINING step to align with training progress
        if self.rank == 0:
            current_train_step = self.steps["train"]
            prefix = f"Loss/{phase}_"
            for name, meter in loss_meters.items():
                raw_name = name[len(prefix):] if name.startswith(prefix) else name
                self._log_scalar(f"Val_Loss/{raw_name}", meter.avg, current_train_step)
            logging.info(f"Validation Epoch {self.epoch} complete. Logged averages at train step {current_train_step}")

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches

        if self.gradient_clipper is not None:
            # setup gradient clipping at the beginning of training
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break

            # measure data loading time
            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            accum_steps = self.accum_steps

            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            # compute gradient and do SGD step
            assert data_iter <= limit_train_batches  # allow for off by one errors
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1].")

            # Log schedulers
            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = f"{i}_" if len(self.optims) > 1 else ("" + f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
                            self._log_scalar(
                                f"Train_Optim/{optim_prefix}{option}",
                                param_group[option],
                                self.steps[phase],
                            )
                self._log_scalar(
                    "Train_Optim/where",
                    self.where,
                    self.steps[phase],
                )

            # Clipping gradients and detecting diverging gradients
            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    meter_key = f"Grad/{key}"
                    if meter_key in loss_meters:
                        loss_meters[meter_key].update(grad_norm)
                    if self.steps[phase] % self.logging_conf.log_freq == 0:
                        self._log_scalar(f"Train_Grad/{key}", grad_norm, self.steps[phase])

            # Optimizer step
            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """
        Run the forward / backward as many times as there are chunks in the batch,
        accumulating the gradients on each backward
        """

        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)

        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        if amp_type == "bfloat16":
            amp_type = torch.bfloat16
        else:
            amp_type = torch.float16

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = self.model.no_sync() if i < accum_steps - 1 else contextlib.nullcontext()

            with ddp_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

    def _apply_batch_repetition(self, batch: Mapping) -> Mapping:
        """
        Applies a data augmentation by concatenating the original batch with a
        flipped version of itself.
        """
        tensor_keys = [
            "images",
            "depths",
            "extrinsics",
            "intrinsics",
            "cam_points",
            "world_points",
            "point_masks",
            "geom_masks",
            "scanner_coords",
        ]
        string_keys = ["seq_name"]

        for key in tensor_keys:
            if key in batch:
                original_tensor = batch[key]
                batch[key] = torch.concatenate([original_tensor, torch.flip(original_tensor, dims=[1])], dim=0)

        for key in string_keys:
            if key in batch:
                batch[key] = batch[key] * 2

        return batch

    def _process_batch(self, batch: Mapping):
        if self.data_conf.train.common_config.repeat_batch:
            batch = self._apply_batch_repetition(batch)

        # Normalize camera extrinsics and points if enabled (default: True)
        if self.data_conf.train.common_config.get("normalize_points", True):
            # Normalize camera extrinsics and points. The function returns new tensors.
            normalized_extrinsics, normalized_cam_points, normalized_world_points, normalized_depths = normalize_camera_extrinsics_and_points_batch(
                extrinsics=batch["extrinsics"],
                cam_points=batch["cam_points"],
                world_points=batch["world_points"],
                depths=batch["depths"],
                point_masks=batch["point_masks"],
            )

            # Replace the original values in the batch with the normalized ones.
            batch["extrinsics"] = normalized_extrinsics
            batch["cam_points"] = normalized_cam_points
            batch["world_points"] = normalized_world_points
            batch["depths"] = normalized_depths

        return batch

    def _step(self, batch, model: nn.Module, phase: str, loss_meters: dict):
        """
        Performs a single forward pass, computes loss, and logs results.

        Returns:
            A dictionary containing the computed losses.
        """
        # Forward pass
        y_hat = model(images=batch["images"], batch=batch)

        # Loss computation
        loss_dict = self.loss(y_hat, batch)
        loss_dict["loss_objective"] = loss_dict["objective"]

        # Combine all data for logging
        log_data = {**{f"pred_{k}": v for k, v in y_hat.items()}, **loss_dict, **batch}

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_tb_visuals(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to TensorBoard."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data["extrinsics"].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in loss_meters:
                    loss_meters[meter_key].update(value, batch_size)

                # Only log batch-level scalars for training to avoid step collision and noise
                if phase == "train" and step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self._log_scalar(f"Train_Loss/{key}", value, step)

        # Log Frame and Slice selection for Slot 2 and Slot 3 (if available)
        if phase == "train" and step % self.logging_conf.log_freq == 0 and self.rank == 0:
            # data["timesteps"] and data["slice_indices"] are [B, S]
            if "timesteps" in data and "slice_indices" in data:
                ts = data["timesteps"]
                sls = data["slice_indices"]
                # S is the number of slots in the sequence
                S = ts.shape[1] if hasattr(ts, "shape") else len(ts[0])
                
                # Slot 2 (index 1)
                if S > 1:
                    self._log_scalar("train_slice_selection/slot2_frame", ts[0, 1].item(), step)
                    self._log_scalar("train_slice_selection/slot2_slice", sls[0, 1].item(), step)
                
                # Slot 3 (index 2)
                if S > 2:
                    self._log_scalar("train_slice_selection/slot3_frame", ts[0, 2].item(), step)
                    self._log_scalar("train_slice_selection/slot3_slice", sls[0, 2].item(), step)

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""

        # Scale frequency by accum_steps to prevent redundant logging of chunks
        freq = self.logging_conf.log_visual_frequency.get(phase, 0)
        if phase == "train":
            freq *= self.accum_steps

        # For validation, we use the training step to keep WandB monotonic
        log_step = step if phase == "train" else self.steps["train"]

        if not (self.logging_conf.log_visuals and freq > 0 and (step % freq == 0) and (self.logging_conf.visuals_keys_to_log is not None)):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            keys_to_log = self.logging_conf.visuals_keys_to_log[phase]["keys_to_log"]
            assert len(keys_to_log) > 0, "Need to include some visual keys to log"
            modality = self.logging_conf.visuals_keys_to_log[phase]["modality"]
            assert modality in [
                "image",
                "video",
            ], "Currently only support video or image logging"

            prefix = "Train" if phase == "train" else "Val"
            name = f"{prefix}_Visuals"

            # Prepare MRI metadata for caption — show all frames' slice and timestep
            caption = None
            if "slice_indices" in batch and "timesteps" in batch:
                try:
                    seq_name = batch["seq_name"][0] if "seq_name" in batch else ""
                    label = "z" if "mri_axial" in seq_name else "s"
                    slices = batch["slice_indices"][0]   # (S,)
                    timesteps = batch["timesteps"][0]    # (S,)
                    S = slices.shape[0]
                    parts = [f"f{i}: {label}={slices[i].item()}, t={timesteps[i].item()}" for i in range(S)]
                    caption = " | ".join(parts)
                except:
                    pass

            def prepare_visual(key, v):
                # If it's world points (S, H, W, 3), permute to (S, 3, H, W)
                if v.dim() == 4 and v.shape[-1] == 3:
                    v = v.permute(0, 3, 1, 2)

                if "world_points" in key:
                    # Rescale coordinates [-1, 1] to [0, 1]
                    v = (v + 1.0) / 2.0
                elif key == "images":
                    # Images are [0, 1]. If they were normalized to [-1, 1], shift to [0, 1]
                    if v.min() < 0:
                        v = (v + 1.0) / 2.0
                return v.clamp(0, 1)

            # Log input images separately
            if "images" in batch and batch["images"][0].dim() >= 3:
                img_grid = torchvision.utils.make_grid(prepare_visual("images", batch["images"][0]), nrow=self.logging_conf.visuals_per_batch_to_log)
                img_grid = img_grid.cpu()
                if img_grid.dtype == torch.bfloat16:
                    img_grid = img_grid.to(torch.float16)
                self._log_visuals(f"{name}_images", img_grid.numpy(), log_step, self.logging_conf.video_logging_fps, caption=caption)

            # Assemble gradient grids (GT and Pred) together
            grad_grids = []
            for key in ["world_points", "pred_world_points"]:
                if key in keys_to_log and key in batch and batch[key][0].dim() >= 3:
                    v = batch[key][0]
                    if key == "pred_world_points":
                        # Mask out padded regions in the prediction to match GT
                        mask = batch["point_masks"][0].unsqueeze(-1)
                        v = v * mask
                    grad_grids.append(torchvision.utils.make_grid(prepare_visual(key, v), nrow=self.logging_conf.visuals_per_batch_to_log))

            if grad_grids:
                grad_visuals = torch.cat(grad_grids, dim=1)  # Stack vertically: Top=GT, Bot=Pred
                grad_visuals = grad_visuals.cpu()
                if grad_visuals.dtype == torch.bfloat16:
                    grad_visuals = grad_visuals.to(torch.float16)
                self._log_visuals(f"{name}_gradients_GT_Top_Pred_Bot", grad_visuals.numpy(), log_step, self.logging_conf.video_logging_fps, caption=caption)

            # --- 3D point cloud logging removed (too heavy for WandB storage) ---

            # --- Added DVF Visualization ---
            if self.wandb_writer and "pred_dvfs" in batch and "gt_dvfs_unnorm" in batch:
                try:
                    import wandb

                    from visual_util import plot_dvf_grid
                except ImportError:
                    wandb = None

                if wandb is not None:
                    # Get the first sequence in the batch
                    pred_dvf = batch["pred_dvfs"][0].detach().cpu().numpy()  # (S, H, W, 3)
                    gt_dvf = batch["gt_dvfs_unnorm"][0].detach().cpu().numpy()  # (S, H, W, 3)
                    img_tx = batch["images"][0].detach().cpu().numpy()  # (S, 3, H, W)

                    # Visualize the first dynamic frame (index 1 if S > 1, else 0)
                    s_idx = min(1, pred_dvf.shape[0] - 1)

                    # Create a descriptive caption
                    caption = f"Phase: {phase}, Step: {log_step}"
                    if "timesteps" in batch:
                        t_idx = batch["timesteps"][0][s_idx].item()
                        caption += f", Frame (t): {t_idx + 1}"
                    if "slice_indices" in batch:
                        s_val = batch["slice_indices"][0][s_idx].item()
                        caption += f", Slice: {s_val}"
                    if "rotations" in batch:
                        rot = batch["rotations"][0][s_idx]
                        caption += f", Rot: [{rot[0]:.0f},{rot[1]:.0f},{rot[2]:.0f}]"

                    fig = plot_dvf_grid(img_tx, gt_dvf, pred_dvf, seq_idx=s_idx, v_min=-5.0, v_max=5.0, ref_image=img_tx[0])
                    self.wandb_writer.log(f"{name}_DVF", wandb.Image(fig, caption=caption), log_step)

                    import matplotlib.pyplot as plt

                    plt.close(fig)
            # ------------------------------------

            # --- Volume reconstruction visualization (direct V_canon vs V_gt loss) ---
            if self.wandb_writer and "V_canon" in batch:
                try:
                    import wandb
                    import matplotlib.pyplot as plt
                except ImportError:
                    wandb = None

                if wandb is not None:
                    import numpy as np
                    V = batch["V_canon"][0].detach().float().cpu().numpy()        # (D, H, W)
                    cov = batch["coverage"][0].detach().float().cpu().numpy()     # (D, H, W)
                    has_gt = "V_gt" in batch
                    V_gt_np = batch["V_gt"][0].detach().float().cpu().numpy() if has_gt else None
                    D, Hv, Wv = V.shape

                    # ── Panel A: MIPs (V_canon, V_gt, error, coverage) ──────────────
                    n_rows = 4 if has_gt else 2
                    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 4 * n_rows), dpi=72)
                    fig.suptitle(f"Volumes — {caption or ''}")
                    v_vmax = float(max(V.max(), V_gt_np.max() if has_gt else 0.0, 1e-3))
                    for col, (lbl, mip) in enumerate([
                        ("axial", V.max(0)), ("coronal", V.max(1)), ("sagittal", V.max(2)),
                    ]):
                        axes[0, col].imshow(mip, cmap="gray", vmin=0, vmax=v_vmax)
                        axes[0, col].set_title(f"V_canon (pred) {lbl}"); axes[0, col].axis("off")
                    if has_gt:
                        for col, mip in enumerate([V_gt_np.max(0), V_gt_np.max(1), V_gt_np.max(2)]):
                            axes[1, col].imshow(mip, cmap="gray", vmin=0, vmax=v_vmax)
                            axes[1, col].set_title("V_gt (phase-0 splat)"); axes[1, col].axis("off")
                        diff_vol = V - V_gt_np  # signed error volume
                        err_max = max(float(np.abs(diff_vol).max()), 1e-3)
                        for col, mip in enumerate([diff_vol.mean(0), diff_vol.mean(1), diff_vol.mean(2)]):
                            axes[2, col].imshow(mip, cmap="RdBu", vmin=-err_max, vmax=err_max)
                            axes[2, col].set_title("V_canon - V_gt (mean)"); axes[2, col].axis("off")
                    c_row = 3 if has_gt else 1
                    c_vmax = float(max(cov.max(), 1e-3))
                    for col, mip in enumerate([cov.max(0), cov.max(1), cov.max(2)]):
                        axes[c_row, col].imshow(mip, cmap="viridis", vmin=0, vmax=c_vmax)
                        axes[c_row, col].set_title("coverage"); axes[c_row, col].axis("off")
                    plt.tight_layout()
                    self.wandb_writer.log(f"{name}_Volume", wandb.Image(fig, caption=caption), log_step)
                    plt.close(fig)

                    # ── Per-z slice strip removed (too heavy: 3×12 = 36 thumbnails) ──

                    # ── Panel C: input slices grid (just show what the model received) ──
                    images_b = batch["images"][0].detach().float().cpu()  # (S, 3, H, W)
                    S_in = images_b.shape[0]
                    I_in_all = images_b.mean(dim=1).numpy()                # (S, H, W)
                    if I_in_all.max() > 2.0:
                        I_in_all = I_in_all / 255.0
                    fig3, ax3 = plt.subplots(1, S_in, figsize=(2.0 * S_in, 2.0), dpi=72)
                    fig3.suptitle(f"Input slices — {caption or ''}")
                    if S_in == 1:
                        ax3 = [ax3]
                    for s in range(S_in):
                        ax3[s].imshow(I_in_all[s], cmap="gray", vmin=0, vmax=1)
                        ax3[s].set_title(f"s={s}", fontsize=7); ax3[s].axis("off")
                    plt.tight_layout()
                    self.wandb_writer.log(f"{name}_InputSlices", wandb.Image(fig3, caption=caption), log_step)
                    plt.close(fig3)
            # ------------------------------------


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return isinstance(data, Sequence) and not isinstance(data, str) and len(data) > 0 and isinstance(data[0], (str, int, float, bool))


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """
    Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        # either a tensor or a list of primitive objects
        # assert len(data) % num_chunks == 0
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {key: get_chunk_from_data(value, chunk_id, num_chunks) for key, value in data.items()}
    elif isinstance(data, str):
        # NOTE: this is a hack to support string keys in the batch
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
