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
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from data.gpu_aug import build_gpu_transforms, gpu_augment_batch
from data.respiratory import RespiratoryConfig
from train_utils.checkpoint import CheckpointSaver
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
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
        # Fully-resolved config snapshot (from launch.py) → logged as wandb run.config.
        self._wandb_config = kwargs.get("_wandb_config", None)

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
        self._setup_backends(cuda)

        # Setup logging directory and configure logger
        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=0,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, 0)

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint: a run's own latest checkpoint in save_dir wins (SLURM auto-requeue
        # / crash resume — epoch+steps+optimizer+scaler intact); the configured seed/base
        # checkpoint (resume_checkpoint_path) is only the cold-start fallback. See
        # resolve_resume_checkpoint for why the priority must be this way round.
        ckpt_to_load = resolve_resume_checkpoint(
            self.checkpoint_conf.save_dir, self.checkpoint_conf.resume_checkpoint_path
        )
        if ckpt_to_load is not None:
            self._load_resuming_checkpoint(ckpt_to_load)

        # ── Diagnostics state (val-only; never touches training) ──────────
        # Cache the dataset's `t_target_fixed` setting so val-only diagnostics can gate on it.
        # When None → multi-phase val sampling: log per-phase PSNR + cardiac-cycle filmstrip.
        # When int → single-phase mode: those diagnostics are skipped (meaningless).
        self.t_target_fixed = None
        # Reference-slice conditioning (docs/25): when True the cardiac-cycle filmstrip must
        # rebuild slot 0 (the reference) at each queried phase, since the model reads the target
        # phase from slot-0's image content rather than a broadcast target_t index.
        self.reference_slot = False
        mri_ds = self._get_mri_dataset()
        if mri_ds is not None:
            self.t_target_fixed = mri_ds.t_target_fixed
            self.reference_slot = getattr(mri_ds, "reference_slot", False)
        # Accumulator for per-phase val PSNR, cleared at the start of each val epoch.
        # Two parallel namespaces: `_full` (whole canonical cube) and `_bbox` (subject's
        # geometric native-FOV region only).
        self._per_phase_val_psnr_full = defaultdict(list)
        self._per_phase_val_psnr_bbox = defaultdict(list)
        # `_motion` = only the voxels that move across the cardiac cycle (the dynamic
        # heart, ~3-5% of the cube). The honest signal: static tissue is excluded, so
        # this PSNR vs its identity baseline tells you whether the model actually
        # corrects motion where it matters. Logged under the `val/psnr/motion/` panel.
        self._per_phase_val_psnr_motion = defaultdict(list)
        # Identity baseline per phase + aggregate mean, populated by _compute_identity_baseline
        # and baked into val_psnr metric names so each panel shows n and baseline in its title.
        self._identity_baseline_full_per_phase = None
        self._identity_baseline_full_mean = None
        self._identity_baseline_bbox_per_phase = None
        self._identity_baseline_bbox_mean = None
        self._identity_baseline_motion_per_phase = None
        self._identity_baseline_motion_mean = None
        # Cumulative count of training batches skipped due to non-finite loss. Without this,
        # NaN/Inf chunks get swallowed by the early-return in _run_steps_on_batch_chunks and
        # loss_meters silently undercount, making the wandb loss curve look healthier than it is.
        self._nan_batch_count = 0
        # ── GPU augmentation pipeline (off by default) ─────────────────────
        # Built from `data.augmentation` (see mri_volume.yaml). `None` → identity
        # passthrough in the trainer hook. Val ALWAYS uses identity — augmentation
        # only ever applies to train. NOTE: `logging` is shadowed by the `logging`
        # kwarg in this __init__ scope, so build_gpu_transforms does its own
        # (module-scope) logging rather than logging here.
        aug_cfg = self.data_conf.get("augmentation", None) if self.data_conf is not None else None
        self.gpu_transforms = build_gpu_transforms(aug_cfg)
        self._aug_tier = (aug_cfg.get("tier", "conservative")
                          if aug_cfg is not None else "conservative")  # for the aug panel caption
        self.val_gpu_transforms = None  # val never AFFINE-augments

        # ── Respiratory-motion augmentation (off by default) ───────────────
        # Unlike affine, respiratory applies in BOTH train and val: train draws
        # iid per epoch from a PRIVATE generator (so it never perturbs the global
        # RNG stream batchaug/dropout draw from — turning breathing on doesn't
        # change affine draws); val draws deterministically per seq_index. It
        # overwrites ONLY the input slices (see gpu_aug.gpu_augment_batch).
        # NOTE: `logging` is shadowed by the `logging` kwarg in this __init__ scope
        # (see the gpu_transforms comment above), so we log respiratory status from
        # the module-scope `RespiratoryConfig.from_cfg` path / leave it to the config.
        self.respiratory_cfg = RespiratoryConfig.from_cfg(
            aug_cfg.get("respiratory", None) if aug_cfg is not None else None)
        self.resp_generator = torch.Generator(device=self.device).manual_seed(
            int(self.seed_value))

        # Compute identity-Δ baseline once at startup.
        if self.mode in ["train", "val"]:
            self._compute_identity_baseline()
            # Motion-mask reference panel (3 val subjects). Data-derived + static across
            # training, so logged once here rather than every val epoch.
            self._log_motion_mask_example(self.steps.get("train", 0))

    def _get_mri_dataset(self):
        """Walk the wrapper chain (DynamicTorchDataset → ComposedDataset → TupleConcatDataset)
        to retrieve the underlying MRIDataset instance. Returns None if val_dataset is unset
        or the chain doesn't match (e.g., non-MRI datasets)."""
        try:
            ds = self.val_dataset
            if ds is None:
                return None
            inner = ds.dataset.base_dataset.datasets[0]
            return inner
        except (AttributeError, IndexError, TypeError):
            return None

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

    def _setup_backends(self, cuda_conf: Dict) -> None:
        """Configures PyTorch CUDA backends. Single-GPU: no process group is created."""
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
        logging.info(f"Model state loaded. Missing keys count: {len(missing) if missing else 0}. Unexpected keys count: {len(unexpected) if unexpected else 0}.")

        # Load optimizer state if available and in training mode (self.optims is only
        # constructed when self.mode != "val"; skip otherwise to avoid AttributeError).
        if "optimizer" in checkpoint and self.mode != "val":
            logging.info("Loading optimizer state dict")
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
        # Single-GPU only: always device 0 (was read from torchrun's LOCAL_RANK env).
        if device == "cuda":
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(0)
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
            self.wandb_writer = instantiate(
                self.logging_conf.wandb_writer,
                wandb_config=self._wandb_config,
                _recursive_=False,
            )

        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        # GradScaler only helps fp16 (prevents underflow). bf16 has fp32 range, so loss
        # scaling is dead weight — disable to keep the train loop honest.
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.optim_conf.amp.enabled and self.optim_conf.amp.amp_dtype == "float16"
        )

        # Freeze specified model parameters if any
        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(f"[Start] Freezing modules: {self.optim_conf.frozen_module_names}")
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(f"[Done] Freezing modules: {self.optim_conf.frozen_module_names}")

        # Log model summary
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

        saver = CheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            epoch=epoch,
        )

        # Single-GPU: self.model is the bare module (no DDP wrap to unwrap).
        model = self.model

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

    @staticmethod
    def _scalar_name(phase: str, key: str) -> str:
        """Map a raw loss/metric dict-key to its subtree'd wandb name.
        Groups under {phase}/: loss/, psnr/ (all PSNR variants), resp/ (breathing),
        metric/ (everything else). full/bbox PSNR get a _mean suffix so the flat
        whole-val scalar doesn't collide with the per-phase val/psnr/full/ group."""
        if key.startswith("loss_"):
            return f"{phase}/loss/{key[len('loss_'):]}"
        if "psnr" in key:
            name = key.replace("metric_psnr_3d_", "")
            if name in ("full", "bbox"):
                name += "_mean"
            return f"{phase}/psnr/{name}"
        if "resp" in key:
            return f"{phase}/resp/{key.replace('metric_resp_', '').replace('metric_', '')}"
        return f"{phase}/metric/{key.replace('metric_', '')}"

    def _log_visuals(self, name: str, data: Any, step: int, fps: int = 4, caption: Optional[str] = None):
        """Logs visual data to both TensorBoard and WandB."""
        if self.tb_writer:
            self.tb_writer.log_visuals(name, data, step, fps)
        if self.wandb_writer:
            self.wandb_writer.log_visuals(name, data, step, fps, caption=caption)

    def _log_resp_disp_scalar(self, batch, step: int, prefix: str):
        """Log the per-slot respiratory displacement magnitude (mm) as scalars under
        `{prefix}/resp/disp_mm_{mean,max}`. No-op when breathing is off (key absent).
        Read-only diagnostic — never affects training."""
        if not self.wandb_writer:
            return
        disp = batch.get("resp_disp_mm")
        if disp is None:
            return
        try:
            mag = disp.float().norm(dim=-1)  # (B, S) per-slot |d| in mm
            self._log_scalar(f"{prefix}/resp/disp_mm_mean", float(mag.mean().item()), step)
            self._log_scalar(f"{prefix}/resp/disp_mm_max", float(mag.max().item()), step)
        except Exception as e:
            logging.warning(f"resp_disp scalar log failed (ignored): {e}")

    def _compute_identity_baseline(self):
        """Run identity-Δ (no motion correction) on the val set and log PSNR as constants.
        Called ONCE at trainer setup. Read-only over the val dataset; never touches the
        training path. Failures are caught and logged — training proceeds either way.
        """
        try:
            from data.composed_dataset import _data_to_batch_tensors as _maybe_to_batch  # type: ignore
        except Exception:
            _maybe_to_batch = None  # we'll do the conversion inline

        mri_ds = self._get_mri_dataset()
        if mri_ds is None:
            logging.warning("Identity baseline: no MRIDataset found, skipping.")
            return

        try:
            import numpy as np
            from loss import compute_volume_intensity_loss

            grid_shape = tuple(mri_ds.gt_grid_shape)
            num_slices = mri_ds.num_slices

            # Bucket PSNR by t_target — TWO parallel namespaces (`_full` over the
            # whole cube, `_bbox` over the subject's geometric content region).
            per_phase_full = defaultdict(list)
            per_phase_bbox = defaultdict(list)
            per_phase_motion = defaultdict(list)

            # Iterate the SAME seq_index range the val loop uses. Val calls
            # get_data(seq_index = 0..N-1) with N = effective limit_val_batches, and val
            # inputs are now seeded per seq_index (random.Random(seq_index)), so mirroring
            # the range here makes the identity reference cover the EXACT (subject,
            # t_target, seeded-input) samples val scores. Falls back to one pass over
            # subjects when limit_val_batches is unset.
            n_subj = len(mri_ds.subjects)
            N = self.limit_val_batches if self.limit_val_batches is not None else n_subj
            if self.t_target_fixed is not None and 0 < n_subj < N:
                N = n_subj   # mirror val's fixed-phase auto-cap (one deterministic pass)
            if getattr(mri_ds, "val_targets", None) is not None:
                N = len(mri_ds.val_targets)   # mirror val's EF-sweep length (30×{ED,ES})
            N = max(int(N), 1)
            for i in range(N):
                data = mri_ds.get_data(seq_index=i, img_per_seq=num_slices)
                # Build a minimal batch dict on device (no model forward; identity Δ).
                def st(k, dt=np.float32):
                    return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(self.device)
                imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
                batch = {
                    "images": imgs,
                    "scanner_coords": st("scanner_coords"),
                }
                if "gt_target_volume" in data:
                    batch["gt_target_volume"] = torch.from_numpy(
                        data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(self.device)
                else:
                    continue  # no GT to compare against; skip
                if "anatomy_bbox" in data:
                    batch["anatomy_bbox"] = torch.from_numpy(
                        np.asarray(data["anatomy_bbox"]).astype(np.int64)).unsqueeze(0).to(self.device)
                if "phases" in data:
                    batch["phases"] = torch.from_numpy(
                        np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(self.device)
                # Apply the SAME deterministic respiratory shift the val loop uses, so the
                # identity (do-nothing, Δ=0) reference is the splat of the SAME breathing-
                # corrupted inputs the model is scored on — NOT the unshifted clean inputs.
                # Otherwise "Δ vs identity" compares a clean-input splat against a
                # corrupted-input model and understates motion correction. No-op when
                # respiratory is disabled (gpu_augment_batch early-returns unchanged).
                batch["timesteps"] = st("timesteps", np.int64)
                batch["slice_indices"] = st("slice_indices", np.float32)  # may be continuous z
                batch["seq_index"] = torch.tensor([[i]], dtype=torch.int64, device=self.device)
                batch = gpu_augment_batch(
                    batch, None, self.device,
                    respiratory_cfg=self.respiratory_cfg, train=False)
                # Identity world_points = scanner_coords (Δ = 0).
                preds = {"world_points": batch["scanner_coords"]}
                out = compute_volume_intensity_loss(preds, batch, grid_shape=grid_shape, tv_weight=0.0)
                t = int(data["t_target"].item() if data["t_target"].ndim == 0 else data["t_target"].flatten()[0].item())
                if "metric_psnr_3d_full" in out:
                    per_phase_full[t].append(out["metric_psnr_3d_full"].item())
                if "metric_psnr_3d_bbox" in out:
                    per_phase_bbox[t].append(out["metric_psnr_3d_bbox"].item())
                if "metric_psnr_3d_motion" in out:
                    per_phase_motion[t].append(out["metric_psnr_3d_motion"].item())

            # Aggregate.
            all_full = [p for ps in per_phase_full.values() for p in ps]
            all_bbox = [p for ps in per_phase_bbox.values() for p in ps]
            all_motion = [p for ps in per_phase_motion.values() for p in ps]
            if not all_full:
                logging.warning("Identity baseline: no subjects yielded a PSNR; skipping.")
                return
            mean_full = float(sum(all_full) / len(all_full))
            mean_bbox = float(sum(all_bbox) / len(all_bbox)) if all_bbox else None
            mean_motion = float(sum(all_motion) / len(all_motion)) if all_motion else None
            per_phase_mean_full = {t: float(sum(ps) / len(ps)) for t, ps in per_phase_full.items()}
            per_phase_mean_bbox = {t: float(sum(ps) / len(ps)) for t, ps in per_phase_bbox.items()}
            per_phase_mean_motion = {t: float(sum(ps) / len(ps)) for t, ps in per_phase_motion.items()}

            # Persist to JSON for offline inspection.
            out_path = os.path.join(self.logging_conf.log_dir, "baseline_identity.json")
            try:
                with open(out_path, "w") as f:
                    json.dump({
                        "full": {"mean_psnr": mean_full, "per_phase_mean": per_phase_mean_full},
                        "bbox": {"mean_psnr": mean_bbox, "per_phase_mean": per_phase_mean_bbox},
                        "motion": {"mean_psnr": mean_motion, "per_phase_mean": per_phase_mean_motion},
                        "per_phase_counts": {t: len(ps) for t, ps in per_phase_full.items()},
                        "t_target_fixed": self.t_target_fixed,
                    }, f, indent=2)
            except Exception as e:
                logging.warning(f"baseline JSON write failed: {e}")

            # Cache for val-time metric-name embedding.
            self._identity_baseline_full_per_phase = per_phase_mean_full
            self._identity_baseline_full_mean = mean_full
            self._identity_baseline_bbox_per_phase = per_phase_mean_bbox
            self._identity_baseline_bbox_mean = mean_bbox
            self._identity_baseline_motion_per_phase = per_phase_mean_motion
            self._identity_baseline_motion_mean = mean_motion
            bbox_str = f"{mean_bbox:.2f}" if mean_bbox is not None else "n/a"
            motion_str = f"{mean_motion:.2f}" if mean_motion is not None else "n/a"
            logging.info(
                f"Identity-Δ baseline: full PSNR = {mean_full:.2f} dB / bbox PSNR = {bbox_str} dB "
                f"/ motion PSNR = {motion_str} dB across {len(all_full)} val sample(s). "
                f"Persisted to {out_path}."
            )
        except Exception as e:
            logging.warning(f"_compute_identity_baseline failed (ignored): {e}")

    def _log_motion_mask_example(self, log_step: int):
        """Render the motion mask for 3 val subjects in one panel under `media_others/`.

        Each row is one subject (val indices 0, 7, 15); columns show, at a mid-bbox
        z-plane: V_gt(ED) | motion magnitude (max-min over phases) | mask overlay on V_gt.
        Data-derived only (no model forward) and static across training, so this is logged
        ONCE at startup to document which voxels the `val/psnr/motion` PSNR is computed over.
        vmin/vmax are per-subject so small-FOV rows aren't washed out. Wrapped, never raises.
        """
        if not self.wandb_writer:
            return
        try:
            import wandb
            import numpy as np
            import matplotlib.pyplot as plt
            from loss import compute_motion_mask, MOTION_MASK_TAU
        except ImportError:
            return

        mri_ds = self._get_mri_dataset()
        if mri_ds is None:
            return
        try:
            # Distinct val subjects; drop any out-of-range / duplicate after wraparound
            # so tiny val sets (e.g. synthetic test data) don't render the same subject twice.
            n_subj = len(mri_ds.subjects)
            subj_indices = [i for i in (0, 7, 15) if i < n_subj]
            if not subj_indices:
                return
            n_rows = len(subj_indices)

            fig, axes = plt.subplots(n_rows, 3, figsize=(9.0, 3.2 * n_rows), dpi=90)
            axes = np.atleast_2d(axes)  # (n_rows, 3) even when n_rows == 1
            for row, subj_idx in enumerate(subj_indices):
                data = mri_ds.get_data(seq_index=subj_idx, img_per_seq=mri_ds.num_slices)
                phases = np.asarray(data["phases"]).astype(np.float32)   # (T, D, H, W)
                ed = phases[0]                                           # (D, H, W)
                motion_mag = phases.max(0) - phases.min(0)              # (D, H, W)
                mask = compute_motion_mask(
                    torch.from_numpy(phases).unsqueeze(0)
                )[0].cpu().numpy()                                      # (D, H, W) bool
                bbox = np.asarray(data["anatomy_bbox"]).astype(int)
                z0, z1 = int(bbox[0]), int(bbox[1])
                z = (z0 + z1) // 2
                frac = float(mask[z0:z1].mean())
                vmax = max(float(ed.max()), 1e-3)
                mmax = max(float(motion_mag.max()), 1e-3)

                ax = axes[row]
                ax[0].imshow(ed[z], cmap="gray", vmin=0, vmax=vmax)
                ax[0].set_title(f"subj {subj_idx}: V_gt ED (z={z})", fontsize=9)
                ax[1].imshow(motion_mag[z], cmap="magma", vmin=0, vmax=mmax)
                ax[1].set_title("motion = max-min", fontsize=9)
                ax[2].imshow(ed[z], cmap="gray", vmin=0, vmax=vmax)
                overlay = np.zeros((*mask[z].shape, 4))
                overlay[mask[z]] = [1, 0, 0, 0.45]
                ax[2].imshow(overlay)
                ax[2].set_title(f"mask tau={MOTION_MASK_TAU} ({frac*100:.1f}% of bbox)", fontsize=9)
                for a in ax:
                    a.set_xticks([]); a.set_yticks([])
            fig.suptitle("Motion mask (val subjects, mid-bbox z)", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            self.wandb_writer.log("media_others/val_motion_mask_example", wandb.Image(fig), log_step)
            plt.close(fig)
        except Exception as e:
            logging.warning(f"motion mask example log failed (ignored): {e}")

    def _log_cardiac_cycle_filmstrip(self, log_step: int):
        """Reconstruct one fixed val subject (idx 0) at all 12 phases and log a 2×12 strip
        (V_gt top / V_canon bot, mid-z slice). Builds ONE input batch and sweeps the global
        target_t query over all phases (decoupled-target design) — faithful 4D cine from a
        single acquisition. Read-only over model + dataset state (no t_target_fixed mutation).
        """
        if not self.wandb_writer:
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs
            import numpy as np
            from loss import compute_volume_intensity_loss
        except ImportError:
            return

        mri_ds = self._get_mri_dataset()
        if mri_ds is None:
            return

        T_total = mri_ds.gt_grid_shape[0]
        subj_idx = 0
        grid_shape = tuple(mri_ds.gt_grid_shape)
        num_slices = mri_ds.num_slices

        canon_frames = []
        gt_frames = []
        refined_frames = []  # populated only when the refiner ran (additive gif)
        try:
            self.model.eval()
            amp_dtype = torch.bfloat16
            # Build the input batch ONCE. In the legacy decoupled-target design the SAME
            # scattered input slices are reused for every target phase — only the global
            # target_t query (and the V_gt) varies. In reference-slice mode (docs/25) slot 0
            # is additionally rebuilt to the target-phase reference at each phase (see loop
            # below); slots 1..S-1 (the scattered inputs) are still reused across phases.
            data = mri_ds.get_data(seq_index=subj_idx, img_per_seq=num_slices)
            def st(k, dt=np.float32):
                return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(self.device)
            imgs = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
            S = imgs.shape[1]
            batch = {
                "images": imgs,
                "scanner_coords": st("scanner_coords"),
                "z_indices": st("z_indices"),
                "t_indices": st("t_indices"),
            }
            # Full canonical phase bundle → per-phase V_gt without re-sampling inputs.
            phases_bundle = torch.from_numpy(
                np.asarray(data["phases"]).astype(np.float32)).to(self.device)  # (T, D, H, W)
            # The filmstrip visualizes the REAL corrupted->clean val task (V_gt stays at the
            # unshifted reference), so val breathing is applied to the INPUT slices when enabled.
            # Legacy (decoupled) mode: inputs are identical for every target phase → apply
            # breathing ONCE and reuse. Reference-slot mode: slot 0 changes per phase → breathing
            # is (re-)applied INSIDE the loop instead. No-op when respiratory is disabled.
            batch["phases"] = phases_bundle.unsqueeze(0)
            batch["timesteps"] = st("timesteps", np.int64)
            batch["slice_indices"] = st("slice_indices", np.float32)  # may be continuous z
            batch["seq_index"] = torch.tensor([[subj_idx]], dtype=torch.int64, device=self.device)
            do_resp = (self.respiratory_cfg is not None
                       and getattr(self.respiratory_cfg, "enable", False))
            if not self.reference_slot:
                batch = gpu_augment_batch(
                    batch, None, self.device,
                    respiratory_cfg=self.respiratory_cfg, train=False)
            model = self.model
            # Reference-slice mode (docs/25): slot 0 defines the queried phase via its IMAGE, not a
            # target_t index, so it must OBSERVE phase t at each query. z_mid (the bbox z-center) is
            # constant across the sweep, so scanner_coords[0]/z_indices[0] from get_data stay valid;
            # only slot-0's phase (and image) change. We set timesteps[0]=t and re-extract through
            # the SAME val input pipeline: when breathing is on slot 0 is corrupted exactly as in
            # val (scattered slots 1..S-1 re-extract identically each phase); else a clean reslice.
            # (target_t_indices is set below but inert when use_target_t=false.)
            import torch.nn.functional as _F
            ref_zmid = None
            hw = batch["images"].shape[-1]
            if self.reference_slot:
                _bb = np.asarray(data["anatomy_bbox"]).astype(np.int64)
                ref_zmid = (int(_bb[0]) + int(_bb[1])) // 2
            for t in range(T_total):
                t_norm = (t / max(1, T_total)) * 2.0 - 1.0  # match dataset normalization
                batch["target_t_indices"] = torch.full(
                    (1, S, 1), t_norm, dtype=torch.float32, device=self.device)
                if self.reference_slot:
                    batch["timesteps"][:, 0] = t  # slot 0 observes the target phase t
                    if do_resp:
                        # Re-extract all inputs via the deterministic val breathing pipeline:
                        # slot 0 → breathing-shifted (t, z_mid); scattered slots identical each phase.
                        batch = gpu_augment_batch(
                            batch, None, self.device,
                            respiratory_cfg=self.respiratory_cfg, train=False)
                    else:
                        ref_up = _F.interpolate(
                            phases_bundle[t, ref_zmid][None, None].float(), size=(hw, hw),
                            mode="bilinear", align_corners=True)  # (1, 1, 518, 518) in [0, 1]
                        batch["images"][:, 0] = ref_up.repeat(1, 3, 1, 1)
                batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)  # (1, D, H, W)
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=amp_dtype):
                    preds = model(batch["images"], batch=batch)
                    out = compute_volume_intensity_loss(
                        {"world_points": preds["world_points"].float()},
                        batch, grid_shape=grid_shape, tv_weight=0.0,
                    )
                V_canon = out["V_canon"][0].float().cpu().numpy()
                V_gt = out["V_gt"][0].float().cpu().numpy()
                # Render 5 planes (mid-2 .. mid+2, clamped) stacked vertically into (5H, W) so
                # the strip/gif shows off-reference planes, not just the mid/reference plane.
                D = V_canon.shape[0]
                mid_d = D // 2
                window = [min(max(mid_d + off, 0), D - 1) for off in (-2, -1, 0, 1, 2)]
                canon_frames.append(np.concatenate([V_canon[c] for c in window], axis=0))
                gt_frames.append(np.concatenate([V_gt[c] for c in window], axis=0))
                if "V_refined" in preds:
                    Vr = preds["V_refined"][0].float().cpu().numpy()
                    refined_frames.append(np.concatenate([Vr[c] for c in window], axis=0))
        except Exception as e:
            logging.warning(f"cardiac filmstrip render failed (ignored): {e}")
            return

        v_vmax = float(max(max(f.max() for f in canon_frames),
                           max(f.max() for f in gt_frames),
                           1e-3))

        # In fixed-phase mode the model only ever trained at t_target_fixed, so V_canon at
        # every other phase is out-of-distribution (V_gt still beats — it's real data).
        # Note this in the caption so the strip/gif isn't misread as a failed reconstruction.
        if self.t_target_fixed is not None:
            mode_note = f" | fixed-t={self.t_target_fixed}: V_canon at t≠{self.t_target_fixed} is out-of-distribution"
        else:
            mode_note = ""

        # NOTE: the static 2×T_total cardiac-cycle still image is intentionally NOT logged —
        # the animated GIF below carries the same information more compactly. (Disabled per
        # request; re-enable this block if you want the static filmstrip back.)
        # fig = None
        # try:
        #     fig = plt.figure(figsize=(1.4 * T_total + 0.5, 6.0), dpi=90)  # taller: cells are 5 stacked planes
        #     gs = _gs.GridSpec(2, T_total + 1, width_ratios=[1.0] * T_total + [0.04], wspace=0.05, hspace=0.18)
        #     for t in range(T_total):
        #         ax = fig.add_subplot(gs[0, t])
        #         ax.imshow(gt_frames[t], cmap="gray", vmin=0, vmax=v_vmax)
        #         ax.set_xticks([]); ax.set_yticks([])
        #         ax.set_title(f"t={t}", fontsize=8)
        #         if t == 0:
        #             ax.set_ylabel("V_gt", fontsize=9)
        #         ax2 = fig.add_subplot(gs[1, t])
        #         im = ax2.imshow(canon_frames[t], cmap="gray", vmin=0, vmax=v_vmax)
        #         ax2.set_xticks([]); ax2.set_yticks([])
        #         if t == 0:
        #             ax2.set_ylabel("V_canon", fontsize=9)
        #     cax = fig.add_subplot(gs[:, T_total]); plt.colorbar(im, cax=cax)
        #     fig.suptitle(f"Cardiac cycle (val subj 0, mid-z ±2, 5 planes) — step={log_step}", fontsize=9)
        #     self.wandb_writer.log("media_others/Val_Visuals_cardiac_cycle",
        #                           wandb.Image(fig, caption=f"step={log_step}{mode_note}"), log_step)
        # except Exception as e:
        #     logging.warning(f"cardiac filmstrip log failed (ignored): {e}")
        # finally:
        #     if fig is not None:
        #         plt.close(fig)                                        # never leak a figure on the error path

        # Animated GIF: 2 rows × 5 cols per frame — top row = V_gt's 5 planes (mid-2..mid+2)
        # laid out horizontally, bottom row = the model's same 5 planes. Cycled over t so the
        # heart beats. (Each stored frame is (5h, W) = 5 planes stacked vertically; we reshape
        # back to the 5 planes and re-tile as 2×5.) wandb.Video → moviepy needs 3-channel RGB.
        def _tile_2x5(gt5, model5, vmax):
            n = 5; h = gt5.shape[0] // n; W = gt5.shape[1]
            def _row(stack5):                                   # (5h, W) -> (h, 5W)
                planes = stack5.reshape(n, h, W)
                return np.concatenate([planes[i] for i in range(n)], axis=1)
            grid = np.concatenate([_row(gt5), _row(model5)], axis=0)   # (2h, 5W)
            g = np.clip(grid / vmax * 255.0, 0, 255).astype(np.uint8)
            return np.stack([g, g, g], axis=0)                  # (3, 2h, 5W)

        try:
            frames = np.stack([_tile_2x5(gt_frames[t], canon_frames[t], v_vmax)
                               for t in range(len(gt_frames))], axis=0)   # (T, 3, 2h, 5W)
            self.wandb_writer.log(
                "media_val_ED_ES/Val_Visuals_cardiac_cycle_gif",
                wandb.Video(frames, fps=4, format="gif",
                            caption=f"step={log_step} — rows: V_gt (top) / V_canon (bottom); "
                                    f"cols: z = mid-2 .. mid+2 (planes 4-8){mode_note}"),
                log_step,
            )
        except Exception as e:
            logging.warning(f"cardiac cycle gif log failed (ignored): {e}")

        # Refiner gif — additive, only when the refiner ran. Same 2×5 (V_gt top / V_refined bottom).
        if len(refined_frames) == len(gt_frames) and refined_frames:
            try:
                rmax = float(max(max(f.max() for f in refined_frames),
                                 max(f.max() for f in gt_frames), 1e-3))
                frames = np.stack([_tile_2x5(gt_frames[t], refined_frames[t], rmax)
                                   for t in range(len(gt_frames))], axis=0)
                self.wandb_writer.log(
                    "media_others/refiner_cardiac_cycle_gif",
                    wandb.Video(frames, fps=4, format="gif",
                                caption=f"step={log_step} — rows: V_gt (top) / V_refined (bottom); "
                                        f"cols: z = mid-2 .. mid+2{mode_note}"),
                    log_step,
                )
            except Exception as e:
                logging.warning(f"refiner cardiac gif log failed (ignored): {e}")

    def _save_val_volumes(self, batch: Mapping, loss_dict: Mapping) -> None:
        """Dump predicted + GT volumes to ${log_dir}/val_volumes/, one pair per (subject, phase).

        The val loop revisits each subject at a few t_target phases over the epoch, so we
        de-dup by (subject, phase) and save each distinct pair only the FIRST time it is seen
        this epoch. Because val is deterministic (shuffle=False), the first-seen order and phases
        are identical every epoch, so the same filenames are overwritten in place (a couple hundred
        MB, NOT limit_val_batches × 2). Under the EF sweep this yields ED + ES per subject.
        Affine is identity — V_canon lives in the dimensionless canonical [-1, 1] grid, not the
        source NIfTI's physical frame, so a physical affine would be misleading.
        """
        if not getattr(self.logging_conf, "save_val_volumes", False):
            return
        if "V_canon" not in loss_dict or "V_gt" not in loss_dict:
            return
        try:
            import nibabel as nib
            import numpy as np

            out_dir = os.path.join(self.logging_conf.log_dir, "val_volumes")
            safe_makedirs(out_dir)

            V_canon = loss_dict["V_canon"].detach().float().cpu().numpy()  # (B, D, H, W)
            V_gt = loss_dict["V_gt"].detach().float().cpu().numpy()
            t_targets = batch.get("t_target")
            seq_names = batch.get("seq_name", [])
            B = V_canon.shape[0]
            affine = np.eye(4, dtype=np.float32)
            saved = self._val_volumes_saved  # per-epoch set of already-dumped keys
            # Additivity: only the EF sweep dedups by (subject, phase) so both ED+ES are kept;
            # every other config keeps the original subject-only dedup (byte-identical behavior).
            _mri_ds = self._get_mri_dataset()
            sweeping = _mri_ds is not None and getattr(_mri_ds, "val_targets", None) is not None

            for b in range(B):
                raw_seq = seq_names[b] if b < len(seq_names) else f"unknown{b}"
                # seq_name is "mri_{mri_mode}_{rel_path}"; strip the first two parts to keep filenames short.
                subject = raw_seq.split("_", 2)[-1] if raw_seq.startswith("mri_") else raw_seq
                t_val = int(t_targets[b].flatten()[0].item()) if t_targets is not None else -1
                # Sweep: dedup by (subject, phase) so ED+ES are both kept. Non-sweep: dedup by
                # subject (original behavior — one pair per subject). Overwritten in place each epoch.
                key = (subject, t_val) if sweeping else subject
                if key in saved:
                    continue
                subj_idx = len(saved)  # deterministic first-seen order
                saved.add(key)
                stem = f"subj{subj_idx:02d}_t{t_val:02d}_{subject}"
                nib.save(nib.Nifti1Image(V_canon[b], affine), os.path.join(out_dir, f"{stem}_pred.nii.gz"))
                nib.save(nib.Nifti1Image(V_gt[b], affine), os.path.join(out_dir, f"{stem}_gt.nii.gz"))
        except Exception as e:
            logging.warning(f"val-volume save failed (ignored): {e}")

    def _save_ef_volume(self, batch: Mapping, loss_dict: Mapping) -> None:
        """On EF-epochs, dump each reconstructed val volume to ef_tmp/pred/ in nnU-Net input
        format (X,Y,Z / 1.4,1.4,12 / _0000). The EF-sweep visits each subject at ED and ES, so
        both phases land here (no dedup). Filenames use the clean subject id (matches the CSV)."""
        if "V_canon" not in loss_dict:
            return
        try:
            import ef_eval
            mri_ds = self._get_mri_dataset()
            vt = getattr(mri_ds, "val_targets", None)
            if vt is None:
                return
            V_canon = loss_dict["V_canon"].detach().float().cpu().numpy()  # (B, D, H, W)
            seqs = batch.get("seq_index")
            for b in range(V_canon.shape[0]):
                si = int(seqs[b].flatten()[0].item())
                subj_idx, t = vt[si % len(vt)]
                subject = os.path.basename(os.path.dirname(mri_ds.subjects[subj_idx]))
                ef_eval.save_pred_volume(V_canon[b], self._ef_pred_dir, subject, int(t))
        except Exception as e:
            logging.warning(f"[ef] save pred volume failed (ignored): {e}")

    def _compute_and_log_ef(self, step: int) -> None:
        """End of an EF-epoch: one batched nnU-Net Task114 seg over the saved ED/ES pred
        volumes, then predicted-vs-GT EF slope/Spearman/MAE logged to wandb. Try/except-wrapped so
        a seg/subprocess failure never touches training."""
        try:
            import glob
            import shutil
            import ef_eval
            mri_ds = self._get_mri_dataset()
            vt = getattr(mri_ds, "val_targets", None)
            csv_path = getattr(mri_ds, "cardiac_phase_csv", None)
            if vt is None or csv_path is None:
                return
            n_vols = len(glob.glob(os.path.join(self._ef_pred_dir, "*_0000.nii.gz")))
            if n_vols == 0:
                logging.warning("[ef] no pred volumes written; skipping EF")
                return
            seg_dir = os.path.join(self.logging_conf.log_dir, "ef_tmp", "seg_pred")
            shutil.rmtree(seg_dir, ignore_errors=True)
            torch.cuda.empty_cache()  # release cached GPU mem so nnU-Net coexists with the model
            ef_eval.run_nnunet(self._ef_pred_dir, seg_dir)
            # (subject_id, ed, es) per val subject. vt = [all ED] + [all ES], so vt[i] and vt[i+N]
            # are the SAME subject's ED and ES (N = half the sweep length — self-consistent with vt).
            N = len(vt) // 2
            subjects_ed_es = [
                (os.path.basename(os.path.dirname(mri_ds.subjects[vt[i][0]])), vt[i][1], vt[i + N][1])
                for i in range(N)
            ]
            m = ef_eval.compute_ef_metrics(seg_dir, subjects_ed_es, csv_path)
            if m is None:
                logging.warning("[ef] too few valid subjects for EF correlation; skipping")
                return
            for k in ("slope", "spearman", "mae_pct"):
                self._log_scalar(f"val/ef/{k}", m[k], step)
            self._log_scalar("val/ef/n", m["n"], step)
            logging.info(f"[ef] epoch {self.epoch}: slope={m['slope']:.3f} "
                         f"spearman={m['spearman']:.3f} mae={m['mae_pct']:.2f}% "
                         f"n={m['n']} (skipped {m['n_skipped']})")
        except Exception as e:
            logging.warning(f"[ef] compute/log EF failed (ignored): {e}")

    def _log_augmentation_to_wandb(self, orig_images, aug_images, step: int) -> None:
        """Log a before/after panel of GPU augmentation for one training subject.

        Top row = original input slices, bottom row = the augmented slices the model
        actually trains on, so it's visible which augmentations are active and how
        strong they are (the caption lists the tier's ops + per-op probabilities).
        Diagnostic only: gated on aug enabled + the epoch cadence at the call
        site, and wrapped in try/except so it can never perturb training.
        """
        if not self.wandb_writer or orig_images is None or aug_images is None:
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs

            def _gray(t):  # (B,S,3,H,W) or (S,3,H,W) -> (S,H,W) in [0,1]
                t = t[0] if t.dim() == 5 else t
                t = t.detach().float().cpu()
                if t.min() < 0:                       # [-1,1] -> [0,1] if needed
                    t = (t + 1.0) / 2.0
                return t.clamp(0, 1).mean(dim=1).numpy()

            orig, aug = _gray(orig_images), _gray(aug_images)
            S = min(orig.shape[0], aug.shape[0], 6)   # a few slots is enough
            # Caption reflects what is ACTUALLY applied (affine tier and/or breathing),
            # not a hardcoded description — so breathing-only runs aren't mislabeled.
            # batchaug doesn't expose which ops actually fired, so we can't enumerate the
            # realized subset; instead label affine ops as PROBABILISTIC candidates (each
            # fires per-subject at its tier probability) and rely on the bottom row being
            # the actual draw. Breathing is always-on per-slot (not probabilistic).
            applied = []
            if self.gpu_transforms is not None:
                applied.append(f"affine[{self._aug_tier}] (candidates flip/rotate/translate/scale "
                               f"+ noise/gamma/bias; each fires per-subject at its tier prob)")
            if self.respiratory_cfg.enable:
                rc = self.respiratory_cfg
                tilt_hi = rc.tilt_max_deg if rc.tilt_max_deg is not None else rc.direction_jitter_deg
                applied.append(f"breathing (ALWAYS-on: A={rc.amplitude_mm:.0f}+/-{rc.amplitude_jitter:.0f}mm/subj, "
                               f"n={rc.cos2n}, tilt<={tilt_hi:.0f}deg/subj)")
            caption = (f"GPU aug = {' + '.join(applied) if applied else 'none'} | top=original, "
                       f"bottom=ACTUAL draw applied (the realized affine subset is visible there); step={step}")
            fig = plt.figure(figsize=(1.6 * S + 0.4, 3.6), dpi=90)
            gs = _gs.GridSpec(2, S, wspace=0.04, hspace=0.12)
            fig.suptitle("Data augmentation -- original (top) vs augmented (bottom)", fontsize=8)
            for s in range(S):
                ax0 = fig.add_subplot(gs[0, s])
                ax0.imshow(orig[s], cmap="gray", vmin=0, vmax=1)
                ax0.set_xticks([]); ax0.set_yticks([])
                if s == 0:
                    ax0.set_ylabel("original", fontsize=8)
                ax1 = fig.add_subplot(gs[1, s])
                ax1.imshow(aug[s], cmap="gray", vmin=0, vmax=1)
                ax1.set_xticks([]); ax1.set_yticks([])
                if s == 0:
                    ax1.set_ylabel("augmented", fontsize=8)
            self.wandb_writer.log("media_others/Train_Visuals_Augmentation", wandb.Image(fig, caption=caption), step)
            plt.close(fig)
        except Exception as e:
            logging.warning(f"augmentation visual log failed (ignored): {e}")

    def _log_volume_and_dvf_to_wandb(self, batch: dict, name: str, step: int, caption: str,
                                     group: str = "media_others"):
        """Log two figures per visual step (matching tools/test_sequential_sampling.py style):
          {name}_Volume : 4 rows × max(S,D) cols — input slices, V_gt, V_canon, signed diff (per z).
          {name}_DVF    : 4 rows × S cols       — input intensity + Δx/Δy/Δz per slot.
        Both use dpi=90 (≈0.6 MB each PNG). Logged under `{group}/`.
        """
        if not self.wandb_writer:
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs
            import numpy as np
        except ImportError:
            return

        # ── Volume figure (per-z grid) ─────────────────────────────────────
        if "V_canon" in batch and "V_gt" in batch:
            V_canon = batch["V_canon"][0].detach().float().cpu().numpy()   # (D, H, W)
            V_gt = batch["V_gt"][0].detach().float().cpu().numpy()
            D, _, _ = V_canon.shape
            # Input row from batch images (518² padded, OK as diagnostic).
            imgs = batch["images"][0].detach().float().cpu()                # (S, 3, H, W)
            if imgs.min() < 0:
                imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1).mean(dim=1).numpy()                     # (S, H, W) gray
            S = imgs.shape[0]
            diff = V_canon - V_gt
            v_vmax = float(max(V_canon.max(), V_gt.max(), 1e-3))
            ERR = 0.1  # signed-diff color range ±ERR for the V_canon-V_gt row
            n_cols = max(S, D)
            t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
            z_picks = batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None
            # Per-slot breathing (respiratory phase r + displacement magnitude |d| mm),
            # present only when respiratory augmentation is on.
            resp_dmag = (batch["resp_disp_mm"][0].float().norm(dim=-1).cpu().numpy()
                         if "resp_disp_mm" in batch else None)
            resp_r = batch["resp_r"][0].float().cpu().numpy() if "resp_r" in batch else None

            fig = plt.figure(figsize=(1.6 * n_cols + 1.6, 7.5), dpi=90)
            gs = _gs.GridSpec(4, n_cols + 1, width_ratios=[1.0] * n_cols + [0.05], wspace=0.04, hspace=0.18)
            fig.suptitle(f"Volumes — {caption}", fontsize=8)

            # Row 0: input slices
            for s in range(S):
                ax = fig.add_subplot(gs[0, s])
                ax.imshow(imgs[s], cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if t_picks is not None and z_picks is not None:
                    ttl = f"t={int(t_picks[s])}, z={int(z_picks[s])}"
                    if resp_r is not None:
                        ttl += f"\nr={resp_r[s]:.2f} |d|={resp_dmag[s]:.0f}mm"
                    ax.set_title(ttl, fontsize=7)
                if s == 0:
                    ax.set_ylabel("input slice", fontsize=8)
            for s in range(S, n_cols):
                fig.add_subplot(gs[0, s]).axis("off")
            fig.add_subplot(gs[0, n_cols]).axis("off")

            def _vol_row(r, vol, cmap, vmin, vmax, ylabel, show_titles=False):
                last_im = None
                for d in range(D):
                    ax = fig.add_subplot(gs[r, d])
                    last_im = ax.imshow(vol[d], cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                    if show_titles:
                        ax.set_title(f"z={d}", fontsize=7)
                    if d == 0:
                        ax.set_ylabel(ylabel, fontsize=8)
                for d in range(D, n_cols):
                    fig.add_subplot(gs[r, d]).axis("off")
                plt.colorbar(last_im, cax=fig.add_subplot(gs[r, n_cols]))

            _vol_row(1, V_gt,    "gray",   0,     v_vmax, "V_gt", show_titles=True)
            _vol_row(2, V_canon, "gray",   0,     v_vmax, "V_canon")
            _vol_row(3, diff,    "RdBu_r", -ERR,  ERR,    f"V_canon-V_gt\n(±{ERR})")

            self.wandb_writer.log(f"{group}/{name}_Volume", wandb.Image(fig, caption=caption), step)
            plt.close(fig)

        # ── DVF figure (per-slot Δx/Δy/Δz) ─────────────────────────────────
        # The model's predicted Δ is recovered as (pred_world_points - scanner_coords).
        pred_dvf = None
        if "pred_world_points" in batch and "scanner_coords" in batch:
            pred_dvf = (batch["pred_world_points"][0] - batch["scanner_coords"][0]).detach().float().cpu().numpy()
        if pred_dvf is not None:
            imgs = batch["images"][0].detach().float().cpu()
            if imgs.min() < 0:
                imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1).mean(dim=1).numpy()
            S = imgs.shape[0]
            t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
            z_picks = batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None
            resp_dmag = (batch["resp_disp_mm"][0].float().norm(dim=-1).cpu().numpy()
                         if "resp_disp_mm" in batch else None)
            resp_r = batch["resp_r"][0].float().cpu().numpy() if "resp_r" in batch else None
            p50 = float(np.percentile(np.abs(pred_dvf), 50))
            p95 = float(np.percentile(np.abs(pred_dvf), 95))
            p99 = float(np.percentile(np.abs(pred_dvf), 99))
            # Normalized [-1,1] residual → physical mm (align_corners convention:
            # 1 norm unit = (size-1)/2 * spacing). In-plane (256 vox @1.4mm) and
            # through-plane (12 vox @12mm) have very different mm/norm, so the colorbars
            # are per-axis (a single norm range would make Δz look ~4x bigger than it is).
            IN_PLANE_MM = (256 - 1) / 2.0 * 1.4      # ≈178.5 mm per norm unit (Δx, Δy)
            THROUGH_MM = (12 - 1) / 2.0 * 12.0       # ≈66.0 mm per norm unit (Δz); 12mm true pitch
            IN_PLANE_R = 15.0                         # in-plane colorbar half-range (mm)
            THROUGH_R = 25.0                          # through-plane colorbar half-range (mm)

            fig = plt.figure(figsize=(1.6 * S + 1.6, 7.5), dpi=90)
            gs = _gs.GridSpec(4, S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.18)
            fig.suptitle(
                f"DVF — {caption}    |Δ|(norm) p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}",
                fontsize=8,
            )
            rows = [
                ("input intensity", imgs,                          "gray",   0,           1.0,        True),
                ("Δx (mm)",         pred_dvf[..., 0] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
                ("Δy (mm)",         pred_dvf[..., 1] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
                ("Δz (mm)",         pred_dvf[..., 2] * THROUGH_MM,  "RdBu_r", -THROUGH_R,  THROUGH_R,  False),
            ]
            for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(rows):
                last_im = None
                for s in range(S):
                    ax = fig.add_subplot(gs[r, s])
                    last_im = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_xticks([]); ax.set_yticks([])
                    if is_top and t_picks is not None and z_picks is not None:
                        ttl = f"t={int(t_picks[s])}, z={int(z_picks[s])}"
                        if resp_r is not None:
                            ttl += f"\nr={resp_r[s]:.2f} |d|={resp_dmag[s]:.0f}mm"
                        ax.set_title(ttl, fontsize=7)
                    if s == 0:
                        ax.set_ylabel(lbl, fontsize=8)
                plt.colorbar(last_im, cax=fig.add_subplot(gs[r, S]))

            self.wandb_writer.log(f"{group}/{name}_DVF", wandb.Image(fig, caption=caption), step)
            plt.close(fig)

    # Subjects shown in the ED-vs-ES panel (the EF sweep reconstructs each at its ED and ES).
    _ED_ES_SUBJECTS = (0, 7, 14, 21)

    def _stash_ed_es(self, batch: Mapping, loss_dict: Mapping) -> None:
        """During val, capture per-z ED and ES reconstructions for the chosen subjects, keyed by
        (subject → role). Rendered together at end of epoch by _log_ed_es_panels. Sweep-only."""
        if "V_canon" not in loss_dict or "V_gt" not in loss_dict:
            return
        try:
            mri_ds = self._get_mri_dataset()
            vt = getattr(mri_ds, "val_targets", None)
            i = int(self._val_iter)
            if vt is None or i >= len(vt):
                return
            subj_idx = vt[i][0]
            if subj_idx not in self._ED_ES_SUBJECTS:
                return
            role = "ED" if i < len(vt) // 2 else "ES"           # blocked layout: all ED then all ES
            imgs = batch["images"][0].detach().float().cpu()    # (S, 3, H, W)
            if imgs.min() < 0:
                imgs = (imgs + 1.0) / 2.0
            imgs = imgs.clamp(0, 1).mean(dim=1).numpy()         # (S, H, W) gray
            self._ed_es_stash.setdefault(subj_idx, {})[role] = {
                "images": imgs,
                "V_gt": loss_dict["V_gt"][0].detach().float().cpu().numpy(),      # (D, H, W)
                "V_canon": loss_dict["V_canon"][0].detach().float().cpu().numpy(),
                "t": int(batch["t_target"][0].flatten()[0].item()) if "t_target" in batch else -1,
                "timesteps": batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None,
                "slices": batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None,
            }
        except Exception as e:
            logging.warning(f"[ed/es] stash failed (ignored): {e}")

    def _log_ed_es_panels(self, step: int) -> None:
        """Render one 6-row ED-vs-ES figure per chosen subject that has both phases stashed:
        rows = input(ED) / V_gt(ED) / V_canon(ED) / input(ES) / V_gt(ES) / V_canon(ES);
        columns = the S input slices (input rows) or the D z-planes (volume rows). Per-z (NOT the
        mid plane, which is the reference slot — the model does nothing there). Contraction reads as
        the ED cavities shrinking to the ES cavities across planes."""
        if not self.wandb_writer:
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs
        except ImportError:
            return
        for subj_idx in self._ED_ES_SUBJECTS:
            rec = self._ed_es_stash.get(subj_idx)
            if not rec or "ED" not in rec or "ES" not in rec:
                continue
            fig = None
            try:
                ed, es = rec["ED"], rec["ES"]
                D = ed["V_gt"].shape[0]
                S = ed["images"].shape[0]
                n_cols = max(S, D)
                vmax = float(max(ed["V_gt"].max(), ed["V_canon"].max(),
                                 es["V_gt"].max(), es["V_canon"].max(), 1e-3))
                # (row-label, kind, data, phase-record) — kind "in"=S input slices, "vol"=D z-planes.
                rows = [
                    ("input ED", "in", ed["images"], ed),
                    ("V_gt ED", "vol", ed["V_gt"], ed),
                    ("V_canon ED", "vol", ed["V_canon"], ed),
                    ("input ES", "in", es["images"], es),
                    ("V_gt ES", "vol", es["V_gt"], es),
                    ("V_canon ES", "vol", es["V_canon"], es),
                ]
                fig = plt.figure(figsize=(1.1 * n_cols + 1.4, 1.5 * len(rows) + 0.8), dpi=90)
                gs = _gs.GridSpec(len(rows), n_cols, wspace=0.04, hspace=0.22)
                fig.suptitle(f"ED vs ES — val subj {subj_idx} (ED t={ed['t']}, ES t={es['t']}) — step={step}",
                             fontsize=9)
                for r, (label, kind, data, prec) in enumerate(rows):
                    ncol = S if kind == "in" else D
                    ts, zs = prec.get("timesteps"), prec.get("slices")
                    for c in range(ncol):
                        ax = fig.add_subplot(gs[r, c])
                        ax.imshow(data[c], cmap="gray", vmin=0, vmax=(1.0 if kind == "in" else vmax))
                        ax.set_xticks([]); ax.set_yticks([])
                        if c == 0:
                            ax.set_ylabel(label, fontsize=8)          # row label on the leftmost cell
                        # titles: (t,z) on BOTH input rows (ED at r0, ES at r3 — different samples,
                        # so both need their own coords), z on the two V_gt rows (r1, r4).
                        if kind == "in" and ts is not None and zs is not None:
                            ax.set_title(f"t{int(ts[c])} z{int(zs[c])}", fontsize=6)
                        elif kind == "vol" and r in (1, 4):
                            ax.set_title(f"z{c}", fontsize=6)
                    for c in range(ncol, n_cols):
                        fig.add_subplot(gs[r, c]).axis("off")
                self.wandb_writer.log(f"media_val_ED_ES/Val_Visuals_subj{subj_idx}_ED_ES",
                                      wandb.Image(fig, caption=f"per-z (mid plane = reference slot)"), step)
            except Exception as e:
                logging.warning(f"[ed/es] panel subj {subj_idx} failed (ignored): {e}")
            finally:
                if fig is not None:
                    plt.close(fig)                                    # never leak a figure on the error path

    def _log_refiner_viz_to_wandb(self, batch: dict, name: str, step: int, caption: str,
                                  group: str = "media_others"):
        """Log one figure `{group}/refiner_{name}_Volume` (4 rows × D cols): V_gt, V_canon (raw
        splat), V_refined (refiner output), and the refined signed-diff (V_refined - V_gt) per z.
        Purely additive: returns immediately unless the refiner ran (`V_refined` present), so it
        NEVER fires — and never touches the existing panels — when the refiner is off.
        """
        if not self.wandb_writer or "V_refined" not in batch or "V_gt" not in batch:
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs
        except ImportError:
            return

        V_refined = batch["V_refined"][0].detach().float().cpu().numpy()  # (D, H, W)
        V_gt = batch["V_gt"][0].detach().float().cpu().numpy()
        V_canon = (batch["V_canon"][0].detach().float().cpu().numpy()
                   if "V_canon" in batch else V_refined)
        D, _, _ = V_refined.shape
        v_vmax = float(max(V_refined.max(), V_canon.max(), V_gt.max(), 1e-3))
        ERR = 0.1
        diff = V_refined - V_gt

        fig = plt.figure(figsize=(1.6 * D + 1.6, 7.5), dpi=90)
        gs = _gs.GridSpec(4, D + 1, width_ratios=[1.0] * D + [0.05], wspace=0.04, hspace=0.18)
        fig.suptitle(f"Refiner — {caption}", fontsize=8)

        def _row(r, vol, cmap, vmin, vmax, ylabel, titles=False):
            last_im = None
            for d in range(D):
                ax = fig.add_subplot(gs[r, d])
                last_im = ax.imshow(vol[d], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if titles:
                    ax.set_title(f"z={d}", fontsize=7)
                if d == 0:
                    ax.set_ylabel(ylabel, fontsize=8)
            plt.colorbar(last_im, cax=fig.add_subplot(gs[r, D]))

        _row(0, V_gt,      "gray",   0,    v_vmax, "V_gt", titles=True)
        _row(1, V_canon,   "gray",   0,    v_vmax, "V_canon (splat)")
        _row(2, V_refined, "gray",   0,    v_vmax, "V_refined")
        _row(3, diff,      "RdBu_r", -ERR, ERR,    f"V_refined-V_gt\n(±{ERR})")

        self.wandb_writer.log(f"{group}/refiner_{name}_Volume", wandb.Image(fig, caption=caption), step)
        plt.close(fig)

    def _log_lookup_to_wandb(self, batch: dict, name: str, step: int, caption: str,
                             group: str = "media_others"):
        """Round-trip / analysis-by-synthesis panel (val-only, GT-referenced).

        For a few input slices spread across depth, sample the reconstruction (V_canon) AND the GT
        volume (V_gt) back at the model's predicted per-pixel coords `p = pred_world_points`, and show
        them beside the raw input slice + the |V_canon−V_gt|@p error map. Reveals WHERE detail is lost
        (renderer softening vs reconstruction error). Rows are chosen by z-depth (not slot index) so
        frames-per-slice>1 doesn't collapse/duplicate depths.

        Column relation: col1 (input I) ≈ col2 (V_canon@pred) BY CONSTRUCTION — sampling the recon
        back where each input pixel was splatted recovers that pixel's intensity, so their gap is the
        renderer's coverage-averaging blur. col2 (V_canon@pred) ≈ col3 (V_gt@pred) BY TRAINING — the
        L1 loss drives V_canon→V_gt, so their gap (the |·| error map, col4) is the reconstruction
        error. The reference row (slot 0, Δ≈0, t_slot=t_target) is the phase-matched control where
        col1=col3 directly; non-reference rows sample V_gt at the WARPED coords (pred=scanner+Δ), so
        the observed→target phase geometry is absorbed by Δ and col1≈col3 holds where the warp is right.
        """
        if not self.wandb_writer:
            return
        needed = ("V_canon", "V_gt", "pred_world_points", "images", "slice_indices")
        if any(k not in batch for k in needed):
            return
        try:
            import wandb
            import matplotlib.pyplot as plt
            from matplotlib import gridspec as _gs
            import numpy as np
            from vggt.utils.splat import sample_volume
        except ImportError:
            return

        V_canon = batch["V_canon"][0].detach().float()                    # (D, H, W)
        V_gt = batch["V_gt"][0].detach().float()
        wp = batch["pred_world_points"][0].detach().float()               # (S, Hs, Ws, 3)
        imgs = batch["images"][0].detach().float().cpu()                  # (S, 3, Hs, Ws)
        if imgs.min() < 0:
            imgs = (imgs + 1.0) / 2.0
        imgs = imgs.clamp(0, 1).mean(dim=1).numpy()                       # (S, Hs, Ws) gray
        z_picks = batch["slice_indices"][0].cpu().numpy()                 # (S,)
        t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
        S, Hs, Ws = imgs.shape

        # Rows: ≤4 slots at evenly-spaced DISTINCT z-depths (robust to frames-per-slice>1). When
        # slot 0 is the target-phase reference (reference_slot), pin it as the FIRST row — a phase-
        # matched control (Δ≈0, so its round-trip should be near-perfect), always shown.
        present = np.unique(z_picks)
        n_pick = min(4, len(present))
        zsel = present[np.linspace(0, len(present) - 1, n_pick).round().astype(int)]
        slots = [int(np.where(z_picks == zt)[0][0]) for zt in zsel]       # first slot at each z
        ref_slot = 0 if getattr(self, "reference_slot", False) else None
        if ref_slot is not None:                                          # pin reference first, ≤4 total
            slots = ([ref_slot] + [s for s in slots if s != ref_slot])[:4]

        vmax = float(max(V_canon.max().item(), V_gt.max().item(), 1e-3))
        ERR = 0.1
        dev = V_canon.device

        fig = plt.figure(figsize=(4 * 2.6, len(slots) * 2.6 + 0.6), dpi=110)
        try:
            gs = _gs.GridSpec(len(slots), 4, wspace=0.06, hspace=0.14)
            fig.suptitle(f"Lookup (round-trip @pred) — {caption}", fontsize=8)
            col_titles = ["input I", "V_canon @ pred", "V_gt @ pred", "|V_canon−V_gt| @ pred"]

            for r, s in enumerate(slots):
                pos = wp[s].reshape(1, -1, 3).to(dev)                      # (1, Hs*Ws, 3)
                rc = sample_volume(V_canon.unsqueeze(0), pos).reshape(Hs, Ws).cpu().numpy()
                rg = sample_volume(V_gt.unsqueeze(0), pos).reshape(Hs, Ws).cpu().numpy()
                err = np.abs(rc - rg)
                zt = int(z_picks[s]); tt = int(t_picks[s]) if t_picks is not None else -1
                is_ref = (ref_slot is not None and s == ref_slot)
                cells = [
                    (imgs[s], "gray",   0, 1.0),
                    (rc,      "gray",   0, vmax),
                    (rg,      "gray",   0, vmax),
                    (err,     "magma",  0, ERR),
                ]
                for c, (data, cmap, vmin, vm) in enumerate(cells):
                    ax = fig.add_subplot(gs[r, c])
                    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vm)
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0:
                        ax.set_title(col_titles[c], fontsize=7)
                    if c == 0:
                        lbl = ("REF " if is_ref else "") + f"slot{s}\nt={tt} z={zt}"
                        ax.set_ylabel(lbl, fontsize=7)

            self.wandb_writer.log(f"{group}/{name}_Lookup", wandb.Image(fig, caption=caption), step)
        finally:
            plt.close(fig)

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
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, 0)

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
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

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
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
        phase = "val"

        # Fresh per-phase accumulators for this val epoch (diagnostic only; train unaffected).
        self._per_phase_val_psnr_full = defaultdict(list)
        self._per_phase_val_psnr_bbox = defaultdict(list)
        self._per_phase_val_psnr_motion = defaultdict(list)
        # Refiner per-phase accumulators (only populated when the refiner ran; additive).
        self._per_phase_val_psnr_bbox_refined = defaultdict(list)
        self._per_phase_val_psnr_motion_refined = defaultdict(list)
        # Per-epoch val iter, read by _log_tb_visuals so subject-specific visuals fire every
        # epoch (not just the first one — self.steps["val"] is monotonic and resumes carry it).
        self._val_iter = 0
        # Per-epoch dedup set so _save_val_volumes writes one pred+gt pair per subject
        # (not one per val iteration), overwritten in place each epoch.
        self._val_volumes_saved = set()

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

        # Fixed-phase mode: t_target is constant, so the val loop's only source of
        # variety (cycling t_target = seq_index % T) collapses — iterating past one
        # pass over the val subjects just re-evaluates byte-identical (subject, phase)
        # samples (val is deterministic). Cap to one pass = len(val subjects) so we
        # don't waste ~limit_val_batches/N_subj× redundant compute (e.g. 200→30).
        if self.t_target_fixed is not None:
            mri_ds = self._get_mri_dataset()
            n_val_subj = len(mri_ds.subjects) if mri_ds is not None else 0
            if 0 < n_val_subj < limit_val_batches:
                logging.info(
                    f"Fixed-phase val (t_target_fixed={self.t_target_fixed}): capping "
                    f"limit_val_batches {limit_val_batches}→{n_val_subj} (one deterministic "
                    f"pass over val subjects; more is redundant)."
                )
                limit_val_batches = n_val_subj

        # EF sweep: cap to the explicit (subject, phase) list length so each pair is hit once.
        _mri_ds = self._get_mri_dataset()
        if _mri_ds is not None and getattr(_mri_ds, "val_targets", None) is not None:
            limit_val_batches = len(_mri_ds.val_targets)

        # Predicted-EF metric: is this an EF-epoch? (opt-in, val-only, cadence-gated,
        # requires the ED/ES sweep). If so, reset the pred dir; volumes are dumped per batch below.
        self._ef_this_epoch = bool(
            getattr(self.logging_conf, "ef_eval_enable", False)
            and _mri_ds is not None and getattr(_mri_ds, "val_targets", None) is not None
            and self.epoch % max(1, getattr(self.logging_conf, "ef_eval_every_n_val_epochs", 5)) == 0
        )
        if self._ef_this_epoch:
            import shutil
            self._ef_pred_dir = os.path.join(self.logging_conf.log_dir, "ef_tmp", "pred")
            shutil.rmtree(self._ef_pred_dir, ignore_errors=True)
            safe_makedirs(self._ef_pred_dir)

        # ED-vs-ES panel: on visual epochs, stash ED+ES reconstructions for the chosen subjects
        # (needs the sweep so each subject is hit at both phases). Rendered at end of epoch.
        self._viz_ed_es = bool(
            getattr(self.logging_conf, "log_visuals", True)
            and _mri_ds is not None and getattr(_mri_ds, "val_targets", None) is not None
            and self.epoch % max(1, getattr(self.logging_conf, "filmstrip_every_n_val_epochs", 5)) == 0
        )
        if self._viz_ed_es:
            self._ed_es_stash = {}

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            self._val_iter = data_iter

            # measure data loading time
            data_time.update(time.time() - end)

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)
            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            # Val never AFFINE-augments, but respiratory (if enabled) applies
            # deterministically per seq_index so val measures the real corrupted->clean task.
            batch = gpu_augment_batch(
                batch, self.val_gpu_transforms, self.device,
                respiratory_cfg=self.respiratory_cfg, train=False)
            if data_iter == 0:
                self._log_resp_disp_scalar(batch, self.steps["train"], "val")

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

            self._save_val_volumes(batch, val_loss_dict)
            if getattr(self, "_ef_this_epoch", False):
                self._save_ef_volume(batch, val_loss_dict)
            if getattr(self, "_viz_ed_es", False):
                self._stash_ed_es(batch, val_loss_dict)

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
        current_train_step = self.steps["train"]
        prefix = f"Loss/{phase}_"
        for name, meter in loss_meters.items():
            raw_name = name[len(prefix):] if name.startswith(prefix) else name
            self._log_scalar(self._scalar_name("val", raw_name), meter.avg, current_train_step)

        # ── Per-phase val PSNR (only when t_target is varying) ──
        # Metric name bakes in n and the identity baseline:
        #   val_psnr/t{k}_n{n}_base{b:.1f}
        # With deterministic stratified val, n is constant (3 for t=0..5, 2 for t=6..11)
        # so each phase keeps one stable panel. If val ever loses determinism, n drifts
        # and new panels appear — that drift is the smoke alarm.
        # Multi-phase mode logs TWO parallel namespaces:
        #   val/psnr/full/t{k}_n{n}_base{b:.1f}   averages over the whole cube
        #   val/psnr/bbox/t{k}_n{n}_base{b:.1f}   averages over the subject's geometric content region only
        # Both go to wandb so you can sanity-check that they track each other; large divergence
        # means something's off (e.g., model hallucinating outside the FOV).
        for namespace, accum, baseline_per_phase, baseline_mean in [
            ("val/psnr/full", self._per_phase_val_psnr_full,
             self._identity_baseline_full_per_phase, self._identity_baseline_full_mean),
            ("val/psnr/bbox", self._per_phase_val_psnr_bbox,
             self._identity_baseline_bbox_per_phase, self._identity_baseline_bbox_mean),
            ("val/psnr/motion", self._per_phase_val_psnr_motion,
             self._identity_baseline_motion_per_phase, self._identity_baseline_motion_mean),
            # Refiner panels — additive; empty (skipped) unless the refiner ran. Same
            # identity baselines as their V_canon counterparts so val/psnr/bbox vs
            # val/psnr/bbox_refined are directly comparable.
            ("val/psnr/bbox_refined", self._per_phase_val_psnr_bbox_refined,
             self._identity_baseline_bbox_per_phase, self._identity_baseline_bbox_mean),
            ("val/psnr/motion_refined", self._per_phase_val_psnr_motion_refined,
             self._identity_baseline_motion_per_phase, self._identity_baseline_motion_mean),
        ]:
            if len(accum) == 0:
                continue
            # Single-phase runs log ONLY the motion panels: full/bbox already go to the
            # standard Loss/val_metric_* meters, so per-phase full/bbox would be redundant
            # (that's why fixed-phase originally skipped this loop entirely). Motion is the
            # one metric the standard meters don't cover, so it must still be logged.
            if self.t_target_fixed is not None and not namespace.startswith("val/psnr/motion"):
                continue
            try:
                all_psnrs = []
                per_phase_means = []
                for t in sorted(accum.keys()):
                    psnrs = accum[t]
                    all_psnrs.extend(psnrs)
                    n = len(psnrs)
                    phase_mean = float(sum(psnrs) / n)
                    per_phase_means.append((t, phase_mean))
                    baseline = (baseline_per_phase or {}).get(t)
                    base_tag = f"_base{baseline:.1f}" if baseline is not None else ""
                    self._log_scalar(
                        f"{namespace}/t{t}_n{n}{base_tag}",
                        phase_mean,
                        current_train_step,
                    )
                n_total = len(all_psnrs)
                overall_mean = float(sum(all_psnrs) / n_total)
                base_tag = f"_base{baseline_mean:.1f}" if baseline_mean is not None else ""
                self._log_scalar(
                    f"{namespace}/mean_n{n_total}{base_tag}",
                    overall_mean,
                    current_train_step,
                )
                # Also emit to the TEXT log (slurm log), not just wandb, so per-phase
                # progress and the identity-baseline delta are readable straight from the
                # log file. Δ>0 means the model beats the no-motion-correction baseline.
                per_phase_str = " ".join(f"t{t}={m:.2f}" for t, m in per_phase_means)
                if baseline_mean is not None:
                    head = f"mean={overall_mean:.2f} (Δ={overall_mean - baseline_mean:+.2f} vs identity {baseline_mean:.2f})"
                else:
                    head = f"mean={overall_mean:.2f}"
                logging.info(f"[val per-phase @ step {current_train_step}] {namespace}: {head} | {per_phase_str}")
            except Exception as e:
                logging.warning(f"per-phase {namespace} log failed (ignored): {e}")

        # ── Cardiac-cycle filmstrip (every N val epochs) ──
        # Useful in BOTH modes: in multi-phase it's the qualitative proof of
        # cross-phase reconstruction; in fixed-ED it shows what the model does at
        # phases it wasn't trained on (degenerate or not — diagnostic).
        filmstrip_every_n = getattr(self.logging_conf, "filmstrip_every_n_val_epochs", 5)
        if self.epoch % filmstrip_every_n == 0:
            self._log_cardiac_cycle_filmstrip(current_train_step)

        # ED-vs-ES per-subject panels (per-z; from the sweep's ED+ES reconstructions)
        if getattr(self, "_viz_ed_es", False):
            self._log_ed_es_panels(current_train_step)

        # Predicted-EF metric (nnU-Net seg of the ED/ES pred volumes → slope/Spearman).
        if getattr(self, "_ef_this_epoch", False):
            self._compute_and_log_ef(current_train_step)

        logging.info(f"Validation Epoch {self.epoch} complete. Logged averages at train step {current_train_step}")

        return True

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        for config in self.gradient_clipper.configs:
            # Skip ONLY the refiner clip group when it has no params (enable_refiner=false),
            # so an OFF run's console/meters stay byte-identical to before the refiner existed.
            # NB: don't skip other empty groups — the aggregator group is also always-empty in
            # mri runs (fully frozen) yet its `Grad/aggregator: 0.0000` meter was historically
            # created + displayed; preserve that.
            if "refiner" in config["module_names"]:
                has_refiner = any(p.requires_grad and "refiner" in n
                                  for n, p in self.model.named_parameters())
                if not has_refiner:
                    continue
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

            with torch.cuda.amp.autocast(enabled=False):
                batch = self._process_batch(batch)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            # GPU augmentation (train only; identity passthrough when self.gpu_transforms is None).
            # On the log cadence, snapshot the pre-aug input slices (read-only clone) so we
            # can log a before/after augmentation example. This never alters the batch the
            # model trains on — gpu_augment_batch runs identically either way.
            _aug_log = (
                (self.gpu_transforms is not None or self.respiratory_cfg.enable)
                and self.logging_conf.log_visuals and data_iter == 0
                and self.epoch % max(1, getattr(self.logging_conf, "filmstrip_every_n_val_epochs", 5)) == 0
            )
            _orig_images = batch["images"].detach().clone() if (_aug_log and "images" in batch) else None
            batch = gpu_augment_batch(
                batch, self.gpu_transforms, self.device,
                respiratory_cfg=self.respiratory_cfg, train=True,
                resp_generator=self.resp_generator)
            if _aug_log:
                self._log_augmentation_to_wandb(_orig_images, batch.get("images"), self.steps["train"])
            if data_iter == 0:
                self._log_resp_disp_scalar(batch, self.steps["train"], "train")

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
                                f"train/optim/{optim_prefix}{option}",
                                param_group[option],
                                self.steps[phase],
                            )
                self._log_scalar(
                    "train/optim/where",
                    self.where,
                    self.steps[phase],
                )
                # Fractional epoch (floor = integer epoch) so cross-run curves can be
                # aligned on training progress, not just the global optimizer step.
                self._log_scalar(
                    "train/optim/epoch",
                    exact_epoch,
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
                        # Logged under train/optim/ alongside lr + where (gradient norms are optimizer diagnostics).
                        self._log_scalar(f"train/optim/grad_{key}", grad_norm, self.steps[phase])

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
            # Single-GPU: no DDP, so no gradient sync to defer — no_sync() was a no-op anyway.
            grad_accum_context = contextlib.nullcontext()

            with grad_accum_context:
                with torch.cuda.amp.autocast(
                    enabled=self.optim_conf.amp.enabled,
                    dtype=amp_type,
                ):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_loss_objective"
                batch_size = chunked_batch["images"].shape[0]

                if not math.isfinite(loss.item()):
                    self._nan_batch_count += 1
                    logging.error(
                        f"Loss is {loss.item()} (phase={phase}, step={self.steps[phase]}, "
                        f"cumulative_nan_batches={self._nan_batch_count}); skipping backward."
                    )
                    self._log_scalar(
                        "train/optim/nan_batches_cumulative",
                        float(self._nan_batch_count),
                        self.steps[phase],
                    )
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss.item(), batch_size)

    def _process_batch(self, batch: Mapping):
        # Passthrough hook. The legacy SfM batch-repetition (repeat_batch) and camera/point
        # normalization (normalize_points) were removed — both were gated off for the MRI
        # pipeline, so this returns the batch unchanged.
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
        batch_size = data["images"].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in loss_meters:
                    loss_meters[meter_key].update(value, batch_size)

                # Only log batch-level scalars for training to avoid step collision and noise
                if phase == "train" and step % self.logging_conf.log_freq == 0:
                    self._log_scalar(self._scalar_name("train", key), value, step)

        # ── Val-only diagnostic: per-phase PSNR accumulation (gated, never touches train) ──
        # Compute per-sample PSNR from V_canon/V_gt and bucket by the batch's t_target.
        # Runs in BOTH phase modes: multi-phase buckets across all 12 t_targets; single-phase
        # (t_target_fixed) buckets everything under its one fixed phase. The logging side then
        # selects which panels to emit — single-phase logs only val/psnr/motion (see the gate below).
        # All conditions must hold; if anything raises, we log a warning and continue —
        # diagnostics must NEVER crash training.
        if (phase == "val"
                and "V_canon" in data and "V_gt" in data and "t_target" in data):
            try:
                from loss import compute_motion_mask
                V_canon = data["V_canon"]
                V_gt = data["V_gt"]
                t_targets = data["t_target"]
                bboxes = data.get("anatomy_bbox", None)   # (B, 6) int64 or None
                # Motion mask from the full phase bundle (post-aug; val never augments).
                motion_masks = compute_motion_mask(data["phases"]) if "phases" in data else None
                B = V_canon.shape[0]
                for b in range(B):
                    Vc = V_canon[b].float()
                    Vg = V_gt[b].float()
                    mse_full = (Vc - Vg).pow(2).mean()
                    psnr_full = 10.0 * torch.log10(1.0 / mse_full.clamp(min=1e-10)).item()
                    tt = t_targets[b]
                    t = int(tt.item() if tt.ndim == 0 else tt.flatten()[0].item())
                    self._per_phase_val_psnr_full[t].append(psnr_full)
                    if bboxes is not None:
                        z0, z1, y0, y1, x0, x1 = [int(v) for v in bboxes[b].tolist()]
                        if (z1 > z0) and (y1 > y0) and (x1 > x0):
                            Vc_bb = Vc[z0:z1, y0:y1, x0:x1]
                            Vg_bb = Vg[z0:z1, y0:y1, x0:x1]
                        else:
                            Vc_bb, Vg_bb = Vc, Vg
                        mse_bbox = (Vc_bb - Vg_bb).pow(2).mean().clamp(min=1e-10)
                        psnr_bbox = (10.0 * torch.log10(1.0 / mse_bbox)).item()
                        self._per_phase_val_psnr_bbox[t].append(psnr_bbox)
                    if motion_masks is not None:
                        m = motion_masks[b]
                        if bool(m.any()):
                            Vc_m = Vc[m]
                            Vg_m = Vg[m]
                            mse_motion = (Vc_m - Vg_m).pow(2).mean().clamp(min=1e-10)
                            psnr_motion = (10.0 * torch.log10(1.0 / mse_motion)).item()
                            self._per_phase_val_psnr_motion[t].append(psnr_motion)
                    # Refiner per-phase PSNR — additive, only when the refiner ran.
                    if "V_refined" in data:
                        Vr = data["V_refined"][b].float()
                        if bboxes is not None:
                            z0, z1, y0, y1, x0, x1 = [int(v) for v in bboxes[b].tolist()]
                            if (z1 > z0) and (y1 > y0) and (x1 > x0):
                                Vr_bb, Vg_bb = Vr[z0:z1, y0:y1, x0:x1], Vg[z0:z1, y0:y1, x0:x1]
                            else:
                                Vr_bb, Vg_bb = Vr, Vg
                            mse_rb = (Vr_bb - Vg_bb).pow(2).mean().clamp(min=1e-10)
                            self._per_phase_val_psnr_bbox_refined[t].append((10.0 * torch.log10(1.0 / mse_rb)).item())
                        if motion_masks is not None and bool(motion_masks[b].any()):
                            mm = motion_masks[b]
                            mse_rm = (Vr[mm] - Vg[mm]).pow(2).mean().clamp(min=1e-10)
                            self._per_phase_val_psnr_motion_refined[t].append((10.0 * torch.log10(1.0 / mse_rm)).item())
            except Exception as e:
                logging.warning(f"per-phase PSNR accumulation failed (ignored): {e}")

        # Refiner train scalars — additive, logged directly (NOT via the meter allowlist, so
        # an OFF run's console/meters are byte-identical to today). Only fires when the refiner
        # ran (keys present). Val is covered by the per-phase val_psnr_*_refined panels above.
        if phase == "train" and step % self.logging_conf.log_freq == 0:
            for key in ("loss_refiner", "loss_refiner_ssim", "metric_ssim_2d_refined",
                        "metric_psnr_3d_full_refined",
                        "metric_psnr_3d_bbox_refined", "metric_psnr_3d_motion_refined"):
                if key in data:
                    val = data[key].item() if torch.is_tensor(data[key]) else data[key]
                    self._log_scalar(self._scalar_name("train", key), val, step)

        # Log Frame and Slice selection for the first few slots (if available).
        # NOTE: with the decoupled-target design, slot 0 is NO LONGER the t_target slice —
        # input phases are sampled independently of t_target. These are just the raw (t, z)
        # picks for the first slots, for sanity-checking the sampler.
        if phase == "train" and step % self.logging_conf.log_freq == 0:
            # data["timesteps"] and data["slice_indices"] are [B, S]
            if "timesteps" in data and "slice_indices" in data:
                ts = data["timesteps"]
                sls = data["slice_indices"]
                # S is the number of slots in the sequence
                S = ts.shape[1] if hasattr(ts, "shape") else len(ts[0])

                # Slot 0
                self._log_scalar("train/slice_selection/slot1_frame", ts[0, 0].item(), step)
                self._log_scalar("train/slice_selection/slot1_slice", sls[0, 0].item(), step)

                # Slot 1
                if S > 1:
                    self._log_scalar("train/slice_selection/slot2_frame", ts[0, 1].item(), step)
                    self._log_scalar("train/slice_selection/slot2_slice", sls[0, 1].item(), step)

                # Slot 2
                if S > 2:
                    self._log_scalar("train/slice_selection/slot3_frame", ts[0, 2].item(), step)
                    self._log_scalar("train/slice_selection/slot3_slice", sls[0, 2].item(), step)

                # Natural-anchor rate: mean number of input slots whose phase == t_target.
                # With decoupled sampling this is no longer guaranteed ≥1 — tracking it
                # surfaces how often the model gets a free zero-motion anchor (training
                # dynamics signal; ~65% have ≥1 at S=12, T=12).
                if "t_target" in data:
                    try:
                        tt = data["t_target"]  # (B, 1)
                        n_anchor = (ts == tt).float().sum(dim=1).mean().item()
                        self._log_scalar("train/slice_selection/n_slots_at_target", n_anchor, step)
                    except Exception as e:
                        logging.warning(f"n_slots_at_target log failed (ignored): {e}")

    def _log_tb_visuals(self, batch: Mapping, phase: str, step: int) -> None:
        """Logs image or video visualizations to TensorBoard."""

        # Scale frequency by accum_steps to prevent redundant logging of chunks
        freq = self.logging_conf.log_visual_frequency.get(phase, 0)
        if phase == "train":
            freq *= self.accum_steps

        # For validation, we use the training step to keep WandB monotonic
        log_step = step if phase == "train" else self.steps["train"]

        # Visual logging policy:
        #   train: log every `log_visual_frequency.train` steps, one random subject (whatever
        #          the shuffled dataloader pulled — diagnostic sampling).
        #   val:   log only at the val_steps listed below. With stratified val sampling
        #          (t_target = seq_index % T_total), val_step k → subject k → t_target = k % T.
        #          (0, 7) targets ED (t=0) + ES (t=7, empirically measured ES median across
        #          val subjects via LV-cavity bright-pixel count; histogram of ES across 30
        #          val subjects peaks at t=7-8). Adjust if the dataset's phase binning
        #          convention shifts. Filmstrip (every N val epochs) covers the full cycle
        #          for subject 0 so this snapshot choice isn't load-bearing.
        #          Kept in sync with the ED/ES panel subjects so the same subjects get both
        #          the Volume/DVF snapshot and the per-z ED/ES panel.
        VAL_VISUAL_SUBJECT_INDICES = self._ED_ES_SUBJECTS
        if phase == "train":
            should_log = freq > 0 and (step % freq == 0)
        else:
            # self._val_iter resets each val epoch — using self.steps["val"] (monotonic,
            # checkpointed) here would skip these visuals after the first epoch / on resume.
            should_log = self._val_iter in VAL_VISUAL_SUBJECT_INDICES
        if not (self.logging_conf.log_visuals and should_log and (self.logging_conf.visuals_keys_to_log is not None)):
            return

        if phase in self.logging_conf.visuals_keys_to_log:
            # Build caption (subject + per-slot z/t info + t_target).
            caption_parts = []
            if "seq_name" in batch:
                try:
                    raw_seq = batch["seq_name"][0]
                    # seq_name is "mri_{mri_mode}_{rel_path}"; strip to the bare subject id.
                    subject = raw_seq.split("_", 2)[-1] if raw_seq.startswith("mri_") else raw_seq
                    caption_parts.append(f"subj={subject}")
                except Exception:
                    pass
            t_target_val = None
            if "t_target" in batch:
                try:
                    t_target_val = int(batch["t_target"][0].item() if hasattr(batch["t_target"][0], "item") else batch["t_target"][0])
                    caption_parts.append(f"t_target={t_target_val}")
                except Exception:
                    pass
            if "slice_indices" in batch and "timesteps" in batch:
                try:
                    slices = batch["slice_indices"][0]
                    timesteps = batch["timesteps"][0]
                    S = slices.shape[0]
                    # Per-slot respiratory phase r + displacement |d| (mm), if breathing on.
                    resp = batch.get("resp_disp_mm")
                    dmag = resp[0].norm(dim=-1) if resp is not None else None  # (S,)
                    rphase = batch.get("resp_r")
                    rphase = rphase[0] if rphase is not None else None         # (S,)
                    def _slot(i):
                        s = f"f{i}: z={slices[i].item()}, t={timesteps[i].item()}"
                        if dmag is not None:
                            s += f", r={rphase[i].item():.2f}, |d|={dmag[i].item():.0f}mm"
                        return s
                    per_slot = " | ".join([_slot(i) for i in range(S)])
                    caption_parts.append(per_slot)
                except Exception:
                    pass
            caption_parts.append(f"step={log_step}")
            caption = "  ".join(caption_parts)

            # Wandb key prefix. Train stays in the media_others/ bucket; val gets a per-subject
            # section (media_val_subj{i}/) so the 4 val subjects' panels group together.
            if phase == "train":
                name, group = "Train_Visuals", "media_others"
            else:
                name, group = "Val_Visuals", f"media_val_subj{self._val_iter}"

            # Render both figures and log. Diagnostic only — a render error (e.g. a
            # shape regression in the per-slot r/|d| titles, or a matplotlib/wandb
            # hiccup) must NEVER crash training/validation, so guard the whole call.
            try:
                self._log_volume_and_dvf_to_wandb(batch, name, log_step, caption, group=group)
            except Exception as e:
                logging.warning(f"volume/DVF visual log failed (ignored): {e}")

            # Refiner panel — additive, only when the refiner ran (key present). Logs to the same
            # per-subject section (group/refiner_*), never touches the existing panels.
            try:
                self._log_refiner_viz_to_wandb(batch, name, log_step, caption, group=group)
            except Exception as e:
                logging.warning(f"refiner visual log failed (ignored): {e}")

            # Round-trip lookup panel — val-only (GT-referenced, in-distribution).
            if phase != "train":
                try:
                    self._log_lookup_to_wandb(batch, name, log_step, caption, group=group)
                except Exception as e:
                    logging.warning(f"lookup visual log failed (ignored): {e}")


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
