# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket

# --- Environment Variable Setup for Performance and Debugging ---
# Helps with memory fragmentation in PyTorch's memory allocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Specifies the threading layer for MKL, can prevent hangs in some environments.
os.environ["MKL_THREADING_LAYER"] = "GNU"
# Provides full Hydra stack traces on error for easier debugging.
os.environ["HYDRA_FULL_ERROR"] = "1"
# Enables asynchronous error handling for NCCL, which can prevent hangs.
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


import gc
import json
import logging
import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from data.gpu_aug import build_gpu_transforms, gpu_augment_batch
from data.respiratory import RespiratoryConfig
from train_utils.checkpoint import CheckpointSaver
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
from train_utils.freeze import freeze_modules
from train_utils.general import *
from train_utils.logging import setup_logging
from train_utils.notify import GradientCollapseAlarm, send_email
from train_utils.optimizer import construct_optimizers

# Fallback when `checkpoint.best_metric` is absent from a config. Heart-segmentation ROI
# PSNR — a real anatomical mask; see the rationale in default.yaml's checkpoint block.
_BEST_METRIC_DEFAULT = "metric_psnr_3d_heartseg"
from train_utils.run_log import RunLog, file_md5
from train_utils.val_logging import resp_offslab_stats, seq_index_to_subject
from trainer_viz import TrainerVizMixin


# Metrics sliced per source / per pitch each val epoch (docs/60). Heart-ROI first: it is
# the headline, and `hole_frac` is docs/38's VETO — stratifying the electors without it
# lets a per-source coverage regression pass the gate. `psnr_bbox` is kept last, demoted,
# as the continuity link to the pre-docs/60 series.
STRATA_METRICS = (
    "metric_psnr_seg_gain_db",     # dB above each subject's OWN floor — the only heart
                                   # number that is comparable across a mixed cohort
    "metric_recov_frac_seg",       # recovered fraction on the SAME mask, unclamped
    "metric_psnr_3d_heartseg",     # raw; fine per subject, cohort-composition-sensitive
    "metric_mae_3d_heartseg",
    "metric_hole_frac_heart",      # docs/38 VETO — must be stratified alongside the electors
    "metric_psnr_3d_bbox",         # demoted, kept as continuity with the pre-docs/60 series
)


class Trainer(TrainerVizMixin):
    """
    A single-GPU trainer. (The multi-GPU DDP apparatus was removed in 284992c —
    this repo trains on one GPU and the 1-process DDP wrap did nothing useful.)

    This class orchestrates the entire training and validation process, including:
    - Initializing the model, optimizers, loss functions, and data loaders.
    - Handling checkpointing for resuming training.
    - Executing the main training and validation loops.
    - Logging metrics and visualizations to wandb.
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
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initializes the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (wandb, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            device: "cuda" or "cpu".
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
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

        # On-disk run log (docs/60): wandb scalars mirrored to metrics.jsonl, plus the
        # per-subject val rows wandb never sees. Built first so _log_scalar can mirror
        # from the very first call.
        self.run_log = RunLog(self.logging_conf.log_dir)

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
        # Stratified val accumulator, keyed (axis, group, metric) → [per-subject values].
        # Cleared each val epoch by _log_val_strata. See _record_val_subject.
        self._val_strata = defaultdict(list)
        # Identity baseline per phase + aggregate mean, populated by _compute_identity_baseline
        # and baked into val_psnr metric names so each panel shows n and baseline in its title.
        self._identity_baseline_full_per_phase = None
        self._identity_baseline_full_mean = None
        self._identity_baseline_bbox_per_phase = None
        self._identity_baseline_bbox_mean = None
        self._identity_baseline_motion_per_phase = None
        self._identity_baseline_motion_mean = None
        # Cumulative count of training batches skipped due to non-finite loss. Without this,
        # NaN/Inf batches get swallowed by the early-return in _run_step_and_backward and
        # loss_meters silently undercount, making the wandb loss curve look healthier than it is.
        self._nan_batch_count = 0
        # Per-phase count of batches whose objective was non-finite when logging was
        # attempted. Distinct from _nan_batch_count (train-only, incremented later, and
        # about skipping the BACKWARD): this one is about keeping the meters clean, and it
        # covers val too. See _log_if_finite.
        self._nonfinite_logged = defaultdict(int)
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

        # One `run_meta.jsonl` line per PROCESS LAUNCH (docs/60). Written here, after the
        # checkpoint load, so `resumed_from_epoch` is known — a requeued run therefore
        # leaves one line per segment and a mid-run code edit shows as a changed git sha.
        self._write_run_meta()

        # Compute identity-Δ baseline once at startup.
        if self.mode in ["train", "val"]:
            self._compute_identity_baseline()
            # Motion-mask reference panel (3 val subjects). Data-derived + static across
            # training, so logged once here rather than every val epoch.
            self._log_motion_mask_example(self.steps.get("train", 0))

    def _write_run_meta(self):
        """Identify WHICH code and cohort produced this run's numbers. The git sha and the
        split md5 are the load-bearing fields, and neither is recoverable from wandb."""
        try:
            def _subjects(ds_attr):
                try:
                    inner = getattr(self, ds_attr).dataset.base_dataset.datasets[0]
                    return len(inner.subjects), getattr(inner, "split_file", None)
                except (AttributeError, IndexError, TypeError):
                    return None, None

            n_train, split_file = _subjects("train_dataset")
            n_val, val_split_file = _subjects("val_dataset")
            split_file = split_file or val_split_file
            manifest = os.path.join(
                os.path.dirname(split_file), "manifest.csv") if split_file else None

            # The val PROTOCOL, not just the cohort: cardiac_phase.csv decides which
            # (subject, t_target) pairs every val number is averaged over, so editing it
            # moves every metric and the identity floor with it.
            mri_ds = self._get_mri_dataset()
            ef_csv = getattr(mri_ds, "cardiac_phase_csv", None) if mri_ds else None
            # Bumped whenever the on-disk VOXELS change (slice-order flips, roll fixes,
            # ROI regeneration). split_md5 only hashes subject NAMES, so without this two
            # runs on different pixels look identical.
            try:
                from data.preprocess import cache_signature
                cache_sig = getattr(mri_ds, "cache_signature", cache_signature())
            except Exception:
                cache_sig = None

            wb = getattr(self.wandb_writer, "run", None)
            self.run_log.meta({
                "event": "launch",
                "log_dir": self.logging_conf.log_dir,
                "mode": self.mode,
                # `is not None`, not truthiness — epoch 0 is falsy, and a resume at epoch 0
                # would otherwise be indistinguishable from a cold start.
                "resumed_from_epoch": int(self.epoch) if self.epoch is not None else None,
                "steps_at_launch": dict(self.steps),
                "max_epochs": self.max_epochs,
                "seed": self.seed_value,
                "limit_train_batches": self.limit_train_batches,
                "limit_val_batches": self.limit_val_batches,
                "n_train_subjects": n_train,
                "n_val_subjects": n_val,
                "split_file": split_file,
                "split_md5": file_md5(split_file) if split_file else None,
                "manifest_md5": file_md5(manifest) if manifest else None,
                "cardiac_phase_csv": ef_csv,
                "cardiac_phase_md5": file_md5(ef_csv) if ef_csv else None,
                "data_cache_signature": cache_sig,
                "wandb_id": getattr(wb, "id", None),
                "wandb_url": getattr(wb, "url", None),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_node": os.environ.get("SLURMD_NODENAME") or socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "config": self._wandb_config,
            })
        except Exception as e:
            logging.warning(f"run_meta write failed (ignored): {e}")

    def _get_mri_dataset(self):
        """Walk the wrapper chain (DynamicTorchDataset → ComposedDataset → TupleIndexedDataset)
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
        # Read here, applied in _compile_attention_blocks() once the model exists.
        self.compile_attention_blocks = bool(
            cuda_conf.get("compile_attention_blocks", False)) if cuda_conf else False
        if torch.cuda.is_available():
            # Configure CUDA backend settings for performance
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

    def _compile_attention_blocks(self) -> None:
        """`torch.compile` the aggregator's frame/global attention blocks in place (docs/40).

        Compiles each Block individually rather than the whole model: the aggregator wraps every
        block in `torch.utils.checkpoint`, and a whole-model compile inlines through those
        wrappers and keeps all 48 blocks' activations live (>44 GB → OOM). Per-block compilation
        leaves the outer `checkpoint()` in eager Python, so checkpointing is fully preserved.

        Uses `nn.Module.compile()`, NOT `mod = torch.compile(mod)`: the former swaps the module's
        internal call implementation, so the module identity and `state_dict()` keys are unchanged
        and checkpoints stay interchangeable with eager runs. The latter would wrap the block in an
        `OptimizedModule` and prefix its keys with `_orig_mod.`.

        `dynamic=True` because S varies per subject (one_frame_per_slice ⇒ S = in-FOV plane count).
        """
        if not self.compile_attention_blocks:
            return
        # Gate on the device this run actually uses, NOT on cuda.is_available(): with
        # `device: cpu` on a GPU node the latter is True, so every block would compile
        # against the Inductor CPU backend (minutes of C++ codegen, or an outright failure)
        # on a run that works fine today, with no warning to explain the stall.
        if self.device.type != "cuda":
            logging.warning(f"compile_attention_blocks requested but device is "
                            f"{self.device}; skipping compilation.")
            return
        agg = getattr(self.model, "aggregator", None)
        if agg is None:
            logging.warning("compile_attention_blocks requested but model has no aggregator; skipping.")
            return
        n = 0
        for blocks in (getattr(agg, "frame_blocks", None), getattr(agg, "global_blocks", None)):
            for block in (blocks or []):
                block.compile(mode="default", dynamic=True)
                n += 1
        logging.info(f"torch.compile applied to {n} aggregator attention blocks "
                     "(mode=default, dynamic=True); first steps pay one-time compilation.")

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training."""
        logging.info(f"Resuming training from {ckpt_path}")

        # Stage the (immutable) base/seed weights onto node-local /tmp before loading —
        # torch.load's seeky small reads are pathologically slow on GPFS (~266s vs ~5s
        # for an ~8GB ckpt), so repeated loads (e.g. smoke runs) reuse one /tmp copy.
        # ONLY the configured resume_checkpoint_path (immutable) is staged; a run's own
        # save_dir checkpoint_last.pt (overwritten each requeue) is loaded directly to
        # avoid ever reusing a stale copy. Byte-identical (pure copy); see checkpoint_stage.py.
        load_path = ckpt_path
        resume_cfg = self.checkpoint_conf.resume_checkpoint_path
        if resume_cfg and os.path.abspath(ckpt_path) == os.path.abspath(resume_cfg):
            load_path = stage_checkpoint_to_local(ckpt_path)
        with g_pathmgr.open(load_path, "rb") as f:
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

        # Instantiate components from configs
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
        # docs/64 tripwire. Both pooled1337 runs sat at grad_aggregator < 1e-6 for ~70
        # epochs (dead ReLU in the DPT head) while grad_point looked healthy, so nothing
        # in the standard logging flagged it. One alarm per clipper group, keyed by the
        # same name the clipper reports.
        alarm_cfg = self.optim_conf.get("grad_collapse_alarm", None)
        self._grad_alarms = {}
        if alarm_cfg is None or alarm_cfg.get("enable", True):
            thr = float(alarm_cfg.get("threshold", 1e-6)) if alarm_cfg else 1e-6
            pat = int(alarm_cfg.get("patience", 200)) if alarm_cfg else 200
            watch = list(alarm_cfg.get("modules", ["aggregator"])) if alarm_cfg else ["aggregator"]
            for name in watch:
                self._grad_alarms[name] = GradientCollapseAlarm(
                    threshold=thr, patience=pat, name=name)
        # GradScaler only helps fp16 (prevents underflow). bf16 has fp32 range, so loss
        # scaling is dead weight — disable to keep the train loop honest.
        self.scaler = torch.amp.GradScaler("cuda",
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

        # Log model summary (before compilation, so it reports the eager module tree)
        model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
        model_summary(self.model, log_file=model_summary_path, logging_func=logging.info)
        logging.info(f"Model summary saved to {model_summary_path}")

        # Compile AFTER freezing: dynamo guards on requires_grad, so compiling first would
        # immediately recompile every block once the freeze flips those flags.
        self._compile_attention_blocks()

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

    def _maybe_save_best_checkpoint(self):
        """Keep a WEIGHTS-ONLY `checkpoint_best.pt` for the best val epoch so far.

        Why this exists (docs/64): both pooled1337 runs peaked around epoch 10-15 and then
        collapsed permanently at epoch ~17. With `save_freq=50` the only files on disk were
        `checkpoint_50.pt` and `checkpoint_last.pt` — BOTH post-collapse. The best weights
        either run ever produced were never written anywhere and are unrecoverable.

        Weights-only on purpose: a best checkpoint is for evaluating/deploying, never for
        resuming (`checkpoint_last.pt` is the resume path). That keeps it ~3.8 GB instead of
        ~8.9 GB — and this repo has already hit `Disk quota exceeded` mid-run — and it also
        sidesteps the docs/37 trap where `resume_checkpoint_path` restores `prev_epoch` and
        silently does zero training.
        """
        if not self.checkpoint_conf.get("save_best", True):
            return
        value = getattr(self, "_last_val_metric", None)
        if value is None or not math.isfinite(value):
            return
        higher_is_better = self.checkpoint_conf.get("best_metric_higher_is_better", True)
        prev = getattr(self, "_best_val_metric", None)
        improved = prev is None or (value > prev if higher_is_better else value < prev)
        if not improved:
            return
        self._best_val_metric = value
        try:
            path = os.path.join(self.checkpoint_conf.save_dir, "checkpoint_best.pt")
            safe_makedirs(self.checkpoint_conf.save_dir)
            tmp = path + ".tmp"
            # Write-then-rename: a kill mid-write must not destroy the previous best.
            torch.save({"model": self.model.state_dict(),
                        "prev_epoch": int(self.epoch),
                        "best_metric_name": self.checkpoint_conf.get(
                            "best_metric", _BEST_METRIC_DEFAULT),
                        "best_metric_value": float(value)}, tmp)
            os.replace(tmp, path)
            logging.info(f"[checkpoint] new best {self.checkpoint_conf.get('best_metric', _BEST_METRIC_DEFAULT)}"
                         f"={value:.4f} at epoch {int(self.epoch)} -> {path}")
        except Exception as e:
            logging.warning(f"[checkpoint] best-checkpoint save failed (ignored): {e}")

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
        """Log a scalar to wandb AND mirror it to `metrics.jsonl` (docs/60). This is the
        only scalar chokepoint, so nothing can reach wandb without reaching disk. The
        mirror is not gated on `wandb_writer` — offline runs still get a full record."""
        if self.wandb_writer:
            self.wandb_writer.log(name, value, step)
        run_log = getattr(self, "run_log", None)
        if run_log is not None:
            run_log.scalar(name, value, step, epoch=getattr(self, "epoch", None))

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

    def _log_resp_disp_scalar(self, batch, step: int, prefix: str):
        """Log the per-slot respiratory displacement magnitude (mm) as scalars under
        `{prefix}/resp/disp_mm_{mean,max}`. No-op when breathing is off (key absent).
        Read-only diagnostic — never affects training.

        NOT gated on `self.wandb_writer`: `_log_scalar` mirrors to disk too, and the
        off-slab numbers below are the one family that cannot be recovered after the fact
        (the corruption is applied on GPU and never persisted), so an offline run must
        still record them."""
        disp = batch.get("resp_disp_mm")
        if disp is None:
            return
        try:
            mag = disp.float().norm(dim=-1)  # (B, S) per-slot |d| in mm
            self._log_scalar(f"{prefix}/resp/disp_mm_mean", float(mag.mean().item()), step)
            self._log_scalar(f"{prefix}/resp/disp_mm_max", float(mag.max().item()), step)
            for k, v in resp_offslab_stats(batch, 0).items():
                self._log_scalar(f"{prefix}/resp/{k}", v, step)
        except Exception as e:
            logging.warning(f"resp_disp scalar log failed (ignored): {e}")

    def run(self):
        """Main entry point to start the training or validation process.

        Records how the run ENDED (docs/60). Without this, `run_meta.jsonl` has a launch
        line and nothing else, so a crashed run and a completed one are indistinguishable
        from the files — you have to guess from whether a `.tmp` checkpoint was left behind.
        A SIGUSR1 requeue calls `os._exit(0)` and so bypasses this; that segment is
        identified instead by the NEXT launch line carrying `resumed_from_epoch`.
        """
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        try:
            if self.mode == "train":
                self.run_train()
                # Optionally run a final validation after all training is done
                self.run_val()
            else:
                self.run_val()
        except BaseException as e:               # BaseException: also record KeyboardInterrupt
            self._log_exit("error", f"{type(e).__name__}: {e}")
            raise
        self._log_exit("completed")

    def _log_exit(self, status, error=None):
        try:
            self.run_log.meta({"event": "exit", "status": status, "error": error,
                               "final_epoch": int(self.epoch),
                               "steps": dict(self.steps)})
        except Exception as e:
            logging.warning(f"exit record failed (ignored): {e}")

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
                self._maybe_save_best_checkpoint()

            self.epoch += 1

        # Step back to the last epoch actually completed. NOTE: deliberately UNGUARDED.
        # A guard (`if ran_an_epoch`) was tried and reverted: when the loop never runs
        # it leaves self.epoch == max_epochs, an epoch that never happened, which flips
        # the trailing run_val()'s cadence gates (e.g. 200 % 5 == 0 fires the heavy
        # nnU-Net EF eval on a restarted-but-already-complete run, and 100 % 3 != 0
        # stops the filmstrip firing). The decremented value is the more accurate label
        # in that case too, so the plain decrement stays.
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
        # Per-epoch val iter, read by _log_visuals_to_wandb so subject-specific visuals fire every
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
            and self.epoch % max(1, getattr(self.logging_conf, "visual_panels_every_n_val_epochs", 3)) == 0
        )
        if self._viz_ed_es:
            self._ed_es_stash = {}

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            self._val_iter = data_iter

            # measure data loading time
            data_time.update(time.time() - end)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            # Val never AFFINE-augments, but respiratory (if enabled) applies
            # deterministically per seq_index so val measures the real corrupted->clean task.
            batch = gpu_augment_batch(
                batch, self.val_gpu_transforms, self.device,
                respiratory_cfg=self.respiratory_cfg, train=False)
            if data_iter == 0:
                self._log_resp_disp_scalar(batch, self.steps["train"], "val")

            amp_type = self._amp_dtype()

            # compute output
            with torch.no_grad():
                with torch.amp.autocast("cuda",
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
            # count==0 means EVERY val batch was non-finite and got skipped by
            # _log_if_finite. AverageMeter.avg is initialised to 0, so publishing it
            # would report a perfect 0.0 for a catastrophically broken epoch. Report NaN
            # instead — that is what this logged before the guard existed.
            value = meter.avg if meter.count else float("nan")
            self._log_scalar(self._scalar_name("val", raw_name), value, current_train_step)
            # Capture the best-checkpoint selection metric (default: heart-seg ROI PSNR, a
            # real anatomical mask — see default.yaml's checkpoint block). Deliberately not
            # bbox/full: those are dominated by static tissue the model gets for free and
            # look fine even when zero motion correction is happening.
            if raw_name == self.checkpoint_conf.get("best_metric", _BEST_METRIC_DEFAULT):
                self._last_val_metric = value

        # ── Per-phase val PSNR (only when t_target is varying) ──
        # Metric name bakes in n and the identity baseline:
        #   val_psnr/t{k}_n{n}_base{b:.1f}
        # With deterministic stratified val, n is constant per phase, so each phase keeps
        # one stable panel. If val ever loses determinism, n drifts and new panels appear —
        # that drift is the smoke alarm. (The old '3 for t=0..5, 2 for t=6..11' no longer
        # holds under ef_val_sweep, which visits each subject at its own ED/ES — docs/59 F20.)
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

        # ── Stratified val means (per source / per pitch bucket) ──
        self._log_val_strata(current_train_step)

        # ── Cardiac-cycle filmstrip (every N val epochs) ──
        # Useful in BOTH modes: in multi-phase it's the qualitative proof of
        # cross-phase reconstruction; in fixed-ED it shows what the model does at
        # phases it wasn't trained on (degenerate or not — diagnostic).
        filmstrip_every_n = max(1, getattr(self.logging_conf, "filmstrip_every_n_val_epochs", 5))
        if self.epoch % filmstrip_every_n == 0:
            for subj_idx in self._ED_ES_SUBJECTS:
                self._log_cardiac_cycle_filmstrip(current_train_step, subj_idx)

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

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            # GPU augmentation (train only; identity passthrough when self.gpu_transforms is None).
            # On the log cadence, snapshot the pre-aug input slices (read-only clone) so we
            # can log a before/after augmentation example. This never alters the batch the
            # model trains on — gpu_augment_batch runs identically either way.
            _aug_log = (
                (self.gpu_transforms is not None or self.respiratory_cfg.enable)
                and self.logging_conf.log_visuals and data_iter == 0
                and self.epoch % max(1, getattr(self.logging_conf, "visual_panels_every_n_val_epochs", 3)) == 0
            )
            # `defer_input_images` (the train default) means the dataset omits `images` —
            # gpu_augment_batch creates it below. The old `"images" in batch` guard was
            # vacuous when written and silently killed this panel once deferral landed
            # (docs/62 §5.1), so rebuild the pre-aug snapshot from `phases`, exactly as
            # trainer_viz._subject_device_batch does. `phases` here is pre-affine,
            # pre-respiratory, so this is the true "before".
            _orig_images = None
            if _aug_log:
                if "images" in batch:
                    _orig_images = batch["images"].detach().clone()
                elif "phases" in batch:
                    from data.gpu_aug import extract_slices_from_phases
                    # `extract_slices_from_phases` returns (B,S,R,R,3) in [0,255]; the
                    # `batch["images"]` contract — and what `_log_augmentation_to_wandb._gray`
                    # expects — is (B,S,3,R,R) in [0,1]. Skipping the conversion does NOT
                    # crash: _gray would clamp [0,255] to all-1.0 and average over H instead
                    # of the channel axis, rendering the "original" row as a near-white sliver.
                    _orig_images = extract_slices_from_phases(
                        batch["phases"].float(), batch["timesteps"], batch["slice_indices"],
                        out_size=int(batch["scanner_coords"].shape[-2]),
                    ).permute(0, 1, 4, 2, 3).contiguous().div(255.0).detach()
                else:
                    logging.warning("[aug-viz] neither 'images' nor 'phases' in batch; panel skipped")
            batch = gpu_augment_batch(
                batch, self.gpu_transforms, self.device,
                respiratory_cfg=self.respiratory_cfg, train=True,
                resp_generator=self.resp_generator)
            if _aug_log:
                self._log_augmentation_to_wandb(_orig_images, batch.get("images"), self.steps["train"])
            if data_iter == 0:
                self._log_resp_disp_scalar(batch, self.steps["train"], "train")

            self._run_step_and_backward(batch, phase, loss_meters)

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
                    # docs/64 tripwire: a dead ReLU in the DPT head silently severs the
                    # gradient to the aggregator. Fed EVERY step (not on log_freq) so the
                    # patience counter counts real steps.
                    alarm = self._grad_alarms.get(key)
                    if alarm is not None:
                        alarm.update(grad_norm, self.steps[phase], epoch=self.epoch)

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

    def _run_step_and_backward(self, batch: Mapping, phase: str,
                               loss_meters: Dict[str, AverageMeter]):
        """One forward + backward. Gradient accumulation was removed (2026-08-01): every
        config sets `accum_steps: 1`, and B is hardcoded to 1 in the loader (the only
        collation that is safe under native-z), so chunking a batch into N>1 pieces would
        have produced EMPTY tensors — the feature was dead and unusable, not just unused."""
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        amp_type = self._amp_dtype()

        with torch.amp.autocast("cuda", enabled=self.optim_conf.amp.enabled, dtype=amp_type):
            loss_dict = self._step(batch, self.model, phase, loss_meters)

        loss = loss_dict["objective"]

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

        self.scaler.scale(loss).backward()
        loss_meters[f"Loss/{phase}_loss_objective"].update(
            loss.item(), batch["images"].shape[0])

    def _amp_dtype(self):
        """Resolve `optim.amp.amp_dtype` once (was open-coded identically in two places)."""
        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        return torch.bfloat16 if amp_type == "bfloat16" else torch.float16

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

        # Skip logging (only) for a non-finite batch, so one NaN can't poison the epoch's
        # AverageMeters. The backward-skip logic in _run_step_and_backward runs later and is
        # unchanged; val had no check at all. docs/60. (Named _run_steps_on_batch_chunks
        # until 2026-08-01, when accumulation was removed — docs/62 §7.)
        if not self._log_if_finite(log_data, phase):
            self.steps[phase] += 1
            return loss_dict

        self._update_and_log_scalars(log_data, phase, self.steps[phase], loss_meters)
        self._log_visuals_to_wandb(log_data, phase, self.steps[phase])

        self.steps[phase] += 1
        return loss_dict

    @staticmethod
    def _pitch_bucket(dz):
        """Coarse (>=10 mm) vs fine (<10 mm) pitch. Two buckets, not the 6 raw val pitches
        (n = 72/52/4/2/2/1) — a per-pitch curve would be one subject dressed as a statistic.
        docs/59 F8."""
        if dz is None:
            return None
        return "coarse_ge10mm" if float(dz) >= 10.0 else "fine_lt10mm"

    def _record_val_subject(self, data: Mapping, b: int, t: int,
                            psnr_full, psnr_bbox, psnr_motion):
        """One per-subject val row to disk, plus the live stratified means.

        Batch size is pinned to 1, so these values ARE per-subject — the AverageMeter just
        averages them away. Writing the raw row is free and makes every later slice
        (vendor, pathology, D, ...) a groupby against manifest.csv. docs/60.
        """
        try:
            mri_ds = self._get_mri_dataset()
            seqs = data.get("seq_index")
            seq_index = int(seqs[b].flatten()[0].item()) if seqs is not None else None
            subject, source = (seq_index_to_subject(mri_ds, seq_index)
                               if (mri_ds is not None and seq_index is not None)
                               else (None, None))
            dz = data.get("dz_mm")
            dz = float(dz.reshape(-1)[b]) if dz is not None else None

            row = {
                "epoch": int(self.epoch),
                "step": int(self.steps["train"]),
                "seq_name": subject,
                "source": source,
                "t_target": t,
                "dz_mm": dz,
                "D": int(data["V_gt"].shape[-3]),
                "S": int(data["images"].shape[1]),
                "seq_index": seq_index,
                "metric_psnr_3d_full": psnr_full,
                "metric_psnr_3d_bbox": psnr_bbox,
                "metric_psnr_3d_motion": psnr_motion,
            }
            # Every metric the loss already computed for this sample. B==1, so these
            # batch-level scalars are per-subject values.
            for k, v in data.items():
                if k.startswith("metric_") and k not in row:
                    try:
                        row[k] = float(v.item() if torch.is_tensor(v) else v)
                    except (TypeError, ValueError, RuntimeError):
                        pass
            # Breathing damage is not a `metric_*` key, so the sweep above misses it — and
            # it is the one quantity that cannot be recovered after the run.
            row.update(resp_offslab_stats(data, b))
            self.run_log.subject_row(row)

            # Live strata (emitted at end of val epoch by _log_val_strata).
            for axis, group in (("source", source),
                                ("pitch", self._pitch_bucket(dz))):
                if group is None:
                    continue
                for metric in STRATA_METRICS:
                    val = row.get(metric)
                    if val is not None:
                        self._val_strata[(axis, group, metric)].append(val)
        except Exception as e:
            logging.warning(f"per-subject val record failed (ignored): {e}")

    def _log_val_strata(self, step: int):
        """Per-source and per-pitch val means, then clear. Two axes only: the pooled mean
        hides a one-source collapse (15 of 133 subjects moving 5 dB shifts it 0.6 dB), and
        pitch is the native-z tripwire. `n` is a separate scalar, not baked into the metric
        name — baking it would orphan the curve on any split edit. docs/60."""
        if not self._val_strata:
            return
        try:
            for (axis, group, metric), values in sorted(self._val_strata.items()):
                if not values:
                    continue
                short = metric.replace("metric_psnr_3d_", "psnr_").replace("metric_", "")
                self._log_scalar(f"val/strata/{axis}/{group}/{short}",
                                 float(sum(values) / len(values)), step)
                # `n` per METRIC, not per group: the heart-ROI metrics are conditional
                # (they need a valid heart_roi_canonical), so a group's metrics can have
                # different counts. One shared `n` name meant last-write-wins, i.e. the
                # count shown belonged to whichever metric sorted last.
                self._log_scalar(f"val/strata/{axis}/{group}/{short}_n", float(len(values)), step)
            groups = sorted({(a, g) for (a, g, _) in self._val_strata})
            logging.info(f"[val strata @ step {step}] " + " | ".join(
                f"{a}:{g}" for a, g in groups))
        except Exception as e:
            logging.warning(f"val strata log failed (ignored): {e}")
        finally:
            self._val_strata.clear()

    def _log_if_finite(self, log_data: Mapping, phase: str) -> bool:
        """True when the objective is finite; otherwise name the offending subject and
        return False. Turns "everything after epoch 40 is NaN" into "subject X did it"."""
        try:
            obj = log_data.get("objective")
            value = obj.item() if torch.is_tensor(obj) else obj
            if value is None or math.isfinite(value):
                return True
            names = log_data.get("seq_name") or ["<unknown>"]
            self._nonfinite_logged[phase] += 1
            logging.error(
                f"Non-finite objective ({value}) in phase={phase} at step "
                f"{self.steps[phase]}; subject(s)={list(names)[:4]}. Skipping the metric "
                f"update for this batch so the epoch's AverageMeters stay finite "
                f"(cumulative {phase}: {self._nonfinite_logged[phase]})."
            )
            # Logged at the TRAIN step like every other val scalar: wandb drops
            # non-monotonic steps, and steps["val"] lags badly, so at steps[phase] this
            # alarm could be discarded exactly when it matters.
            self._log_scalar(f"{phase}/optim/nonfinite_logged_cumulative",
                             float(self._nonfinite_logged[phase]), self.steps["train"])
            return False
        except Exception as e:
            logging.warning(f"finiteness guard failed (ignored, logging anyway): {e}")
            return True

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values to wandb."""
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
                    psnr_motion = None
                    if motion_masks is not None:
                        m = motion_masks[b]
                        if bool(m.any()):
                            Vc_m = Vc[m]
                            Vg_m = Vg[m]
                            mse_motion = (Vc_m - Vg_m).pow(2).mean().clamp(min=1e-10)
                            psnr_motion = (10.0 * torch.log10(1.0 / mse_motion)).item()
                            self._per_phase_val_psnr_motion[t].append(psnr_motion)
                    self._record_val_subject(
                        data, b, t, psnr_full,
                        psnr_bbox if bboxes is not None else None, psnr_motion)
            except Exception as e:
                logging.warning(f"per-phase PSNR accumulation failed (ignored): {e}")

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


