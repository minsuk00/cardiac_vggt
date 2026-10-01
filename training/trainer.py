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


import gc
import logging
import math
import time
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from training.data.gpu_aug import build_gpu_transforms, gpu_augment_batch
from training.data.loader import build_loader
from training.data.respiratory import RespiratoryConfig
from training.monitor import Monitor
from training.train_utils.checkpoint import make_checkpoint, restore_checkpoint, robust_torch_save
from vggt.utils.checkpoint_stage import stage_checkpoint_to_local
from training.train_utils.freeze import freeze_modules
from training.train_utils.general import (
    AverageMeter, DurationMeter, ProgressMeter, copy_data_to_device, model_summary,
    resolve_resume_checkpoint, safe_makedirs, set_seeds,
)
from training.train_utils.logging import setup_logging
from training.train_utils.optimizer import construct_optimizer
from training.train_utils.run_log import RunLog, file_md5

# Fallback when `checkpoint.best_metric` is absent from a config. Heart-segmentation ROI
# PSNR — a real anatomical mask; see the rationale in default.yaml's checkpoint block.
_BEST_METRIC_DEFAULT = "metric_psnr_3d_heartseg"


def forward(model, loss_fn, batch, phase: str):
    """Model forward + loss -> (predictions, loss dict). `loss_objective` aliases
    `objective` so the scalar logs can name it like every other `loss_*` key."""
    y_hat = model(images=batch["images"], batch=batch)
    loss_dict = loss_fn(y_hat, batch, val_metrics=(phase == "val"))
    loss_dict["loss_objective"] = loss_dict["objective"]
    return y_hat, loss_dict


def train_step(model, loss_fn, optim, clipper, batch, where: float, amp: bool):
    """One optimizer step -> (predictions, loss dict, objective is finite, grad norms).

    zero_grad -> forward under bf16 autocast -> backward -> schedulers set to `where` ->
    per-group clipping -> optimizer step. A non-finite objective skips the backward; the
    clipper and the optimizer step still run, but every gradient is None after
    `zero_grad(set_to_none=True)`, so they leave the parameters and optimizer state as
    they were. Gradient accumulation does not exist (batch size is pinned to 1)."""
    optim.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
        y_hat, loss_dict = forward(model, loss_fn, batch, "train")
    finite = math.isfinite(loss_dict["objective"].item())
    if finite:
        loss_dict["objective"].backward()
    optim.step_schedulers(where)
    grad_norms = clipper()
    optim.optimizer.step()
    return y_hat, loss_dict, finite, grad_norms


class Trainer:
    """Single-GPU training / validation loop (DDP was removed in 284992c).

    Builds the model, loss, data, optimizer and checkpoint state, then runs the epochs.
    Everything diagnostic — scalars, val metrics by phase/source, per-subject rows, volume
    dumps, EF, wandb panels — lives in `self.monitor` (training/monitor.py).
    """

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
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
        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (wandb, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: "train" for training and validation, "val" for validation only.
            seed_value: A random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run validation.
            cuda: Hydra config for CUDA-specific settings (e.g., cuDNN).
            limit_train_batches: Limit the number of training batches per epoch (for debugging).
            limit_val_batches: Limit the number of validation batches per epoch (for debugging).
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Dictionary of environment variables to set.
        """
        # NOTE: the `logging` kwarg shadows the logging module inside __init__.
        if env_variables:
            for variable_name, value in env_variables.items():
                os.environ[variable_name] = value
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim
        # Fully-resolved config snapshot (from launch.py) → logged as wandb run.config.
        self._wandb_config = kwargs.get("_wandb_config", None)

        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value
        # Training progress in [0, 1]; the schedulers are functions of it.
        self.where = 0.0

        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(0)
        self._setup_backends(cuda)

        safe_makedirs(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            log_level_primary=self.logging_conf.log_level_primary,
        )
        set_seeds(seed_value, self.max_epochs, 0)

        # On-disk run log (docs/60): every scalar mirrored to metrics.jsonl, plus the
        # per-subject val rows wandb never sees.
        self.run_log = RunLog(self.logging_conf.log_dir)

        self._setup_components()
        self._setup_dataloaders()

        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", ":.4f")
        self.optim = (construct_optimizer(self.model, self.optim_conf)
                      if self.mode != "val" else None)

        # A run's own latest checkpoint in save_dir wins (SLURM requeue / crash resume —
        # epoch+steps+optimizer intact); the configured seed/base checkpoint
        # (resume_checkpoint_path) is only the cold-start fallback. See
        # resolve_resume_checkpoint for why the priority must be this way round.
        ckpt_to_load = resolve_resume_checkpoint(
            self.checkpoint_conf.save_dir, self.checkpoint_conf.resume_checkpoint_path
        )
        if ckpt_to_load is not None:
            self._load_resuming_checkpoint(ckpt_to_load)
        self._restore_best_val_metric()

        # GPU augmentation (train only; None = identity). Respiratory motion is separate
        # and applies in train AND val: train draws from a PRIVATE generator (so turning
        # breathing on never shifts the global RNG stream batchaug/dropout draw from), val
        # deterministically per seq_index. It overwrites only the input slices.
        aug_cfg = self.data_conf.get("augmentation", None) if self.data_conf is not None else None
        self.gpu_transforms = build_gpu_transforms(aug_cfg)
        self.respiratory_cfg = RespiratoryConfig.from_cfg(
            aug_cfg.get("respiratory", None) if aug_cfg is not None else None)
        self.resp_generator = torch.Generator(device=self.device).manual_seed(
            int(self.seed_value))

        vol = getattr(self.loss_conf, "volume", None)
        self.monitor = Monitor(
            self.logging_conf, self.run_log, self.wandb_writer, self.val_ds, self.device,
            epoch=self.epoch,
            respiratory_cfg=self.respiratory_cfg,
            gpu_transforms=self.gpu_transforms,
            aug_tier=aug_cfg.get("tier", "aggressive") if aug_cfg is not None else "aggressive",
            splat_res=None if vol is None else vol.get("splat_res"),
            val_len=self._val_len(),
            grad_alarm_cfg=self.optim_conf.get("grad_collapse_alarm", None),
        )

        # One `run_meta.jsonl` line per PROCESS LAUNCH (docs/60). Written after the
        # checkpoint load, so `resumed_from_epoch` is known — a requeued run therefore
        # leaves one line per segment and a mid-run code edit shows as a changed git sha.
        self._write_run_meta()

        if self.mode in ["train", "val"]:
            self.monitor.on_startup(self.steps.get("train", 0))

    # ── build ────────────────────────────────────────────────────────────────

    def _setup_backends(self, cuda_conf: Dict) -> None:
        """Configures PyTorch CUDA backends. Single-GPU: no process group is created."""
        # Read here, applied in _compile_attention_blocks() once the model exists.
        self.compile_attention_blocks = bool(
            cuda_conf.get("compile_attention_blocks", False)) if cuda_conf else False
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

    def _setup_components(self):
        """wandb writer, model, loss, gradient clipper; freeze, then compile. Keep the
        order: it is part of what a reproduced run has to match."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0}

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
        # bf16 only: it has fp32 range, so no GradScaler / loss scaling is needed.
        if self.optim_conf.amp.enabled:
            assert self.optim_conf.amp.amp_dtype == "bfloat16", (
                f"only bfloat16 AMP is supported, got {self.optim_conf.amp.amp_dtype}")

        if getattr(self.optim_conf, "frozen_module_names", None):
            logging.info(f"[Start] Freezing modules: {self.optim_conf.frozen_module_names}")
            self.model = freeze_modules(
                self.model,
                patterns=self.optim_conf.frozen_module_names,
            )
            logging.info(f"[Done] Freezing modules: {self.optim_conf.frozen_module_names}")

        # Summary before compilation, so it reports the eager module tree.
        model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
        model_summary(self.model, log_file=model_summary_path, logging_func=logging.info)
        logging.info(f"Model summary saved to {model_summary_path}")

        # Compile AFTER freezing: dynamo guards on requires_grad, so compiling first would
        # immediately recompile every block once the freeze flips those flags.
        self._compile_attention_blocks()

        logging.info("Successfully initialized training components.")

    def _compile_attention_blocks(self) -> None:
        """`torch.compile` the DINO and aggregator attention blocks in place (docs/40).

        Compiles each Block individually rather than the whole model. Per-block compilation leaves
        the outer `checkpoint()` calls in eager Python, so checkpointing is fully preserved.

        Uses `nn.Module.compile()`, NOT `mod = torch.compile(mod)`: the former swaps the module's
        internal call implementation, so the module identity and `state_dict()` keys are unchanged
        and checkpoints stay interchangeable with eager runs. The latter would wrap the block in an
        `OptimizedModule` and prefix its keys with `_orig_mod.`.

        `dynamic=True` because S varies per subject (one_frame_per_slice ⇒ S = in-FOV plane count).
        """
        if not self.compile_attention_blocks:
            return
        agg = getattr(self.model, "aggregator", None)
        if agg is None:
            logging.warning("compile_attention_blocks requested but model has no aggregator; skipping.")
            return
        pe = getattr(agg, "patch_embed", None)
        # The in-repo DINOv2 ViT stores its 24 layers at `.blocks`; the DINOv3 adapter
        # (vggt/models/dinov3.py) wraps the HF DINOv3ViTModel at `.model`, whose layers
        # live at `.layer` — cover both so neither backbone silently skips compilation.
        dino_blocks = getattr(pe, "blocks", None)
        if dino_blocks is None:
            dino_blocks = getattr(getattr(pe, "model", None), "layer", None)
        n = 0
        for blocks in (dino_blocks, getattr(agg, "frame_blocks", None),
                       getattr(agg, "global_blocks", None)):
            for block in (blocks or []):
                block.compile(mode="default", dynamic=True)
                n += 1
        logging.info(f"torch.compile applied to {n} DINO + aggregator attention blocks "
                     "(mode=default, dynamic=True); first steps pay one-time compilation.")

    def _setup_dataloaders(self):
        """Instantiates the train/val MRIDatasets; `_loader` wraps one per epoch."""
        self.train_ds = None
        self.val_ds = None

        if self.mode in ["train", "val"]:
            self.val_ds = instantiate(self.data_conf.get("val", None), _recursive_=False)

        if self.mode in ["train"]:
            self.train_ds = instantiate(self.data_conf.train, _recursive_=False)

    def _loader(self, dataset, shuffle):
        return build_loader(dataset, seed=self.seed_value, epoch=int(self.epoch),
                            shuffle=shuffle, num_workers=self.data_conf.num_workers)

    def _val_len(self) -> int:
        """Val samples per epoch (seq_index 0..N-1); the identity baseline covers the same.

        The EF sweep visits each (subject, ED/ES) pair exactly once. A fixed target phase
        makes every pass over the subjects byte-identical (val is deterministic and t_target
        is its only source of variety), so it is capped to one pass."""
        ds = self.val_ds
        if ds is None:
            return 0
        if getattr(ds, "val_targets", None) is not None:
            return len(ds.val_targets)
        n_subj = len(ds.subjects)
        n = n_subj if self.limit_val_batches is None else self.limit_val_batches
        if ds.t_target_fixed is not None and 0 < n_subj < n:
            logging.info(
                f"Fixed-phase val (t_target_fixed={ds.t_target_fixed}): capping "
                f"limit_val_batches {n}→{n_subj} (one deterministic "
                f"pass over val subjects; more is redundant)."
            )
            n = n_subj
        return n

    def _write_run_meta(self):
        """Identify WHICH code and cohort produced this run's numbers. The git sha and the
        split md5 are the load-bearing fields, and neither is recoverable from wandb."""
        try:
            def _subjects(ds):
                if ds is None:
                    return None, None
                return len(ds.subjects), getattr(ds, "split_file", None)

            n_train, split_file = _subjects(self.train_ds)
            n_val, val_split_file = _subjects(self.val_ds)
            split_file = split_file or val_split_file
            manifest = os.path.join(
                os.path.dirname(split_file), "manifest.csv") if split_file else None

            # The val PROTOCOL, not just the cohort: cardiac_phase.csv decides which
            # (subject, t_target) pairs every val number is averaged over, so editing it
            # moves every metric and the identity floor with it.
            mri_ds = self.val_ds
            ef_csv = getattr(mri_ds, "cardiac_phase_csv", None) if mri_ds else None
            # Bumped whenever the on-disk VOXELS change (slice-order flips, roll fixes,
            # ROI regeneration). split_md5 only hashes subject NAMES, so without this two
            # runs on different pixels look identical.
            try:
                from training.data.preprocess import cache_signature
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

    # ── checkpoints ──────────────────────────────────────────────────────────

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Load model (+ optimizer unless val) and progress from `ckpt_path`."""
        logging.info(f"Resuming training from {ckpt_path}")

        # Only the configured (immutable) resume_checkpoint_path is staged to node-local
        # /tmp — GPFS torch.load is ~266s vs ~5s for an ~8GB ckpt. A run's own
        # checkpoint_last.pt (overwritten each requeue) is loaded directly so a stale copy
        # can never be reused. Byte-identical; see checkpoint_stage.py.
        load_path = ckpt_path
        resume_cfg = self.checkpoint_conf.resume_checkpoint_path
        if resume_cfg and os.path.abspath(ckpt_path) == os.path.abspath(resume_cfg):
            load_path = stage_checkpoint_to_local(ckpt_path)
        with open(load_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        self.epoch, self.steps, self.ckpt_time_elapsed = restore_checkpoint(
            checkpoint, self.model, self.optim, strict=self.checkpoint_conf.strict)

    def _restore_best_val_metric(self):
        """Seed the best-so-far score from an existing `checkpoint_best.pt`, so a requeued
        process does not overwrite a better checkpoint with its first val epoch."""
        self._best_val_metric = None
        path = os.path.join(self.checkpoint_conf.save_dir, "checkpoint_best.pt")
        if not os.path.exists(path):
            return
        try:
            # mmap: only the small pickle is read, not the ~3.8 GB of weights.
            ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            self._best_val_metric = float(ck["best_metric_value"])
            logging.info(f"[checkpoint] best so far {self._best_val_metric:.4f} (from {path})")
        except Exception as e:
            logging.warning(f"[checkpoint] could not read {path} (ignored): {e}")

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
            # Write-then-rename: a kill mid-write must not destroy the previous best.
            robust_torch_save({"model": self.model.state_dict(),
                               "prev_epoch": int(self.epoch),
                               "best_metric_name": self.checkpoint_conf.get(
                                   "best_metric", _BEST_METRIC_DEFAULT),
                               "best_metric_value": float(value)}, path)
            logging.info(f"[checkpoint] new best {self.checkpoint_conf.get('best_metric', _BEST_METRIC_DEFAULT)}"
                         f"={value:.4f} at epoch {int(self.epoch)} -> {path}")
        except Exception as e:
            logging.warning(f"[checkpoint] best-checkpoint save failed (ignored): {e}")

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """Save `checkpoint_last` (and `checkpoint_{epoch}` every `save_freq` epochs)."""
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint_last"]
            if self.checkpoint_conf.save_freq > 0 and int(epoch) % self.checkpoint_conf.save_freq == 0 and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint = make_checkpoint(self.model, self.optim, epoch, self.steps,
                                     self.time_elapsed_meter.val)
        for ckpt_name in checkpoint_names:
            checkpoint_path = os.path.join(checkpoint_folder, f"{ckpt_name}.pt")
            logging.info(f"Saving checkpoint at epoch {epoch} to {checkpoint_path}")
            robust_torch_save(checkpoint, checkpoint_path)

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self):
        """Train (+ a final val) or validate only, and record how the run ENDED (docs/60):
        without the exit line a crashed run and a completed one look the same on disk. A
        SIGUSR1 requeue calls `os._exit(0)` and bypasses this; that segment is identified
        by the NEXT launch line carrying `resumed_from_epoch`.
        """
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        try:
            if self.mode == "train":
                self.run_train()
                self.run_val()
                self._maybe_save_best_checkpoint()
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

            dataloader = self._loader(self.train_ds, shuffle=True)
            self.train_epoch(dataloader)
            self.save_checkpoint(self.epoch)

            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # The val after the last epoch is the trailing run_val() in run().
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
        if self.val_ds is None:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        dataloader = self._loader(self.val_ds, shuffle=False)
        self.val_epoch(dataloader)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def _loss_meters(self, phase: str) -> Dict[str, AverageMeter]:
        names = [f"Loss/{phase}_{key}" for key in self.monitor.scalar_keys(phase)]
        return {name: AverageMeter(name, ":.4f") for name in names}

    def train_epoch(self, train_loader):
        batch_time = AverageMeter("Batch Time", ":.4f")
        data_time = AverageMeter("Data Time", ":.4f")
        mem = AverageMeter("Mem (GB)", ":.4f")
        phase = "train"

        loss_meters = self._loss_meters(phase)
        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", ":.4f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        self.monitor.begin_train(self.epoch)
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else self.limit_train_batches

        self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break
            data_time.update(time.time() - end)

            step = self.steps[phase]
            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            snapshot = self.monitor.aug_snapshot(batch, data_iter)
            batch = gpu_augment_batch(
                batch, self.gpu_transforms, self.device,
                respiratory_cfg=self.respiratory_cfg, train=True,
                resp_generator=self.resp_generator)
            self.monitor.aug_panel(snapshot, batch, step)
            if data_iter == 0:
                self.monitor.resp_disp(batch, step, "train")

            # Scheduler progress. Always < 1: epoch < max_epochs and data_iter < limit.
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs
            y_hat, loss_dict, finite, grad_norms = train_step(
                self.model, self.loss, self.optim, self.gradient_clipper, batch,
                self.where, amp=self.optim_conf.amp.enabled)
            self.monitor.on_step(y_hat, loss_dict, batch, phase, step, step, loss_meters, data_iter)
            self.steps[phase] += 1
            self.monitor.after_train_step(self.steps[phase], self.optim, self.where, exact_epoch,
                                          grad_norms, finite, loss_dict, loss_meters)
            del y_hat, loss_dict

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

    @torch.no_grad()
    def val_epoch(self, val_loader):
        batch_time = AverageMeter("Batch Time", ":.4f")
        data_time = AverageMeter("Data Time", ":.4f")
        mem = AverageMeter("Mem (GB)", ":.4f")
        phase = "val"

        self.monitor.begin_val(self.epoch)
        loss_meters = self._loss_meters(phase)
        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[batch_time, data_time, mem, self.time_elapsed_meter, *loss_meters.values()],
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        self.model.eval()
        end = time.time()
        limit_val_batches = self._val_len()

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= limit_val_batches:
                break
            data_time.update(time.time() - end)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)
            # Val never affine-augments; breathing (if enabled) applies deterministically
            # per seq_index, so val measures the real corrupted->clean task.
            batch = gpu_augment_batch(
                batch, None, self.device,
                respiratory_cfg=self.respiratory_cfg, train=False)
            if data_iter == 0:
                self.monitor.resp_disp(batch, self.steps["train"], "val")

            with torch.amp.autocast("cuda", enabled=self.optim_conf.amp.enabled,
                                    dtype=torch.bfloat16):
                y_hat, loss_dict = forward(self.model, self.loss, batch, phase)
                self.monitor.on_step(y_hat, loss_dict, batch, phase, self.steps[phase],
                                     self.steps["train"], loss_meters, data_iter)
            self.steps[phase] += 1
            self.monitor.on_val_batch(batch, loss_dict)
            del y_hat, loss_dict

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # Val scalars are logged at the TRAIN step, to align with training progress.
        current_train_step = self.steps["train"]
        means = self.monitor.end_val(current_train_step, loss_meters, self.model)
        # Best-checkpoint selection metric (default: heart-seg ROI PSNR, a real anatomical
        # mask). Not bbox/full: those are dominated by static tissue the model gets for free.
        best_metric = self.checkpoint_conf.get("best_metric", _BEST_METRIC_DEFAULT)
        if best_metric in means:
            self._last_val_metric = means[best_metric]
        logging.info(f"Validation Epoch {self.epoch} complete. Logged averages at train step {current_train_step}")
