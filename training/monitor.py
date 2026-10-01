# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Everything diagnostic around the training loop, owned by one `Monitor`.

The trainer calls a few hooks; nothing here feeds back into training:

  startup      on_startup                       identity-Δ baseline, motion-mask panel
  train step   aug_snapshot / aug_panel, resp_disp, on_step, after_train_step
  val epoch    begin_val, resp_disp, on_step, on_val_batch, end_val

`scalar` is the single scalar chokepoint: every scalar goes to wandb AND `metrics.jsonl`
(docs/60). Every hook that renders or writes files is try/except-wrapped, so a
diagnostic failure is logged and ignored instead of raised into training.
"""

import json
import logging
import math
import os
import shutil
from collections import defaultdict
from typing import Mapping

import torch

import monitor_panels as panels
from data.gpu_aug import extract_slices_from_phases, gpu_augment_batch
from data.preprocess import NUM_PHASES
from ef_eval import IN_PLANE_MM
from train_utils.general import safe_makedirs
from train_utils.notify import GradientCollapseAlarm
from train_utils.val_logging import (
    N_FILM_PLANES, load_subject_groups, pick_one_index_per_source, pick_planes, pick_visual_indices,
    resp_offslab_stats, seq_index_to_subject, to_float,
)

# The three per-phase val PSNR regions: the whole cube, the subject's geometric content
# bbox, and only the voxels that move across the cardiac cycle (the honest signal — static
# tissue is free). Logged under val/psnr/<kind>/.
PSNR_KINDS = ("full", "bbox", "motion")

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


def pitch_bucket(dz):
    """Coarse (>=10 mm) vs fine (<10 mm) pitch. Two buckets, not the 6 raw val pitches
    (n = 72/52/4/2/2/1) — a per-pitch curve would be one subject dressed as a statistic.
    docs/59 F8."""
    if dz is None:
        return None
    return "coarse_ge10mm" if float(dz) >= 10.0 else "fine_lt10mm"


def scalar_name(phase: str, key: str) -> str:
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


def _psnr(a, b):
    return (10.0 * torch.log10(1.0 / (a - b).pow(2).mean().clamp(min=1e-10))).item()


class Monitor:
    """Diagnostics for one run. `epoch` is the trainer's epoch, kept in sync by
    `begin_train` / `begin_val`; it tags every scalar row on disk and drives the cadences."""

    def __init__(self, log_conf, run_log, wandb_writer, val_ds, device, *, epoch=0,
                 respiratory_cfg=None, gpu_transforms=None, aug_tier="aggressive",
                 splat_res=None, val_len=0, grad_alarm_cfg=None):
        self.log_conf = log_conf
        self.run_log = run_log
        self.wandb = wandb_writer
        self.val_ds = val_ds
        self.device = device
        self.epoch = epoch
        self.respiratory_cfg = respiratory_cfg
        self.gpu_transforms = gpu_transforms
        self.aug_tier = aug_tier            # for the augmentation panel caption
        self.splat_res = splat_res          # loss.volume.splat_res, so baseline/filmstrip splat like val
        self.val_len = val_len              # samples per val epoch (seq_index 0..val_len-1)

        # Val-set facts the diagnostics gate on. t_target_fixed=None → multi-phase val
        # (per-phase panels); an int → single-phase, where only the motion panels are logged.
        # reference_slot (docs/25): the filmstrip rebuilds slot 0 at each queried phase.
        self.t_target_fixed = val_ds.t_target_fixed if val_ds is not None else None
        self.reference_slot = getattr(val_ds, "reference_slot", False) if val_ds is not None else False
        # EF sweep: [(subj_idx, ED) for all] + [(subj_idx, ES) for all], or None.
        self.val_targets = getattr(val_ds, "val_targets", None) if val_ds is not None else None

        # Identity-Δ baseline, kind -> {"mean_psnr": float|None, "per_phase_mean": {t: float}};
        # empty until on_startup. Baked into the per-phase val metric names.
        self.identity_baseline = {}
        # Per-phase val PSNR, kind -> {t_target: [per-sample PSNR]}; reset each val epoch.
        self.val_psnr = {kind: defaultdict(list) for kind in PSNR_KINDS}
        # (axis, group, metric) -> [per-subject values]; emitted and cleared each val epoch.
        self.val_strata = defaultdict(list)
        # Train batches whose backward was skipped (non-finite objective), and per phase the
        # batches whose meters/scalars were skipped for the same reason.
        self.nan_batch_count = 0
        self.nonfinite_logged = defaultdict(int)

        # docs/64 tripwire. Both pooled1337 runs sat at grad_aggregator < 1e-6 for ~70
        # epochs (dead ReLU in the DPT head) while grad_point looked healthy, so nothing
        # in the standard logging flagged it. One alarm per clipper group, keyed by the
        # same name the clipper reports.
        self.grad_alarms = {}
        if grad_alarm_cfg is None or grad_alarm_cfg.get("enable", True):
            cfg = grad_alarm_cfg
            thr = float(cfg.get("threshold", 1e-6)) if cfg else 1e-6
            pat = int(cfg.get("patience", 200)) if cfg else 200
            watch = list(cfg.get("modules", ["aggregator"])) if cfg else ["aggregator"]
            for name in watch:
                self.grad_alarms[name] = GradientCollapseAlarm(
                    threshold=thr, patience=pat, name=name)

        self._visual_subjects = None
        # Per-val-epoch state, reset by begin_val.
        self._val_iter = 0
        self._val_volumes_saved = set()
        self.ef_this_epoch = False
        self.viz_ed_es = False
        self._ed_es_stash = {}
        self._ef_pred_dir = None

    # ── scalars ──────────────────────────────────────────────────────────────

    def scalar(self, name: str, value, step: int):
        """Log a scalar to wandb AND mirror it to `metrics.jsonl` (docs/60). The mirror is
        not gated on the wandb writer — offline runs still get a full record."""
        if self.wandb:
            self.wandb.log(name, value, step)
        if self.run_log is not None:
            self.run_log.scalar(name, value, step, epoch=self.epoch)

    def scalar_keys(self, phase: str):
        """Batch keys averaged into the epoch meters (and logged per step in train)."""
        if self.log_conf.scalar_keys_to_log and phase in self.log_conf.scalar_keys_to_log:
            return self.log_conf.scalar_keys_to_log[phase].keys_to_log
        return ["loss_objective"] if phase == "val" else []

    def resp_disp(self, batch, step: int, prefix: str):
        """Per-slot respiratory displacement (mm) under `{prefix}/resp/`, plus the off-slab
        stats. No-op when breathing is off. Not gated on wandb: the off-slab numbers cannot
        be recovered after the run (the corruption is applied on GPU, never persisted)."""
        disp = batch.get("resp_disp_mm")
        if disp is None:
            return
        try:
            mag = disp.float().norm(dim=-1)  # (B, S) per-slot |d| in mm
            self.scalar(f"{prefix}/resp/disp_mm_mean", float(mag.mean().item()), step)
            self.scalar(f"{prefix}/resp/disp_mm_max", float(mag.max().item()), step)
            for k, v in resp_offslab_stats(batch, 0).items():
                self.scalar(f"{prefix}/resp/{k}", v, step)
        except Exception as e:
            logging.warning(f"resp_disp scalar log failed (ignored): {e}")

    # ── startup ──────────────────────────────────────────────────────────────

    def on_startup(self, step: int):
        """Identity-Δ baseline over the val samples, then the (static) motion-mask panel."""
        self._compute_identity_baseline()
        if self.wandb and self.val_ds is not None:
            panels.log_motion_mask_example(self.wandb, self.val_ds, self.visual_subjects, step)

    @property
    def visual_subjects(self):
        """Val subject indices shown in every visual panel: one per source (and vendor, when
        the manifest has it), so a source-specific geometry regression is visible at a glance.
        A fixed `(0, 7, 14, 21)` showed only the two smallest sources of the sorted pooled
        split. Computed once; falls back to those indices without a usable val set."""
        if self._visual_subjects is not None:
            return self._visual_subjects
        picks = None
        try:
            mri_ds = self.val_ds
            if mri_ds is not None and getattr(mri_ds, "subjects", None):
                vendors = load_subject_groups(mri_ds.split_file, "vendor")
                picks = (pick_visual_indices(mri_ds.subjects, vendors) if vendors
                         else pick_one_index_per_source(mri_ds.subjects))
        except Exception as e:
            logging.warning(f"visual subject pick failed (ignored): {e}")
        if not picks:
            n = 0
            try:
                n = len(self.val_ds.subjects)
            except Exception:
                pass
            picks = tuple(i for i in (0, 7, 14, 21) if n == 0 or i < n)
        self._visual_subjects = picks
        logging.info(f"Visual panel val subjects (source/vendor coverage): {picks}")
        return picks

    def _subject_device_batch(self, data, seq_index):
        """`get_data(...)` numpy dict → batched device tensors: the core every caller needs.

        Callers add their own extras (`gt_target_volume`/`anatomy_bbox` for the identity
        baseline, `z_indices`/`phases` for the filmstrip). One copy on purpose: when native-z
        added `dz_mm`/`z_scale`, a site that missed them would not crash, it would splat at
        the wrong z-scale and silently move the identity floor (the docs/59 F14 pattern).
        """
        import numpy as np

        def st(k, dt=np.float32):
            return torch.from_numpy(np.stack(data[k]).astype(dt)).unsqueeze(0).to(self.device)

        def scalar(k, dt=np.float32):
            return torch.from_numpy(np.asarray(data[k]).astype(dt)).unsqueeze(0).to(self.device)

        timesteps = st("timesteps", np.int64)
        slice_indices = st("slice_indices", np.float32)

        # These callers hit `get_data` directly, bypassing the gpu_augment_batch that
        # materialises a deferred `images` (`defer_input_images`, the training default), so
        # build it here — in float32, matching the dataset's own extraction exactly.
        if "images" in data:
            images = st("images").permute(0, 1, 4, 2, 3).contiguous() / 255.0
        else:
            phases = torch.from_numpy(
                np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(self.device)
            images = extract_slices_from_phases(
                phases, timesteps, slice_indices,
                out_size=int(np.stack(data["scanner_coords"]).shape[-2]),
            ).permute(0, 1, 4, 2, 3).contiguous() / 255.0

        return {
            "images": images,
            "scanner_coords": st("scanner_coords"),
            "z_scale": scalar("z_scale"),
            "dz_mm": scalar("dz_mm"),
            "timesteps": timesteps,
            "slice_indices": slice_indices,
            "seq_index": torch.tensor([[seq_index]], dtype=torch.int64, device=self.device),
        }

    def _compute_identity_baseline(self):
        """Identity-Δ (no motion correction) PSNR over the SAME seq_index range val scores
        (seq_index 0..val_len-1, each input deterministically seeded and breathing-corrupted
        exactly as in val). Written to baseline_identity.json with per-subject floors
        (docs/60) and kept in `identity_baseline` for the per-phase metric names."""
        mri_ds = self.val_ds
        if mri_ds is None:
            logging.warning("Identity baseline: no MRIDataset found, skipping.")
            return

        try:
            import numpy as np
            from loss import compute_volume_intensity_loss

            num_slices = mri_ds.num_slices
            per_phase = {kind: defaultdict(list) for kind in PSNR_KINDS}
            # Per-subject rows (docs/60): raw per-subject PSNR is not comparable across the
            # pooled cohort (the ceiling varies with D, dz, FOV), so later per-subject val
            # numbers are read against each subject's OWN floor. A subject sitting 5-10 dB
            # below its cohort here is an F1-class geometry bug, visible before any step.
            per_subject = []

            N = max(int(self.val_len), 1)
            for i in range(N):
                # Per-subject isolation (docs/59 F5-adjacent): one bad subject (e.g. the
                # docs/59 F19 budget guard raising) must not wipe the whole baseline, which
                # would silently drop the `_base{b}` suffix from every val metric name.
                try:
                    data = mri_ds.get_data(seq_index=i, img_per_seq=num_slices)
                    if "gt_target_volume" not in data:
                        continue  # no GT to compare against; skip
                    batch = self._subject_device_batch(data, i)
                    batch["gt_target_volume"] = torch.from_numpy(
                        data["gt_target_volume"].astype(np.float32)).unsqueeze(0).to(self.device)
                    if "anatomy_bbox" in data:
                        batch["anatomy_bbox"] = torch.from_numpy(
                            np.asarray(data["anatomy_bbox"]).astype(np.int64)).unsqueeze(0).to(self.device)
                    if "phases" in data:
                        batch["phases"] = torch.from_numpy(
                            np.asarray(data["phases"]).astype(np.float32)).unsqueeze(0).to(self.device)
                    # The SAME deterministic breathing shift as val, so "Δ vs identity"
                    # compares like with like. With breathing off this still builds
                    # `images_splat` (native render content, docs/73).
                    batch = gpu_augment_batch(
                        batch, None, self.device,
                        respiratory_cfg=self.respiratory_cfg, train=False)
                    preds = {"world_points": batch["scanner_coords"]}     # Δ = 0
                    out = compute_volume_intensity_loss(preds, batch, splat_res=self.splat_res)
                    t = int(data["t_target"].item() if data["t_target"].ndim == 0 else data["t_target"].flatten()[0].item())
                    if "metric_psnr_3d_full" in out:
                        per_phase["full"][t].append(out["metric_psnr_3d_full"].item())
                    if "metric_psnr_3d_bbox" in out:
                        per_phase["bbox"][t].append(out["metric_psnr_3d_bbox"].item())
                    subject, source = seq_index_to_subject(mri_ds, i)
                    per_subject.append({
                        "seq_index": i,
                        "subject": subject,
                        "source": source,
                        "t_target": t,
                        "D": int(batch["gt_target_volume"].shape[-3]),
                        "dz_mm": float(np.asarray(data["dz_mm"]).reshape(-1)[0]),
                        "psnr_full": to_float(out.get("metric_psnr_3d_full")),
                        "psnr_bbox": to_float(out.get("metric_psnr_3d_bbox")),
                        "psnr_motion": to_float(out.get("metric_psnr_3d_motion")),
                    })
                    # A subject with no moving voxels reports 0.0 dB motion PSNR (the loss
                    # stays branchless); keep it out of the baseline.
                    if "metric_psnr_3d_motion" in out and float(out.get("metric_motion_frac", 1.0)) > 0:
                        per_phase["motion"][t].append(out["metric_psnr_3d_motion"].item())
                except Exception as e:
                    logging.warning(f"identity baseline: subject seq_index={i} failed, "
                                    f"skipping it (ignored): {e}")

            if not any(per_phase["full"].values()):
                logging.warning("Identity baseline: no subjects yielded a PSNR; skipping.")
                return
            baseline = {}
            for kind in PSNR_KINDS:
                values = [p for ps in per_phase[kind].values() for p in ps]
                baseline[kind] = {
                    "mean_psnr": float(sum(values) / len(values)) if values else None,
                    "per_phase_mean": {t: float(sum(ps) / len(ps)) for t, ps in per_phase[kind].items()},
                }

            out_path = os.path.join(self.log_conf.log_dir, "baseline_identity.json")
            try:
                with open(out_path, "w") as f:
                    json.dump({
                        **baseline,
                        "per_phase_counts": {t: len(ps) for t, ps in per_phase["full"].items()},
                        "t_target_fixed": self.t_target_fixed,
                        "per_subject": per_subject,
                    }, f, indent=2)
            except Exception as e:
                logging.warning(f"baseline JSON write failed: {e}")

            self.identity_baseline = baseline
            n_full = sum(len(ps) for ps in per_phase["full"].values())
            fmt = {k: (f"{baseline[k]['mean_psnr']:.2f}" if baseline[k]["mean_psnr"] is not None
                       else "n/a") for k in PSNR_KINDS}
            logging.info(
                f"Identity-Δ baseline: full PSNR = {fmt['full']} dB / bbox PSNR = {fmt['bbox']} dB "
                f"/ motion PSNR = {fmt['motion']} dB across {n_full} val sample(s). "
                f"Persisted to {out_path}."
            )
        except Exception as e:
            logging.warning(f"_compute_identity_baseline failed (ignored): {e}")

    # ── train ────────────────────────────────────────────────────────────────

    def begin_train(self, epoch: int):
        self.epoch = epoch

    def _visual_panel_epoch(self) -> bool:
        return self.epoch % max(1, getattr(self.log_conf, "visual_panels_every_n_val_epochs", 3)) == 0

    def aug_snapshot(self, batch, data_iter: int):
        """On the augmentation-panel cadence, a read-only copy of the PRE-augmentation input
        slices (B,S,3,R,R) in [0,1]; otherwise None. Never alters the batch."""
        if not ((self.gpu_transforms is not None or self.respiratory_cfg.enable)
                and self.log_conf.log_visuals and data_iter == 0
                and self._visual_panel_epoch()):
            return None
        if "images" in batch:
            return batch["images"].detach().clone()
        if "phases" in batch:
            # `defer_input_images` (the train default) omits `images`; rebuild the "before"
            # from `phases` (pre-affine, pre-breathing). extract_slices_from_phases returns
            # (B,S,R,R,3) in [0,255]; the `images` contract is (B,S,3,R,R) in [0,1].
            return extract_slices_from_phases(
                batch["phases"].float(), batch["timesteps"], batch["slice_indices"],
                out_size=int(batch["scanner_coords"].shape[-2]),
            ).permute(0, 1, 4, 2, 3).contiguous().div(255.0).detach()
        logging.warning("[aug-viz] neither 'images' nor 'phases' in batch; panel skipped")
        return None

    def aug_panel(self, snapshot, batch, step: int):
        """Before/after augmentation panel from an `aug_snapshot` (no-op for None)."""
        aug_images = batch.get("images")
        if not self.wandb or snapshot is None or aug_images is None:
            return
        try:
            # Label what is actually configured. batchaug does not report which ops fired,
            # so affine ops are listed as per-subject probabilistic candidates; the bottom
            # row is the actual draw. Breathing is always on per slot.
            applied = []
            if self.gpu_transforms is not None:
                applied.append(f"affine[{self.aug_tier}] (candidates flip/rotate/translate/scale "
                               f"+ noise/gamma/bias; each fires per-subject at its tier prob)")
            if self.respiratory_cfg.enable:
                rc = self.respiratory_cfg
                tilt_hi = rc.tilt_max_deg if rc.tilt_max_deg is not None else rc.direction_jitter_deg
                applied.append(f"breathing (ALWAYS-on: A={rc.amplitude_mm:.0f}+/-{rc.amplitude_jitter:.0f}mm/subj, "
                               f"n={rc.cos2n}, tilt<={tilt_hi:.0f}deg/subj)")
            caption = (f"GPU aug = {' + '.join(applied) if applied else 'none'} | top=original, "
                       f"bottom=ACTUAL draw applied (the realized affine subset is visible there); step={step}")
            panels.log_augmentation(self.wandb, snapshot, aug_images, caption, step)
        except Exception as e:
            logging.warning(f"augmentation visual log failed (ignored): {e}")

    def after_train_step(self, step: int, optim, where: float, exact_epoch: float,
                         grad_norms: Mapping, finite: bool, loss_dict: Mapping, meters: dict):
        """Optimizer-side diagnostics for the step just taken (`step` = steps after it):
        the non-finite counter, lr/wd/where/epoch, per-group grad norms, collapse alarms."""
        if not finite:
            self.nan_batch_count += 1
            logging.error(
                f"Loss is {loss_dict['objective'].item()} (phase=train, step={step}, "
                f"cumulative_nan_batches={self.nan_batch_count}); skipping backward."
            )
            self.scalar("train/optim/nan_batches_cumulative", float(self.nan_batch_count), step)

        log_now = step % self.log_conf.log_freq == 0
        if log_now:
            param_group = optim.optimizer.param_groups[0]
            for option in optim.schedulers:
                self.scalar(f"train/optim/{option}", param_group[option], step)
            self.scalar("train/optim/where", where, step)
            # Fractional epoch, so cross-run curves can be aligned on training progress.
            self.scalar("train/optim/epoch", exact_epoch, step)

        for key, grad_norm in grad_norms.items():
            meter_key = f"Grad/{key}"
            if meter_key in meters:
                meters[meter_key].update(grad_norm)
            if log_now:
                self.scalar(f"train/optim/grad_{key}", grad_norm, step)
            # Fed EVERY step (not on log_freq) so the patience counter counts real steps.
            alarm = self.grad_alarms.get(key)
            if alarm is not None:
                alarm.update(grad_norm, step, epoch=self.epoch)

    # ── train + val ──────────────────────────────────────────────────────────

    def on_step(self, y_hat, loss_dict, batch, phase: str, step: int, train_step: int,
                meters: dict, data_iter: int):
        """Meters, scalars, val per-phase PSNR / per-subject rows and visuals for one step.

        `step` is steps[phase] before this step; `train_step` is steps["train"] — every val
        scalar and panel is logged at the train step (wandb drops non-monotonic steps).
        A non-finite objective skips all of it, so one NaN cannot poison the epoch meters."""
        if phase == "val":
            self._val_iter = data_iter
        log_data = {**{f"pred_{k}": v for k, v in y_hat.items()}, **loss_dict, **batch}
        if not self._log_if_finite(log_data, phase, step, train_step):
            return
        self._update_and_log_scalars(log_data, phase, step, train_step, meters)
        self._log_visuals(log_data, phase, step, train_step)

    def _log_if_finite(self, log_data: Mapping, phase: str, step: int, train_step: int) -> bool:
        """True when the objective is finite; otherwise name the offending subject and
        return False. Turns "everything after epoch 40 is NaN" into "subject X did it"."""
        try:
            obj = log_data.get("objective")
            value = obj.item() if torch.is_tensor(obj) else obj
            if value is None or math.isfinite(value):
                return True
            names = log_data.get("seq_name") or ["<unknown>"]
            self.nonfinite_logged[phase] += 1
            logging.error(
                f"Non-finite objective ({value}) in phase={phase} at step "
                f"{step}; subject(s)={list(names)[:4]}. Skipping the metric "
                f"update for this batch so the epoch's AverageMeters stay finite "
                f"(cumulative {phase}: {self.nonfinite_logged[phase]})."
            )
            self.scalar(f"{phase}/optim/nonfinite_logged_cumulative",
                        float(self.nonfinite_logged[phase]), train_step)
            return False
        except Exception as e:
            logging.warning(f"finiteness guard failed (ignored, logging anyway): {e}")
            return True

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, train_step: int,
                                meters: dict):
        keys_to_log = self.scalar_keys(phase)
        batch_size = data["images"].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key].item() if torch.is_tensor(data[key]) else data[key]
                meter_key = f"Loss/{phase}_{key}"
                if meter_key in meters:
                    meters[meter_key].update(value, batch_size)
                # Per-step scalars only in train (val logs its epoch means at end_val).
                if phase == "train" and step % self.log_conf.log_freq == 0:
                    self.scalar(scalar_name("train", key), value, step)

        if phase == "val" and "V_canon" in data and "V_gt" in data and "t_target" in data:
            self._accumulate_val_psnr(data, train_step)

        if phase == "train" and step % self.log_conf.log_freq == 0:
            self._log_slot_selection(data, step)

    def _accumulate_val_psnr(self, data: Mapping, train_step: int):
        """Per-sample full / bbox / motion PSNR bucketed by t_target, plus the per-subject
        row. Runs in both phase modes; end_val decides which panels to emit."""
        try:
            from metrics import compute_motion_mask
            V_canon = data["V_canon"]
            V_gt = data["V_gt"]
            t_targets = data["t_target"]
            bboxes = data.get("anatomy_bbox", None)   # (B, 6) int64 or None
            # Motion mask from the full phase bundle (val never augments).
            motion_masks = compute_motion_mask(data["phases"]) if "phases" in data else None
            B = V_canon.shape[0]
            for b in range(B):
                Vc = V_canon[b].float()
                Vg = V_gt[b].float()
                mse_full = (Vc - Vg).pow(2).mean()
                psnr_full = 10.0 * torch.log10(1.0 / mse_full.clamp(min=1e-10)).item()
                tt = t_targets[b]
                t = int(tt.item() if tt.ndim == 0 else tt.flatten()[0].item())
                self.val_psnr["full"][t].append(psnr_full)
                psnr_bbox = None
                if bboxes is not None:
                    z0, z1, y0, y1, x0, x1 = [int(v) for v in bboxes[b].tolist()]
                    if (z1 > z0) and (y1 > y0) and (x1 > x0):
                        Vc_bb = Vc[z0:z1, y0:y1, x0:x1]
                        Vg_bb = Vg[z0:z1, y0:y1, x0:x1]
                    else:
                        Vc_bb, Vg_bb = Vc, Vg
                    psnr_bbox = _psnr(Vc_bb, Vg_bb)
                    self.val_psnr["bbox"][t].append(psnr_bbox)
                psnr_motion = None
                if motion_masks is not None:
                    m = motion_masks[b]
                    if bool(m.any()):
                        psnr_motion = _psnr(Vc[m], Vg[m])
                        self.val_psnr["motion"][t].append(psnr_motion)
                self._record_val_subject(data, b, t, train_step, psnr_full, psnr_bbox, psnr_motion)
        except Exception as e:
            logging.warning(f"per-phase PSNR accumulation failed (ignored): {e}")

    def _record_val_subject(self, data: Mapping, b: int, t: int, train_step: int,
                            psnr_full, psnr_bbox, psnr_motion):
        """One per-subject val row to disk, plus the live stratified means.

        Batch size is pinned to 1, so these values ARE per-subject — the AverageMeter just
        averages them away. Writing the raw row makes every later slice (vendor, pathology,
        D, ...) a groupby against manifest.csv. docs/60.
        """
        try:
            mri_ds = self.val_ds
            seqs = data.get("seq_index")
            seq_index = int(seqs[b].flatten()[0].item()) if seqs is not None else None
            subject, source = (seq_index_to_subject(mri_ds, seq_index)
                               if (mri_ds is not None and seq_index is not None)
                               else (None, None))
            dz = data.get("dz_mm")
            dz = float(dz.reshape(-1)[b]) if dz is not None else None

            row = {
                "epoch": int(self.epoch),
                "step": int(train_step),
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
            # Every metric the loss already computed for this sample (B == 1).
            for k, v in data.items():
                if k.startswith("metric_") and k not in row:
                    try:
                        row[k] = float(v.item() if torch.is_tensor(v) else v)
                    except (TypeError, ValueError, RuntimeError):
                        pass
            # Breathing damage is not a `metric_*` key, and cannot be recovered after the run.
            row.update(resp_offslab_stats(data, b))
            self.run_log.subject_row(row)

            for axis, group in (("source", source), ("pitch", pitch_bucket(dz))):
                if group is None:
                    continue
                for metric in STRATA_METRICS:
                    val = row.get(metric)
                    if val is not None:
                        self.val_strata[(axis, group, metric)].append(val)
        except Exception as e:
            logging.warning(f"per-subject val record failed (ignored): {e}")

    def _log_slot_selection(self, data: Mapping, step: int):
        """Raw (t, z) picks of the first three slots, and how many input slots happen to sit
        at the target phase (a free zero-motion anchor) — sampler sanity checks."""
        if "timesteps" not in data or "slice_indices" not in data:
            return
        ts = data["timesteps"]
        sls = data["slice_indices"]
        S = ts.shape[1] if hasattr(ts, "shape") else len(ts[0])
        for slot in range(min(S, 3)):
            self.scalar(f"train/slice_selection/slot{slot + 1}_frame", ts[0, slot].item(), step)
            self.scalar(f"train/slice_selection/slot{slot + 1}_slice", sls[0, slot].item(), step)
        if "t_target" in data:
            try:
                tt = data["t_target"]  # (B, 1)
                n_anchor = (ts == tt).float().sum(dim=1).mean().item()
                self.scalar("train/slice_selection/n_slots_at_target", n_anchor, step)
            except Exception as e:
                logging.warning(f"n_slots_at_target log failed (ignored): {e}")

    def _log_visuals(self, batch: Mapping, phase: str, step: int, train_step: int) -> None:
        """Volume / DVF (and, in val, Lookup) panels.

        train: every `log_visual_frequency.train` steps, whatever subject the loader drew.
        val:   the visual subjects only, at their ES sample (the ED/ES panel already shows
               both phases); keyed by the sample's own seq_index, mapped through the sweep.
        """
        freq = self.log_conf.log_visual_frequency.get(phase, 0)
        log_step = step if phase == "train" else train_step

        val_idx = None
        if phase == "train":
            should_log = freq > 0 and (step % freq == 0)
        else:
            # Must not raise (runs before the per-figure guards below). _val_iter is the
            # fallback for a batch without seq_index; it equals seq_index at batch size 1.
            try:
                _seqs = batch.get("seq_index")
                val_idx = int(_seqs[0].flatten()[0].item()) if _seqs is not None else int(self._val_iter)
            except Exception:
                val_idx = int(self._val_iter)
            # Compare the SUBJECT index: under the sweep ([all ED] + [all ES]) seq_index
            # equals it only in the ED half.
            _vt = self.val_targets
            sweep_idx = val_idx % len(_vt) if _vt else None
            subj_idx = _vt[sweep_idx][0] if _vt else val_idx
            val_sid, _ = seq_index_to_subject(self.val_ds, val_idx)
            is_es = _vt is None or sweep_idx >= len(_vt) // 2
            should_log = is_es and subj_idx in self.visual_subjects
        if not (self.log_conf.log_visuals and should_log):
            return

        # Caption: subject + t_target + per-slot z/t (+ breathing r/|d|) + step.
        caption_parts = []
        if "seq_name" in batch:
            try:
                raw_seq = batch["seq_name"][0]
                # seq_name is "mri_{mri_mode}_{rel_path}"; strip to the bare subject id.
                subject = raw_seq.split("_", 2)[-1] if raw_seq.startswith("mri_") else raw_seq
                caption_parts.append(f"subj={subject}")
            except Exception:
                pass
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
                resp = batch.get("resp_disp_mm")
                dmag = resp[0].norm(dim=-1) if resp is not None else None  # (S,)
                rphase = batch.get("resp_r")
                rphase = rphase[0] if rphase is not None else None         # (S,)
                def _slot(i):
                    s = f"f{i}: z={slices[i].item()}, t={timesteps[i].item()}"
                    if dmag is not None:
                        s += f", r={rphase[i].item():.2f}, |d|={dmag[i].item():.0f}mm"
                    return s
                caption_parts.append(" | ".join([_slot(i) for i in range(S)]))
            except Exception:
                pass
        caption_parts.append(f"step={log_step}")
        caption = "  ".join(caption_parts)

        if phase == "train":
            name, group = "Train_Visuals", "media_others"
        else:
            # One wandb section per visual subject: index (stable sort) + id (readable).
            name, group = "Val_Visuals", f"media_val_subj{subj_idx}_{val_sid or 'unknown'}"

        try:
            panels.log_volume_and_dvf(self.wandb, batch, name, log_step, caption, group=group)
        except Exception as e:
            logging.warning(f"volume/DVF visual log failed (ignored): {e}")
        # Round-trip lookup panel — val only (GT-referenced).
        if phase != "train":
            try:
                panels.log_lookup(self.wandb, batch, name, log_step, caption, group=group,
                                  reference_slot=self.reference_slot)
            except Exception as e:
                logging.warning(f"lookup visual log failed (ignored): {e}")

    # ── val ──────────────────────────────────────────────────────────────────

    def begin_val(self, epoch: int):
        """Reset the per-epoch val state and decide this epoch's heavy diagnostics:
        predicted EF (needs the ED/ES sweep) and the ED-vs-ES panels."""
        self.epoch = epoch
        self.val_psnr = {kind: defaultdict(list) for kind in PSNR_KINDS}
        self._val_iter = 0
        # One volume dump per subject (per (subject, phase) under the sweep), per epoch.
        self._val_volumes_saved = set()

        sweep = self.val_ds is not None and self.val_targets is not None
        self.ef_this_epoch = bool(
            getattr(self.log_conf, "ef_eval_enable", False) and sweep
            and self.epoch % max(1, getattr(self.log_conf, "ef_eval_every_n_val_epochs", 5)) == 0
        )
        if self.ef_this_epoch:
            self._ef_pred_dir = os.path.join(self.log_conf.log_dir, "ef_tmp", "pred")
            shutil.rmtree(self._ef_pred_dir, ignore_errors=True)
            safe_makedirs(self._ef_pred_dir)

        self.viz_ed_es = bool(
            getattr(self.log_conf, "log_visuals", True) and sweep and self._visual_panel_epoch())
        if self.viz_ed_es:
            self._ed_es_stash = {}

    def on_val_batch(self, batch: Mapping, loss_dict: Mapping):
        """Per-batch val side outputs: volume dumps, EF inputs, ED/ES stash."""
        self._save_val_volumes(batch, loss_dict)
        if self.ef_this_epoch:
            self._save_ef_volume(batch, loss_dict)
        if self.viz_ed_es:
            self._stash_ed_es(batch, loss_dict)

    def end_val(self, step: int, meters: dict, model) -> dict:
        """Log the epoch's val means and every end-of-epoch diagnostic at train step `step`;
        return {raw metric key: epoch mean}."""
        prefix = "Loss/val_"
        means = {}
        for name, meter in meters.items():
            raw_name = name[len(prefix):] if name.startswith(prefix) else name
            # count==0: every batch was non-finite. AverageMeter.avg starts at 0, which would
            # report a perfect 0.0 for a broken epoch; report NaN.
            value = meter.avg if meter.count else float("nan")
            self.scalar(scalar_name("val", raw_name), value, step)
            means[raw_name] = value

        self._log_per_phase_psnr(step)
        self._log_val_strata(step)

        # Cardiac-cycle filmstrip: cross-phase reconstruction (or, in fixed-phase mode,
        # what the model does at phases it never trained on).
        if self.epoch % max(1, getattr(self.log_conf, "filmstrip_every_n_val_epochs", 5)) == 0:
            for subj_idx in self.visual_subjects:
                self._log_cardiac_cycle_filmstrip(model, step, subj_idx)

        if self.viz_ed_es:
            panels.log_ed_es_panels(self.wandb, self._ed_es_stash, self.visual_subjects,
                                    self.val_ds, step)
        if self.ef_this_epoch:
            self._compute_and_log_ef(step)
        return means

    def _log_per_phase_psnr(self, step: int):
        """`val/psnr/<kind>/t{k}_n{n}_base{b:.1f}` per phase and `.../mean_n{n}_base{b:.1f}`.

        n and the identity baseline are baked into the name: val is deterministic, so each
        phase keeps one stable panel, and n drifting (new panels appearing) is the smoke
        alarm for lost determinism. Single-phase runs log only the motion panels — full/bbox
        are already the standard val meters."""
        for kind in PSNR_KINDS:
            namespace = f"val/psnr/{kind}"
            accum = self.val_psnr[kind]
            base = self.identity_baseline.get(kind, {})
            baseline_per_phase, baseline_mean = base.get("per_phase_mean"), base.get("mean_psnr")
            if len(accum) == 0:
                continue
            if self.t_target_fixed is not None and kind != "motion":
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
                    self.scalar(f"{namespace}/t{t}_n{n}{base_tag}", phase_mean, step)
                n_total = len(all_psnrs)
                overall_mean = float(sum(all_psnrs) / n_total)
                base_tag = f"_base{baseline_mean:.1f}" if baseline_mean is not None else ""
                self.scalar(f"{namespace}/mean_n{n_total}{base_tag}", overall_mean, step)
                # Also to the text log: Δ>0 means the model beats the identity baseline.
                per_phase_str = " ".join(f"t{t}={m:.2f}" for t, m in per_phase_means)
                if baseline_mean is not None:
                    head = f"mean={overall_mean:.2f} (Δ={overall_mean - baseline_mean:+.2f} vs identity {baseline_mean:.2f})"
                else:
                    head = f"mean={overall_mean:.2f}"
                logging.info(f"[val per-phase @ step {step}] {namespace}: {head} | {per_phase_str}")
            except Exception as e:
                logging.warning(f"per-phase {namespace} log failed (ignored): {e}")

    def _log_val_strata(self, step: int):
        """Per-source and per-pitch val means, then clear. The pooled mean hides a one-source
        collapse, and pitch is the native-z tripwire. `n` is its own scalar per metric (the
        heart-ROI metrics are conditional, so counts differ), not baked into the name, so a
        split edit does not orphan the curve. docs/60."""
        if not self.val_strata:
            return
        try:
            for (axis, group, metric), values in sorted(self.val_strata.items()):
                if not values:
                    continue
                short = metric.replace("metric_psnr_3d_", "psnr_").replace("metric_", "")
                self.scalar(f"val/strata/{axis}/{group}/{short}",
                            float(sum(values) / len(values)), step)
                self.scalar(f"val/strata/{axis}/{group}/{short}_n", float(len(values)), step)
            groups = sorted({(a, g) for (a, g, _) in self.val_strata})
            logging.info(f"[val strata @ step {step}] " + " | ".join(
                f"{a}:{g}" for a, g in groups))
        except Exception as e:
            logging.warning(f"val strata log failed (ignored): {e}")
        finally:
            self.val_strata.clear()

    def _log_cardiac_cycle_filmstrip(self, model, step: int, subj_idx: int = 0):
        """Reconstruct one val subject at all 12 phases and log a GIF (V_gt top / V_canon
        bottom, N_FILM_PLANES planes spanning the stack). One input batch is built and reused;
        in reference-slot mode slot 0 is rebuilt at each phase so it observes that phase.
        Read-only over model and dataset."""
        if not self.wandb:
            return
        try:
            import numpy as np
            import torch.nn.functional as _F
            from loss import compute_volume_intensity_loss
        except ImportError:
            return

        mri_ds = self.val_ds
        if mri_ds is None:
            return

        canon_frames = []
        gt_frames = []
        try:
            model.eval()
            data = mri_ds.get_data(seq_index=subj_idx, img_per_seq=mri_ds.num_slices)
            batch = self._subject_device_batch(data, subj_idx)
            batch["z_indices"] = torch.from_numpy(
                np.stack(data["z_indices"]).astype(np.float32)).unsqueeze(0).to(self.device)
            # Full canonical phase bundle → per-phase V_gt without re-sampling inputs.
            phases_bundle = torch.from_numpy(
                np.asarray(data["phases"]).astype(np.float32)).to(self.device)  # (T, D, H, W)
            batch["phases"] = phases_bundle.unsqueeze(0)
            # The real corrupted->clean val task: val breathing on the INPUT slices (V_gt
            # stays at the unshifted reference). Without a reference slot the inputs are the
            # same for every phase, so breathing is applied once; with one, slot 0 changes
            # per phase and the inputs are re-extracted inside the loop.
            do_resp = (self.respiratory_cfg is not None
                       and getattr(self.respiratory_cfg, "enable", False))
            if not self.reference_slot:
                batch = gpu_augment_batch(
                    batch, None, self.device,
                    respiratory_cfg=self.respiratory_cfg, train=False)
            ref_zmid = None
            hw = batch["images"].shape[-1]
            if self.reference_slot:
                # z_mid (bbox z-center) is constant across the sweep, so slot 0's
                # scanner_coords / z_indices from get_data stay valid; only its phase changes.
                _bb = np.asarray(data["anatomy_bbox"]).astype(np.int64)
                ref_zmid = (int(_bb[0]) + int(_bb[1])) // 2
            for t in range(NUM_PHASES):
                if self.reference_slot:
                    batch["timesteps"][:, 0] = t  # slot 0 observes the target phase t
                    if do_resp:
                        # Re-extract every input through the deterministic val breathing
                        # pipeline (the scattered slots come out identical each phase).
                        batch = gpu_augment_batch(
                            batch, None, self.device,
                            respiratory_cfg=self.respiratory_cfg, train=False)
                    else:
                        ref_up = _F.interpolate(
                            phases_bundle[t, ref_zmid][None, None].float(), size=(hw, hw),
                            mode="bilinear", align_corners=True)  # (1, 1, R, R) in [0, 1]
                        batch["images"][:, 0] = ref_up.repeat(1, 3, 1, 1)
                        # Rebuild images_splat for slot 0 = phase t (native render, docs/73).
                        batch = gpu_augment_batch(
                            batch, None, self.device, respiratory_cfg=None, train=False)
                batch["gt_target_volume"] = phases_bundle[t].unsqueeze(0)  # (1, D, H, W)
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    preds = model(batch["images"], batch=batch)
                    out = compute_volume_intensity_loss(
                        {"world_points": preds["world_points"].float()},
                        batch, splat_res=self.splat_res,
                    )
                V_canon = out["V_canon"][0].float().cpu().numpy()
                V_gt = out["V_gt"][0].float().cpu().numpy()
                # N_FILM_PLANES planes evenly spanning the stack, stacked into (N*H, W).
                window = pick_planes(V_canon.shape[0], N_FILM_PLANES)
                canon_frames.append(np.concatenate([V_canon[c] for c in window], axis=0))
                gt_frames.append(np.concatenate([V_gt[c] for c in window], axis=0))
        except Exception as e:
            logging.warning(f"cardiac filmstrip render failed (ignored): {e}")
            return

        v_vmax = float(max(max(f.max() for f in canon_frames),
                           max(f.max() for f in gt_frames),
                           1e-3))
        # Fixed-phase models never trained at other phases: say so in the caption, so an
        # out-of-distribution V_canon is not misread as a failed reconstruction.
        if self.t_target_fixed is not None:
            mode_note = f" | fixed-t={self.t_target_fixed}: V_canon at t≠{self.t_target_fixed} is out-of-distribution"
        else:
            mode_note = ""
        panels.log_cardiac_cycle_gif(self.wandb, gt_frames, canon_frames, v_vmax,
                                     mri_ds, subj_idx, mode_note, step)

    def _save_val_volumes(self, batch: Mapping, loss_dict: Mapping) -> None:
        """Dump pred + GT volumes to ${log_dir}/val_volumes/, once per subject (per
        (subject, phase) under the EF sweep, so ED and ES are both kept) per epoch. Val is
        deterministic, so the same filenames are overwritten in place every epoch.

        The affine carries the real voxel sizes (docs/60): in-plane 1.4 mm for everyone, z
        this subject's own pitch (docs/59 F14). The array is in splat order (D,H,W)=(Z,Y,X),
        so the diagonal is (dz, 1.4, 1.4) — the axes are NOT permuted to (X,Y,Z).
        """
        if not getattr(self.log_conf, "save_val_volumes", False):
            return
        if "V_canon" not in loss_dict or "V_gt" not in loss_dict:
            return
        try:
            import nibabel as nib
            import numpy as np

            out_dir = os.path.join(self.log_conf.log_dir, "val_volumes")
            safe_makedirs(out_dir)

            V_canon = loss_dict["V_canon"].detach().float().cpu().numpy()  # (B, D, H, W)
            V_gt = loss_dict["V_gt"].detach().float().cpu().numpy()
            t_targets = batch.get("t_target")
            seq_names = batch.get("seq_name", [])
            B = V_canon.shape[0]
            dz_all = batch.get("dz_mm")
            saved = self._val_volumes_saved
            sweeping = self.val_ds is not None and self.val_targets is not None

            for b in range(B):
                raw_seq = seq_names[b] if b < len(seq_names) else f"unknown{b}"
                # seq_name is "mri_{mri_mode}_{rel_path}"; keep the short subject part.
                subject = raw_seq.split("_", 2)[-1] if raw_seq.startswith("mri_") else raw_seq
                t_val = int(t_targets[b].flatten()[0].item()) if t_targets is not None else -1
                key = (subject, t_val) if sweeping else subject
                if key in saved:
                    continue
                subj_idx = len(saved)  # deterministic first-seen order
                saved.add(key)
                stem = f"subj{subj_idx:02d}_t{t_val:02d}_{subject}"
                dz = float(dz_all.reshape(-1)[b]) if dz_all is not None else 1.0
                affine = np.diag([dz, IN_PLANE_MM, IN_PLANE_MM, 1.0]).astype(np.float32)
                nib.save(nib.Nifti1Image(V_canon[b], affine), os.path.join(out_dir, f"{stem}_pred.nii.gz"))
                nib.save(nib.Nifti1Image(V_gt[b], affine), os.path.join(out_dir, f"{stem}_gt.nii.gz"))
        except Exception as e:
            logging.warning(f"val-volume save failed (ignored): {e}")

    def _save_ef_volume(self, batch: Mapping, loss_dict: Mapping) -> None:
        """On EF epochs, dump each reconstructed volume to ef_tmp/pred/ in nnU-Net input
        format (X,Y,Z / (1.4, 1.4, this subject's own dz) / _0000 — docs/59 F14). The sweep
        visits each subject at ED and ES, so both land here; filenames use the clean subject
        id (matches the cardiac-phase CSV)."""
        if "V_canon" not in loss_dict:
            return
        try:
            import ef_eval
            vt = self.val_targets
            if vt is None:
                return
            V_canon = loss_dict["V_canon"].detach().float().cpu().numpy()  # (B, D, H, W)
            seqs = batch.get("seq_index")
            for b in range(V_canon.shape[0]):
                si = int(seqs[b].flatten()[0].item())
                subj_idx, t = vt[si % len(vt)]
                subject = os.path.basename(os.path.dirname(self.val_ds.subjects[subj_idx]))
                dz = float(batch["dz_mm"].reshape(-1)[b])
                ef_eval.save_pred_volume(V_canon[b], self._ef_pred_dir, subject, int(t), dz_mm=dz)
        except Exception as e:
            logging.warning(f"[ef] save pred volume failed (ignored): {e}")

    def _compute_and_log_ef(self, step: int) -> None:
        """End of an EF epoch: one batched segmentation of the saved ED/ES volumes, then
        predicted-vs-GT EF slope / Spearman / MAE (overall and per pathology group)."""
        try:
            import glob
            import ef_eval
            mri_ds = self.val_ds
            vt = self.val_targets
            csv_path = getattr(mri_ds, "cardiac_phase_csv", None)
            if vt is None or csv_path is None:
                return
            n_vols = len(glob.glob(os.path.join(self._ef_pred_dir, "*_0000.nii.gz")))
            if n_vols == 0:
                logging.warning("[ef] no pred volumes written; skipping EF")
                return
            seg_dir = os.path.join(self.log_conf.log_dir, "ef_tmp", "seg_pred")
            shutil.rmtree(seg_dir, ignore_errors=True)
            torch.cuda.empty_cache()  # release cached GPU mem so the segmenter has room
            backend = getattr(self.log_conf, "ef_seg_backend", "corseg")
            lv_label = ef_eval.segment(self._ef_pred_dir, seg_dir, backend=backend)
            # vt = [all ED] + [all ES]: vt[i] and vt[i+N] are the same subject.
            N = len(vt) // 2
            subjects_ed_es = [
                (os.path.basename(os.path.dirname(mri_ds.subjects[vt[i][0]])), vt[i][1], vt[i + N][1])
                for i in range(N)
            ]
            groups = load_subject_groups(getattr(mri_ds, "split_file", None), "pathology_label")
            m = ef_eval.compute_ef_metrics(seg_dir, subjects_ed_es, csv_path,
                                           groups=groups, lv_label=lv_label)
            if m is None:
                logging.warning("[ef] too few valid subjects for EF correlation; skipping")
                return
            for k in ("slope", "spearman", "mae_pct"):
                self.scalar(f"val/ef/{k}", m[k], step)
            self.scalar("val/ef/n", m["n"], step)
            for name, g in (m.get("by_group") or {}).items():
                for k in ("slope", "spearman", "mae_pct", "n"):
                    self.scalar(f"val/ef/{name}/{k}", g[k], step)
            logging.info(f"[ef] epoch {self.epoch}: slope={m['slope']:.3f} "
                         f"spearman={m['spearman']:.3f} mae={m['mae_pct']:.2f}% "
                         f"n={m['n']} (skipped {m['n_skipped']})"
                         + "".join(f" | {name}: slope={g['slope']:.3f} n={g['n']}"
                                   for name, g in sorted((m.get("by_group") or {}).items())))
        except Exception as e:
            logging.warning(f"[ef] compute/log EF failed (ignored): {e}")

    def _stash_ed_es(self, batch: Mapping, loss_dict: Mapping) -> None:
        """Keep the ED and ES reconstructions of the visual subjects for end_val's ED-vs-ES
        panels. Keyed by the sample's own seq_index (blocked sweep: all ED, then all ES)."""
        if "V_canon" not in loss_dict or "V_gt" not in loss_dict:
            return
        try:
            vt = self.val_targets
            seqs = batch.get("seq_index")
            i = int(seqs[0].flatten()[0].item()) if seqs is not None else int(self._val_iter)
            if vt is None or i >= len(vt):
                return
            subj_idx = vt[i][0]
            if subj_idx not in self.visual_subjects:
                return
            role = "ED" if i < len(vt) // 2 else "ES"
            self._ed_es_stash.setdefault(subj_idx, {})[role] = {
                "images": panels._gray_slices(batch["images"][0]),          # (S, H, W) gray
                "V_gt": loss_dict["V_gt"][0].detach().float().cpu().numpy(),      # (D, H, W)
                "V_canon": loss_dict["V_canon"][0].detach().float().cpu().numpy(),
                "t": int(batch["t_target"][0].flatten()[0].item()) if "t_target" in batch else -1,
                "timesteps": batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None,
                "slices": batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None,
            }
        except Exception as e:
            logging.warning(f"[ed/es] stash failed (ignored): {e}")
