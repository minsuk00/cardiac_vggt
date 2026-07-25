# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""wandb / matplotlib visualisation + diagnostic logging for :class:`Trainer`.

Split out of ``trainer.py`` verbatim: these methods are read-only observers of
trainer state (they never mutate it and never participate in the training step),
so they live in a mixin to keep ``trainer.py`` focused on the training loop.
Heavy plotting deps (matplotlib, wandb, nibabel) stay function-local, exactly as
they were before the move, so importing this module is cheap.
"""

import json
import logging
import os
from collections import defaultdict
from typing import Mapping

import torch

from data.gpu_aug import gpu_augment_batch
from train_utils.general import safe_makedirs


class TrainerVizMixin:
    """Visualisation / diagnostic logging methods mixed into :class:`Trainer`."""

    def _compute_identity_baseline(self):
        """Run identity-Δ (no motion correction) on the val set and log PSNR as constants.
        Called ONCE at trainer setup. Read-only over the val dataset; never touches the
        training path. Failures are caught and logged — training proceeds either way.
        """
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
            try:
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
            finally:
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
                with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype):
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
            try:
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
            finally:
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
            try:
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
            finally:
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
            try:
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
            finally:
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
            # Index by the SAMPLE's own seq_index (as _save_ef_volume does), not the
            # per-BATCH counter _val_iter. They coincide only while the val batch size is
            # 1 (max_img_per_gpu // img_nums == 1); raising max_img_per_gpu would other-
            # wise silently mislabel which subject/phase each stashed panel belongs to.
            seqs = batch.get("seq_index")
            i = int(seqs[0].flatten()[0].item()) if seqs is not None else int(self._val_iter)
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
        try:
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
        finally:
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

    def _log_visuals_to_wandb(self, batch: Mapping, phase: str, step: int) -> None:
        """Dispatches the per-step image/video visualizations to wandb."""

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
        val_idx = None
        if phase == "train":
            should_log = freq > 0 and (step % freq == 0)
        else:
            # Identify the val sample by its OWN seq_index rather than the per-batch
            # counter _val_iter (same reasoning as _stash_ed_es): they are equal only
            # while the val batch size is 1, and keying on _val_iter would otherwise
            # select the wrong samples and mislabel the per-subject wandb section.
            # Defensive — this runs BEFORE the per-figure try/except blocks below and
            # this method's call site (trainer.py) is NOT guarded, so it must not raise.
            # _val_iter is the fallback: it resets each val epoch, whereas
            # self.steps["val"] is monotonic and would skip these visuals after the
            # first epoch / on resume.
            try:
                _seqs = batch.get("seq_index")
                val_idx = int(_seqs[0].flatten()[0].item()) if _seqs is not None else int(self._val_iter)
            except Exception:
                val_idx = int(self._val_iter)
            should_log = val_idx in VAL_VISUAL_SUBJECT_INDICES
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
                name, group = "Val_Visuals", f"media_val_subj{val_idx}"

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
