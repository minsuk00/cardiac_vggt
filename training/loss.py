# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch
from data.preprocess import Z_HALF_MM
from vggt.utils.splat import sample_volume, splat_to_volume, splat_predictions


# Threshold for the cardiac-motion mask (see compute_motion_mask). A canonical
# voxel counts as "moving" if its intensity swings by more than this across the
# 12 cardiac phases. 0.05 (on the [0, 1] normalized scale) isolates the LV/RV
# blood pool + myocardium (~3-5% of in-bbox voxels) — see tools/preview_motion_mask.py.
MOTION_MASK_TAU = 0.05


def compute_motion_mask(phases, tau=MOTION_MASK_TAU):
    """Per-voxel cardiac-motion mask from the full canonical phase bundle.

    motion[z,y,x] = max_t phases[t,z,y,x] - min_t phases[t,z,y,x]; mask = motion > tau.
    Static tissue (chest wall, liver, fat) barely changes across the cycle → False;
    the dynamic heart → True. No segmentation needed.

    Args:
        phases: (B, T, D, H, W) phase bundle (the batch["phases"] field; post-aug
                if augmentation ran, since the trainer overwrites it in place).
        tau: intensity-swing threshold on the normalized [0, 1] scale.

    Returns:
        (B, D, H, W) bool mask on the same device as `phases`.
    """
    # Reduce over T FIRST (in the input dtype) so we never materialize a float32
    # copy of the full (B, T, D, H, W) bundle — amax/amin are exact selections, so
    # fp16 input loses nothing here; only the small (B, D, H, W) swing is upcast.
    swing = phases.amax(dim=1) - phases.amin(dim=1)   # (B, D, H, W)
    return swing.float() > tau


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.

    Supports:
    - Camera loss
    - Depth loss
    - Point loss
    - Volume intensity loss (unsupervised slice-to-volume)
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """

    def __init__(self, volume=None, **kwargs):
        super().__init__()
        # Volume-intensity loss configuration (the only active task).
        self.volume = volume

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        # Direct volume-to-volume loss against the GT phase-0 volume loaded from disk.
        if "world_points" in predictions and self.volume is not None and self.volume.get("weight", 0) > 0:
            vol_loss_dict = compute_volume_intensity_loss(predictions, batch, **self.volume)
            vol_loss = (vol_loss_dict["loss_volume"] + vol_loss_dict["loss_pos_tv"]
                        + vol_loss_dict.get("loss_diffusion", 0.0)
                        + vol_loss_dict.get("loss_gather", 0.0)) * self.volume["weight"]
            total_loss = total_loss + vol_loss
            loss_dict.update(vol_loss_dict)

        loss_dict["objective"] = total_loss

        return loss_dict


def diffusion_loss_l2(field):
    """L2 diffusion (Tikhonov) smoothness regularizer ‖∇u‖² on a displacement field.

    Mean of SQUARED in-plane (H, W) neighbor differences over a (B, S, H, W, C) field.
    Unlike `tv_loss` (L1, edge-preserving, ∝|ω|), the squared gradient is ∝ω² and
    smoothness-promoting — the VoxelMorph diffusion regularizer that actively suppresses
    high-frequency (e.g. ViT-patch-period) ripples in the predicted warp. fp32-forced so
    the reduction stays accurate under autocast(bf16).
    """
    with torch.amp.autocast("cuda", enabled=False):
        f = field.float()
        return (
            (f[:, :, 1:, :, :] - f[:, :, :-1, :, :]).pow(2).mean()
            + (f[:, :, :, 1:, :] - f[:, :, :, :-1, :]).pow(2).mean()
        )


def compute_volume_intensity_loss(predictions, batch, tv_weight=0.1,
                                  diffusion_weight=0.0, gather_weight=0.0, **kwargs):
    """Direct volume-to-volume loss: splat input pixels to V_canon, compare to V_gt.

    Pipeline:
        input slices → per-pixel predicted positions → splat into V_canon (B, D, H, W)
        loss = |V_canon - V_gt|  averaged over voxels with GT anatomy.

    V_gt is the target-phase NIfTI loaded from disk by the dataset (`batch["gt_target_volume"]`),
    resampled to the canonical grid in the same per-axis normalized [-1, 1] frame as scanner_coords.
    The target phase (slot 0's t_idx) is sampled per call by the dataset.

    Args:
        predictions: dict with "world_points" (B, S, H, W, 3) — per-pixel canonical position in [-1, 1].
        batch: dict with "images" (B, S, 3, H, W), "gt_target_volume" (B, D, H, W), "z_scale" (B,).
            grid_shape is DERIVED from gt_target_volume's own shape (docs/58, native-z) — D varies
            per subject, so it can no longer be a fixed config constant.
        tv_weight: weight for the spatial smoothness regularizer on pos_pred.
    """
    if "gt_target_volume" not in batch:
        raise RuntimeError("compute_volume_intensity_loss requires batch['gt_target_volume'].")
    if "z_scale" not in batch:
        raise RuntimeError("compute_volume_intensity_loss requires batch['z_scale'] (docs/58).")

    pos_pred = predictions["world_points"]
    V_gt = batch["gt_target_volume"]
    grid_shape = tuple(V_gt.shape[1:])
    # batch_size is always 1 in this pipeline, so a single scalar z_scale describes the whole
    # batch. Guard it (docs/59 F7): under native-z, two subjects with the same D but different
    # dz collate SUCCESSFULLY, and row 1 would then be splatted at row 0's scale — a silent
    # 20%+ through-plane geometry error. Different-D pairs already crash loudly on collate.
    _zs = batch["z_scale"].reshape(-1)
    if not bool((_zs == _zs[0]).all()):
        raise RuntimeError(
            f"z_scale is not uniform across the batch: {_zs.tolist()}. One scalar z_scale is "
            "applied to every row, so mixing slice pitches would splat rows 1..B-1 at row 0's "
            "scale — a silent through-plane geometry error (docs/59 F7)."
        )
    z_scale = float(batch["z_scale"].reshape(-1)[0])

    V_canon, coverage = splat_predictions(predictions, batch, grid_shape, z_scale)

    if V_gt.shape != V_canon.shape:
        raise RuntimeError(f"gt_target_volume {tuple(V_gt.shape)} must match V_canon {tuple(V_canon.shape)}")

    B = V_canon.shape[0]

    # Main loss: naive L1 over the full canonical volume.
    # Previously we masked by (V_gt > 1e-3) — i.e. only voxels where the GT
    # phase-0 NIfTI had anatomy contributed. That kept the average focused on
    # tissue voxels but also gave the model a "free pass" to over-predict
    # intensity anywhere V_gt was zero (lungs, background outside the body),
    # which showed up as ghost blobs in the V_canon - V_gt panel.
    # Naive L1 over every voxel penalizes those over-predictions symmetrically.
    # valid = (V_gt > 1e-3).float()
    # denom = valid.sum().clamp(min=1.0)
    # loss_volume = ((V_canon - V_gt).abs() * valid).sum() / denom
    loss_volume = (V_canon - V_gt).abs().mean()

    # Plain TV on pos_pred — mean absolute difference between H/W neighbors. fp32-forced
    # so the reduction stays accurate under autocast(bf16).
    with torch.amp.autocast("cuda", enabled=False):
        pos_fp = pos_pred.float()
        loss_pos_tv = (
            (pos_fp[:, :, 1:, :, :] - pos_fp[:, :, :-1, :, :]).abs().mean()
            + (pos_fp[:, :, :, 1:, :] - pos_fp[:, :, :, :-1, :]).abs().mean()
        ) * tv_weight

    # Optional L2 diffusion regularizer ‖∇u‖² on the DISPLACEMENT field (VoxelMorph-style).
    # Penalizes squared in-plane gradients of the residual DVF (the true displacement u),
    # not the absolute world position — so a smooth warp pays nothing and the ViT-patch
    # ripple is actively suppressed. diffusion_weight=0.0 ⇒ exactly 0.0 ⇒ no-op (bit-identical).
    if diffusion_weight > 0:
        u = predictions.get("dvfs", pos_pred)   # residual DVF if present, else world_points
        loss_diffusion = diffusion_loss_l2(u) * diffusion_weight
    else:
        loss_diffusion = pos_pred.new_zeros(())

    # Optional coverage-free GATHER-placement auxiliary (docs/37, docs/38). For each input pixel at
    # its predicted world position p, SAMPLE V_gt at p ("pull") and match it to that pixel's own
    # intensity I:  L = |sample_volume(V_gt, p) − I|. No coverage division ⇒ it restores the sharp
    # through-plane placement gradient the splat's ÷coverage flattens into a plateau; the splat L1
    # stays primary (keeps V_canon complete/coherent). No anatomical/motion mask — only the standard
    # padded-pixel gate (intensity>1e-3), same as splat_predictions (docs/38). One grid_sample —
    # cheaper than the splat. Uses the SAME input intensity as `splat_predictions`. padding_mode
    # 'zeros' in sample_volume means predicting p outside the FOV samples 0 → mismatch → the aux
    # discourages moving pixels out of bounds. gather_weight=0.0 ⇒ exactly 0.0 ⇒ no-op (bit-identical).
    if gather_weight > 0:
        with torch.amp.autocast("cuda", enabled=False):
            # Same branchless [0,255]→[0,1] rescale as `splat_predictions` (reciprocal-multiply,
            # so it stays bit-identical to the pre-refactor `gi / 255.0` — see splat.py).
            inv_scale_gi = torch.where((batch["images"] > 2.0).any(), 1.0 / 255.0, 1.0)
            gi = batch["images"].float().mean(dim=2) * inv_scale_gi   # (B, S, H, W) input intensity
            gi = gi.reshape(gi.shape[0], -1)                  # (B, S*H*W)
            gs = sample_volume(V_gt.float(), pos_pred.float().reshape(gi.shape[0], -1, 3), z_scale)  # (B, S*H*W)
            gmask = (gi > 1e-3).float()                       # only real acquired (non-padded) pixels
            loss_gather = ((gs - gi).abs() * gmask).sum() / gmask.sum().clamp(min=1.0) * gather_weight
    else:
        loss_gather = pos_pred.new_zeros(())

    out = {
        "loss_volume": loss_volume,
        "loss_pos_tv": loss_pos_tv,
        "loss_diffusion": loss_diffusion,
        "loss_gather": loss_gather,
        "V_canon": V_canon,
        "V_gt": V_gt,
        "coverage": coverage,
    }

    with torch.no_grad():
        # ── Full-volume metrics (over all D*H*W voxels of the canonical cube) ──
        # These match what `loss_volume` averages over and what the L1 loss is
        # actually optimizing. For small-FOV subjects the cube has many (V_gt=0,
        # V_canon≈0) padded voxels that inflate PSNR — see `_bbox` companion below.
        mse_full = ((V_canon - V_gt) ** 2).mean()
        psnr_full = 10.0 * torch.log10(torch.tensor(1.0, device=mse_full.device) / mse_full.clamp(min=1e-10))
        out["metric_mae_3d_full"] = loss_volume.detach()
        out["metric_mse_3d_full"] = mse_full
        out["metric_psnr_3d_full"] = psnr_full
        out["metric_gt_coverage_frac"] = (V_gt > 1e-3).float().mean()  # data property
        out["metric_coverage_frac"] = (coverage > 1e-3).float().mean()
        out["metric_coverage_mean"] = coverage.mean()
        if "scanner_coords" in batch:
            out["metric_mean_disp_norm"] = (pos_pred - batch["scanner_coords"]).abs().sum(-1).mean()
        if V_canon.is_cuda:
            try:
                # PER-SLICE 2D SSIM. REPLACES the old `metric_ssim_3d_full` (docs/59 F11):
                # `fused_ssim3d` slides an 11-tap (radius-5) window in ALL three dims with zero
                # padding, so the fraction of z-planes contaminated by the padded edge depends on
                # D — which under native-z varies 5-21 ACROSS SUBJECTS. Measured on one structured
                # volume cropped to different depths (same content, same error field, so only D
                # changes): 3D reads 0.9929@D=5 -> 0.9939@D=32, while this per-slice form reads
                # 0.9947 -> 0.9946 (~10x flatter). Reshaping (B,D,H,W) -> (B*D,1,H,W) treats z as
                # a batch dim, removing the z-padding entirely: the window is only ever in-plane
                # and D just sets how many slices are averaged.
                #
                # The 3D metric was DROPPED rather than kept alongside: its only argument was
                # continuity with pre-native-z runs, but those are not comparable anyway (V_gt
                # frame, normalization and grid all changed), so logging both would just be two
                # numbers where one is knowingly wrong.
                from fused_ssim import fused_ssim
                # (B, D, H, W) -> (B*D, 1, H, W): every slice is an independent 2D image.
                pred_s = V_canon.reshape(-1, 1, *V_canon.shape[-2:]).float().contiguous()
                targ_s = V_gt.reshape(-1, 1, *V_gt.shape[-2:]).float().contiguous()
                out["metric_ssim_2d_full"] = fused_ssim(pred_s, targ_s, train=False)
            except Exception:
                pass

        # ── Bbox-cropped metrics (only voxels inside the subject's native FOV) ──
        # Each subject's `anatomy_bbox` was derived geometrically from a content
        # mask propagated through the same spatial transforms as the data — not
        # from intensity thresholding. For small-FOV subjects this excludes the
        # padded zeros that inflate the full-volume PSNR; for large-FOV subjects
        # (bbox = full cube) bbox metrics ≡ full metrics. Bbox SSIM is skipped
        # because per-sample shape varies and `fused_ssim3d` wants a fixed size.
        if "anatomy_bbox" in batch:
            # Vectorized with spatial coordinate masks instead of a per-sample Python loop, so
            # the metric costs no `.tolist()` host-device syncs. (This code never runs inside a
            # compiled region — only the aggregator's attention blocks are compiled — so the
            # motivation is eager sync removal, not graph breaks.)
            bboxes = batch["anatomy_bbox"].to(V_canon.device)   # (B, 6) int64
            D, H, W = V_canon.shape[1], V_canon.shape[2], V_canon.shape[3]
            z_idx = torch.arange(D, device=V_canon.device).view(1, -1, 1, 1)
            y_idx = torch.arange(H, device=V_canon.device).view(1, 1, -1, 1)
            x_idx = torch.arange(W, device=V_canon.device).view(1, 1, 1, -1)

            z0, z1 = bboxes[:, 0:1, None, None], bboxes[:, 1:2, None, None]
            y0, y1 = bboxes[:, 2:3, None, None], bboxes[:, 3:4, None, None]
            x0, x1 = bboxes[:, 4:5, None, None], bboxes[:, 5:6, None, None]

            bbox_mask = (z_idx >= z0) & (z_idx < z1) & (y_idx >= y0) & (y_idx < y1) & (x_idx >= x0) & (x_idx < x1)
            valid_bbox = (z1 > z0) & (y1 > y0) & (x1 > x0)
            bbox_mask = torch.where(valid_bbox, bbox_mask, torch.ones_like(bbox_mask))

            # `torch.where(mask, diff, 0)` rather than `diff * mask.float()`: NaN * 0.0 == NaN,
            # so multiplying lets a single non-finite voxel ANYWHERE in the cube (a diverged
            # step, bad aug) turn this whole metric into NaN. The pre-vectorization loop read
            # only in-ROI voxels via boolean indexing and so was immune. The trainer's
            # non-finite guard skips backward but logs scalars first, and val has no guard.
            err = V_canon - V_gt
            zero = torch.zeros((), device=V_canon.device, dtype=err.dtype)
            diff_sq = torch.where(bbox_mask, err ** 2, zero)
            diff_abs = torch.where(bbox_mask, err.abs(), zero)
            mask_sum = bbox_mask.float().sum(dim=(1, 2, 3)).clamp(min=1.0)

            mse_bbox = diff_sq.sum(dim=(1, 2, 3)) / mask_sum
            mae_bbox = diff_abs.sum(dim=(1, 2, 3)) / mask_sum
            psnr_bbox = 10.0 * torch.log10(torch.tensor(1.0, device=V_canon.device) / mse_bbox.clamp(min=1e-10))

            out["metric_mae_3d_bbox"] = mae_bbox.mean()
            out["metric_mse_3d_bbox"] = mse_bbox.mean()
            out["metric_psnr_3d_bbox"] = psnr_bbox.mean()

        # ── Motion-masked metrics (only voxels that move across the cardiac cycle) ──
        if "phases" in batch:
            # Vectorized (no per-sample `bool(m.any())` host sync). Masked with torch.where,
            # not a float multiply — see the bbox block above for why (NaN * 0.0 == NaN).
            motion_mask = compute_motion_mask(batch["phases"])  # (B, D, H, W) bool
            motion_cnt = motion_mask.float().sum(dim=(1, 2, 3))
            err = V_canon - V_gt
            zero = torch.zeros((), device=V_canon.device, dtype=err.dtype)
            diff_sq = torch.where(motion_mask, err ** 2, zero)
            diff_abs = torch.where(motion_mask, err.abs(), zero)

            mse_m = diff_sq.sum(dim=(1, 2, 3)) / motion_cnt.clamp(min=1.0)
            mae_m = diff_abs.sum(dim=(1, 2, 3)) / motion_cnt.clamp(min=1.0)
            psnr_m = 10.0 * torch.log10(torch.tensor(1.0, device=V_canon.device) / mse_m.clamp(min=1e-10))

            # A sample with zero moving voxels has mse_m == 0 ⇒ psnr_m == 100 dB (the clamp
            # floor), which would silently inflate the batch mean, so average over valid
            # samples only. Kept branchless (no `if n_valid > 0`) to preserve the sync-free /
            # zero-graph-break property of the whole train step: a Python-level decision here
            # costs 4 graph breaks, measured. The degenerate case where NO sample moves
            # therefore reports 0.0 rather than omitting the keys; it cannot occur for real
            # cardiac data (it needs a subject whose 12 phases are identical), and consumers
            # that aggregate this metric filter it via `metric_motion_frac == 0` — see
            # `TrainerVizMixin._compute_identity_baseline`, which feeds baseline_identity.json.
            valid_m = (motion_cnt > 0).float()
            denom_m = valid_m.sum().clamp(min=1.0)
            out["metric_psnr_3d_motion"] = (psnr_m * valid_m).sum() / denom_m
            out["metric_mae_3d_motion"] = (mae_m * valid_m).sum() / denom_m
            out["metric_motion_frac"] = motion_mask.float().mean()

        # ── Anatomy heart-ROI PSNR (val-only) ────────────────────────────────────
        # Same masked PSNR as the motion metric above, but the ROI is the nnU-Net
        # whole-heart segmentation (union over the 12 phases, dilated) resampled onto
        # the canonical grid — one anatomy-defined region per subject, shared with the
        # SVR baselines. Complements motion PSNR (which restricts to *moving* voxels);
        # this covers the whole heart incl. the static blood-pool interior. Gated to
        # val (requires_grad) + skips the startup identity pass, so training numerics
        # are bit-identical. Surfaces in wandb via keys_to_log in mri_volume.yaml.
        if (not pos_pred.requires_grad) and (pos_pred is not batch.get("scanner_coords")) \
                and "heart_roi_canonical" in batch:
            seg_roi = batch["heart_roi_canonical"].bool()   # (B, D, H, W)
            psnr_seg_list, mae_seg_list = [], []
            for b in range(B):
                m = seg_roi[b]
                if not bool(m.any()):
                    continue
                Vc = V_canon[b][m]
                Vg = V_gt[b][m]
                mse_s = ((Vc - Vg) ** 2).mean().clamp(min=1e-10)
                psnr_seg_list.append(10.0 * torch.log10(1.0 / mse_s))
                mae_seg_list.append((Vc - Vg).abs().mean())
            if psnr_seg_list:
                out["metric_psnr_3d_heartseg"] = torch.stack(psnr_seg_list).mean()
                out["metric_mae_3d_heartseg"] = torch.stack(mae_seg_list).mean()
                out["metric_heartseg_frac"] = seg_roi.float().mean()

        # ── VAL-ONLY ship-decision + breathing metrics (docs/37) ─────────────────
        # These quantify targeted improvements that aggregate PSNR buries: an oracle-
        # normalized recoverable-fraction (rescales out the un-fixable appearance wall),
        # a heart/static PSNR split, a coverage-hole tripwire, and breathing through-
        # plane recovery vs the EXACT simulated shift. Gated to val via requires_grad
        # (train forward has grad ⇒ skipped ⇒ training cost + numerics bit-identical);
        # each part try/except-wrapped ⇒ never raises into the loop. Extra splats run
        # only in val. The cardiac-motion mask (compute_motion_mask) is the heart ROI —
        # no segmentation needed. See docs/37 for the design + the stop-grad test.
        # `pos_pred is not batch["scanner_coords"]` also skips the startup identity-baseline
        # pass (which calls this with world_points = scanner_coords) so the extra splats run
        # only in REAL val. To surface in wandb, the metric_* keys below must be listed in
        # `logging.scalar_keys_to_log.val.keys_to_log` (mri_volume.yaml).
        if (not pos_pred.requires_grad) and (pos_pred is not batch.get("scanner_coords")) \
                and "phases" in batch and "scanner_coords" in batch:
            # (1) recov_frac_heart + psnr_static + hole_frac_heart (vs GT, heart ROI)
            try:
                heart = compute_motion_mask(batch["phases"])            # (B,D,H,W) bool
                # identity splat (Δ=0, real corrupted input content) — exact forward path
                V_id, _ = splat_predictions({"world_points": batch["scanner_coords"]}, batch, grid_shape, z_scale)
                # oracle splat (Δ=0, TRUE target-phase content sampled at each pixel's home) —
                # the recoverable ceiling; the model→oracle gap is the appearance wall (docs 19-21).
                inv_scale = torch.where((batch["images"] > 2.0).any(), 1.0 / 255.0, 1.0)
                intensity = batch["images"].float().mean(dim=2) * inv_scale
                scan_flat = batch["scanner_coords"].reshape(B, -1, 3)
                w = (intensity.reshape(B, -1) > 1e-3).float()
                V_or, _ = splat_to_volume(scan_flat, sample_volume(V_gt, scan_flat, z_scale), grid_shape, z_scale, weight=w)
                recov, mse_id_l, mse_mo_l, mse_or_l, holes, static_psnr = [], [], [], [], [], []
                for b in range(B):
                    m = heart[b]
                    if bool(m.any()):
                        g = V_gt[b][m]
                        mse_id = ((V_id[b][m] - g) ** 2).mean()
                        mse_mo = ((V_canon[b][m] - g) ** 2).mean()
                        mse_or = ((V_or[b][m] - g) ** 2).mean()
                        mse_id_l.append(mse_id); mse_mo_l.append(mse_mo); mse_or_l.append(mse_or)
                        holes.append((coverage[b][m] < 0.5).float().mean())
                        span = mse_id - mse_or                          # recoverable span (identity → ceiling)
                        if float(span) > 1e-6:                          # skip if oracle ≯ identity (recov undefined;
                            recov.append(((mse_id - mse_mo) / span).clamp(-0.5, 1.5))  # signed clamp on a signed denom is wrong)
                    st = (V_gt[b] > 1e-3) & (~heart[b])                 # content that does NOT beat (control)
                    if bool(st.any()):
                        mse_s = ((V_canon[b][st] - V_gt[b][st]) ** 2).mean().clamp(min=1e-10)
                        static_psnr.append(10.0 * torch.log10(1.0 / mse_s))
                if mse_id_l:
                    out["metric_mse_heart_identity"] = torch.stack(mse_id_l).mean()
                    out["metric_mse_heart_model"] = torch.stack(mse_mo_l).mean()
                    out["metric_mse_heart_oracle"] = torch.stack(mse_or_l).mean()
                    out["metric_hole_frac_heart"] = torch.stack(holes).mean()
                if recov:
                    out["metric_recov_frac_heart"] = torch.stack(recov).mean()
                if static_psnr:
                    out["metric_psnr_3d_static"] = torch.stack(static_psnr).mean()
            except Exception as e:
                logging.warning(f"docs/38 recov/static val metric failed (ignored): {e}")

            # (2) breathing through-plane recovery vs the EXACT applied sim shift.
            # predicted Δz per slot (mm) vs applied SI (resp_disp_mm[...,0]) →
            # slope/corr/EPE + deep-breath-ignored. Brings tools/exp_4wok_analysis.py online.
            # No-op when breathing is off (resp_disp_mm absent). Slot 0 (reference anchor) is
            # INCLUDED — matched to eval's run_vggt.py:resp_diag so the two numbers are comparable.
            # Per-subject then meter-averaged ⇒ EPE is the robust headline; slope is clamped so one
            # low-applied-variance subject can't dominate; corr is SIGNED Pearson (differs from the
            # offline abs-corr in exp_4wok_analysis.py — see docs/38).
            if "resp_disp_mm" in batch:
                try:
                    # Z_HALF_MM, NOT a per-subject (D-1)/2*dz: z_norm is now PHYSICAL
                    # (z_mm / Z_HALF_MM, docs/58), so one normalized z-unit is always exactly
                    # Z_HALF_MM mm for every subject by construction — unlike the old
                    # index-based scheme where it was each subject's own half-span (which
                    # only looked constant because D/dz never varied). Using the per-subject
                    # half-span here would systematically understate Δz in mm.
                    through_mm = Z_HALF_MM
                    dvf = pos_pred - batch["scanner_coords"]           # (B,S,H,W,3) normalized residual
                    img_int = batch["images"].float().mean(dim=2)      # (B,S,H,W)
                    disp = batch["resp_disp_mm"].float()               # (B,S,3) = (d_D,d_H,d_W) mm
                    sl, co, epe, deep_ign = [], [], [], []
                    for b in range(B):
                        xs, ys = [], []
                        for s in range(dvf.shape[1]):                  # all slots incl. slot 0 (reference)
                            msk = img_int[b, s] > 0.05
                            if bool(msk.any()):
                                xs.append(disp[b, s, 0])
                                ys.append(dvf[b, s, :, :, 2][msk].mean() * through_mm)
                        if len(xs) < 3:
                            continue
                        x = torch.stack(xs); y = torch.stack(ys)
                        xd, yd = x - x.mean(), y - y.mean()
                        epe.append((y - x).abs().mean())               # EPE penalizes gain AND scatter (robust)
                        co.append((xd * yd).sum() / (xd.norm() * yd.norm()).clamp(min=1e-8))
                        sl.append(((xd * yd).sum() / (xd * xd).sum().clamp(min=1e-8)).clamp(-3.0, 3.0))
                        deep = x.abs() >= 12.0
                        if bool(deep.any()):
                            deep_ign.append((y[deep].abs() < 2.0).float().mean())
                    if sl:
                        out["metric_resp_slope_dz"] = torch.stack(sl).mean()
                        out["metric_resp_corr_dz"] = torch.stack(co).mean()
                        out["metric_resp_epe_dz_mm"] = torch.stack(epe).mean()
                    if deep_ign:
                        out["metric_resp_frac_deep_ignored"] = torch.stack(deep_ign).mean()
                except Exception as e:
                    logging.warning(f"docs/38 breathing val metric failed (ignored): {e}")

    return out


