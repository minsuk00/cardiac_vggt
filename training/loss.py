# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass

import torch
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
            # Deep-supervised refiner term (present only when enable_refiner=true; λ already
            # folded into loss_refiner). OFF ⇒ key absent ⇒ vol_loss unchanged (bitwise).
            if "loss_refiner" in vol_loss_dict:
                vol_loss = vol_loss + vol_loss_dict["loss_refiner"] * self.volume["weight"]
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


def compute_volume_intensity_loss(predictions, batch, grid_shape=(12, 256, 256), tv_weight=0.1,
                                  diffusion_weight=0.0, gather_weight=0.0,
                                  refiner_lambda=1.0, refiner_ssim_weight=0.0, **kwargs):
    """Direct volume-to-volume loss: splat input pixels to V_canon, compare to V_gt.

    Pipeline:
        input slices → per-pixel predicted positions → splat into V_canon (B, D, H, W)
        loss = |V_canon - V_gt|  averaged over voxels with GT anatomy.

    V_gt is the target-phase NIfTI loaded from disk by the dataset (`batch["gt_target_volume"]`),
    resampled to the canonical grid in the same per-axis normalized [-1, 1] frame as scanner_coords.
    The target phase (slot 0's t_idx) is sampled per call by the dataset.

    Args:
        predictions: dict with "world_points" (B, S, H, W, 3) — per-pixel canonical position in [-1, 1].
        batch: dict with "images" (B, S, 3, H, W), "gt_target_volume" (B, D, H, W).
        grid_shape: (D, H_v, W_v) canonical volume resolution; must match gt_target_volume.
        tv_weight: weight for the spatial smoothness regularizer on pos_pred.
    """
    if "gt_target_volume" not in batch:
        raise RuntimeError("compute_volume_intensity_loss requires batch['gt_target_volume'].")

    pos_pred = predictions["world_points"]
    V_gt = batch["gt_target_volume"]

    # Refiner path: VGGT.forward already splatted (so the refiner could run inside the
    # DDP-wrapped forward). Reuse those exact tensors. Otherwise splat here — the SAME
    # `splat_predictions` helper the forward uses, so V_canon is byte-identical either way.
    if "V_canon" in predictions:
        V_canon = predictions["V_canon"]
        coverage = predictions["coverage"]
    else:
        V_canon, coverage = splat_predictions(predictions, batch, grid_shape)

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
            gi = batch["images"].float().mean(dim=2)          # (B, S, H, W) input intensity
            if gi.max() > 2.0:
                gi = gi / 255.0
            gi = gi.reshape(gi.shape[0], -1)                  # (B, S*H*W)
            gs = sample_volume(V_gt.float(), pos_pred.float().reshape(gi.shape[0], -1, 3))  # (B, S*H*W)
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

    # ── Refiner (optional): deep-supervised L_post on the refined volume ──────────
    # Present ONLY when VGGT.forward ran the refiner (enable_refiner=true). L_pre above
    # (loss_volume on the raw splat) keeps the point head honest; this L_post trains the
    # refiner. λ (refiner_lambda) is folded in here; MultitaskLoss adds loss_refiner to the
    # objective.
    #
    # L_post = λ·L1 + w_ssim·(1 − SSIM_2d). L1 is mean-seeking and caps high-frequency
    # detail; the optional SSIM term (Zhao 2017, "Loss Functions for Image Restoration")
    # rewards matching GT's local structure/contrast ⇒ pushes sharpness while still
    # penalising hallucinated/misplaced edges (it's reference-based, two-sided). SSIM is
    # contrast-normalised so it's blind to a uniform intensity offset — L1 covers that.
    # 2D per-slice, NOT 3D: the cube is anisotropic (12 mm Z over 12 slices vs 1.4 mm
    # in-plane), so SSIM runs on each axial (H,W) slice independently. w_ssim=0 ⇒ no-op
    # (L1-only, bit-identical to the pre-SSIM refiner).
    V_refined = predictions.get("V_refined")
    if V_refined is not None:
        out["V_refined"] = V_refined
        loss_refiner = refiner_lambda * (V_refined - V_gt).abs().mean()
        if refiner_ssim_weight > 0.0 and V_refined.is_cuda:
            from fused_ssim import fused_ssim
            # (B, D, H, W) → (B·D, 1, H, W): D folds into the batch dim, each in-plane
            # (H,W)=(Y,X) slice is one single-channel SSIM image. Contiguous fp32 (fused_ssim
            # is CUDA-only, assumes [0,1] data range — our normalised intensity is ~[0,1]).
            # Only the first arg (prediction) carries gradient; V_gt is the reference.
            Bv, Dv, Hv, Wv = V_refined.shape
            pred_s = V_refined.float().reshape(Bv * Dv, 1, Hv, Wv).contiguous()
            targ_s = V_gt.float().reshape(Bv * Dv, 1, Hv, Wv).contiguous()
            ssim_2d = fused_ssim(pred_s, targ_s, train=True)
            out["loss_refiner_ssim"] = refiner_ssim_weight * (1.0 - ssim_2d)
            out["metric_ssim_2d_refined"] = ssim_2d.detach()
            loss_refiner = loss_refiner + out["loss_refiner_ssim"]
        out["loss_refiner"] = loss_refiner

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
                from fused_ssim import fused_ssim3d
                pred_m = V_canon.unsqueeze(1).float().contiguous()
                targ_m = V_gt.unsqueeze(1).float().contiguous()
                out["metric_ssim_3d_full"] = fused_ssim3d(pred_m, targ_m, train=False)
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
            bboxes = batch["anatomy_bbox"]   # (B, 6) int64
            psnr_bbox_list, mae_bbox_list, mse_bbox_list = [], [], []
            for b in range(B):
                z0, z1, y0, y1, x0, x1 = [int(v) for v in bboxes[b].tolist()]
                # Empty bbox safety: fall back to full cube (matches the
                # full-volume metric for that sample). Can happen after
                # aggressive aug clears the volume.
                if (z1 <= z0) or (y1 <= y0) or (x1 <= x0):
                    Vc = V_canon[b]
                    Vg = V_gt[b]
                else:
                    Vc = V_canon[b, z0:z1, y0:y1, x0:x1]
                    Vg = V_gt[b, z0:z1, y0:y1, x0:x1]
                mse_b = ((Vc - Vg) ** 2).mean().clamp(min=1e-10)
                psnr_bbox_list.append(10.0 * torch.log10(1.0 / mse_b))
                mae_bbox_list.append((Vc - Vg).abs().mean())
                mse_bbox_list.append(mse_b)
            out["metric_mae_3d_bbox"] = torch.stack(mae_bbox_list).mean()
            out["metric_mse_3d_bbox"] = torch.stack(mse_bbox_list).mean()
            out["metric_psnr_3d_bbox"] = torch.stack(psnr_bbox_list).mean()

        # ── Motion-masked metrics (only voxels that move across the cardiac cycle) ──
        # The dynamic heart is ~3-5% of the cube; full/bbox PSNR is dominated by
        # static tissue and barely moves between a good and a bad model. Restricting
        # to (max_t - min_t > tau) isolates the region the model must actually get
        # right. Mask is derived from the full phase bundle batch["phases"].
        if "phases" in batch:
            motion_mask = compute_motion_mask(batch["phases"])  # (B, D, H, W) bool
            psnr_motion_list, mae_motion_list = [], []
            for b in range(B):
                m = motion_mask[b]
                if not bool(m.any()):
                    continue  # no moving voxels (shouldn't happen for real cardiac data)
                Vc = V_canon[b][m]
                Vg = V_gt[b][m]
                mse_m = ((Vc - Vg) ** 2).mean().clamp(min=1e-10)
                psnr_motion_list.append(10.0 * torch.log10(1.0 / mse_m))
                mae_motion_list.append((Vc - Vg).abs().mean())
            if psnr_motion_list:
                out["metric_psnr_3d_motion"] = torch.stack(psnr_motion_list).mean()
                out["metric_mae_3d_motion"] = torch.stack(mae_motion_list).mean()
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
                V_id, _ = splat_predictions({"world_points": batch["scanner_coords"]}, batch, grid_shape)
                # oracle splat (Δ=0, TRUE target-phase content sampled at each pixel's home) —
                # the recoverable ceiling; the model→oracle gap is the appearance wall (docs 19-21).
                intensity = batch["images"].float().mean(dim=2)
                if intensity.max() > 2.0:
                    intensity = intensity / 255.0
                scan_flat = batch["scanner_coords"].reshape(B, -1, 3)
                w = (intensity.reshape(B, -1) > 1e-3).float()
                V_or, _ = splat_to_volume(scan_flat, sample_volume(V_gt, scan_flat), grid_shape, weight=w)
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
                    through_mm = (V_canon.shape[1] - 1) / 2.0 * 12.0    # (D-1)/2*12 = 66 mm / norm z-unit
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

        # ── Refined-volume metrics (only when the refiner ran) ───────────────────
        # Mirror full/bbox/motion PSNR on V_refined so we can see, per phase, whether
        # the refiner beats the raw splat. Separate block (existing V_canon metrics
        # untouched ⇒ OFF path bitwise-identical).
        if V_refined is not None:
            mse_rf = ((V_refined - V_gt) ** 2).mean()
            out["metric_psnr_3d_full_refined"] = 10.0 * torch.log10(
                torch.tensor(1.0, device=mse_rf.device) / mse_rf.clamp(min=1e-10))
            if "anatomy_bbox" in batch:
                bboxes = batch["anatomy_bbox"]
                psnr_rf_bbox = []
                for b in range(B):
                    z0, z1, y0, y1, x0, x1 = [int(v) for v in bboxes[b].tolist()]
                    if (z1 <= z0) or (y1 <= y0) or (x1 <= x0):
                        Vc, Vg = V_refined[b], V_gt[b]
                    else:
                        Vc = V_refined[b, z0:z1, y0:y1, x0:x1]
                        Vg = V_gt[b, z0:z1, y0:y1, x0:x1]
                    mse_b = ((Vc - Vg) ** 2).mean().clamp(min=1e-10)
                    psnr_rf_bbox.append(10.0 * torch.log10(1.0 / mse_b))
                out["metric_psnr_3d_bbox_refined"] = torch.stack(psnr_rf_bbox).mean()
            if "phases" in batch:
                # Reuse the motion_mask already computed for the V_canon metrics above
                # (same batch["phases"]); recomputing would be redundant.
                psnr_rf_motion = []
                for b in range(B):
                    m = motion_mask[b]
                    if not bool(m.any()):
                        continue
                    mse_m = ((V_refined[b][m] - V_gt[b][m]) ** 2).mean().clamp(min=1e-10)
                    psnr_rf_motion.append(10.0 * torch.log10(1.0 / mse_m))
                if psnr_rf_motion:
                    out["metric_psnr_3d_motion_refined"] = torch.stack(psnr_rf_motion).mean()

    return out


