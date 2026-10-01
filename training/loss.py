# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from metrics import base_metrics, val_metrics as _val_metrics
from vggt.utils.splat import sample_volume, splat_preds_native
# Private aliases kept for importers that predate the move to vggt.utils.splat
# (evaluation/src/engine/run_vggt*.py, tools/interp/*).
from vggt.utils.splat import splat_preds_native as _splat_preds_native  # noqa: F401
from vggt.utils.splat import resize_field as _resize_field  # noqa: F401


class MultitaskLoss(torch.nn.Module):
    """The training objective. Despite the inherited name there is exactly ONE task left:
    the unsupervised volume-intensity loss (`compute_volume_intensity_loss`).

    Upstream VGGT's camera / depth / point / tracking losses were removed along with the
    supervised-DVF pipeline — the name and the `_target_: loss.MultitaskLoss` config key are
    kept only so existing configs and checkpoints keep resolving.
    """

    def __init__(self, volume=None):
        super().__init__()
        # Volume-intensity loss configuration (the only active task).
        self.volume = volume

    def forward(self, predictions, batch, val_metrics=False) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            val_metrics: also compute the val-only metrics (see metrics.val_metrics).

        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        # Direct volume-to-volume loss against the GT phase-0 volume loaded from disk.
        if "world_points" in predictions and self.volume is not None and self.volume.get("weight", 0) > 0:
            vol_kw = {k: v for k, v in self.volume.items() if k != "weight"}
            vol_loss_dict = compute_volume_intensity_loss(predictions, batch, **vol_kw,
                                                          val_metrics=val_metrics)
            vol_loss = (vol_loss_dict["loss_volume"]
                        + vol_loss_dict.get("loss_diffusion", 0.0)
                        + vol_loss_dict.get("loss_gather", 0.0)
                        + vol_loss_dict.get("loss_heart", 0.0)) * self.volume["weight"]
            total_loss = total_loss + vol_loss
            loss_dict.update(vol_loss_dict)

        loss_dict["objective"] = total_loss

        return loss_dict


def diffusion_loss_l2(field):
    """L2 diffusion (Tikhonov) smoothness regularizer ‖∇u‖² on a displacement field.

    Mean of SQUARED in-plane (H, W) neighbor differences over a (B, S, H, W, C) field.
    The squared gradient is ∝ω² and smoothness-promoting — the VoxelMorph diffusion regularizer that actively suppresses
    high-frequency (e.g. ViT-patch-period) ripples in the predicted warp. fp32-forced so
    the reduction stays accurate under autocast(bf16).
    """
    with torch.amp.autocast("cuda", enabled=False):
        f = field.float()
        return (
            (f[:, :, 1:, :, :] - f[:, :, :-1, :, :]).pow(2).mean()
            + (f[:, :, :, 1:, :] - f[:, :, :, :-1, :]).pow(2).mean()
        )


def compute_volume_intensity_loss(predictions, batch,
                                  diffusion_weight=0.0, gather_weight=0.0,
                                  heart_weight=0.0, splat_res=None, val_metrics=False):
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
        val_metrics: also compute the VAL-ONLY metrics (metrics.val_metrics: heart-seg PSNR,
            docs/38 ship-decision + breathing). The trainer passes True only in the val phase;
            the train step, identity baseline and filmstrip leave it False.

    Returns the loss terms, V_canon / V_gt / coverage, then metrics.base_metrics (always)
    and metrics.val_metrics (when val_metrics=True), in that key order.
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

    V_canon, coverage = splat_preds_native(predictions, batch, grid_shape, z_scale, splat_res)

    if V_gt.shape != V_canon.shape:
        raise RuntimeError(f"gt_target_volume {tuple(V_gt.shape)} must match V_canon {tuple(V_canon.shape)}")

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

    # ── ARM heart-L1 (docs/69 follow-up): ADDITIVE heart-ROI L1 on top of the full L1 ──
    # Motivation: the full-volume L1 above averages over a volume in which the heart is only
    # ~4.5-6.5% of voxels (measured: metric_heartseg_frac on native-z runs), so ~95% of the
    # gradient is spent on chest wall / lungs / background / padding, none of which move with
    # the heart. This term upweights the heart WITHOUT masking.
    #
    # ADDITIVE, NOT A MASK — deliberately. The masked variant (`V_gt > 1e-3`, still commented
    # out above) was removed because it gave the model a free pass to over-predict intensity
    # wherever V_gt was zero (ghost blobs). Keeping the full L1 as the primary term preserves
    # that penalty; this only adds pressure inside the ROI.
    #
    # `heart_roi_canonical` is the DILATED WHOLE-HEART nnU-Net ROI (union over the 12 phases)
    # on the canonical grid — it includes atria/RV, so this is not an "LV loss". It is loaded
    # by MRIDataset.get_data with NO split gate, so it is present for train as well as val
    # (only the psnr_3d_heartseg METRIC is val-gated).
    #
    # Availability is per-sample: get_data omits the key when the on-disk ROI shape does not
    # match the canonical grid. That never fires on native-z (measured: 0 warnings across a
    # full pooled run) but fires on ~28/29 subjects under the fixed12 hub wrapper, which pads
    # phases/mask to D=12 and leaves the ROI at native D. So this arm is for the NATIVE-Z
    # pipeline; under fixed12 the term would silently vanish. Hence the explicit raise below
    # rather than a silent 0.0 — a weight that is asked for but cannot be applied is a bug.
    #
    # ⚠️ Training on this ROI makes metric_psnr_3d_heartseg a TRAINING TARGET, so it is no
    # longer an independent val check. Judge this arm on amp_ratio / motion PSNR instead.
    loss_heart = pos_pred.new_zeros(())
    if heart_weight > 0:
        if "heart_roi_canonical" not in batch:
            raise RuntimeError(
                "loss.volume.heart_weight > 0 but batch has no 'heart_roi_canonical'. The ROI is "
                "dropped per-sample when its on-disk shape != the canonical grid (see "
                "MRIDataset.get_data) — this is expected under the fixed12 hub wrapper, which "
                "pads phases/mask to D=12 but leaves the ROI at native D. Use native-z, or "
                "set heart_weight=0."
            )
        roi = batch["heart_roi_canonical"].bool()
        # BRANCHLESS on purpose — no `if n_roi > 0`. `heart_weight` is a Python float so the
        # outer `if` is free, but a tensor-valued test would force a GPU sync + graph break on
        # EVERY train step, and cuda.compile_attention_blocks is True by default. Same reason
        # the motion metric in metrics.base_metrics is branchless ("a Python-level decision here
        # costs 4 graph breaks, measured"). The empty-ROI case needs no branch anyway: the numerator is
        # exactly 0 when the mask is empty, and clamp(min=1) keeps the denominator safe, so the
        # expression already evaluates to 0.0 — verified against the branched version.
        # Mean over ROI voxels only, so the term is on the same per-voxel scale as loss_volume;
        # heart_weight is then a straight relative weight.
        loss_heart = ((V_canon - V_gt).abs() * roi).sum() / roi.sum().clamp(min=1) * heart_weight

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
    # cheaper than the splat. Uses the model-resolution `images` intensity (the primary splat
    # renders native `images_splat` since docs/73 — same content, different sampling). padding_mode
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
        "loss_diffusion": loss_diffusion,
        "loss_gather": loss_gather,
        "loss_heart": loss_heart,   # ARM heart-L1 (already scaled by heart_weight)
        "V_canon": V_canon,
        "V_gt": V_gt,
        "coverage": coverage,
    }
    out.update(base_metrics(V_canon, V_gt, coverage, pos_pred, batch))
    if val_metrics:
        out.update(_val_metrics(V_canon, V_gt, coverage, pos_pred, batch, z_scale, splat_res))
    return out
