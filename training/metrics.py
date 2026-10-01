"""Reconstruction metrics for the volume-intensity objective (split out of loss.py).

`base_metrics` runs on every call of `compute_volume_intensity_loss` (train, val, identity
baseline, filmstrip). `val_metrics` is the VAL-ONLY block (heart-seg PSNR + the docs/38
ship-decision and breathing metrics); the caller opts in explicitly via
`compute_volume_intensity_loss(..., val_metrics=True)`, which the trainer passes only in
the val phase. Neither touches the objective: both run under no_grad.
"""

import logging

import torch
from data.preprocess import Z_HALF_MM
from vggt.utils.splat import (resize_field, sample_volume, splat_inputs,
                              splat_preds_native, splat_to_volume)


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


def _psnr_from_mse(mse):
    """PSNR (dB) for signals in [0, 1]. The 1e-10 floor caps a perfect match at 100 dB —
    pinned by tests/test_loss_masked_metrics_golden.py::test_psnr_clamp_floor_is_pinned."""
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-10))


def _masked_stats_vec(err, mask):
    """Per-sample masked MSE / MAE / voxel-count — VECTORIZED and BRANCHLESS.

    For the TRAIN-PATH metrics (bbox, motion), which run every step. Branchless is
    load-bearing: a Python-level decision here costs 4 graph breaks (measured), and the
    `torch.where` (rather than `err * mask.float()`) is what keeps a single non-finite voxel
    ANYWHERE in the cube from poisoning the metric — NaN * 0.0 == NaN.

    Returns (mse, mae, count), each (B,). Count is returned unclamped so callers can decide
    what an empty mask means; the division uses a clamped denominator.
    """
    zero = torch.zeros((), device=err.device, dtype=err.dtype)
    count = mask.float().sum(dim=(1, 2, 3))
    denom = count.clamp(min=1.0)
    mse = torch.where(mask, err ** 2, zero).sum(dim=(1, 2, 3)) / denom
    mae = torch.where(mask, err.abs(), zero).sum(dim=(1, 2, 3)) / denom
    return mse, mae, count


def _masked_mse(a, b, mask):
    """Masked MSE for ONE sample, via boolean indexing.

    For the VAL-ONLY metrics (heartseg, docs/38 heart + seg), which already run inside a
    `for b in range(B)` loop. Boolean indexing never reads outside the mask, so it is
    NaN-safe by construction, and the host syncs it costs are free off the train path.
    Callers must guard `mask.any()` themselves — `a[empty].mean()` is NaN.
    """
    return ((a[mask] - b[mask]) ** 2).mean()


@torch.no_grad()
def base_metrics(V_canon, V_gt, coverage, pos_pred, batch):
    """Metrics computed on every call (train + val + identity baseline + filmstrip).
    Returns an ordered dict of metric_* scalars."""
    out = {}
    # ── Full-volume metrics (over all D*H*W voxels of the canonical cube) ──
    # These match what `loss_volume` averages over and what the L1 loss is
    # actually optimizing. For small-FOV subjects the cube has many (V_gt=0,
    # V_canon≈0) padded voxels that inflate PSNR — see `_bbox` companion below.
    mse_full = ((V_canon - V_gt) ** 2).mean()
    psnr_full = 10.0 * torch.log10(torch.tensor(1.0, device=mse_full.device) / mse_full.clamp(min=1e-10))
    out["metric_mae_3d_full"] = (V_canon - V_gt).abs().mean().detach()
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

        mse_bbox, mae_bbox, _ = _masked_stats_vec(V_canon - V_gt, bbox_mask)
        psnr_bbox = _psnr_from_mse(mse_bbox)

        out["metric_mae_3d_bbox"] = mae_bbox.mean()
        out["metric_mse_3d_bbox"] = mse_bbox.mean()
        out["metric_psnr_3d_bbox"] = psnr_bbox.mean()

    # ── Motion-masked metrics (only voxels that move across the cardiac cycle) ──
    if "phases" in batch:
        # Vectorized (no per-sample `bool(m.any())` host sync). Masked with torch.where,
        # not a float multiply — see the bbox block above for why (NaN * 0.0 == NaN).
        motion_mask = compute_motion_mask(batch["phases"])  # (B, D, H, W) bool
        mse_m, mae_m, motion_cnt = _masked_stats_vec(V_canon - V_gt, motion_mask)
        psnr_m = _psnr_from_mse(mse_m)

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
    return out


@torch.no_grad()
def val_metrics(V_canon, V_gt, coverage, pos_pred, batch, z_scale, splat_res=None):
    """VAL-ONLY metrics: heart-seg PSNR + the docs/38 ship-decision and breathing metrics.

    Run only when the caller passes `val_metrics=True` (the trainer's val phase with real
    predictions) — never in train, so training cost + numerics are untouched. Each docs/38
    part is try/except-wrapped ⇒ never raises into the loop. Returns an ordered dict.
    """
    out = {}
    B = V_canon.shape[0]
    grid_shape = tuple(V_gt.shape[1:])

    # ── Anatomy heart-ROI PSNR (val-only) ────────────────────────────────────
    # Same masked PSNR as the motion metric in base_metrics, but the ROI is the nnU-Net
    # whole-heart segmentation (union over the 12 phases, dilated) resampled onto
    # the canonical grid — one anatomy-defined region per subject, shared with the
    # SVR baselines. Complements motion PSNR (which restricts to *moving* voxels);
    # this covers the whole heart incl. the static blood-pool interior.
    if "heart_roi_canonical" in batch:
        seg_roi = batch["heart_roi_canonical"].bool()   # (B, D, H, W)
        psnr_seg_list, mae_seg_list = [], []
        for b in range(B):
            m = seg_roi[b]
            if not bool(m.any()):
                continue
            psnr_seg_list.append(_psnr_from_mse(_masked_mse(V_canon[b], V_gt[b], m)))
            mae_seg_list.append((V_canon[b][m] - V_gt[b][m]).abs().mean())
        if psnr_seg_list:
            out["metric_psnr_3d_heartseg"] = torch.stack(psnr_seg_list).mean()
            out["metric_mae_3d_heartseg"] = torch.stack(mae_seg_list).mean()
            out["metric_heartseg_frac"] = seg_roi.float().mean()

    # ── VAL-ONLY ship-decision + breathing metrics (docs/37) ─────────────────
    # These quantify targeted improvements that aggregate PSNR buries: an oracle-
    # normalized recoverable-fraction (rescales out the un-fixable appearance wall),
    # a heart/static PSNR split, a coverage-hole tripwire, and breathing through-
    # plane recovery vs the EXACT simulated shift. Extra splats run only here.
    # The cardiac-motion mask (compute_motion_mask) is the heart ROI — no segmentation
    # needed. See docs/37 for the design + the stop-grad test. To surface in wandb, the
    # metric_* keys below must be listed in `logging.scalar_keys_to_log.val.keys_to_log`.
    if "phases" in batch and "scanner_coords" in batch:
        # (1) recov_frac_heart + psnr_static + hole_frac_heart (vs GT, heart ROI)
        try:
            heart = compute_motion_mask(batch["phases"])            # (B,D,H,W) bool
            # identity splat (Δ=0, real corrupted input content) — exact forward path
            V_id, _ = splat_preds_native({"world_points": batch["scanner_coords"]}, batch, grid_shape, z_scale, splat_res)
            # oracle splat (Δ=0, TRUE target-phase content sampled at each pixel's home) —
            # the recoverable ceiling; the model→oracle gap is the appearance wall (docs 19-21).
            # Same point set / weight gate as V_id and V_canon (native when images_splat
            # exists, at splat_res when set), so the recov_frac ratio compares one splat
            # pipeline throughout.
            imsp = splat_inputs(batch, splat_res)
            if imsp is not None:
                sc = resize_field(batch["scanner_coords"], *imsp.shape[-2:])
                scan_flat = sc.reshape(B, -1, 3)
                w = (imsp.float().reshape(B, -1) > 1e-3).float()
            else:
                inv_scale = torch.where((batch["images"] > 2.0).any(), 1.0 / 255.0, 1.0)
                intensity = batch["images"].float().mean(dim=2) * inv_scale
                scan_flat = batch["scanner_coords"].reshape(B, -1, 3)
                w = (intensity.reshape(B, -1) > 1e-3).float()
            V_or, _ = splat_to_volume(scan_flat, sample_volume(V_gt, scan_flat, z_scale), grid_shape, z_scale, weight=w)
            recov, mse_id_l, mse_mo_l, mse_or_l, holes, static_psnr = [], [], [], [], [], []
            for b in range(B):
                m = heart[b]
                if bool(m.any()):
                    mse_id = _masked_mse(V_id[b], V_gt[b], m)
                    mse_mo = _masked_mse(V_canon[b], V_gt[b], m)
                    mse_or = _masked_mse(V_or[b], V_gt[b], m)
                    mse_id_l.append(mse_id); mse_mo_l.append(mse_mo); mse_or_l.append(mse_or)
                    holes.append((coverage[b][m] < 0.5).float().mean())
                    span = mse_id - mse_or                          # recoverable span (identity → ceiling)
                    if float(span) > 1e-6:                          # skip if oracle ≯ identity (recov undefined;
                                                                    # signed clamp on a signed denom is wrong)
                        # UNCLAMPED as of 2026-08-01, matching recov_frac_seg. The old
                        # `.clamp(-0.5, 1.5)` censored 98.9% of rows to -0.5 early in
                        # training — it destroyed data, and the only reason it survived
                        # was continuity with wandb curves that native-z had already
                        # invalidated. The `span > 1e-6` guard above keeps the
                        # denominator positive, which is the part that actually matters.
                        recov.append((mse_id - mse_mo) / span)
                st = (V_gt[b] > 1e-3) & (~heart[b])                 # content that does NOT beat (control)
                if bool(st.any()):
                    static_psnr.append(_psnr_from_mse(_masked_mse(V_canon[b], V_gt[b], st)))
            if mse_id_l:
                out["metric_mse_heart_identity"] = torch.stack(mse_id_l).mean()
                out["metric_mse_heart_model"] = torch.stack(mse_mo_l).mean()
                out["metric_mse_heart_oracle"] = torch.stack(mse_or_l).mean()
                out["metric_hole_frac_heart"] = torch.stack(holes).mean()
            if recov:
                out["metric_recov_frac_heart"] = torch.stack(recov).mean()
            if static_psnr:
                out["metric_psnr_3d_static"] = torch.stack(static_psnr).mean()

            # Same floor/ceiling on the SEGMENTATION ROI (docs/60). Two reasons:
            #  1. `metric_psnr_3d_heartseg` is a fine per-subject number but is NOT
            #     comparable ACROSS subjects — measured, 79% of its between-subject
            #     spread is shared with the whole-volume PSNR, i.e. intrinsic scan
            #     difficulty rather than model quality. Referencing it to each
            #     subject's OWN identity floor removes that shared term, so
            #     `psnr_seg_gain_db` can be averaged over a heterogeneous cohort.
            #  2. `recov_frac_heart` uses the motion mask while `psnr_heartseg` uses
            #     the segmentation — two different "heart"s. This puts a recovered
            #     fraction on the SAME mask as the PSNR.
            # V_id/V_or are already computed above, so this is just two more masks.
            seg = batch.get("heart_roi_canonical")
            if seg is not None:
                seg = seg.bool()
                gain, recov_s, sid, smo, sor = [], [], [], [], []
                for b in range(B):
                    m = seg[b]
                    if not bool(m.any()):
                        continue
                    # id/model clamped (they divide each other for the dB gain below);
                    # oracle unclamped (it only ever appears in a difference).
                    mse_id = _masked_mse(V_id[b], V_gt[b], m).clamp(min=1e-10)
                    mse_mo = _masked_mse(V_canon[b], V_gt[b], m).clamp(min=1e-10)
                    mse_or = _masked_mse(V_or[b], V_gt[b], m)
                    sid.append(mse_id); smo.append(mse_mo); sor.append(mse_or)
                    # dB the model gains over doing nothing, on this subject's own ROI.
                    gain.append(10.0 * torch.log10(mse_id / mse_mo))
                    span = mse_id - mse_or
                    if float(span) > 1e-6:
                        # NOT clamped, unlike recov_frac_heart: the `span > 0` guard
                        # already makes the denominator positive, and the clamp there
                        # censors 98.9% of rows to -0.5 early in training.
                        recov_s.append((mse_id - mse_mo) / span)
                if gain:
                    out["metric_psnr_seg_gain_db"] = torch.stack(gain).mean()
                    out["metric_mse_seg_identity"] = torch.stack(sid).mean()
                    out["metric_mse_seg_model"] = torch.stack(smo).mean()
                    out["metric_mse_seg_oracle"] = torch.stack(sor).mean()
                if recov_s:
                    out["metric_recov_frac_seg"] = torch.stack(recov_s).mean()
        except Exception as e:
            logging.warning(f"docs/38 recov/static val metric failed (ignored): {e}")

        # (2) breathing through-plane recovery vs the EXACT applied sim shift.
        # predicted Δz per slot (mm) vs applied SI (resp_disp_mm[...,0]) →
        # slope/corr/EPE + deep-breath-ignored. Brings tools/_archive/exp_4wok_analysis.py online.
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
