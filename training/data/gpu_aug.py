"""batchaug GPU augmentation pipeline for VGGT-MRI.

Operates on the cached canonical `(B, T=12, D, H=256, W=256)` phases tensor
(`D` = this subject's own native slice count, 5-21 — native-z, docs/58; NOT a fixed 12)
that `MRIDataset.get_data` puts in the batch under the `phases` key. One spatial
affine is sampled per subject (B-dim); the same affine is applied to all 12
T-phases (T as channel) AND to the `content_mask`, so:

    * augmented phases stay consistent across cardiac phases (no rotation jitter
      between t=0 and t=11)
    * scanner_coords don't need updating — they're a pure geometric mapping from
      pixel-index to canonical-cube-coord, decoupled from image content
    * `anatomy_bbox` is recomputed from the augmented mask (anatomy has moved)

After aug, the trainer:
    1. Re-derives `V_gt` = `phases_aug[b, t_target[b]]`
    2. Re-extracts `S` input slices from `phases_aug` at the original (t, z) pairs
    3. Recomputes `anatomy_bbox` from the augmented content_mask

batchaug backend is forced to `"pytorch"` at import, and A/B-measured 2026-07-24
(`tools/ab_batchaug_backend.py`, docs/49) — keep it. batchaug's triton backend
overrides ONLY intensity transforms (`triton/geometric/` is empty), so of our 3
active transforms just RandAdjustContrastd + RandBiasFieldd differ; the expensive
RandAffined is the same code either way. Measured on an A40 with seeded-paired
interleaved timing (200 rounds): triton is **not faster** — full pipeline
-0.048 ms +/- 0.0044 SEM (0.993x, triton marginally SLOWER), and isolated at
prob=1.0 -0.013 ms +/- 0.0104 (0.990x, null). Aug is <0.2% of a train step
anyway, and triton is not bitwise identical (~2e-6). So: no upside, real
reproducibility cost.

Two measurement traps, both of which produced WRONG published numbers before
being caught (docs/49): timing must be **seeded-paired** (both backends given
identical draws per round) or the affine's Bernoulli gate injects ~1.6 ms of
noise that swamps the effect — unseeded sd 1.139 ms vs seeded 0.062 ms; and
cross-process GPU clock drift moved pytorch's own median 2.84 -> 2.34 ms, which
is how an earlier run manufactured a phantom "1.18x triton win". Note also that
intensity-transform COST is *not* probability-gated: both backends compute
unconditionally and gate only the output via `torch.where`.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

try:
    import batchaug as _B
    _B.set_backend("pytorch")
except ImportError:  # pragma: no cover — batchaug is a hard dep for aug
    _B = None

from data.respiratory import (
    extract_slices_with_respiratory_vec,
    sample_resp_disp,
)

# Local constants (kept in sync with preprocess.py / mri_dataset.py).
INPUT_IMG_SIZE = 518   # DINOv2 input — must match MRIDataset.target_size


# ──────────────────────────────────────────────────────────────────────────────
# Factory: build the batchaug Compose from config
# ──────────────────────────────────────────────────────────────────────────────
def build_gpu_transforms(aug_cfg=None):
    """Build a `batchaug.Compose` from the config, or return `None` if disabled.

    The trainer treats `None` as identity and skips the aug step entirely.

    Args:
        aug_cfg: object/dict with fields:
            enable (bool, default False)
            tier   ("conservative" | "moderate", default "conservative")

    Returns:
        `batchaug.Compose` or `None`.
    """
    enable = bool(getattr(aug_cfg, "enable", False)) if aug_cfg is not None else False
    if not enable:
        logging.info("GPU augmentation disabled (data.augmentation.enable=False)")
        return None
    if _B is None:
        raise RuntimeError(
            "batchaug is not importable but aug is enabled. "
            "Install via: pip install --no-deps -e /home/minsukc/MRI2CT/batchaug/"
        )

    tier = getattr(aug_cfg, "tier", "conservative")
    logging.info(f"GPU augmentation enabled: tier={tier}")
    keys = ["phases", "content_mask"]
    mode_dict = {"phases": "bilinear", "content_mask": "nearest"}

    if tier == "conservative":
        # Conservative tier — IN-DISTRIBUTION-PRIORITY (mild). Broadens the natural orientation
        # spread (rotate ±45°) WITHOUT chasing OOD tails, gentle photometric, modest fire-probs.
        # NO through-plane rotation (slices are physically anisotropic 12 mm Z vs 1.4 mm X/Y),
        # NO elastic. Gaussian noise is commented out for ALL tiers (rationale inline below).
        # batchaug is POSITIONAL, not semantic: each tuple slot i maps to tensor
        # spatial dim i+2 (our dims are D=0, H=1, W=2 after the channel). BUT
        # `rotate_range` is special — its slots are PLANES of rotation, not axes:
        #   slot 0 → rotation in the H-W plane (about D)  = IN-PLANE  ← what we want
        #   slot 1 → rotation in the D-W plane            = through-plane
        #   slot 2 → rotation in the D-H plane            = through-plane
        # So in-plane rotation goes in rotate_range slot 0, NOT slot 2.
        # (translate_range / scale_range ARE per-axis (D, H, W): slot 0 = D, so
        #  freezing slot 0 there correctly disables through-plane shift/scale.)
        transforms = [
            # RandFlipd DISABLED here — flip lives in the AGGRESSIVE tier only (2026-08-01, user
            # decision). It is the newest addition (re-enabled 2026-07-31, docs/58 §10c) and was
            # NOT part of the moderate arm that docs/46 §3 C2 measured and shipped, so keeping
            # conservative/moderate flip-free preserves the validated configuration. Rationale
            # for having it at all (mirror-equivariant objective; 29% of the pooled CMRx cohort
            # is mirrored on disk) is recorded at the aggressive tier below.
            # _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.5,
                rotate_range=(float(np.deg2rad(45)), 0.0, 0.0),  # in-plane (H-W); mild — broaden natural spread, don't chase OOD tails
                translate_range=(0.0, 6.0, 6.0),                 # H, W only (D frozen)
                scale_range=(0.0, 0.05, 0.05),                   # H, W only; small → anisotropy (independent H/W) stays negligible
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask.
            _B.RandAdjustContrastd(keys=["phases"], prob=0.4, gamma=(0.8, 1.3)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.4, degree=3, coeff_range=(-0.10, 0.10)),  # symmetric → zero-mean shading (both brighten & darken)
            # RandGaussianNoised DISABLED (all tiers): models i.i.d. noise, but real OOD/real-time
            # degradation is structured (aliasing/motion-blur); it mostly corrupts clean signal.
            # _B.RandGaussianNoised(keys=["phases"], prob=0.3, std=(0.0, 0.02)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    if tier == "moderate":
        # Moderate tier — RECOMMENDED OOD-AWARE DEFAULT (synthesized from a 2-stance debate).
        # IN-PLANE ONLY. Headline = FULL-CIRCLE (±180°) rotation to cover the measured MIITT
        # orientation gap (MIITT clusters ~180° off CMRx's mode), at moderate prob (0.6) so
        # natural orientation is still anchored (~40% of samples). Gamma = the key contrast
        # lever; bias-field models coil shading; scale kept small (near-isotropic); flip OFF,
        # Gaussian noise OFF (see conservative note). Plane semantics as in conservative:
        # rotate slot 0 = in-plane (H-W); translate/scale slots (D,H,W) with D frozen.
        transforms = [
            # RandFlipd DISABLED — see conservative note. This tier is the shipped docs/46 §3 C2
            # arm, which was measured WITHOUT flip; flip is aggressive-only.
            # _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.6,
                rotate_range=(float(np.deg2rad(180)), 0.0, 0.0),  # FULL-CIRCLE: ±180° uniform IS the whole circle (angles mod 360°); 360° would just duplicate orientations
                translate_range=(0.0, 16.0, 16.0),               # H, W only (D frozen)
                scale_range=(0.0, 0.05, 0.05),                   # H, W only; small → near-isotropic (avoids ellipse distortion)
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask. Gamma = the key OOD contrast lever.
            _B.RandAdjustContrastd(keys=["phases"], prob=0.6, gamma=(0.7, 1.5)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.5, degree=3, coeff_range=(-0.5, 0.5)),  # symmetric → zero-mean shading (brighten & darken); clamp handles residual overshoot
            # RandGaussianNoised DISABLED — see conservative note.
            # _B.RandGaussianNoised(keys=["phases"], prob=0.5, std=(0.0, 0.03)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    if tier == "aggressive":
        # Aggressive tier — MAX OOD coverage. IN-PLANE ONLY. Full-circle rotation like moderate
        # but at higher prob + wider gamma/bias-field and larger translate; scale still capped
        # (±8%) to bound ellipse distortion; flip ON (this tier ONLY), Gaussian noise OFF. Same
        # plane semantics (rotate slot 0 = in-plane H-W; translate/scale (D,H,W) with D frozen).
        transforms = [
            # RandFlipd — AGGRESSIVE-ONLY as of 2026-08-01 (it was briefly on in all tiers,
            # 2026-07-31, docs/58 §10c). Why it is justified at all: (1) the training objective
            # is EXACTLY mirror-equivariant — inputs, `gt_target_volume` and `scanner_coords`
            # all derive from the same array, so a consistent W-mirror is a measured no-op
            # (splat residual 1.25e-06); RV location is observable in every input slice, not
            # prior knowledge. (2) 29% of the pooled CMRx cohort is mirrored on disk anyway, so
            # the cohort already contains both handednesses. Known cost, unmeasured: the head
            # outputs a VECTOR field, so mirror-invariance needs a coupled spatial flip AND a Δx
            # sign negation — a harder symmetry than nnU-Net's label-map mirroring, spent from a
            # fixed capacity/step budget. That cost is why it is confined to this tier.
            _B.RandFlipd(keys=keys, prob=0.5, spatial_axis=[2]),
            _B.RandAffined(
                keys=keys,
                prob=0.9,
                rotate_range=(float(np.deg2rad(180)), 0.0, 0.0),  # FULL-CIRCLE in-plane rotation (±180° = whole circle)
                translate_range=(0.0, 20.0, 20.0),               # H, W only (D frozen)
                scale_range=(0.0, 0.08, 0.08),                   # H, W only; capped ±8% to bound anisotropic ellipse distortion
                padding_mode="zeros",
            ),
            # Photometric — apply ONLY to `phases`, not the mask.
            _B.RandAdjustContrastd(keys=["phases"], prob=0.75, gamma=(0.6, 1.7)),
            _B.RandBiasFieldd(keys=["phases"], prob=0.7, degree=3, coeff_range=(-0.6, 0.6)),  # symmetric → zero-mean shading
            # RandGaussianNoised DISABLED — see conservative note.
            # _B.RandGaussianNoised(keys=["phases"], prob=0.6, std=(0.0, 0.05)),
        ]
        return _B.Compose(transforms=transforms, lazy=True, mode=mode_dict)

    raise ValueError(f"unknown aug tier: {tier!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper: bbox of a 3D mask (GPU-friendly, no python loop over voxels)
# ──────────────────────────────────────────────────────────────────────────────
# The post-aug bbox is the SAME computation the dataset runs pre-aug — one implementation,
# two call sites: `preprocess.compute_geometric_bbox` at cache-read time (mri_dataset.py),
# and here after an affine has moved the content mask. It used to be duplicated verbatim.
from data.preprocess import compute_geometric_bbox as recompute_bbox_gpu


# ──────────────────────────────────────────────────────────────────────────────
# Helper: re-extract S slices from an augmented (B, T, D, H, W) tensor
# ──────────────────────────────────────────────────────────────────────────────
def extract_slices_from_phases(phases, t_seq, z_seq):
    """Pull S slices per batch element from an augmented phases tensor and
    bilinear-upsample each to `(INPUT_IMG_SIZE, INPUT_IMG_SIZE)` for DINOv2.

    Args:
        phases: `(B, T, D, H=256, W=256)` float.
        t_seq:  `(B, S)` int64 — t index per slot.
        z_seq:  `(B, S)` z index per slot — int OR continuous float (linearly
                interpolated between the two bracketing z planes; exact at integer z).

    Returns:
        `(B, S, 518, 518, 3)` float in `[0, 255]` — RGB-replicated, ready to
        replace `batch["images"]` after a `permute(0, 1, 4, 2, 3) / 255` in
        the trainer (matches the ComposedDataset contract).
    """
    Bsize, T, D, H, W = phases.shape
    S = t_seq.shape[1]
    b_idx = torch.arange(Bsize, device=phases.device).view(Bsize, 1).expand(Bsize, S)
    t_seq = t_seq.long()
    # Continuous-z safe: linear blend between the two bracketing z planes. At integer z the
    # fraction is 0 → exact plane, so the discrete-grid path is numerically unchanged.
    z_f = z_seq.float()
    z0 = torch.floor(z_f).long().clamp(0, D - 1)
    z1 = (z0 + 1).clamp(0, D - 1)
    frac = (z_f - z0.float()).view(Bsize, S, 1, 1)
    s0 = phases[b_idx, t_seq, z0]               # (B, S, H, W)
    s1 = phases[b_idx, t_seq, z1]
    slices_canon = (1.0 - frac) * s0 + frac * s1
    slices_canon = slices_canon.reshape(Bsize * S, 1, H, W)
    upsampled = F.interpolate(
        slices_canon,
        size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
        mode="bilinear",
        align_corners=True,
    )                                                    # (B*S, 1, 518, 518)
    upsampled = upsampled.view(Bsize, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE)
    upsampled = (upsampled * 255.0).clamp(0.0, 255.0)
    # RGB-replicate to (B, S, 518, 518, 3).
    return upsampled.unsqueeze(-1).expand(Bsize, S, INPUT_IMG_SIZE, INPUT_IMG_SIZE, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Main entry: apply aug to the batch in place
# ──────────────────────────────────────────────────────────────────────────────
def gpu_augment_batch(batch, transforms, device,
                      respiratory_cfg=None, train=True, resp_generator=None):
    """Apply GPU augmentations to a batch and re-derive the dependent fields.

    Two INDEPENDENT augmentations, each separately gated:

      * **Affine/photometric** (`transforms`, whole-subject): warps `phases`+
        `content_mask` and re-derives `gt_target_volume`/`anatomy_bbox`/`images`.
        Affects BOTH target and inputs. `None` → skipped.
      * **Respiratory** (`respiratory_cfg.enable`, per-input-slice): a deform-then-
        reslice shift that overwrites **ONLY** `images` — the target,
        `scanner_coords`, `gt_target_volume`, `anatomy_bbox`, `content_mask`, and
        `phases` stay at the unshifted end-expiration reference (the model learns to
        CORRECT breathing, blind to `r`). Runs even when affine is off, and AFTER
        affine when both are on.

    Train uses the private `resp_generator` (iid per epoch); val seeds breathing
    deterministically per row from `batch["seq_index"]` (required when val + resp).

    If neither augmentation is active, the batch is returned unchanged.

    Required batch keys:
        phases           (B, T, D, H, W) float16/float32
        content_mask     (B, D, H, W)    uint8        (affine)
        t_target         (B, 1)          int64        (affine)
        timesteps        (B, S)          int64 — original t per slot
        slice_indices    (B, S)          float32 — original z per slot (may be continuous)
        seq_index        (B, 1)          int64 — required for val respiratory
    """
    do_affine = transforms is not None
    do_resp = respiratory_cfg is not None and getattr(respiratory_cfg, "enable", False)
    if not do_affine and not do_resp:
        # A missing `images` means the dataset deferred it to us (`defer_input_images`), so
        # we must still produce it even with every augmentation off — otherwise the batch
        # would reach the model with no input. This is the guarantee that makes deferral safe.
        if "images" not in batch:
            # float32, matching the dataset's own extraction in dtype and algorithm — but
            # NOT bit-for-bit: the dataset built this on CPU in a worker, we build it on
            # CUDA, and the 256→518 bilinear `F.interpolate` differs by up to 1 ULP
            # (measured 5.96e-08 = 2^-24 on values in [0,1]; docs/62 §3). Everything else —
            # dtype, align_corners, the *255→clamp→/255 order, RGB replication — matches
            # exactly. Only reachable with BOTH affine and respiratory off, which nothing
            # ships, and 6e-8 is ~5 orders below the bf16 the model computes in.
            # (The augmented branches below extract from the fp16 `phases` as they always
            # have, and ARE bit-identical to pre-deferral — proven in docs/62 §2.1.)
            phases_cur = batch["phases"].to(device=device, dtype=torch.float32, non_blocking=True)
            images = extract_slices_from_phases(
                phases_cur, batch["timesteps"], batch["slice_indices"])
            batch["images"] = images.permute(0, 1, 4, 2, 3).contiguous() / 255.0
        return batch

    phases = batch["phases"]                 # (B, T, D, H, W) any float
    Bsize = phases.shape[0]
    affine_applied = False

    # ── Affine/photometric (whole-subject) ───────────────────────────────────
    if do_affine:
        mask = batch["content_mask"]         # (B, D, H, W) uint8
        phases_f = phases.to(device=device, dtype=torch.float32, non_blocking=True)
        # batchaug grid_sample needs float; mask keeps 0/1 under nearest interp.
        mask_f = mask.to(device=device, dtype=torch.float32, non_blocking=True).unsqueeze(1)
        aug_dict = {"phases": phases_f, "content_mask": mask_f}
        try:
            aug_dict = transforms(aug_dict)
        except Exception as e:
            # Aug must never crash training; log and fall through with identity affine.
            logging.warning(f"gpu_augment_batch: aug pipeline failed (ignored): {e}")
        else:
            # Photometric ops (esp. the multiplicative bias field) can push intensities
            # above the [0,1] normalization range. The input-slice extractors clamp to
            # [0,1], but gt_target_volume is derived here — so clamp the SHARED source ONCE
            # to keep gt and the re-extracted inputs mutually consistent. Otherwise V_gt can
            # exceed 1 while the splat's clamped inputs cap V_canon at ~1, leaving an
            # unlearnable L1 residual (the point head predicts position, not intensity).
            phases_aug = aug_dict["phases"].clamp(0.0, 1.0)    # (B, T, D, H, W) float32
            mask_aug = aug_dict["content_mask"].squeeze(1)      # (B, D, H, W) float (0/1)
            mask_aug_u8 = (mask_aug > 0.5).to(torch.uint8)

            t_target = batch["t_target"]
            if t_target.ndim > 1:
                t_target = t_target.squeeze(-1)  # (B,)
            gt_target_volume = phases_aug[torch.arange(Bsize, device=device), t_target]

            bboxes = torch.stack([recompute_bbox_gpu(mask_aug_u8[b]) for b in range(Bsize)])

            batch["phases"] = phases_aug.to(phases.dtype)
            batch["content_mask"] = mask_aug_u8
            batch["gt_target_volume"] = gt_target_volume
            batch["anatomy_bbox"] = bboxes
            affine_applied = True

    # ── Re-extract input slices EXACTLY ONCE ──────────────────────────────────
    # Respiratory (if on) overwrites `images` with the breathing-shifted reslice;
    # gt/bbox/scanner_coords above stay at the reference. Extracting once avoids a
    # wasted (and immediately-discarded) affine extraction. If affine failed AND
    # respiratory is off, nothing changed → leave the dataset's `images` untouched.
    if do_resp:
        S = batch["timesteps"].shape[1]
        seq_index = batch.get("seq_index")
        if not train and seq_index is None:
            raise ValueError(
                "respiratory val augmentation requires batch['seq_index'] for determinism"
            )
        phases_cur = batch["phases"].to(device=device, non_blocking=True)
        D = phases_cur.shape[2]                    # this subject's own native slice count (native-z)
        # One scalar dz for the whole batch — valid only at batch_size==1. Guard it
        # (docs/59 F7): same-D-different-dz subjects collate fine, and row 1 would then be
        # breathed at row 0's scale, a silent through-plane geometry error.
        _dz = batch["dz_mm"].reshape(-1)
        if not bool((_dz == _dz[0]).all()):
            raise RuntimeError(
                f"dz_mm is not uniform across the batch: {_dz.tolist()}. One scalar dz is applied "
                "to every row, so mixing slice pitches would breathe rows 1..B-1 at row 0's scale "
                "— a silent through-plane geometry error (docs/59 F7)."
            )
        dz = float(batch["dz_mm"].reshape(-1)[0])   # this subject's own native z spacing (mm)
        # z-plane per slot — the burst-grouping key for group_by_burst (one breath per plane).
        # Ignored unless respiratory_cfg.group_by_burst is set (default → per-slot iid, unchanged).
        group_ids = batch["slice_indices"].round().long().to(device)
        disp, resp_r = sample_resp_disp(
            Bsize, S, respiratory_cfg, device,
            train=train, seq_index=seq_index, generator=resp_generator, group_ids=group_ids,
            n_planes=D,
        )                                                       # (B,S,3) mm, (B,S) phase
        images = extract_slices_with_respiratory_vec(
            phases_cur, batch["timesteps"], batch["slice_indices"], disp,
            spacing=(dz, 1.4, 1.4),
        )                                                       # (B, S, 518, 518, 3) [0,255]
        batch["images"] = images.permute(0, 1, 4, 2, 3).contiguous() / 255.0
        # Surface per-slot displacement (canonical D,H,W mm) + respiratory phase r for
        # diagnostics only — captions + the resp scalar. Inert for model/loss (read-only).
        batch["resp_disp_mm"] = disp
        batch["resp_r"] = resp_r
    elif affine_applied or "images" not in batch:
        # `"images" not in batch` covers the deferred case where the affine BUILD failed
        # (caught above, affine_applied stays False) and respiratory is off — without it the
        # batch would reach the model with no input at all.
        phases_cur = batch["phases"].to(device=device, non_blocking=True)
        images = extract_slices_from_phases(
            phases_cur, batch["timesteps"], batch["slice_indices"],
        )                                                       # (B, S, 518, 518, 3) [0,255]
        batch["images"] = images.permute(0, 1, 4, 2, 3).contiguous() / 255.0
    return batch
