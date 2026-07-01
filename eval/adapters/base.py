"""Shared canonical RTFB pipeline + `BaseRTFBAdapter`.

The free functions (`percentile_scale`, `assign_canonical_z`, `to_canonical_inplane`)
and `_build_batch_core` are lifted VERBATIM from the original `tools/eval_ocmr_inference.py`
so OCMR inference stays numerically bit-identical (see tests/test_eval_ocmr_equivalence.py).
The ONLY change vs the original `build_batch` is that the in-plane spacing is passed in as
an argument (`inplane`) instead of read from `meta["inplane_mm"]` — a pure parameter-threading
move, no numeric op touched. Per-dataset variation is isolated to the 3 abstract seams below.
"""
import numpy as np
import torch
import torch.nn.functional as F

INPUT_IMG_SIZE = 518
TARGET_INPLANE_MM = 1.4
GRID_SHAPE = (12, 256, 256)          # (D, H, W) canonical splat grid
D_CANON = GRID_SHAPE[0]
CANON_Z_SPACING_MM = 12.0            # canonical plane spacing = CMRx true pitch (was 8mm thickness); docs/27
PCT_LO, PCT_HI = 0.5, 99.9           # matches ScaleIntensityByT0PercentilesD
# in-plane: (256-1)/2 * 1.4 mm ; through-plane: (12-1)/2 * 12.0 mm  (norm[-1,1] -> mm)
MM_PER_NORM = (178.5, 178.5, 66.0)
DEFAULT_CKPT = ("scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/"
                "ckpts/checkpoint_last.pt")
# Reference-slot model (docs/25): trained under mri_volume.yaml with use_reference_token=true.
# The aggregator is set-attention over S, so it's agnostic to how many input slots we feed at
# inference (S need not match training's S) — the only requirement is slot 0 carrying the
# target-phase reference marked via the native camera_token.
# Default = the mri_volume_diffusion run (wandb 4wokxzov): architecturally IDENTICAL to the
# reference model (inherits mri_volume: reference_slot, z-only, point head, no refiner) — it only
# swaps the DVF smoothness loss (L2 ‖∇u‖² for L1 TV), so load_rtfb_model_reference loads it as-is.
DEFAULT_CKPT_REFERENCE = ("scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/"
                          "ckpts/checkpoint_last.pt")


def percentile_scale(cine):
    """Single per-subject (vmin, vmax) over ALL nonzero voxels of the whole cine.
    Frame-selection-invariant so different random draws share one intensity scale.
    Mirrors preprocess.py's clip-and-rescale to [0, 1]."""
    nz = cine[cine > 0]
    if nz.size == 0:                      # degenerate all-zero cine: fall back to all voxels
        nz = cine.reshape(-1)            # (matches preprocess.py's nonzero->all fallback)
    vmin = np.percentile(nz, PCT_LO)
    vmax = np.percentile(nz, PCT_HI)
    return float(vmin), float(max(vmax, vmin + 1e-6))


def assign_canonical_z(positions):
    """Map each physical slice to a canonical z-index using the TRUE slice spacing
    (center-to-center along the stack axis, ~10 mm for OCMR), not the 8 mm thickness.
    Canonical planes are CANON_Z_SPACING_MM (12 mm) apart — the CMRx true pitch.
    Returns list of (z_canon_idx, slice_idx) for slices landing in [0, D-1];
    on collision keeps the slice closest to that plane center. No through-plane interp."""
    pos = np.asarray(positions, dtype=np.float64)       # (nS, 3) scanner mm
    axis = pos[-1] - pos[0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    d = (pos - pos[0]) @ axis                            # signed depth along stack (mm)
    d = d - d.mean()                                     # center the stack
    cont = d / CANON_Z_SPACING_MM + (D_CANON - 1) / 2.0  # continuous canonical index
    idx = np.floor(cont + 0.5).astype(int)  # round-half-up: deterministic (np.round uses banker's
    #                                         rounding → even/odd-dependent collisions on exact .5)
    best = {}                                            # z_canon -> (slice, |residual|)
    for s, (k, c) in enumerate(zip(idx, cont)):
        if 0 <= k <= D_CANON - 1:
            res = abs(c - k)
            if k not in best or res < best[k][1]:
                best[k] = (s, res)
    return sorted((k, s) for k, (s, _) in best.items())  # [(z_canon, slice_idx), ...]


def to_canonical_inplane(slice2d, inplane_mm):
    """(H, W) at native in-plane mm -> (256, 256) at 1.4 mm (bilinear resample + center
    crop/pad), matching Spacingd + ResizeWithPadOrCropd."""
    H, W = slice2d.shape
    sh = int(round(H * inplane_mm[1] / TARGET_INPLANE_MM))
    sw = int(round(W * inplane_mm[0] / TARGET_INPLANE_MM))
    t = torch.from_numpy(slice2d)[None, None].float()
    r = F.interpolate(t, size=(sh, sw), mode="bilinear", align_corners=True)[0, 0]
    out = torch.zeros(256, 256)
    # center crop/pad
    y0s, x0s = max(0, (sh - 256) // 2), max(0, (sw - 256) // 2)
    y0d, x0d = max(0, (256 - sh) // 2), max(0, (256 - sw) // 2)
    hh, ww = min(sh, 256), min(sw, 256)
    out[y0d:y0d + hh, x0d:x0d + ww] = r[y0s:y0s + hh, x0s:x0s + ww]
    return out  # (256, 256), values already normalized [0,1]


def _build_batch_core(cine, inplane, scale, z_map, rng, device):
    """One random frame per slice -> model batch (images[0,1], scanner_coords, z_indices).

    Verbatim copy of the original `build_batch` body; `inplane` is the only sourcing
    change (was `meta["inplane_mm"]`)."""
    vmin, vmax = scale
    n_frames = cine.shape[0]
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    x_norm = (px / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)

    imgs, coords, z_idx, picks = [], [], [], []
    for z_canon, slice_idx in z_map:
        f = int(rng.integers(n_frames))
        raw = cine[f, slice_idx]
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)                  # (256,256) [0,1]
        up = F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        imgs.append(np.repeat(up[None], 3, axis=0))                  # (3,518,518)
        z_val = z_canon / max(1, D_CANON - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        z_idx.append([z_val])
        picks.append((z_canon, slice_idx, f, up))
    S = len(imgs)
    batch = {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),          # (1,S,3,518,518) [0,1]
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),  # (1,S,518,518,3)
        "z_indices": torch.tensor(z_idx, dtype=torch.float32)[None].to(device),        # (1,S,1)
    }
    return batch, S, picks


def _build_batch_multiframe_core(cine, inplane, scale, z_map, device, frames_per_slice, frames_for_reference):
    """Multi-frame + reference-slot batch (docs/25 + docs/28) for INFERENCE, additive alongside
    `_build_batch_core` (kept verbatim for historical one-frame-per-slice callers). Reproducible
    (seeded), not random per run — random coverage/extras is a TRAINING-time augmentation only.

    Frames are a SHORT CONSECUTIVE burst of each slice's own recording (not evenly spaced across
    its full length) — the whole point of the project is a short acquisition burst per slice, so
    this truncates each slice's real-time dwell to a brief window. EVERY plane's burst — the
    reference's `frames_for_reference` frames (swept as the query) AND each other plane's
    `frames_per_slice` frames — starts at a RANDOM frame index (seeded), NOT frame 0: a real
    short acquisition of a slice lands at an arbitrary point in the cycle, so starting all planes
    at frame 0 would make the first cardiac phase trivially observed everywhere. The seed is
    fixed, so the starts are identical across runs (comparable across model checkpoints).

    The z-plane nearest canonical mid-depth is the reference. Its `frames_for_reference` frames
    are ALL fed to the model — one sits in slot 0 (the query, cycled by `reference_sweep` via the
    native camera_token anchor, docs/25), the REST ride along as companions so the model sees the
    full reference-plane cine as context (docs/28 — a per-slice cine makes phase/amplitude/
    through-plane motion observable). Companions are built once and held constant across the
    sweep — only slot 0 changes (mirrors the trainer's cardiac-cycle filmstrip).
    """
    vmin, vmax = scale
    n_frames_total = cine.shape[0]
    frames_for_reference = max(1, frames_for_reference)   # ≥1 so the reference burst / sweep is non-empty
    frames_per_slice = max(1, frames_per_slice)
    rng = np.random.default_rng(0)   # reproducible per-plane burst starts (not random per run)
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    x_norm = (px / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)

    def extract(slice_idx, f):
        raw = cine[f, slice_idx]
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)
        return F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                             mode="bilinear", align_corners=True)[0, 0].numpy()

    def random_burst(k):
        """k consecutive real frames starting at a random index (linear stream, no wrap)."""
        k = min(k, n_frames_total)
        s0 = int(rng.integers(max(1, n_frames_total - k + 1)))
        return list(range(s0, s0 + k))

    ref_i = int(np.argmin([abs(z - (D_CANON - 1) / 2.0) for z, _ in z_map]))
    ref_z, ref_slice = z_map[ref_i]
    rest = z_map[:ref_i] + z_map[ref_i + 1:]

    ref_frame_indices = random_burst(frames_for_reference)    # reference burst also starts random (seeded)
    slots = [(ref_z, ref_slice, ref_frame_indices[0])]        # slot 0: placeholder, overwritten per sweep step
    # The reference plane's OTHER frames ride along as companions too (not discarded) — gives
    # the model the full reference-plane cine as context, not just whichever frame is queried.
    slots += [(ref_z, ref_slice, f) for f in ref_frame_indices]
    for z_canon, slice_idx in rest:
        slots += [(z_canon, slice_idx, f) for f in random_burst(frames_per_slice)]

    imgs, coords, z_idx, picks = [], [], [], []
    for z_canon, slice_idx, f in slots:
        up = extract(slice_idx, f)
        imgs.append(np.repeat(up[None], 3, axis=0))                  # (3,518,518)
        z_val = z_canon / max(1, D_CANON - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        z_idx.append([z_val])
        picks.append((z_canon, slice_idx, f, up))
    S = len(imgs)
    batch = {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),          # (1,S,3,518,518) [0,1]
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),  # (1,S,518,518,3)
        "z_indices": torch.tensor(z_idx, dtype=torch.float32)[None].to(device),        # (1,S,1)
    }
    ref_ctx = dict(z_canon=ref_z, slice_idx=ref_slice, inplane=inplane, scale=scale,
                   cine_slice=cine[:, ref_slice],             # (n_frames,H,W), for sweeping slot 0
                   frame_indices=ref_frame_indices)
    return batch, S, picks, ref_ctx


class BaseRTFBAdapter:
    """Real-time-free-breathing cine -> canonical model batch.

    Subclasses implement the 3 per-dataset seams; the canonical pipeline (intensity
    normalization, in-plane resample, scattered single-frame-per-slice sampling, 518
    upsample, scanner_coords) is shared and identical across datasets.
    """

    # ── abstract seams (the ONLY per-dataset variation) ──────────────────
    def load(self):
        """-> cine[frame, slice, H, W] float32 (the full continuous real-time cine)."""
        raise NotImplementedError

    def inplane_mm(self):
        """-> (sx, sy) native in-plane spacing in mm."""
        raise NotImplementedError

    def slice_positions_mm(self):
        """-> (nS, 3) per-slice scanner positions in mm (for canonical-z assignment)."""
        raise NotImplementedError

    # ── concrete pipeline ────────────────────────────────────────────────
    def build_batch(self, rng, device):
        """Sample one random frame per in-FOV canonical z plane -> (batch, S, picks)."""
        cine = self.load()
        scale = percentile_scale(cine)
        z_map = assign_canonical_z(self.slice_positions_mm())
        return _build_batch_core(cine, self.inplane_mm(), scale, z_map, rng, device)

    def build_batch_multiframe(self, device, frames_per_slice=5, frames_for_reference=30):
        """Deterministic multi-frame + reference-slot sampling for INFERENCE (docs/25, docs/28)
        -> (batch, S, picks, ref_ctx). `ref_ctx` carries the reference plane's real frame stack
        for `reference_sweep`. No randomness — that's a training-only augmentation."""
        cine = self.load()
        scale = percentile_scale(cine)
        z_map = assign_canonical_z(self.slice_positions_mm())
        return _build_batch_multiframe_core(cine, self.inplane_mm(), scale, z_map, device,
                                            frames_per_slice, frames_for_reference)
