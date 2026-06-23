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
CANON_Z_SPACING_MM = 8.0             # canonical plane spacing
PCT_LO, PCT_HI = 0.5, 99.9           # matches ScaleIntensityByT0PercentilesD
# in-plane: (256-1)/2 * 1.4 mm ; through-plane: (12-1)/2 * 8.0 mm  (norm[-1,1] -> mm)
MM_PER_NORM = (178.5, 178.5, 44.0)
DEFAULT_CKPT = ("scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/"
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
    (center-to-center along the stack axis, ~10 mm), not the 8 mm thickness.
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
