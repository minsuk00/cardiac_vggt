"""In-plane canonicalization helpers: native (H,W) mm grid -> the shared 256x256 @ 1.4 mm grid.

Moved out of the archived `inference/adapters/base.py` (now `inference/_archive/adapters/base.py`)
when the RTFB adapter stack was retired. These two functions are the only part of that module
that SURVIVED the native-z refactor (docs/58) unchanged: they touch the in-plane axes only, which
are still resampled to a fixed shared grid. Everything else in the old `base.py` was built on the
retired fixed-12-plane cube (`GRID_SHAPE`, `D_CANON`, `CANON_Z_SPACING_MM`, `MM_PER_NORM`) and is
kept in the archive as a frozen record, not as importable code.

They live here rather than in the archive because `baselines/fetal_cmr_4d/` still imports them and
is outside the evaluation/inference rebuild. Nothing in the current eval harness uses them —
`MRIDataset` / `training/data/preprocess.py` do their own resampling, and that is the one path any
new code should go through.
"""
import numpy as np
import torch
import torch.nn.functional as F

TARGET_INPLANE_MM = 1.4
PCT_LO, PCT_HI = 0.5, 99.9           # matches ScaleIntensityByT0PercentilesD


def percentile_scale(cine):
    """Single per-subject (vmin, vmax) over ALL nonzero voxels of the whole cine.
    Frame-selection-invariant so different random draws share one intensity scale.
    Mirrors preprocess.py's clip-and-rescale to [0, 1]."""
    nz = cine[cine > 0]
    if nz.size == 0:                      # degenerate all-zero cine: fall back to all voxels
        nz = cine.reshape(-1)            # (matches preprocess.py's nonzero->all fallback)
    # `float()` BEFORE the span guard is load-bearing under numpy 2 (docs/49):
    # numpy 2 returns a float32 scalar from np.percentile on float32 input (numpy 1
    # returned float64), and float32 cannot resolve `+ 1e-6` once vmin >= 32 — so
    # the divide-by-zero guard silently became a no-op, collapsing the span to 0
    # and yielding an all-NaN normalized cine (which _nanmean then quietly drops)
    # instead of the finite fallback. Widening to a python float restores it at
    # zero cost. No-op under numpy 1.
    #
    # NOT done deliberately: casting `nz` to float64 before np.percentile would
    # also reproduce numpy 1's values exactly, but costs +34% time and 2.5x peak
    # memory (measured) to recover a ~1.9e-4 difference on a [0,1] image — far
    # below PSNR/Dice/EF noise. Not worth it; see docs/49.
    vmin = float(np.percentile(nz, PCT_LO))
    vmax = float(np.percentile(nz, PCT_HI))
    return vmin, max(vmax, vmin + 1e-6)


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
