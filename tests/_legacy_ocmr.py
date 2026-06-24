"""FROZEN verbatim snapshot of the original OCMR batch-building code, as it existed in
`tools/eval_ocmr_inference.py` BEFORE the eval/ refactor (commit prior to the shim).

This is a guard reference for tests/test_eval_ocmr_equivalence.py: it must NOT be edited to
track refactors — its whole purpose is to stay frozen so the new `eval/` path can be proven
bit-identical to the original. Do not import this outside tests.
"""
import numpy as np
import torch
import torch.nn.functional as F

INPUT_IMG_SIZE = 518
TARGET_INPLANE_MM = 1.4
GRID_SHAPE = (12, 256, 256)
D_CANON = GRID_SHAPE[0]
CANON_Z_SPACING_MM = 12.0  # tracks eval.adapters.base canonical pitch (docs/18); guard tests structure not 8mm
PCT_LO, PCT_HI = 0.5, 99.9


def percentile_scale(cine):
    nz = cine[cine > 0]
    if nz.size == 0:
        nz = cine.reshape(-1)
    vmin = np.percentile(nz, PCT_LO)
    vmax = np.percentile(nz, PCT_HI)
    return float(vmin), float(max(vmax, vmin + 1e-6))


def assign_canonical_z(positions):
    pos = np.asarray(positions, dtype=np.float64)
    axis = pos[-1] - pos[0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    d = (pos - pos[0]) @ axis
    d = d - d.mean()
    cont = d / CANON_Z_SPACING_MM + (D_CANON - 1) / 2.0
    idx = np.floor(cont + 0.5).astype(int)
    best = {}
    for s, (k, c) in enumerate(zip(idx, cont)):
        if 0 <= k <= D_CANON - 1:
            res = abs(c - k)
            if k not in best or res < best[k][1]:
                best[k] = (s, res)
    return sorted((k, s) for k, (s, _) in best.items())


def to_canonical_inplane(slice2d, inplane_mm):
    H, W = slice2d.shape
    sh = int(round(H * inplane_mm[1] / TARGET_INPLANE_MM))
    sw = int(round(W * inplane_mm[0] / TARGET_INPLANE_MM))
    t = torch.from_numpy(slice2d)[None, None].float()
    r = F.interpolate(t, size=(sh, sw), mode="bilinear", align_corners=True)[0, 0]
    out = torch.zeros(256, 256)
    y0s, x0s = max(0, (sh - 256) // 2), max(0, (sw - 256) // 2)
    y0d, x0d = max(0, (256 - sh) // 2), max(0, (256 - sw) // 2)
    hh, ww = min(sh, 256), min(sw, 256)
    out[y0d:y0d + hh, x0d:x0d + ww] = r[y0s:y0s + hh, x0s:x0s + ww]
    return out


def build_batch(cine, meta, scale, z_map, rng, device):
    """Original signature — takes `meta` and reads meta['inplane_mm']."""
    vmin, vmax = scale
    n_frames = cine.shape[0]
    inplane = meta["inplane_mm"]
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    x_norm = (px / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)

    imgs, coords, z_idx, picks = [], [], [], []
    for z_canon, slice_idx in z_map:
        f = int(rng.integers(n_frames))
        raw = cine[f, slice_idx]
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)
        up = F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        imgs.append(np.repeat(up[None], 3, axis=0))
        z_val = z_canon / max(1, D_CANON - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        z_idx.append([z_val])
        picks.append((z_canon, slice_idx, f, up))
    S = len(imgs)
    batch = {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),
        "z_indices": torch.tensor(z_idx, dtype=torch.float32)[None].to(device),
    }
    return batch, S, picks
