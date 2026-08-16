"""MIITT geometry helpers — native <-> canonical placement that MIRRORS
`inference.adapters.base.build_canonical_bundle` EXACTLY, so anything we place
(the SVRTK recon, the heart ROI) is co-located voxel-for-voxel with the GT bundle
that VGGT scores against.

Why not affine-resample the recon straight to canonical? The GT is built by
*discrete slice placement* (each native slice snapped to its nearest 12 mm plane).
A physical affine resample would put the recon's content at its TRUE z, offset up
to ~6 mm from the snapped GT slice -> an unfair through-plane penalty that VGGT
(which also snaps) never pays. So we bring the recon back to the NATIVE grid first
(exact, same world frame) and then run the SAME placement -> perfect co-location.
"""
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc

from inference.adapters.base import (
    assign_canonical_z, to_canonical_inplane, percentile_scale, D_CANON,
)


def native_grid(adapter):
    """(shape_xyz, affine) of the adapter's native gated volume — the grid the
    SVRTK recon is resampled back onto before placement."""
    img = nib.load(adapter.nii_path)
    return img.shape[:3], img.affine


def resample_to_native(recon_path, adapter):
    """SVRTK recon (1.4 mm iso, in the native world frame) -> the native gated grid
    (X,Y,Z) via affine resample. Same frame => this is an exact geometric downsample,
    NOT a re-centering. Returns (Z,H,W) = (slice,Y,X) to match adapter.load() order."""
    shape_xyz, aff = native_grid(adapter)
    out = nibproc.resample_from_to(nib.load(recon_path), (shape_xyz, aff), order=1, cval=0.0)
    xyz = np.asarray(out.dataobj, dtype=np.float32)      # (X,Y,Z)
    return np.transpose(xyz, (2, 1, 0))                  # (Z,H,W)


def place_to_canonical(vol_or_cine, adapter, normalize=False, binary=False):
    """Place native slices onto the canonical (12,256,256) grid EXACTLY like
    build_canonical_bundle: in-plane -> 1.4 mm/256 (center crop-pad), z snapped to
    nearest integer plane, collisions dropped by assign_canonical_z, empty planes 0.

    vol_or_cine: (Z,H,W) single volume or (T,Z,H,W) cine, native order (from load()).
    normalize:  percentile-rescale to [0,1] with the whole-input scale (GT-consistent).
    binary:     threshold >0.5 after bilinear placement (for masks).
    Returns (12,256,256) or (T,12,256,256) float32.
    """
    a = np.asarray(vol_or_cine, dtype=np.float32)
    single = (a.ndim == 3)
    if single:
        a = a[None]                                      # (1,Z,H,W)
    T, Z, H, W = a.shape
    if normalize:
        vmin, vmax = percentile_scale(a)
        a = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    inpl = adapter.inplane_mm()
    z_map = assign_canonical_z(adapter.slice_positions_mm())
    out = np.zeros((T, D_CANON, 256, 256), np.float32)
    for z_canon, slice_idx in z_map:
        zc = min(max(int(np.floor(float(z_canon) + 0.5)), 0), D_CANON - 1)
        for t in range(T):
            out[t, zc] = to_canonical_inplane(a[t, slice_idx], inpl).numpy()
    if binary:
        out = (out > 0.5).astype(np.float32)
    return out[0] if single else out
