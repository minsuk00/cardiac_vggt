"""Canonical-grid monai preprocess pipeline for VGGT-MRI.

Produces a cached `(T=12, 1, X=256, Y=256, Z=12)` float16 tensor per subject in
monai (X, Y, Z) axis order. The downstream consumer (`mri_dataset.get_data`)
permutes to splat-order `(T, D=12, H=256, W=256)` once at cache-load time.

Key invariants this pipeline preserves:
- All subjects map to the same physical cube: 358.4 mm × 358.4 mm × 144 mm.
- Intensity is normalized against phase_00's percentiles for ALL 12 phases,
  so the unsupervised |V_canon - V_gt| loss isn't biased by per-phase drift
  (matches the legacy contract at mri_dataset.py:155-169).
- No interpolation of input slices at training time — Spacingd does the
  resample once, results are cached on /tmp.
"""

from __future__ import annotations

import os

import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
    CastToTyped,
    Compose,
    ConcatItemsd,
    EnsureChannelFirstd,
    LoadImaged,
    MapTransform,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
)

TARGET_SPACING = (1.4, 1.4, 12.0)       # mm per voxel; ~4% X-downsample for most subjects, Y near-identity.
                                        # Z=12mm = CMRx TRUE slice pitch (8mm thickness + 4mm gap, CMRxRecon2024
                                        # protocol). Source NIfTI affines were relabeled 8→12 on disk (the 8mm was
                                        # slice thickness) so this Spacingd is a Z-identity — see docs/27 +
                                        # tools/relabel_slice_spacing.py.
TARGET_SHAPE = (256, 256, 12)           # (X, Y, Z) in monai order
NUM_PHASES = 12
HALF_EXTENT_MM_SPLAT = (72.0, 179.2, 179.2)   # (D=Z, H=Y, W=X) — splat-order. 12/2*12.0, 256/2*1.4, 256/2*1.4


# ──────────────────────────────────────────────────────────────────────────────
# Custom transforms
# ──────────────────────────────────────────────────────────────────────────────
class ScaleIntensityByT0PercentilesD(MapTransform):
    """Compute intensity percentiles on `ref_key` and apply the same clip-and-
    rescale to all `keys`.

    Preserves mri_dataset.py:155-169's invariant: every phase is normalized
    against t=0's intensity statistics so the unsupervised |V_canon - V_gt|
    loss isn't biased by per-phase intensity drift. Output is float in [0, 1].
    """

    # Percentiles are computed over NON-ZERO ref voxels only (the FOV): the canonical
    # cube zero-pads small subjects, and exact-zero padding would otherwise skew the
    # stat (e.g. pin the low percentile to 0). FOV air is positive (Rician noise), so
    # only the zero-padding is dropped. NOTE on (lower, upper): 0.5/99.9 = robust clip
    # of the dark floor + bright outlier tail (flow/blood-pool spikes). Do NOT retune
    # these to chase PSNR — normalization rescales both recon and GT, so PSNR ∝
    # 20·log10(range) shifts cosmetically WITHOUT changing reconstruction. Changing the
    # region or percentiles invalidates checkpoints; the monai cache auto-rebuilds via
    # cache_signature() (PersistentDataset keys on file paths, not the transform).
    def __init__(self, keys, ref_key, lower: float = 0.5, upper: float = 99.9, eps: float = 1e-8):
        super().__init__(keys, allow_missing_keys=False)
        self.ref_key = ref_key
        self.lower = lower
        self.upper = upper
        self.eps = eps

    def __call__(self, data):
        d = dict(data)
        ref = d[self.ref_key]
        ref_t = ref.as_tensor() if hasattr(ref, "as_tensor") else ref
        ref_f = ref_t.reshape(-1).float()
        # Drop exact-zero padding so it can't skew the percentiles; keep FOV voxels
        # (air is positive). Fall back to all voxels if the ref is all-zero (degenerate).
        nz = ref_f[ref_f != 0]
        if nz.numel() > 0:
            ref_f = nz
        # torch.quantile is exact but slow on big tensors; use it because the volumes
        # are small (256·256·12 ≈ 786k voxels), well under the kernel-launch cost.
        vmin = torch.quantile(ref_f, self.lower / 100.0).item()
        vmax = torch.quantile(ref_f, self.upper / 100.0).item()
        denom = max(vmax - vmin, self.eps)
        for k in self.key_iterator(d):
            v = d[k]
            v_f = v.float() if hasattr(v, "float") else torch.as_tensor(v).float()
            d[k] = ((v_f - vmin) / denom).clamp(0.0, 1.0)
        return d


class AddOnesMaskD(MapTransform):
    """Add a `content_mask` key — an all-ones tensor with the same shape and
    MetaTensor metadata as `ref_key`. Insert this AFTER `EnsureChannelFirstd`
    and BEFORE `Orientationd`/`Spacingd`/`ResizeWithPadOrCropd`, and include
    `content_mask` in those transforms' `keys`. The mask is then warped /
    resampled / padded identically to the phases, so at pipeline end:

        content_mask == 1   →   this canonical voxel came from native data
        content_mask == 0   →   this canonical voxel was zero-padded

    The bbox of the final mask is the **geometric** subject-FOV region in the
    canonical cube — no anatomy intensity involved. Robust to spatial aug too:
    apply the same aug affine to the mask and the bbox stays correct.
    """

    def __init__(self, ref_key: str = "phase_00", output_key: str = "content_mask"):
        super().__init__([ref_key], allow_missing_keys=False)
        self.ref_key = ref_key
        self.output_key = output_key

    def __call__(self, data):
        d = dict(data)
        ref = d[self.ref_key]
        if hasattr(ref, "as_tensor"):
            ones = torch.ones_like(ref.as_tensor(), dtype=torch.float32)
            # Preserve MetaTensor wrapping so subsequent transforms (Orientationd,
            # Spacingd, etc.) use the SAME affine to reorder/resample as the phases.
            mask = MetaTensor(
                ones,
                meta=dict(ref.meta) if hasattr(ref, "meta") else {},
                applied_operations=list(getattr(ref, "applied_operations", [])),
            )
            if hasattr(ref, "affine"):
                mask.affine = ref.affine.clone()
        else:
            mask = torch.ones_like(ref, dtype=torch.float32)
        d[self.output_key] = mask
        return d


class StripMetaD(MapTransform):
    """Drop the MetaTensor subclass on `keys` (returns plain `torch.Tensor`).

    Needed because `PersistentDataset`'s `torch.load(weights_only=True)` returns
    plain tensors on cache hit, while cache miss runs the live transforms which
    yield MetaTensor. With `batch_size > 1`, MONAI's `collate_meta_tensor_fn`
    chokes when a batch mixes the two states. Stripping at the end of the
    cached pipeline makes both states uniform.

    Adapted from /home/minsukc/MRI2CT/src/common/data.py:227-247.
    """

    def __init__(self, keys):
        super().__init__(keys, allow_missing_keys=True)

    def __call__(self, data):
        d = dict(data)
        for k in self.key_iterator(d):
            v = d[k]
            if hasattr(v, "as_tensor"):
                d[k] = v.as_tensor()
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Utility: geometric bbox from content_mask
# ──────────────────────────────────────────────────────────────────────────────
def compute_geometric_bbox(content_mask, padding: int = 0) -> torch.Tensor:
    """Bbox of the subject's content region in canonical voxel indices, splat order.

    Purely **geometric** — based on the 1/0 content mask that was warped through
    the same spatial transforms (Orientationd, Spacingd, ResizeWithPadOrCropd,
    and any runtime aug) as the phases. No intensity thresholding on anatomy.

    Args:
        content_mask: `(D, H, W)` tensor in splat order. 1 = subject's native
            FOV reached this voxel; 0 = zero-pad. Any dtype is fine; the test
            is `mask > 0` which catches both bool and 0/1 numeric.
        padding: optional extra voxels of safety margin on each side, clamped
            to volume bounds. Default 0 (tight geometric bbox).

    Returns:
        `(z0, z1, y0, y1, x0, x1)` `int64` tensor. `[z0:z1, y0:y1, x0:x1]` is
        a valid Python slice (z1/y1/x1 are exclusive ends). Falls back to the
        full cube if the mask is empty (post-aug edge case).
    """
    if content_mask.ndim != 3:
        raise ValueError(
            f"compute_geometric_bbox expects (D, H, W), got {tuple(content_mask.shape)}"
        )
    D, H, W = content_mask.shape
    mask = content_mask > 0
    if not mask.any():
        return torch.tensor([0, D, 0, H, 0, W], dtype=torch.int64, device=content_mask.device)
    nz = mask.nonzero()
    z0, y0, x0 = nz.min(dim=0).values.tolist()
    z_max, y_max, x_max = nz.max(dim=0).values.tolist()
    z1, y1, x1 = z_max + 1, y_max + 1, x_max + 1  # exclusive
    z0 = max(0, z0 - padding); z1 = min(D, z1 + padding)
    y0 = max(0, y0 - padding); y1 = min(H, y1 + padding)
    x0 = max(0, x0 - padding); x1 = min(W, x1 + padding)
    return torch.tensor([z0, z1, y0, y1, x0, x1], dtype=torch.int64, device=content_mask.device)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline factories
# ──────────────────────────────────────────────────────────────────────────────
def get_canonical_transforms(
    target_spacing=TARGET_SPACING,
    target_shape=TARGET_SHAPE,
    num_phases: int = NUM_PHASES,
    lower: float = 0.5,
    upper: float = 99.9,
    storage_dtype=torch.float16,
):
    """Build the deterministic monai pipeline for `PersistentDataset`.

    Output dict has one key, `phases`, of shape `(T=num_phases, 1, X, Y, Z)` in
    monai axis order. `mri_dataset.get_data` is responsible for permuting to
    splat-order at load time.
    """
    phase_keys = [f"phase_{t:02d}" for t in range(num_phases)]
    # Mask propagates through the spatial transforms alongside the phases so its
    # final shape is canonically (1, 256, 256, 12) in monai (X, Y, Z) order and
    # marks which voxels came from native data vs zero-pad.
    spatial_keys = phase_keys + ["content_mask"]
    # Per-key interpolation mode: phases bilinear (smooth), mask nearest (binary).
    spacing_modes = ["bilinear"] * len(phase_keys) + ["nearest"]
    return Compose(
        [
            LoadImaged(keys=phase_keys, image_only=True),
            EnsureChannelFirstd(keys=phase_keys),
            AddOnesMaskD(ref_key=phase_keys[0], output_key="content_mask"),
            Orientationd(keys=spatial_keys, axcodes="LPS"),
            Spacingd(keys=spatial_keys, pixdim=target_spacing, mode=spacing_modes),
            ResizeWithPadOrCropd(
                keys=spatial_keys,
                spatial_size=target_shape,
                mode="constant",   # zero-pad small subjects
                value=0,
            ),
            ScaleIntensityByT0PercentilesD(
                keys=phase_keys, ref_key=phase_keys[0], lower=lower, upper=upper
            ),
            ConcatItemsd(keys=phase_keys, name="phases", dim=0),
            CastToTyped(keys=["phases"], dtype=storage_dtype),
            CastToTyped(keys=["content_mask"], dtype=torch.uint8),
            StripMetaD(keys=["phases", "content_mask"]),
        ]
    )


def build_data_dicts(subject_sax_dirs, num_phases: int = NUM_PHASES):
    """Map each subject's `…/sax` directory to a `data` dict consumable by the
    canonical pipeline.

    Args:
        subject_sax_dirs: iterable of paths like
            `/scratch/.../Cine_combined/Test_P001/sax`
        num_phases: how many `sax_frame_{tt}.nii.gz` files to expect (default 12).

    Returns:
        `[{phase_00: path, ..., phase_{N-1:02d}: path, subj_id: str, sax_dir: str}, ...]`
    """
    items = []
    for sax_dir in subject_sax_dirs:
        subj_id = os.path.basename(os.path.dirname(sax_dir))
        d = {
            f"phase_{t:02d}": os.path.join(sax_dir, "3d_recon", f"sax_frame_{t:02d}.nii.gz")
            for t in range(num_phases)
        }
        d["subj_id"] = subj_id
        d["sax_dir"] = sax_dir
        items.append(d)
    return items


def cache_signature() -> str:
    """Short hash of the params that determine cached tensor *content*.

    monai `PersistentDataset` keys its cache on the input data dict (file paths),
    NOT on the transform — so changing spacing / shape / normalization would silently
    reuse a stale cache. Append this to the cache dir so any such change auto-routes
    to a fresh subdir (old cache is harmlessly orphaned on /tmp). NORM_REGION/percentiles
    are folded in so the non-zero 0.5/99.9 change can't no-op on a warm node.
    """
    import hashlib

    sig = repr((TARGET_SPACING, TARGET_SHAPE, NUM_PHASES, "nonzero", 0.5, 99.9))
    return hashlib.md5(sig.encode()).hexdigest()[:10]


def default_cache_dir() -> str:
    """Node-local NVMe cache dir for `PersistentDataset`.

    `/tmp` is intentional: local NVMe is faster than networked /scratch, and
    cache rebuild is a one-time-per-node cost that's acceptable per SLURM job.
    Mirrors MRI2CT's `default_monai_cache_dir`.
    """
    user = os.environ.get("USER", "default")
    return f"/tmp/vggt-mri_{user}_monai_cache"
