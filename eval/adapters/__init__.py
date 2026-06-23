"""RTFB dataset adapters: real-time cine on disk -> canonical model batch.

Every adapter is a `BaseRTFBAdapter` subclass implementing 3 seams
(`load` / `inplane_mm` / `slice_positions_mm`); the shared canonical pipeline
(percentile norm + in-plane resample + scattered-frame sampling + 518 upsample +
scanner_coords) lives in `base.py` and is identical across datasets.
"""
from eval.adapters.base import (
    BaseRTFBAdapter,
    percentile_scale,
    assign_canonical_z,
    to_canonical_inplane,
    INPUT_IMG_SIZE,
    TARGET_INPLANE_MM,
    GRID_SHAPE,
    D_CANON,
    CANON_Z_SPACING_MM,
    PCT_LO,
    PCT_HI,
    MM_PER_NORM,
    DEFAULT_CKPT,
)
from eval.adapters.ocmr import OCMRAdapter
from eval.adapters.goettingen import GoettingenAdapter
from eval.adapters.miitt import MIITTAdapter

__all__ = [
    "BaseRTFBAdapter",
    "OCMRAdapter",
    "GoettingenAdapter",
    "MIITTAdapter",
    "percentile_scale",
    "assign_canonical_z",
    "to_canonical_inplane",
    "INPUT_IMG_SIZE",
    "TARGET_INPLANE_MM",
    "GRID_SHAPE",
    "D_CANON",
    "CANON_Z_SPACING_MM",
    "PCT_LO",
    "PCT_HI",
    "MM_PER_NORM",
    "DEFAULT_CKPT",
]
