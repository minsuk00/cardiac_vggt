"""OCMR bit-identical guard for the eval/ adapter refactor.

Asserts the new `eval/` path produces byte-identical batches to the FROZEN original OCMR
code (tests/_legacy_ocmr.py). The data-free test always runs; the real-subject test runs
only when reconstructed OCMR data is present on disk.

Run on CPU (the adapter refactor is entirely pre-model; no autocast / GPU nondeterminism).
"""
import glob
import os

import numpy as np
import pytest
import torch

from eval.adapters.base import BaseRTFBAdapter, percentile_scale, assign_canonical_z
from eval.adapters.ocmr import OCMRAdapter
import tests._legacy_ocmr as legacy

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OCMR_RECON = os.path.join(_ROOT, "scratch", "data", "ocmr", "recon")


def _assert_batches_equal(b_old, S_old, p_old, b_new, S_new, p_new):
    assert S_old == S_new
    for k in ("images", "scanner_coords", "z_indices"):
        torch.testing.assert_close(b_old[k], b_new[k], rtol=0, atol=0)  # bit-exact
    # picks: (z_canon, slice_idx, frame, up_image) — all must match
    assert len(p_old) == len(p_new)
    for (zo, so, fo, uo), (zn, sn, fn, un) in zip(p_old, p_new):
        assert (zo, so, fo) == (zn, sn, fn)
        np.testing.assert_array_equal(uo, un)


class _FakeAdapter(BaseRTFBAdapter):
    """Drives the new pipeline from in-memory arrays (no disk / SimpleITK needed)."""
    def __init__(self, cine, inplane, positions):
        self._cine, self._inplane, self._positions = cine, inplane, positions

    def load(self):
        return self._cine

    def inplane_mm(self):
        return self._inplane

    def slice_positions_mm(self):
        return self._positions


def test_build_batch_param_threading_is_identity():
    """The new _build_batch_core (inplane as arg) == legacy build_batch (meta['inplane_mm'])."""
    rng_seed = 0
    F_, Z, H, W = 9, 8, 60, 64           # synthetic continuous cine
    cine = np.random.default_rng(123).random((F_, Z, H, W)).astype(np.float32)
    inplane = [1.8, 1.7]
    # synthesize a slice stack ~10 mm apart so several land in [0, D-1]
    positions = np.stack([np.zeros(Z), np.zeros(Z), np.arange(Z) * 10.0], axis=1)

    scale = percentile_scale(cine)
    z_map = assign_canonical_z(positions)
    # legacy uses its OWN copies of percentile_scale/assign_canonical_z — confirm they agree first
    assert scale == legacy.percentile_scale(cine)
    assert z_map == legacy.assign_canonical_z(positions)

    b_old, S_old, p_old = legacy.build_batch(
        cine, {"inplane_mm": inplane}, scale, z_map, np.random.default_rng(rng_seed), "cpu")
    b_new, S_new, p_new = _FakeAdapter(cine, inplane, positions).build_batch(
        np.random.default_rng(rng_seed), "cpu")
    _assert_batches_equal(b_old, S_old, p_old, b_new, S_new, p_new)


_real_subjects = (
    [d for d in sorted(glob.glob(os.path.join(OCMR_RECON, "*")))
     if os.path.exists(os.path.join(d, "sax_cine.nii.gz"))]
    if os.path.isdir(OCMR_RECON) else []
)


@pytest.mark.skipif(not _real_subjects, reason="real OCMR recon data absent")
def test_ocmr_adapter_matches_legacy_on_real_subject():
    """OCMRAdapter vs frozen legacy on a real reconstructed OCMR subject, bit-exact."""
    import SimpleITK as sitk
    import json
    sd = _real_subjects[0]
    # legacy path
    cine = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(sd, "sax_cine.nii.gz"))).astype(np.float32)
    meta = json.load(open(os.path.join(sd, "meta.json")))
    scale = legacy.percentile_scale(cine)
    z_map = legacy.assign_canonical_z(meta["slice_positions_mm"])
    b_old, S_old, p_old = legacy.build_batch(cine, meta, scale, z_map, np.random.default_rng(0), "cpu")
    # new path
    b_new, S_new, p_new = OCMRAdapter(sd).build_batch(np.random.default_rng(0), "cpu")
    _assert_batches_equal(b_old, S_old, p_old, b_new, S_new, p_new)
