"""OCMR bit-identical guard for the inference/ adapter refactor.

Asserts the new `inference/` path produces byte-identical batches to the FROZEN original OCMR
code (tests/_legacy_ocmr.py). The data-free test always runs; the real-subject test runs
only when reconstructed OCMR data is present on disk.

Run on CPU (the adapter refactor is entirely pre-model; no autocast / GPU nondeterminism).
"""
import glob
import os

import numpy as np
import pytest
import torch

from inference.adapters.base import BaseRTFBAdapter, percentile_scale, assign_canonical_z
from inference.adapters.ocmr import OCMRAdapter
import tests._legacy_ocmr as legacy

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OCMR_RECON = os.path.join(_ROOT, "scratch", "data", "ocmr", "recon", "rtfb")


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
    [os.path.dirname(f)  # recon/rtfb/<exam_id>/<subject>/sax_cine.nii.gz (grouped by patient/exam)
     for f in sorted(glob.glob(os.path.join(OCMR_RECON, "*", "*", "sax_cine.nii.gz")))
     if not os.path.relpath(f, OCMR_RECON).startswith("_")]  # skip _failed_* exam dirs
    if os.path.isdir(OCMR_RECON) else []
)


def test_continuous_z_keeps_all_slices_at_fractional_depth():
    """continuous_z=True keeps every in-range slice at its own fractional z (no snap, no
    collision dedup); default (snap) collapses a finer-than-12mm stack and loses slices."""
    from inference.adapters.base import D_CANON, CANON_Z_SPACING_MM
    Z = 8
    positions = np.stack([np.zeros(Z), np.zeros(Z), np.arange(Z) * 10.0], axis=1)  # 10mm pitch

    snap = assign_canonical_z(positions, continuous_z=False)
    cont = assign_canonical_z(positions, continuous_z=True)

    # 10mm-into-12mm ⇒ collisions ⇒ snap drops slices; continuous keeps them all.
    assert len(cont) == Z                      # every slice survives
    assert len(snap) < Z                       # at least one collision dropped
    # snapped z are ints; continuous z are floats, at least one strictly non-integer.
    assert all(isinstance(z, (int, np.integer)) for z, _ in snap)
    assert all(isinstance(z, float) for z, _ in cont)
    assert any(abs(z - round(z)) > 1e-6 for z, _ in cont)
    # continuous z matches the exact geometric formula d/12 + (D-1)/2, centered.
    d = np.arange(Z) * 10.0; d = d - d.mean()
    expect = d / CANON_Z_SPACING_MM + (D_CANON - 1) / 2.0
    got = {s: z for z, s in cont}
    for s in range(Z):
        assert abs(got[s] - expect[s]) < 1e-9


def test_continuous_z_batch_feeds_fractional_z_indices():
    """The batch built with continuous_z=True carries the fractional z into z_indices/scanner_coords
    (float-safe downstream), with one slot per surviving slice."""
    from inference.adapters.base import D_CANON
    F_, Z, H, W = 9, 8, 60, 64
    cine = np.random.default_rng(1).random((F_, Z, H, W)).astype(np.float32)
    positions = np.stack([np.zeros(Z), np.zeros(Z), np.arange(Z) * 10.0], axis=1)
    b, S, picks = _FakeAdapter(cine, [1.8, 1.7], positions).build_batch(
        np.random.default_rng(0), "cpu", continuous_z=True)
    assert S == Z                                              # all slices kept
    z_vals = b["z_indices"][0, :, 0].numpy()
    # z_val = z_canon/(D-1)*2 - 1 ⇒ recover z_canon and check it's fractional for some slot.
    z_canon = (z_vals + 1.0) / 2.0 * (D_CANON - 1)
    assert np.any(np.abs(z_canon - np.round(z_canon)) > 1e-4)  # genuinely off-grid depths reach the model


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
