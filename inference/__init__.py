"""Model-loading helpers for the evaluation harness.

Formerly the permanent RTFB (real-time free-breathing) inference package. That whole design —
per-dataset adapters turning raw cine into a hand-built canonical batch — was retired on
2026-08-16 and now lives in `_archive/`. Two things made it obsolete at once:

1. **The native-z refactor (docs/58)** deleted the fixed 12-plane cube the adapters were built
   around, so their geometry constants (`GRID_SHAPE`, `D_CANON`, `CANON_Z_SPACING_MM`,
   `MM_PER_NORM`) no longer describe anything real.
2. **Every source now has an `MRIDataset` entry** — OCMR was the last holdout until
   `tools/convert_ocmr_to_12phase.py` — so batches come from `MRIDataset.get_data`, the same code
   training uses. There is exactly one implementation of the geometry contract instead of three
   hand-written copies that drifted apart.

RTFB itself is out of scope for the current harness (gated + breathing-simulated only).

What remains here: `canonical_inplane.py` (the two in-plane helpers that outlived the adapters)
and the CMRxRecon segmentation-metrics scripts.
"""
