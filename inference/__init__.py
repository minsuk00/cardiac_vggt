"""Model-loading helpers for the evaluation harness.

Batches come from `MRIDataset.get_data`, the same code training uses, so there is exactly one
implementation of the geometry contract. The retired per-dataset RTFB adapters live in
`_archive/` (docs/79).

What remains here: `load_run.py` (build a model from its run's own config) and
`canonical_inplane.py` (two in-plane resampling helpers).
"""
