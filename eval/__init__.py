"""Permanent OOD real-time-free-breathing (RTFB) inference package for VGGT-MRI.

Houses the dataset adapters (data -> canonical model batch) and the shared inference
primitives (batch -> reconstructed volume) used to run the trained model on
out-of-distribution real-time cine (OCMR, Göttingen, MIITT). See docs/06, docs/16.

This is eval-only tooling: it deliberately BYPASSES `training/data/datasets/mri_dataset.py`
and never touches training code.
"""
