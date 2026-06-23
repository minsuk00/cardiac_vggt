"""Göttingen adapter: radial real-time free-breathing bSSFP recon (docs/16).

Reads the native 4D recon `<subj>.nii.gz` (X, Y, Z, T ~127 frames) DIRECTLY and feeds the
real slices through the shared canonical pipeline — replacing the former placeholder-12-phase
MRIDataset path. This drops the old 6→8 mm through-plane interpolation (each canonical input
is now a faithful single acquired slice, not an interpolated blend) and samples from the full
continuous frame pool.

Geometry is carried in the NIfTI affine (`diag(1.6, 1.6, 6.0)`); there is no meta.json, so
in-plane mm is the native 1.6 mm and per-slice positions are synthesized as an evenly-spaced
stack at the 6 mm native slice spacing.
"""
import numpy as np
import nibabel as nib

from eval.adapters.base import BaseRTFBAdapter

INPLANE_MM = (1.6, 1.6)
SLICE_SPACING_MM = 6.0


class GoettingenAdapter(BaseRTFBAdapter):
    def __init__(self, nii_path):
        self.nii_path = nii_path
        self._a = nib.load(nii_path).get_fdata().astype(np.float32)  # (X, Y, Z, T)

    def load(self):
        # (X, Y, Z, T) -> (frame=T, slice=Z, H=Y, W=X)
        return np.transpose(self._a, (3, 2, 1, 0))

    def inplane_mm(self):
        return INPLANE_MM

    def slice_positions_mm(self):
        nS = self._a.shape[2]
        return np.stack([np.zeros(nS), np.zeros(nS),
                         np.arange(nS) * SLICE_SPACING_MM], axis=1)
