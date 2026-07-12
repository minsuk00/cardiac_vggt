"""ACDC adapter: ACDC (Automated Cardiac Diagnosis Challenge) gated SAX cine.

Reads a patient's 4D cine `patientXXX_4d.nii.gz` (X, Y, Z, T) — the full ECG-gated
breath-hold short-axis stack (150 patients under `scratch/data/ACDC/{training,testing}/`).
Same canonical pipeline as the MIITT/OCMR gated arms (one random cardiac phase per slice,
then reference-slot sweep / breathing sim).

Two ACDC-specific quirks handled here (unlike MIITT's fixed placeholder constants):
- **Orientation:** ACDC is stored in mixed orientations (114 LPS / 36 LAS across the 150). The
  model is trained only on LPS (`training/data/preprocess.py` `Orientationd(axcodes="LPS")`) and
  the sibling gated data (MIITT/OCMR) is LPS-native, so the volume is reoriented to LPS on load —
  otherwise the 36 LAS patients reach the model AP-flipped.
- **Spacing:** in-plane spacing and slice pitch VARY per patient, so both are read from the NIfTI
  header (in the reoriented axis order) rather than hard-coded (observed: in-plane ~1.3-1.8 mm,
  slice pitch ~5-10 mm; `zooms[2]` is the true center-to-center pitch, matches the affine).
"""
import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation

from inference.adapters.base import BaseRTFBAdapter


class ACDCGatedAdapter(BaseRTFBAdapter):
    """ECG-gated breath-hold ACDC SAX cine: `patientXXX_4d.nii.gz` (X, Y, Z, T), reoriented to LPS."""

    def __init__(self, nii_path):
        self.nii_path = nii_path
        img = nib.load(nii_path)
        data = img.get_fdata().astype(np.float32)             # (X, Y, Z, T), file-native orientation
        # Reorient the 3 spatial axes to LPS (T is left untouched); identity no-op for LPS patients.
        xfm = ornt_transform(io_orientation(img.affine), axcodes2ornt(("L", "P", "S")))
        self._a = np.ascontiguousarray(apply_orientation(data, xfm), dtype=np.float32)
        zin = img.header.get_zooms()[:3]
        self._zooms = tuple(float(zin[int(xfm[i, 0])]) for i in range(3))   # spatial zooms in LPS order

    def load(self):
        # (X, Y, Z, T) -> (frame=T, slice=Z, H=Y, W=X)
        return np.transpose(self._a, (3, 2, 1, 0))

    def inplane_mm(self):
        return (self._zooms[0], self._zooms[1])

    def slice_positions_mm(self):
        nS = self._a.shape[2]
        pitch = self._zooms[2]                                 # header slice spacing (center-to-center)
        return np.stack([np.zeros(nS), np.zeros(nS),
                         np.arange(nS) * pitch], axis=1)
