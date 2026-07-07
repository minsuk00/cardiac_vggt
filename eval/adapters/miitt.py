"""MIITT adapter: U-Michigan paired gated+RT cine, real-time arm (see project memory).

Reads the converted real-time NIfTI `<vol>/realtime/sax/4d_recon.nii.gz`
(X=128, Y=128, Z=12-13, T=180 frames), produced by `tools/convert_miitt_to_nifti.py`.

Spacing is REAL (from J. Hamilton, 2026-07-04): real-time FOV 300x300 mm / 128 matrix ->
2.3x2.3 mm in-plane; 8 mm slice thickness + 2 mm gap -> 10 mm center-to-center.
"""
import numpy as np
import nibabel as nib

from eval.adapters.base import BaseRTFBAdapter

INPLANE_MM = (2.3, 2.3)          # real: FOV 300 mm / 128 matrix
SLICE_SPACING_MM = 10.0          # real: 8 mm thickness + 2 mm gap
SPACING_IS_PLACEHOLDER = False


class MIITTAdapter(BaseRTFBAdapter):
    SPACING_IS_PLACEHOLDER = SPACING_IS_PLACEHOLDER

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


# ── gated (ECG breath-hold) arm ─────────────────────────────────────────────
GATED_INPLANE_MM = (1.5, 1.5)    # real: FOV 336x270 mm / 224x180 matrix -> 1.5 mm isotropic


class MIITTGatedAdapter(BaseRTFBAdapter):
    """ECG-gated breath-hold cine arm: `<vol>/gated/sax/4d_recon.nii.gz`
    (X=224, Y=180, Z=12-13, T=30 cardiac phases). Same canonical pipeline as the RT arm —
    one random cardiac phase per slice (scattered breath-hold acquisition), then target_t sweep."""
    SPACING_IS_PLACEHOLDER = SPACING_IS_PLACEHOLDER

    def __init__(self, nii_path):
        self.nii_path = nii_path
        self._a = nib.load(nii_path).get_fdata().astype(np.float32)  # (X, Y, Z, T=30)

    def load(self):
        return np.transpose(self._a, (3, 2, 1, 0))   # (frame=T, slice=Z, H=Y, W=X)

    def inplane_mm(self):
        return GATED_INPLANE_MM

    def slice_positions_mm(self):
        nS = self._a.shape[2]
        return np.stack([np.zeros(nS), np.zeros(nS),
                         np.arange(nS) * SLICE_SPACING_MM], axis=1)
