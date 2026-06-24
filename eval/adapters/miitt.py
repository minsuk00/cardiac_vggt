"""MIITT adapter: U-Michigan paired gated+RT cine, real-time arm (see project memory).

Reads the converted real-time NIfTI `<vol>/realtime/sax/4d_recon.nii.gz`
(X=128, Y=128, Z=13, T=180 frames), produced by `tools/convert_miitt_to_nifti.py`.

WARNING — PLACEHOLDER spacing. The MIITT .mat files carry no spatial metadata; the
in-plane (2.1 mm) and slice (8.0 mm) spacings are ESTIMATES. They are fine for qualitative
beating-heart inference but NOT for physical distances / EF. `run_rtfb.py` prints a warning
when `SPACING_IS_PLACEHOLDER`.

2.1 mm in-plane is LV-size-calibrated (docs/23 follow-up): at 2.6 mm the MIITT LV cavity came
out ~20% larger than the in-dist GT reference (59 vs 49 mm ⌀); 2.6 × 49/59 ≈ 2.1 mm brings it
in line and reduces the low-res-splat gridding. Replace with the true FOV/matrix when available.
"""
import numpy as np
import nibabel as nib

from eval.adapters.base import BaseRTFBAdapter

INPLANE_MM = (2.1, 2.1)          # PLACEHOLDER — LV-size-calibrated (docs/23); was 2.6
SLICE_SPACING_MM = 8.0           # PLACEHOLDER
SPACING_IS_PLACEHOLDER = True


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
