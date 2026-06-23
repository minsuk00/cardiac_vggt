"""OCMR adapter: reconstructed real-time free-breathing SAX cine (docs/06).

Reads `<subj_dir>/sax_cine.nii.gz` (4D: frame, slice, H, W) + `meta.json`. Geometry
(`inplane_mm`, `slice_positions_mm`) comes from the meta — same sources as the original
`tools/eval_ocmr_inference.py`, so the produced batch is bit-identical.
"""
import json
import os

import numpy as np
import SimpleITK as sitk

from eval.adapters.base import BaseRTFBAdapter


class OCMRAdapter(BaseRTFBAdapter):
    def __init__(self, subj_dir):
        self.subj_dir = subj_dir
        self._img = sitk.ReadImage(os.path.join(subj_dir, "sax_cine.nii.gz"))
        self._meta = json.load(open(os.path.join(subj_dir, "meta.json")))

    def load(self):
        return sitk.GetArrayFromImage(self._img).astype(np.float32)  # (frame, slice, H, W)

    def inplane_mm(self):
        return self._meta["inplane_mm"]

    def slice_positions_mm(self):
        return self._meta["slice_positions_mm"]
