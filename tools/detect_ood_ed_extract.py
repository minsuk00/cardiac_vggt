"""Step 1 of OOD ED detection: extract each RT-OOD subject's MID-VENTRICULAR slice across ALL
frames as nnU-Net Task114 inputs. A later nnU-Net pass segments them; max LV blood-pool area
over frames = ED (written to ed_frames.json by detect_ood_ed_pick.py).

The mid-ventricular plane is the median in-FOV canonical z (same pick as the render's reference
slot). Each frame's slice is put on the canonical 256x256 / 1.4mm grid (same as the model input)
so nnU-Net Task114 (M&Ms SAX) sees an in-distribution image.

Run: micromamba run -n svr python tools/detect_ood_ed_extract.py
"""
import os, sys, json
import numpy as np
import nibabel as nib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from eval.adapters.base import percentile_scale, assign_canonical_z, to_canonical_inplane
from eval.adapters.ocmr import OCMRAdapter
from eval.adapters.goettingen import GoettingenAdapter
from eval.adapters.miitt import MIITTAdapter
from tools.render_reference_ed_targeted import (
    OCMR_SUBJECTS, GOTT_SUBJECTS, MIITT_SUBJECTS, OCMR_RECON, GOTT_RECON, MIITT_RECON,
    mid_ventricular_entry,
)

OUT = os.path.join(_ROOT, "result", "reference_models_io", "ed_detect")
SPACING = (1.4, 1.4, 12.0)


def adapters():
    for s in OCMR_SUBJECTS:
        yield "OCMR", s, OCMRAdapter(os.path.join(OCMR_RECON, s))
    for s in GOTT_SUBJECTS:
        yield "Goett", s, GoettingenAdapter(os.path.join(GOTT_RECON, s, s + ".nii.gz"))
    for s in MIITT_SUBJECTS:
        nii = os.path.join(MIITT_RECON, s, "realtime", "sax", "4d_recon.nii.gz")
        if os.path.exists(nii):
            yield "MIITT", s, MIITTAdapter(nii)


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}  # "{ds}__{subj}" -> {"slice_idx":..,"n_frames":..}
    for ds, subj, ad in adapters():
        try:
            cine = ad.load()                                  # (T, Z, H, W)
            vmin, vmax = percentile_scale(cine)
            z_map = assign_canonical_z(ad.slice_positions_mm())
            (z_canon, slice_idx), _ = mid_ventricular_entry(z_map)
            T = cine.shape[0]
            for f in range(T):
                norm = np.clip((cine[f, slice_idx] - vmin) / (vmax - vmin), 0.0, 1.0)
                canon = to_canonical_inplane(norm, ad.inplane_mm()).numpy()  # (256,256) [0,1]
                arr = (canon.T)[..., None].astype(np.float32)               # (X,Y,1) nnU-Net order
                tag = f"{ds}__{subj}__f{f:03d}"
                nib.save(nib.Nifti1Image(arr, np.diag([*SPACING, 1.0])),
                         os.path.join(OUT, f"{tag}_0000.nii.gz"))
            meta[f"{ds}__{subj}"] = {"slice_idx": int(slice_idx), "z_canon": int(z_canon), "n_frames": int(T)}
            print(f"  {ds}/{subj}: {T} frames @ mid slice {slice_idx} (z={z_canon})", flush=True)
        except Exception as e:
            print(f"  skip {ds}/{subj}: {e}", flush=True)
    json.dump(meta, open(os.path.join(OUT, "extract_meta.json"), "w"), indent=2)
    print(f"done -> {OUT}  ({len(meta)} subjects)")


if __name__ == "__main__":
    main()
