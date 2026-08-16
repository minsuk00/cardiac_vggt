"""Measure LV volume vs cardiac phase (contraction curve + EF) on a fetal_cmr_4d 4D cine recon,
via nnU-Net Task114. Extracts every (slice, phase) of the recon onto the canonical 256x256/1.4mm
grid, ready for `nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS`.

Two-step (like the gater): this writes inputs; a bash nnU-Net pass segments; recon_ef_pick.py
aggregates LV volume per phase -> EF.

Run: micromamba run -n svr python baselines/fetal_cmr_4d/recon_ef.py <recon_cine.nii.gz> <outdir>
"""
import os, sys, json
import numpy as np
import nibabel as nib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from inference.canonical_inplane import percentile_scale, to_canonical_inplane


def main():
    recon_path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    im = nib.load(recon_path)
    rc = im.get_fdata().astype(np.float32)          # (X, Y, Z, P)
    inpl = tuple(float(z) for z in im.header.get_zooms()[:2])
    X, Y, Z, P = rc.shape
    vmin, vmax = percentile_scale(np.clip(rc, 0, None).transpose(3, 2, 1, 0))  # (P,Z,Y,X)
    for p in range(P):
        for z in range(Z):
            norm = np.clip((np.clip(rc[:, :, z, p], 0, None) - vmin) / (vmax - vmin), 0.0, 1.0)
            canon = to_canonical_inplane(norm, inpl).numpy()      # (256,256)
            arr = (canon.T)[..., None].astype(np.float32)
            nib.save(nib.Nifti1Image(arr, np.diag([1.4, 1.4, 12.0, 1.0])),
                     os.path.join(out, f"rec__p{p:02d}__z{z:02d}_0000.nii.gz"))
    json.dump({"recon": recon_path, "P": P, "Z": Z, "inplane_mm": list(inpl),
               "canon_voxel_mm3": 1.4 * 1.4 * float(im.header.get_zooms()[2])},
              open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(f"wrote {P*Z} slices -> {out}  (P={P} phases, Z={Z} slices)")


if __name__ == "__main__":
    main()
