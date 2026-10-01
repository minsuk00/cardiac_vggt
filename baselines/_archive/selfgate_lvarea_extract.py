"""Self-gating step 1/3 — extract every (slice, frame) of a MIITT RT stack as an nnU-Net
Task114 input, on the canonical 256x256 / 1.4 mm grid (in-distribution for the M&Ms net).

A later nnU-Net pass segments them; the per-slice LV blood-pool AREA curve over frames gives the
per-slice ED anchor (area local-max) + ES (area min) — the LV-seg self-gating of doc 35, doing the
job the multi-orientation inter-slice sync can't do single-orientation (doc 34).

Unlike tools/detect_ood_ed_extract.py (mid slice only), this does ALL Z slices — the gater needs a
per-slice anchor.

Run: micromamba run -n svr python baselines/fetal_cmr_4d/selfgate_lvarea_extract.py Volunteer1
"""
import os, sys, json
import numpy as np
import nibabel as nib

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from inference.canonical_inplane import percentile_scale, to_canonical_inplane
# ⚠️ BROKEN as of 2026-08-16: the RTFB adapter stack was retired to
# `inference/_archive/adapters/` when every source moved onto MRIDataset. This script is the only
# remaining caller of MIITTAdapter and it was already stale — `INPLANE_MM` there is the PLACEHOLDER
# spacing, superseded by the real 1.5 x 1.5 x 10 mm protocol values (docs/78). The self-gating SVR
# baseline this feeds (docs/35) was never built, so nothing is regressing today. To revive it, read
# the raw cine through `scratch/data/MIITT_sax/` + MRIDataset instead of resurrecting the adapter.
from inference.adapters.miitt import MIITTAdapter, INPLANE_MM

MIITT_RECON = os.path.join(_ROOT, "scratch/data/MIITT/nifti")
OUTROOT = os.path.join(_ROOT, "scratch/fetal_cmr_4d/recon")
SPACING = (1.4, 1.4, 12.0)


def main():
    vol = sys.argv[1] if len(sys.argv) > 1 else "Volunteer1"
    nii = os.path.join(MIITT_RECON, vol, "realtime", "sax", "4d_recon.nii.gz")
    ad = MIITTAdapter(nii)
    cine = ad.load()                          # (T=180, Z=13, H=128, W=128)
    T, Z = cine.shape[0], cine.shape[1]
    vmin, vmax = percentile_scale(cine)       # whole-cine percentile, matches detect_ood tool

    out = os.path.join(OUTROOT, vol, "selfgate", "lvseg_inputs")
    os.makedirs(out, exist_ok=True)
    for z in range(Z):
        for f in range(T):
            norm = np.clip((cine[f, z] - vmin) / (vmax - vmin), 0.0, 1.0)
            canon = to_canonical_inplane(norm, INPLANE_MM).numpy()   # (256,256) [0,1]
            arr = (canon.T)[..., None].astype(np.float32)            # (X,Y,1) nnU-Net order
            nib.save(nib.Nifti1Image(arr, np.diag([*SPACING, 1.0])),
                     os.path.join(out, f"{vol}__z{z:02d}__f{f:03d}_0000.nii.gz"))
        print(f"  z{z:02d}: {T} frames", flush=True)
    json.dump({"vol": vol, "Z": int(Z), "T": int(T),
               "vmin": float(vmin), "vmax": float(vmax), "inplane_mm": list(INPLANE_MM)},
              open(os.path.join(OUTROOT, vol, "selfgate", "extract_meta.json"), "w"), indent=2)
    print(f"done -> {out}  ({Z*T} images)")


if __name__ == "__main__":
    main()
