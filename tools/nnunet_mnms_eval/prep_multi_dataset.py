"""Prep nnU-Net v1 Task114 (M&Ms) inputs from MULTIPLE datasets for a trustworthiness
check of the whole-heart-ROI plan (CMRxRecon canonical + MIITT gated/RT + OCMR gated/RTFB).

Each case = one 3D SAX volume (a single cardiac frame) written as (X,Y,Z) NIfTI named
<case>_0000.nii.gz with a real spacing/affine so nnU-Net resamples correctly. nnU-Net does its
own per-image z-score, so raw intensities pass through.

Goettingen is skipped: its recon is a whole-heart RAS volume (160,160,24), NOT SAX — Task114
(SAX-trained) needs a reslice first.
"""
import os, glob, sys
import numpy as np
import nibabel as nib

OUT = "/home/minsukc/vggt/scratch/data/nnunet_mnms/multi_ds_inputs"
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, "/home/minsukc/vggt/training")
sys.path.insert(0, "/home/minsukc/vggt")


def save3d(arr_xyz, affine, case):
    nib.save(nib.Nifti1Image(np.asarray(arr_xyz).astype(np.float32), affine),
             os.path.join(OUT, f"{case}_0000.nii.gz"))
    print(f"  wrote {case}  shape={arr_xyz.shape}")


# ---------------- CMRxRecon canonical (more val subjects) ----------------
def prep_cmrx(n_extra=8, start=6):
    from data.preprocess import build_data_dicts, get_canonical_transforms
    # val subjects, in split order; the existing seg covered indices 0..5
    val = []
    with open("/home/minsukc/vggt/training/splits/random_8_1_1.txt") as f:
        f_on = False
        for line in f:
            s = line.strip()
            if s == "[val]": f_on = True; continue
            if s == "[test]": break
            if f_on and s and not s.startswith("#"): val.append(s)
    subs = val[start:start + n_extra]
    root = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
    sax_dirs = [os.path.join(root, s, "sax") for s in subs]
    dicts = build_data_dicts(sax_dirs)
    tfm = get_canonical_transforms()
    aff = np.diag([1.4, 1.4, 12.0, 1.0])
    print(f"[CMRxRecon] {len(subs)} extra subjects (val idx {start}..{start+n_extra-1})")
    for i, d in enumerate(dicts):
        try:
            out = tfm(d)
            phases = np.asarray(out["phases"])          # (T,X,Y,Z)
            v0 = phases[0].astype(np.float32)            # t=0 canonical (X,Y,Z)
            save3d(v0, aff, f"cmrx_subj{start+i:02d}_{d['subj_id']}")
        except Exception as e:
            print(f"  CMRx {d['subj_id']} FAILED: {e}")


# ---------------- MIITT gated + realtime ----------------
def prep_miitt(frame=0):
    root = "/home/minsukc/vggt/scratch/data/MIITT/nifti"
    subs = sorted(os.listdir(root))
    print(f"[MIITT] {len(subs)} subjects x {{gated,rt}}")
    for s in subs:
        for regime, sub in [("gated", "gated"), ("rt", "realtime")]:
            f = os.path.join(root, s, sub, "sax", "4d_recon.nii.gz")
            if not os.path.exists(f):
                print(f"  MIITT {s}/{regime} MISSING"); continue
            im = nib.load(f)
            arr = np.asarray(im.dataobj)                 # (X,Y,Z,T)
            fr = min(frame, arr.shape[-1] - 1)
            save3d(arr[..., fr], im.affine, f"miitt_{s}_{regime}")


# ---------------- OCMR gated + rtfb SAX ----------------
def prep_ocmr(frame=0, n_gated=8, n_rtfb=10):
    base = "/home/minsukc/vggt/scratch/data/ocmr/recon"
    for regime, cap in [("gated", n_gated), ("rtfb", n_rtfb)]:
        files = sorted(glob.glob(os.path.join(base, regime, "exam_*", "sax__*", "sax_cine.nii.gz")))[:cap]
        print(f"[OCMR/{regime}] {len(files)} exams")
        for f in files:
            exam = f.split("/recon/")[1].split("/")[1]   # exam_XXXX
            im = nib.load(f)
            arr = np.asarray(im.dataobj)                  # (X,Y,Z,T)
            fr = min(frame, arr.shape[-1] - 1)
            save3d(arr[..., fr], im.affine, f"ocmr_{exam}_{regime}")


if __name__ == "__main__":
    prep_cmrx()
    prep_miitt()
    prep_ocmr()
    print("\nDONE. inputs ->", OUT)
    print("cases:", len(glob.glob(os.path.join(OUT, "*_0000.nii.gz"))))
