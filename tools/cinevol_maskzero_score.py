"""CiNeVol through the SAME pipeline as every mask-limited baseline: zero outside the heart mask,
then the standing scorer (PSF blur, 6-DOF NCC registration, score inside the padded mask).

CiNeVol is a continuous representation; cinevol/reconstruct.py evaluates it on the mask's whole
bounding box, so the saved `cinevol` arm carries unconstrained values OUTSIDE the mask where
NeSVoR / SVRTK / Fetal CMR 4D carry zeros. This writes a copy arm `cinevol_maskzero` =
vol_t* x mask_heart_pad10 (the fitting mask), then scores it with evaluation/src/score/run.py's
own functions. The only harness-side adaptation is registering the new arm name for the PSF
operator (pose_psf.PSF_METHODS is keyed on arm name and `_maskzero` is not one of its known
suffixes) -- done here in-process, evaluation/ is untouched.

EF / Dice are NOT rerun: ef_dice.py multiplies every volume by the same mask before segmentation,
so the two arms are identical there.

Never overwrites: existing cinevol_maskzero dirs / metrics.json are kept and skipped; the cohort
summary is written under a NEW name if one exists.

    PYTHONPATH=training:. python tools/cinevol_maskzero_score.py --arm af24 [--sources ...]
"""
import argparse
import json
import os
import sys
import time

import nibabel as nib
import nibabel.processing as nibproc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "evaluation", "src", "score")]
import paths                                                       # noqa: E402
import pose_psf                                                    # noqa: E402
import image_metrics                                               # noqa: E402
import aggregate                                                   # noqa: E402

# `cinevol_maskzero` (23 subjects, 2026-09-21) was the first attempt with the half-slice bug above and
# is left in place, unused; the corrected arm has its own name so the two can never be mixed.
SRC_ARM, ARM = "cinevol", "cinevol_masked"
pose_psf.PSF_METHODS = tuple(pose_psf.PSF_METHODS) + (ARM,)      # same PSF treatment as cinevol
SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")


def make_zeroed(ds, s):
    src = paths.recon_dir(ds, s, SRC_ARM, "breath")
    dst = paths.recon_dir(ds, s, ARM, "breath")
    if dst.is_dir():
        return "kept"
    tmp = f"{dst}.partial{os.getpid()}"
    os.makedirs(tmp)
    mask_img = nib.load(str(paths.heart_mask_pad(ds, s)))
    T = json.load(open(paths.manifest(ds, s)))["T"]
    m = None
    for t in range(T):
        v = nib.load(os.path.join(src, f"vol_t{t:02d}.nii.gz"))
        if m is None:                                             # mask resampled once onto the recon grid
            # NOT nibabel resample_from_to(order=0, cval=0): its constant boundary treats every
            # recon voxel beyond the LAST slice centre (stack index > D-1, i.e. the outer half of the
            # first/last 12 mm slice) as outside -> the end slices were cut in half (measured:
            # 0 % mask coverage in those half-slices; docs/116 s4c). A stack voxel is a 12 mm slab,
            # so membership is nearest-slice with the index CLAMPED, i.e. mode="nearest".
            from scipy.ndimage import map_coordinates
            ijk = np.stack(np.meshgrid(*[np.arange(n) for n in v.shape], indexing="ij"), -1).reshape(-1, 3)
            sidx = nib.affines.apply_affine(np.linalg.inv(mask_img.affine) @ v.affine, ijk)
            m = map_coordinates(np.asarray(mask_img.dataobj, np.float32), sidx.T, order=0, mode="nearest").reshape(v.shape) > 0.5
        nib.save(nib.Nifti1Image(np.asarray(v.dataobj, np.float32) * m, v.affine), os.path.join(tmp, f"vol_t{t:02d}.nii.gz"))
    st = json.load(open(os.path.join(src, "stamp.json")))
    st.update(derived_from=SRC_ARM, zeroed_outside=paths.HEART_MASK_PAD)
    json.dump(st, open(os.path.join(tmp, "stamp.json"), "w"))
    for f in ("provenance.txt", "total_wall.sec"):               # timing/provenance are the fit's, unchanged
        if os.path.exists(os.path.join(src, f)):
            open(os.path.join(tmp, f), "w").write(open(os.path.join(src, f)).read())
    os.rename(tmp, dst)
    return "made"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="af24")
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    a = ap.parse_args()
    t0 = time.time()
    for src in a.sources:
        ds = f"{src}_{a.arm}"
        keep, _ = paths.filter_by_split(ds, paths.subjects(ds), "test")
        made = scored = 0
        for s in keep:
            made += make_zeroed(ds, s) == "made"
            if not paths.metrics(ds, s, ARM).is_file():
                image_metrics.score_subject(ds, s, ARM)
                scored += 1
        if paths.summary(ds, ARM, "test").is_file():
            print(f"### {ds}: summary exists, not re-aggregating (never overwrite)", flush=True)
        else:
            aggregate.aggregate(ds, ARM, "test")                  # writes metric_results/test/<ds>/<ARM>.json
        print(f"### {ds}: {len(keep)} subjects, {made} zeroed copies made, {scored} scored  ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
