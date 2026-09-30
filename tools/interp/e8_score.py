"""E8 scoring: per arm, LV volume over the 180 real frames with the reference plane EXCLUDED
(2D nnU-Net Task114 segs from tools/rt_lv_curve_2d.py dump + nnUNet_predict -m 2d), vs the
reference plane's own LV area (the real, observed cardiac signal).
  r      = corr(LV_noref(arm), LV_area_ref(full))   (does the rest of the heart beat with the reference?)
  swing  = (p95 - p5) of LV_noref, relative to the full model's
Usage: python tools/interp/e8_score.py <e8_dir> <seg_dir>
"""
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

ARMS = ["full", "ko_ref_L3", "self_L12_17", "naive"]


def lv_per_plane(seg_dir, tag):
    files = sorted(glob.glob(os.path.join(seg_dir, f"{tag}_f*.nii.gz")))
    if not files:
        return None, None
    vox = float(np.prod(nib.load(files[0]).header.get_zooms()[:3])) / 1000.0
    return np.stack([(np.asarray(nib.load(f).dataobj) == 1).sum(axis=(0, 1)) for f in files]), vox   # (F, Z)


def main(e8, seg):
    res = {}
    for subj in sorted(os.listdir(e8)):
        if not os.path.exists(os.path.join(e8, subj, "full", "meta.txt")):
            continue                                   # e.g. the lvseg2d/ work dir
        meta = dict(l.strip().split("=") for l in open(os.path.join(e8, subj, "full", "meta.txt")))
        ref_z = int(meta["ref_z"])
        short = "afib" if "AFib" in subj else "vol1"
        full, vox = lv_per_plane(seg, f"{short}_full")
        ref_area = full[:, ref_z].astype(float)
        out = {}
        for arm in ARMS:
            a, _ = lv_per_plane(seg, f"{short}_{arm}")
            if a is None:
                continue
            noref = np.delete(a, ref_z, axis=1).sum(1) * vox
            r = float(np.corrcoef(noref, ref_area)[0, 1]) if noref.std() > 0 else float("nan")
            out[arm] = {"r_vs_ref": r, "swing_ml": float(np.percentile(noref, 95) - np.percentile(noref, 5)),
                        "curve_ml": noref.tolist()}
        base = out["full"]["swing_ml"]
        for arm in out:
            out[arm]["swing_rel"] = out[arm]["swing_ml"] / base
            print(f"{subj[:40]:<40} {arm:<12} r={out[arm]['r_vs_ref']:+.2f}  swing={out[arm]['swing_ml']:6.1f} mL "
                  f"({100 * out[arm]['swing_rel']:.0f}% of full)")
        out["ref_area_px"] = ref_area.tolist()
        res[subj] = out
    json.dump(res, open(os.path.join(e8, "e8_scores.json"), "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
