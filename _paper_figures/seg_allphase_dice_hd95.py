"""LV/MYO/RV Dice + HD95 at EVERY cardiac phase (pred frame t vs GT frame t), from the kept nnU-Net
segmentations of the rhythm arms (scratch/eval/_rhythm24_segs/<arm>/<method>__<src>_<arm>/).

Same per-frame definitions as evaluation/src/score/ef_dice.py dice()/hd95() (label 1/2/3, NaN on empty
masks, symmetric 95th-percentile surface distance in mm). HD95 is computed on the bbox of a|b (+2 voxels),
which gives identical distances to the full-volume EDT (max |diff| 0.0, checked) at ~8x less CPU.
ED/ES = argmax/argmin of the GT LV volume, as in ef_dice.py, so the ED/ES entries reproduce the stored
dice_breath_*_ED/ES and hd95_breath_*_ED/ES exactly (checked: max |diff| 0 on af12 + hrv12, 9 methods).

    PYTHONPATH=training:. python _paper_figures/seg_allphase_dice_hd95.py --procs 8 --out temp/allphase_seg/seg_allphase.json
"""
import argparse
import json
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG = f"{ROOT}/scratch/eval/_rhythm24_segs"
SRCS = ["acdc", "cmrx2023", "cmrx2024", "cmrx2025", "mnms"]
VGGT = ["vggt_final518_diff1000_ep300", "vggt_final518_base_ep300", "vggt_final518_nogather_ep300",
        "vggt_final518_hw0_ep300"]
METHODS = {"af12": VGGT + ["cinevol_motion_masked", "fetal_cmr_4d", "svrtk3d_debug_scatter", "niftymic_scatter",
                           "dangi_scatter_4gpu", "nesvor_4gpu_scatter"],
           "hrv12": VGGT + ["cinevol_masked", "fetal_cmr_4d", "svrtk3d_debug_scatter", "niftymic_scatter",
                            "dangi_scatter", "nesvor_scatter"]}
LABS = {"LV": 1, "MYO": 2, "RV": 3}
T = 12


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def hd95(a, b, spacing):
    if not a.any() or not b.any():
        return float("nan")
    idx = np.argwhere(a | b)
    lo, hi = np.maximum(idx.min(0) - 2, 0), idx.max(0) + 3
    sl = tuple(slice(l, h) for l, h in zip(lo, hi))
    a, b = a[sl], b[sl]
    sa = a & ~binary_erosion(a)
    sb = b & ~binary_erosion(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def job(args):
    arm, src, subj, sidx, methods = args
    coh = f"{src}_{arm}"
    gdir = f"{ROOT}/scratch/eval/{coh}/out/{subj}/seg_gt_3d_fullres"
    gt = [np.asarray(nib.load(f"{gdir}/seg_t{t:02d}.nii.gz").dataobj) for t in range(T)]
    vol = np.array([(g == 1).sum() for g in gt])
    out = {"arm": arm, "subject": subj, "src": src, "ed": int(vol.argmax()), "es": int(vol.argmin()), "m": {}}
    for m in methods:
        d = f"{SEG}/{arm}/{m}__{coh}"
        ps = [f"{d}/{coh}__s{sidx:03d}__breath__t{t:02d}.nii.gz" for t in range(T)]
        if not all(os.path.isfile(p) for p in ps):
            continue
        res = {f"{k}_{n}": [] for k in ("dice", "hd95") for n in LABS}
        for t in range(T):
            img = nib.load(ps[t])
            sp = img.header.get_zooms()[:3]
            p = np.asarray(img.dataobj)
            for n, lab in LABS.items():
                a, b = p == lab, gt[t] == lab
                res[f"dice_{n}"].append(dice(a, b))
                res[f"hd95_{n}"].append(hd95(a, b, sp))
        out["m"][m] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=list(METHODS))
    a = ap.parse_args()
    jobs = []
    for arm in a.arms:
        for src in SRCS:
            man = json.load(open(f"{SEG}/{arm}/{METHODS[arm][0]}__{src}_{arm}/ef_manifest.json"))["subjects"]
            jobs += [(arm, src, r["subject"], r["sidx"], METHODS[arm]) for r in man]
    print(len(jobs), "subject-arm jobs", flush=True)
    with Pool(a.procs) as pool:
        res = list(pool.imap_unordered(job, jobs))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
