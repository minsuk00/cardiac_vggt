"""Visual check of the EF/Dice segmenter (nnU-Net Task114 3d_fullres) on every method: for one subject,
all SAX slices at GT end-diastole and end-systole, one row per method, the scored cine (x the heart
ROI, exactly what nnU-Net was given) with its kept segmentation overlaid (LV red, MYO green, RV blue).

Reads only kept outputs: <subject>/cine_gt.nii.gz + seg_gt_3d_fullres/ (GT), <subject>/<method>/
cine_breath.nii.gz + scratch/eval/_rhythm24_segs/<arm>/<method>__<cohort>/ (pred, via ef_manifest.json).
ED/ES = argmax/argmin of the GT LV volume; the same two frames are shown for every method.

    PYTHONPATH=training:. python tools/render_seg_check.py hrv12 ACDC_patient102 [more subjects] \
        [--methods ...] [--out figs/seg_check]
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

METHODS = {"hrv12": ["vggt_final518_diff1000_ep300", "cinevol_masked", "fetal_cmr_4d", "svrtk3d_debug_scatter",
                     "niftymic_scatter", "dangi_scatter"],
           "af12": ["vggt_final518_diff1000_ep300", "cinevol_motion_masked", "fetal_cmr_4d", "svrtk3d_debug_scatter",
                    "niftymic_scatter", "dangi_scatter_4gpu"]}
COLORS = {1: (1, 0.2, 0.2), 2: (0.2, 1, 0.2), 3: (0.3, 0.5, 1)}


def seg_files(arm, method, cohort, subj):
    d = os.path.join(ROOT, "scratch", "eval", "_rhythm24_segs", arm, f"{method}__{cohort}")
    if not os.path.isdir(d) and method == "cinevol_masked" and arm == "af12":
        d = os.path.join(ROOT, "scratch", "cinevol", "segs", arm, f"{method}__{cohort}")
    rows = json.load(open(os.path.join(d, "ef_manifest.json")))["subjects"]
    sidx = [r["sidx"] for r in rows if r["subject"] == subj][0]
    return sorted(glob.glob(os.path.join(d, f"{cohort}__s{sidx:03d}__breath__t*.nii.gz")))


def ef_of(cohort, method, subj):
    p = paths.RESULTS / "test" / cohort / "ef" / f"{method}.json"
    for r in json.load(open(p))["per_subject"]:
        if r["subject"] == subj:
            return r.get("ef_breath"), r.get("ef_gt")
    return None, None


def overlay(ax, img, seg):
    ax.imshow(img.T, cmap="gray", origin="lower", vmin=0, vmax=np.percentile(img, 99.5) + 1e-6)
    rgba = np.zeros(seg.T.shape + (4,))
    for lab, c in COLORS.items():
        rgba[seg.T == lab] = (*c, 0.35)
    ax.imshow(rgba, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("subjects", nargs="+")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "figs", "seg_check"))
    a = ap.parse_args()
    methods = a.methods or METHODS[a.arm]
    os.makedirs(a.out, exist_ok=True)
    for subj in a.subjects:
        cohort = [c for c in glob.glob(os.path.join(ROOT, "scratch", "eval", f"*_{a.arm}", "out", subj))][0].split("/")[-3]
        roi = np.asarray(nib.load(str(paths.heart_mask_pad(cohort, subj))).dataobj) > 0
        gt = np.asarray(nib.load(str(paths.cine_gt(cohort, subj))).dataobj, np.float32)
        T = gt.shape[3]
        gseg = [np.asarray(nib.load(f).dataobj) for f in
                sorted(glob.glob(str(paths.seg_gt_dir(cohort, subj, "3d_fullres") / "seg_t*.nii.gz")))][:T]
        lv = np.array([(s == 1).sum() for s in gseg])
        ed, es = int(lv.argmax()), int(lv.argmin())
        xs, ys = np.where(roi.any(2))
        bx = slice(max(xs.min() - 4, 0), xs.max() + 5); by = slice(max(ys.min() - 4, 0), ys.max() + 5)
        zs = [z for z in range(roi.shape[2]) if roi[..., z].any()]
        rows = [("GT", gt, gseg, None)]
        for m in methods:
            cine = np.asarray(nib.load(str(paths.cine(cohort, subj, m, "breath"))).dataobj, np.float32)
            segs = [np.asarray(nib.load(f).dataobj) for f in seg_files(a.arm, m, cohort, subj)]
            rows.append((m, cine, segs, ef_of(cohort, m, subj)))
        egt = rows[1][3][1] if len(rows) > 1 else None
        for tag, t in (("ED", ed), ("ES", es)):
            fig, axs = plt.subplots(len(rows), len(zs), figsize=(1.6 * len(zs), 1.7 * len(rows)), squeeze=False)
            for i, (name, vol, segs, ef) in enumerate(rows):
                for j, z in enumerate(zs):
                    overlay(axs[i, j], (vol[..., z, t] * roi[..., z])[bx, by], segs[t][..., z][bx, by])
                    if i == 0:
                        axs[i, j].set_title(f"z{z}", fontsize=8)
                lab = name.replace("vggt_final518_", "VGGT ").replace("_ep300", "")
                if ef is not None and ef[0] is not None:
                    lab += f"\nEF {ef[0]:.0f}%"
                elif ef is not None:
                    lab += "\nEF undefined"
                axs[i, 0].set_ylabel(lab, fontsize=8)
            fig.suptitle(f"{a.arm} {subj}  {tag} (GT frame t{t:02d}; GT EF {egt:.0f}%)  "
                         f"LV red / MYO green / RV blue", fontsize=10)
            fig.tight_layout()
            out = os.path.join(a.out, f"{a.arm}_{subj}_{tag}.png")
            fig.savefig(out, dpi=90); plt.close(fig)
            print("->", out)


if __name__ == "__main__":
    main()
