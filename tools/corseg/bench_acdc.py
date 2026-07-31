"""CorSeg vs nnU-Net Task114 head-to-head on ACDC, against REAL human GT.

Why ACDC: it is the only cohort here with human segmentations, and it is zero-shot for BOTH models
(Task114 was trained on M&Ms; CorSeg's private 12-centre set excludes ACDC). So it is the honest
way to answer "is CorSeg better than what we use now".

Two input conditions, to answer the heart-ROI question:
  full : the ACDC frame as-is (what CorSeg's paper assumes -- full field of view)
  roi  : cropped to our own heart-ROI bbox (tools/nnunet_mnms_eval/build_heart_roi.build_roi with
         the project defaults in_mm=6, z_extend=1), derived from the HUMAN GT so the crop is
         identical for both models and neither is advantaged.

Label conventions (all remapped to common 1=LV cavity, 2=myocardium, 3=RV):
  ACDC GT   : 1=RV,      2=MYO, 3=LV cav
  Task114   : 1=LV cav,  2=MYO, 3=RV
  CorSeg    : 1=MYO,     2=LV cav, 3=RV

Subcommands:
  stage  : write ROI-cropped images + cropped GT (and a full-FOV GT copy) for both models to read
  score  : Dice of a prediction dir vs GT dir, per structure, split ED/ES
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "nnunet_mnms_eval"))
from build_heart_roi import build_roi  # noqa: E402

STRUCTS = ("LV", "MYO", "RV")
# per-convention: label value for (LV cavity, myocardium, RV)
CONV = {
    "acdc":   {"LV": 3, "MYO": 2, "RV": 1},
    "t114":   {"LV": 1, "MYO": 2, "RV": 3},
    "corseg": {"LV": 2, "MYO": 1, "RV": 3},
}


def dice(a, b):
    s = a.sum() + b.sum()
    return float("nan") if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


# ─────────────────────────── stage ───────────────────────────
def stage(args):
    imgs = sorted(glob.glob(os.path.join(args.img_dir, "*_0000.nii.gz")))
    if args.limit:
        imgs = imgs[: args.limit]
    for d in (args.out_img, args.out_gt):
        os.makedirs(d, exist_ok=True)
    n, skipped = 0, 0
    meta = {}
    for f in imgs:
        case = os.path.basename(f)[: -len("_0000.nii.gz")]
        gf = os.path.join(args.gt_dir, case + ".nii.gz")
        if not os.path.exists(gf):
            skipped += 1
            continue
        im, gm = nib.load(f), nib.load(gf)
        img = np.asarray(im.dataobj, dtype=np.float32)
        gt = np.asarray(gm.dataobj).astype(np.uint8)
        if img.shape != gt.shape:
            skipped += 1
            continue
        zooms = [float(z) for z in im.header.get_zooms()[:3]]
        roi = build_roi(gt > 0, zooms, in_mm=args.in_mm, z_extend=args.z_extend)
        if not roi.any():
            skipped += 1
            continue
        xs, ys, zs = np.where(roi > 0)
        sl = (slice(xs.min(), xs.max() + 1), slice(ys.min(), ys.max() + 1),
              slice(zs.min(), zs.max() + 1))
        # GOTCHA: ACDC's affine 3x3 is diag(-1,-1,1) -- the REAL spacing lives only in pixdim
        # (header.get_zooms()). Copying im.affine into a fresh Nifti1Image therefore silently
        # stamps 1.0 mm spacing, which wrecks any spacing-aware preprocessing downstream (both
        # CorSeg's 1.25 mm resample and nnU-Net's own). So build a clean diagonal affine from the
        # true zooms -- the same "clean diagonal affine from voxel spacings" the CorSeg paper uses.
        aff = np.diag(list(zooms) + [1.0])
        aff[:3, 3] = np.array([sl[0].start, sl[1].start, sl[2].start], float) * np.array(zooms)
        nib.save(nib.Nifti1Image(img[sl], aff), os.path.join(args.out_img, case + "_0000.nii.gz"))
        nib.save(nib.Nifti1Image(gt[sl], aff), os.path.join(args.out_gt, case + ".nii.gz"))
        meta[case] = {"full_shape": list(img.shape), "roi_shape": list(img[sl].shape),
                      "zooms": zooms}
        n += 1
    json.dump(meta, open(os.path.join(args.out_img, "roi_meta.json"), "w"), indent=2)
    fr = np.mean([np.prod(v["roi_shape"]) / np.prod(v["full_shape"]) for v in meta.values()])
    print(f"staged {n} ROI-cropped cases (skipped {skipped}) -> {args.out_img}")
    print(f"mean ROI/full voxel fraction = {fr:.3f}")


# ─────────────────────────── score ───────────────────────────
def score(args):
    preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    preds = [p for p in preds if not os.path.basename(p).startswith(("corseg_meta", "roi_meta"))]
    pc, gc = CONV[args.pred_conv], CONV[args.gt_conv]
    per = {ph: {s: [] for s in STRUCTS} for ph in ("ED", "ES")}
    cases, missing, mismatch = 0, 0, 0
    for pf in preds:
        case = os.path.basename(pf)[:-7]
        gf = os.path.join(args.gt_dir, case + ".nii.gz")
        if not os.path.exists(gf):
            missing += 1
            continue
        pred = np.asarray(nib.load(pf).dataobj).astype(np.int16)
        gt = np.asarray(nib.load(gf).dataobj).astype(np.int16)
        if pred.shape != gt.shape:
            print("SHAPE MISMATCH", case, pred.shape, gt.shape)
            mismatch += 1
            continue
        ph = "ED" if case.endswith("_ED") else "ES"
        for s in STRUCTS:
            d = dice(pred == pc[s], gt == gc[s])
            if not np.isnan(d):
                per[ph][s].append(d)
        cases += 1

    allc = {s: per["ED"][s] + per["ES"][s] for s in STRUCTS}
    out = {"tag": args.tag, "pred_dir": args.pred_dir, "n_cases": cases,
           "missing_gt": missing, "shape_mismatch": mismatch}
    print(f"\n===== {args.tag} =====  (n={cases} cases, missing {missing}, mismatch {mismatch})")
    print(f"{'phase':6s} {'n':>4s} | " + " ".join(f"{s:>14s}" for s in STRUCTS))
    for ph in ("ED", "ES"):
        cells, n = [], max(len(per[ph][s]) for s in STRUCTS)
        for s in STRUCTS:
            v = per[ph][s]
            cells.append(f"{np.mean(v):.3f}+-{np.std(v):.3f}" if v else "   n/a   ")
            out[f"{ph}_{s}"] = float(np.mean(v)) if v else None
        print(f"{ph:6s} {n:4d} | " + " ".join(f"{c:>14s}" for c in cells))
    cells = []
    for s in STRUCTS:
        cells.append(f"{np.mean(allc[s]):.3f}+-{np.std(allc[s]):.3f}")
        out[f"ALL_{s}"] = float(np.mean(allc[s]))
        out[f"ALL_{s}_std"] = float(np.std(allc[s]))
    print(f"{'ALL':6s} {cases:4d} | " + " ".join(f"{c:>14s}" for c in cells))
    mean3 = float(np.mean([np.mean(allc[s]) for s in STRUCTS]))
    out["mean3"] = mean3
    print(f"mean over 3 structures: {mean3:.4f}")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print("wrote", args.out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("stage")
    st.add_argument("--img_dir", required=True)
    st.add_argument("--gt_dir", required=True)
    st.add_argument("--out_img", required=True)
    st.add_argument("--out_gt", required=True)
    st.add_argument("--in_mm", type=float, default=6.0)
    st.add_argument("--z_extend", type=int, default=1)
    st.add_argument("--limit", type=int, default=0)
    st.set_defaults(func=stage)

    sc = sub.add_parser("score")
    sc.add_argument("--pred_dir", required=True)
    sc.add_argument("--gt_dir", required=True)
    sc.add_argument("--pred_conv", required=True, choices=list(CONV))
    sc.add_argument("--gt_conv", default="acdc", choices=list(CONV))
    sc.add_argument("--tag", default="")
    sc.add_argument("--out", default=None)
    sc.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
