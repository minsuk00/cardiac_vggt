"""Dangi baseline centre-error diagnostic on the eval bundles (docs/100 §7).

For every finished subject under <out_root>/<ds>/out/<subj>/dangi_scatter/recon_breath/centres.json:
  c_gt[z]      LV-cavity centroid of seg_gt/seg_t{k} (label 1) on the native grid, mm
  c_in[z]      true centre of the scatter input slice = c_gt - breath disp (h,w)  (simulator: d>0 samples
               at +d, so content appears at -d)
  c_pred[z]    Dangi's prediction mapped Dangi-grid -> native
Reports (mm, over planes with a measurable LV):
  regressor     |c_pred - c_in|            how well the CNN finds the centre on our scatter slices
  input         |c_in - c_gt| = |disp_hw|   in-plane misalignment before any correction
  after/<anchor> |c_pred + shift - c_gt|   residual in-plane misalignment after Dangi, per anchor
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANGI_MM, NATIVE_MM, SIZE_D, SIZE_N = 1.5625, 1.4, 192, 256


def dangi_to_native_px(c):
    new_shape = int(np.rint(SIZE_N * NATIVE_MM / DANGI_MM))
    start = -((SIZE_D - new_shape) // 2)
    return (np.asarray(c) + start) * (DANGI_MM / NATIVE_MM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=os.path.join(ROOT, "temp/dangi/eval"))
    ap.add_argument("--arm", default="dangi_scatter")
    ap.add_argument("--phases", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    rows = {k: [] for k in ("regressor", "input", "after/reference", "after/mean", "after/image_center", "after/oracle")}
    n_subj = 0
    for cj in sorted(glob.glob(f"{args.out_root}/*/out/*/{args.arm}/recon_breath/centres.json")):
        ds, subj = cj.split("/")[-6], cj.split("/")[-4]
        sd = f"{ROOT}/scratch/eval/{ds}/out/{subj}"
        man = json.load(open(f"{sd}/manifest.json"))
        ref = man["scatter"]["ref_plane"]
        disp = np.asarray(man["breath"]["disp_dhw_mm"])  # (D,3) = (d,h,w) mm
        cent = json.load(open(cj))
        n_subj += 1
        for k in args.phases:
            seg = np.asarray(nib.load(f"{sd}/seg_gt/seg_t{k:02d}.nii.gz").dataobj)
            pred = dangi_to_native_px(cent[f"t{k:02d}"]["centres_px"]) * NATIVE_MM  # (D,2) x,y mm
            c_gt, ok = np.full_like(pred, np.nan), np.zeros(len(pred), bool)
            for z in range(len(pred)):
                xs, ys = np.nonzero(seg[:, :, z] == 1)
                if len(xs) >= 20:
                    c_gt[z] = [xs.mean() * NATIVE_MM, ys.mean() * NATIVE_MM]
                    ok[z] = True
            if not ok[ref] or ok.sum() < 3:
                continue
            c_in = c_gt - disp[:, [2, 1]]  # (x,y) <- (w,h)
            anchors = {"reference": pred[ref], "mean": pred[ok].mean(0),
                       "image_center": dangi_to_native_px([96, 96]) * NATIVE_MM, "oracle": c_gt[ref]}
            err = lambda a, b: np.linalg.norm((a - b)[ok], axis=1)
            rows["regressor"] += err(pred, c_in).tolist()
            rows["input"] += err(c_in, c_gt).tolist()
            for name, a in anchors.items():
                # output content centre = true input centre + applied shift (anchor - prediction)
                rows[f"after/{name}"] += err(c_in + (a - pred), c_gt).tolist()
    print(f"{n_subj} subjects, phases {args.phases}, {len(rows['input'])} planes")
    for k, v in rows.items():
        v = np.asarray(v)
        print(f"{k:20s} mean {v.mean():5.2f}  median {np.median(v):5.2f}  p90 {np.percentile(v, 90):5.2f} mm")


if __name__ == "__main__":
    main()
