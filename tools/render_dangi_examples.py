"""Render Dangi baseline examples: all z-planes (input / Dangi output / GT) with predicted (+) and GT (o)
LV centres, plus XZ and YZ long-axis cuts through the reference plane's GT centre (before / after / GT).
Output: temp/dangi/figs/<ds>_<subj>_t<k>.png + an index.html. Reads only temp/dangi/eval and the bundles.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DANGI_MM, NATIVE_MM = 1.5625, 1.4


def dangi_to_native_px(c):
    start = -((192 - int(np.rint(256 * NATIVE_MM / DANGI_MM))) // 2)
    return (np.asarray(c) + start) * (DANGI_MM / NATIVE_MM)


def render(ds, subj, k, arm, out_root, fig_dir):
    sd = f"{ROOT}/scratch/eval/{ds}/out/{subj}"
    od = f"{out_root}/{ds}/out/{subj}/{arm}/recon_breath"
    man = json.load(open(f"{sd}/manifest.json"))
    ref, dz = man["scatter"]["ref_plane"], man["dz_mm"]
    stack = "scatter" if arm.endswith("_scatter") else "breath"
    inp = nib.load(f"{sd}/{stack}/stack_t{k:02d}.nii.gz").get_fdata()
    out = nib.load(f"{od}/vol_t{k:02d}.nii.gz").get_fdata()
    gt = nib.load(f"{sd}/gt/gt_t{k:02d}.nii.gz").get_fdata()
    seg = np.asarray(nib.load(f"{sd}/seg_gt/seg_t{k:02d}.nii.gz").dataobj)
    c = json.load(open(f"{od}/centres.json"))[f"t{k:02d}"]
    pred = dangi_to_native_px(c["centres_px"])
    shifts_px = np.asarray(c["shifts_mm"]) / NATIVE_MM
    D = inp.shape[2]
    c_gt = np.full((D, 2), np.nan)
    for z in range(D):
        xs, ys = np.nonzero(seg[:, :, z] == 1)
        if len(xs) >= 20:
            c_gt[z] = [xs.mean(), ys.mean()]
    vmax = np.percentile(gt, 99.5)
    fig = plt.figure(figsize=(2.0 * D + 6, 8.5))
    gs = fig.add_gridspec(3, D + 3, width_ratios=[1] * D + [0.3, 1.2, 1.2])
    for z in range(D):
        for r, (v, name) in enumerate([(inp, "input"), (out, "Dangi"), (gt, "GT")]):
            ax = fig.add_subplot(gs[r, z])
            ax.imshow(v[:, :, z].T, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.plot(*pred[z], "+", c="orange", ms=10, mew=2)
                ax.set_title(f"z{z}{' REF' if z == ref else ''}\n{c['shifts_mm'][z][0]:+.0f},{c['shifts_mm'][z][1]:+.0f} mm", fontsize=8)
            if r == 1:
                ax.plot(*(pred[z] + shifts_px[z]), "+", c="orange", ms=10, mew=2)
            if not np.isnan(c_gt[z, 0]):
                ax.plot(*c_gt[z], "o", mfc="none", mec="lime", ms=10, mew=1.5)
            if z == 0:
                ax.set_ylabel(name, fontsize=11)
    # long-axis cuts through the reference plane's GT centre
    cx, cy = (c_gt[ref] if not np.isnan(c_gt[ref, 0]) else np.nanmean(c_gt, 0)).round().astype(int)
    for r, (v, name) in enumerate([(inp, "input"), (out, "Dangi"), (gt, "GT")]):
        for j, (cut, lab) in enumerate([(v[:, cy, :].T, f"XZ @ y={cy}"), (v[cx, :, :].T, f"YZ @ x={cx}")]):
            ax = fig.add_subplot(gs[r, D + 1 + j])
            ax.imshow(cut, cmap="gray", origin="lower", vmin=0, vmax=vmax, aspect=dz / NATIVE_MM, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lab, fontsize=9)
    fig.suptitle(f"{ds}/{subj}  phase {k}  arm {arm}  (+ predicted centre, o GT LV centre; titles = applied shift x,y mm)", fontsize=11)
    fig.tight_layout()
    path = f"{fig_dir}/{ds}_{subj}_{arm}_t{k:02d}.png"
    fig.savefig(path, dpi=60)
    plt.close(fig)
    return os.path.basename(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=f"{ROOT}/temp/dangi/eval")
    ap.add_argument("--arm", default="dangi_scatter")
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--phases", type=int, nargs="+", default=[0, 6])
    ap.add_argument("--fig-dir", default=f"{ROOT}/temp/dangi/figs")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    names = []
    for ds_dir in sorted(glob.glob(f"{args.out_root}/*/out")):
        ds = ds_dir.split("/")[-2]
        subjects = sorted(os.listdir(ds_dir))[: args.per_source]
        for subj in subjects:
            if not os.path.exists(f"{ds_dir}/{subj}/{args.arm}/recon_breath/centres.json"):
                continue
            for k in args.phases:
                names.append(render(ds, subj, k, args.arm, args.out_root, args.fig_dir))
                print(names[-1], flush=True)
    with open(f"{args.fig_dir}/index_{args.arm}.html", "w") as f:
        f.write("<html><body style='background:#111;color:#ddd;font-family:sans-serif'>"
                f"<h2>Dangi baseline examples — {args.arm}</h2>")
        for n in names:
            f.write(f"<h3>{n}</h3><img src='{n}' style='width:100%'>")
        f.write("</body></html>")


if __name__ == "__main__":
    main()
