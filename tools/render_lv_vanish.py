#!/usr/bin/env python
"""Show WHY a method's EF is undefined on a subject: the frames where nnU-Net finds no LV.

ef_dice sets ESV/EF to None when the LV label vanishes in >= 1 of the scored frames
(ef_dice.py `vols_of`). The campaign's seg dirs live in node-local /tmp and are deleted, so this
re-runs the SAME chain on just the named subjects -- ef_dice's own `_dump_cine` + `_heart_roi`
(same heart-ROI crop), then nnU-Net Task114 3d_fullres -- and renders, per subject, one PNG:

  rows    = GT | method          (each with its own nnU-Net segmentation overlaid)
  columns = ED frame, the frame(s) where the method's LV vanishes, and their neighbours
  bottom  = LV voxel count per frame, GT vs method

Every slice that contains GT LV is shown for the vanished frame (never just the mid slice).

Usage:
  PYTHONPATH=training:. python tools/render_lv_vanish.py --arm af24 --method fetal_cmr_4d \
      --subjects acdc:ACDC_patient110 acdc:ACDC_patient126 mnms:MNMs_K7L2Y6 cmrx2023:CMRx23_Test_P028
"""
import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import nibabel as nib                                              # noqa: E402
import numpy as np                                                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation"),
                os.path.join(ROOT, "evaluation", "src", "score")]
import paths                                                       # noqa: E402
import ef_dice                                                     # noqa: E402

ENV_SH = "/home/minsukc/vggt/tools/nnunet_mnms_eval/env.sh"
COL = {1: (1.0, 0.25, 0.25), 2: (0.25, 0.9, 0.35), 3: (0.3, 0.55, 1.0)}     # LV, MYO, RV


def segment(cohort, subj, method, work):
    """-> {'gt': (img4d, seg4d), 'pred': (img4d, seg4d)} on the ROI-cropped volumes nnU-Net saw."""
    inp, seg = os.path.join(work, "in"), os.path.join(work, "seg")
    os.makedirs(inp); os.makedirs(seg)
    roi = ef_dice._heart_roi(cohort, subj)
    src = {"gt": paths.cine_gt(cohort, subj),
           "pred": os.path.join(str(ef_dice.method_dir(cohort, subj, method)), "cine_breath.nii.gz")}
    T = {k: ef_dice._dump_cine(p, inp, cohort, 0, k, roi) for k, p in src.items()}
    subprocess.run(["micromamba", "run", "-n", "nnunet", "bash", "-c",
                    f"source '{ENV_SH}' && nnUNet_predict -i '{inp}' -o '{seg}' -t 114 -m 3d_fullres "
                    f"-tr nnUNetTrainerV2_MMS"], check=True, stdout=subprocess.DEVNULL)
    out = {}
    for k in src:
        im = np.stack([np.asarray(nib.load(f"{inp}/{cohort}__s000__{k}__t{t:02d}_0000.nii.gz").dataobj)
                       for t in range(T[k])], -1)
        sg = np.stack([np.asarray(nib.load(f"{seg}/{cohort}__s000__{k}__t{t:02d}.nii.gz").dataobj)
                       for t in range(T[k])], -1)
        out[k] = (im, sg)
    return out


def overlay(ax, img, seg, vmax, title, flag=False):
    rgb = np.repeat(np.clip(img.T / vmax, 0, 1)[..., None], 3, -1)
    for lab, c in COL.items():
        m = seg.T == lab
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(c)
    ax.imshow(rgb, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color="#c00000" if flag else "black", weight="bold" if flag else "normal")


def render(cohort, subj, method, d, out):
    (gi, gs), (pi, ps) = d["gt"], d["pred"]
    T = gi.shape[3]
    lv_g = (gs == 1).sum((0, 1, 2)); lv_p = (ps == 1).sum((0, 1, 2))
    gone = [int(t) for t in np.where(lv_p == 0)[0]]
    ed = int(lv_g.argmax())
    zs = [int(z) for z in np.where((gs == 1).any((0, 1, 3)))[0]]           # every slice with GT LV
    # crop to the heart ROI bbox so the anatomy is readable
    nz = np.argwhere(gi[..., ed] > 0); (x0, y0, _), (x1, y1, _) = nz.min(0), nz.max(0) + 1
    vmax = float(np.percentile(gi[gi > 0], 99.5))
    frames = sorted({ed, *gone, *[(t - 1) % T for t in gone], *[(t + 1) % T for t in gone]})[:8]
    n_img_rows = 2 * len(zs)
    fig = plt.figure(figsize=(1.75 * len(frames) + 1.2, 1.55 * n_img_rows + 2.6), dpi=130)
    gsp = fig.add_gridspec(n_img_rows + 1, len(frames), height_ratios=[1] * n_img_rows + [1.5])
    for zi, z in enumerate(zs):
        for ci, t in enumerate(frames):
            for ri, (im, sg, nm) in enumerate(((gi, gs, "GT"), (pi, ps, method.replace("_cmr_4d", "")))):
                ax = fig.add_subplot(gsp[2 * zi + ri, ci])
                lost = ri == 1 and t in gone
                n = int((sg[..., z, t] == 1).sum())
                overlay(ax, im[x0:x1, y0:y1, z, t], sg[x0:x1, y0:y1, z, t], vmax,
                        f"{nm} z{z} f{t}{' ED' if t == ed else ''}  LV={n}", flag=lost)
    ax = fig.add_subplot(gsp[n_img_rows, :])
    ax.plot(lv_g, "o-", color="#2f6f4e", label="GT"); ax.plot(lv_p, "s-", color="#c0392b", label=method)
    for t in gone:
        ax.axvline(t, color="#c0392b", ls=":", lw=1)
    ax.set_xlabel("frame"); ax.set_ylabel("LV voxels (whole volume)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"LV vanishes in frame(s) {gone}  ->  ESV, EF undefined", fontsize=9)
    fig.suptitle(f"{cohort} / {subj}   [{method}]\nred=LV  green=MYO  blue=RV   (nnU-Net 3d_fullres, heart-ROI crop)",
                 fontsize=8.5, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor="white"); plt.close(fig)
    return gone, lv_p.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--method", default="fetal_cmr_4d")
    ap.add_argument("--subjects", nargs="+", required=True, help="source:subject")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "figs", "rhythm24", f"{a.arm}_lv_vanish")
    os.makedirs(outdir, exist_ok=True)
    for item in a.subjects:
        src, subj = item.split(":")
        cohort = f"{src}_{a.arm}"
        with tempfile.TemporaryDirectory(prefix="lvvanish_") as work:
            d = segment(cohort, subj, a.method, work)
            out = os.path.join(outdir, f"{subj}__{a.method}.png")
            gone, curve = render(cohort, subj, a.method, d, out)
        print(f"{subj}: LV=0 in frames {gone}   curve={curve}\n  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
