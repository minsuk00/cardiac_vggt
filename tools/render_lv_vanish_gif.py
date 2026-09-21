#!/usr/bin/env python
"""HOW a reconstruction loses its LV: per-slice x per-frame LV map + an animated comparison.

For each subject, runs ef_dice's own crop (`_heart_roi`, `_dump_cine`) + nnU-Net Task114 3d_fullres
on GT and on each named method, KEEPS the masks on GPFS
(scratch/eval/_rhythm24_segs/<arm>/lv_vanish/<subject>/), and writes to figs/rhythm24/<arm>_lv_vanish/:

  <subj>__lvmap.png   LV voxel count per (slice, frame) for GT and every method -- answers "is it a
                      few slices or the whole phase?" at a glance; whole-volume curves underneath.
  <subj>__recon.gif   24 frames; rows = GT / methods, columns = EVERY slice that holds GT LV, with the
                      segmentation overlaid (red LV, green MYO, blue RV) and the LV curves + a cursor.

Usage:
  PYTHONPATH=training:. SPLIT=test SEG_CFG=3d_fullres python tools/render_lv_vanish_gif.py --arm af24 \
      --subjects acdc:ACDC_patient110 mnms:MNMs_K7L2Y6
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import nibabel as nib                                              # noqa: E402
import numpy as np                                                 # noqa: E402
from PIL import Image                                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation"),
                os.path.join(ROOT, "evaluation", "src", "score")]
import paths                                                       # noqa: E402
import ef_dice                                                     # noqa: E402

ENV_SH = "/home/minsukc/vggt/tools/nnunet_mnms_eval/env.sh"
COL = {1: (1.0, 0.2, 0.2), 2: (0.2, 0.9, 0.3), 3: (0.3, 0.55, 1.0)}
NAME = {"gt": "GT", "fetal_cmr_4d": "Fetal CMR 4D", "vggt_final518_diff1000_ep300": "VGGT diff1000",
        "vggt_final518_base_ep300": "VGGT base"}
LINE = {"gt": "#2f6f4e", "fetal_cmr_4d": "#c0392b", "vggt_final518_diff1000_ep300": "#2c6fbb",
        "vggt_final518_base_ep300": "#8e44ad"}


def segment(cohort, subj, methods, keep):
    """-> {key: (img X,Y,Z,T ; seg X,Y,Z,T)} on the ROI-cropped volumes. Masks are kept in `keep`."""
    inp, seg = os.path.join(keep, "in"), os.path.join(keep, "seg")
    roi = ef_dice._heart_roi(cohort, subj)
    src = {"gt": str(paths.cine_gt(cohort, subj))}
    src.update({m: os.path.join(str(ef_dice.method_dir(cohort, subj, m)), "cine_breath.nii.gz") for m in methods})
    if not (os.path.isdir(seg) and len(os.listdir(seg)) >= 24 * len(src)):
        shutil.rmtree(keep, ignore_errors=True); os.makedirs(inp); os.makedirs(seg)
        for k, p in src.items():
            assert ef_dice._dump_cine(p, inp, cohort, 0, k, roi) > 0, p
        cmd = ["micromamba", "run", "-n", "nnunet", "bash", "-c",
               f"source '{ENV_SH}' && nnUNet_predict -i '{inp}' -o '{seg}' -t 114 -m 3d_fullres -tr nnUNetTrainerV2_MMS"]
        for attempt in range(6):        # `micromamba run` collides on ~/.cache/mamba/proc with other jobs
            if subprocess.run(cmd, stdout=subprocess.DEVNULL).returncode == 0:
                break
            time.sleep(20)
        else:
            sys.exit(f"nnUNet_predict failed 6 times for {subj}")
    out = {}
    for k in src:
        T = len([f for f in os.listdir(inp) if f"__{k}__t" in f])
        ld = lambda d, suf: np.stack([np.asarray(nib.load(f"{d}/{cohort}__s000__{k}__t{t:02d}{suf}.nii.gz").dataobj)   # noqa: E731
                                      for t in range(T)], -1)
        out[k] = (ld(inp, "_0000").astype(np.float32), ld(seg, ""))
    return out


def overlay(img, seg, vmax):
    rgb = np.repeat(np.clip(img.T / vmax, 0, 1)[..., None], 3, -1)
    for lab, c in COL.items():
        m = seg.T == lab
        rgb[m] = 0.55 * rgb[m] + 0.45 * np.array(c)
    return rgb


def lvmap(d, keys, title, out):
    fig, axs = plt.subplots(2, len(keys), figsize=(4.6 * len(keys), 6.2), dpi=120,
                            gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    vmax = max(float((d[k][1] == 1).sum((0, 1)).max()) for k in keys)
    for i, k in enumerate(keys):
        m = (d[k][1] == 1).sum((0, 1))                                    # (Z, T)
        im = axs[0, i].imshow(m, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, origin="lower")
        zz, tt = np.where((m == 0) & ((d["gt"][1] == 1).sum((0, 1)) > 0))   # GT has LV here, this one does not
        axs[0, i].scatter(tt, zz, marker="x", color="red", s=22, linewidths=1.2)
        gone = np.where(m.sum(0) == 0)[0]
        for t in gone:
            axs[0, i].axvline(t, color="red", lw=1.5, alpha=.6)
        axs[0, i].set_title(f"{NAME.get(k, k)}\nLV voxels per (slice, frame)   whole-volume LV = 0 in frames {gone.tolist()}",
                            fontsize=9)
        axs[0, i].set_xlabel("frame"); axs[0, i].set_ylabel("slice z")
        plt.colorbar(im, ax=axs[0, i], fraction=0.046)
        for kk in keys:
            axs[1, i].plot((d[kk][1] == 1).sum((0, 1, 2)), color=LINE.get(kk, "k"), lw=2.2 if kk == k else 1,
                           alpha=1 if kk == k else .35, label=NAME.get(kk, kk))
        axs[1, i].set_xlabel("frame"); axs[1, i].set_ylabel("LV voxels (volume)"); axs[1, i].grid(alpha=.3)
        axs[1, i].legend(fontsize=7)
    fig.suptitle(f"{title}    red x = GT has LV in that (slice, frame) but this volume has none", fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(out, facecolor="white"); plt.close(fig)


def gif(d, keys, title, out):
    gs_ = d["gt"][1]
    zs = [int(z) for z in np.where((gs_ == 1).any((0, 1, 3)))[0]]
    T = gs_.shape[3]
    nz = np.argwhere(d["gt"][0][..., 0] > 0); (x0, y0, _), (x1, y1, _) = nz.min(0), nz.max(0) + 1
    vmax = {k: float(np.percentile(d[k][0][d[k][0] > 0], 99.5)) for k in keys}
    curves = {k: (d[k][1] == 1).sum((0, 1, 2)) for k in keys}
    frames = []
    for t in range(T):
        fig = plt.figure(figsize=(1.5 * len(zs) + 0.6, 1.5 * len(keys) + 2.3), dpi=100)
        g = fig.add_gridspec(len(keys) + 1, len(zs), height_ratios=[1] * len(keys) + [1.35], hspace=0.08, wspace=0.03)
        for r, k in enumerate(keys):
            im, sg = d[k]
            for c, z in enumerate(zs):
                ax = fig.add_subplot(g[r, c])
                ax.imshow(overlay(im[x0:x1, y0:y1, z, t], sg[x0:x1, y0:y1, z, t], vmax[k]), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                n = int((sg[..., z, t] == 1).sum())
                lost = n == 0 and int((gs_[..., z, t] == 1).sum()) > 0
                for sp in ax.spines.values():
                    sp.set_color("red" if lost else "#888888"); sp.set_linewidth(2.2 if lost else 0.5)
                if r == 0:
                    ax.set_title(f"z={z}", fontsize=8)
                if c == 0:
                    ax.set_ylabel(NAME.get(k, k), fontsize=8)
                ax.text(0.03, 0.04, f"LV {n}", transform=ax.transAxes, fontsize=6.5, color="white",
                        bbox=dict(fc="red" if lost else "black", ec="none", alpha=.6, pad=1))
        ax = fig.add_subplot(g[len(keys), :])
        for k in keys:
            ax.plot(curves[k], "o-", ms=3, color=LINE.get(k, "k"), label=NAME.get(k, k))
        ax.axvline(t, color="#f39c12", lw=2)
        ax.set_xlim(-0.5, T - 0.5); ax.set_xlabel("frame"); ax.set_ylabel("LV voxels"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=len(keys), loc="upper center")
        fig.suptitle(f"{title}   frame {t:2d}/{T}   red box = GT has LV in this slice, this volume has none", fontsize=9)
        buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight"); plt.close(fig)
        buf.seek(0); frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))
    w, h = min(f.size[0] for f in frames), min(f.size[1] for f in frames)
    frames = [f.crop((0, 0, w, h)) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=450, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subjects", nargs="+", required=True, help="source:subject")
    ap.add_argument("--methods", nargs="+", default=["fetal_cmr_4d", "vggt_final518_diff1000_ep300"])
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "figs", "rhythm24", f"{a.arm}_lv_vanish")
    os.makedirs(outdir, exist_ok=True)
    for item in a.subjects:
        src, subj = item.split(":")
        cohort = f"{src}_{a.arm}"
        keep = os.path.join(ROOT, "scratch", "eval", "_rhythm24_segs", a.arm, "lv_vanish", subj)
        d = segment(cohort, subj, a.methods, keep)
        keys = ["gt"] + a.methods
        lvmap(d, keys, f"{cohort} / {subj}", os.path.join(outdir, f"{subj}__lvmap.png"))
        gif(d, keys, f"{cohort} / {subj}", os.path.join(outdir, f"{subj}__recon.gif"))
        m = (d[a.methods[0]][1] == 1).sum((0, 1)); g = (d["gt"][1] == 1).sum((0, 1))
        gone = np.where(m.sum(0) == 0)[0].tolist()
        part = [(int(t), int(((m[:, t] == 0) & (g[:, t] > 0)).sum()), int((g[:, t] > 0).sum())) for t in range(m.shape[1])]
        print(f"{subj}: whole-volume LV=0 in frames {gone}; per frame (frame, slices lost, slices with GT LV): "
              f"{[p for p in part if p[1] > 0]}", flush=True)


if __name__ == "__main__":
    main()
