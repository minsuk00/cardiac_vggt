#!/usr/bin/env python
"""Qualitative recon comparison GIF: GT | method A | method B, three slices, all frames.

Reads the SCORED cines (cine_gt.nii.gz and <method>/cine_breath.nii.gz -- the exact volumes
image_metrics scored, already on the GT grid with each method's gauge/pose treatment), so what is
shown is what was measured. Rows = the heart's middle slice and the slices 2 below / 2 above it
(middle = centre of the padded heart mask's z-extent, clamped to it); columns = GT and the methods.
Cropped to the padded heart ROI + margin. One fixed intensity window per column for the whole clip.

Usage:
  PYTHONPATH=training:. python tools/render_recon_compare_gif.py --arm af24 \
      --subjects cmrx2024:CMRx24_Test_P017 mnms:MNMs_K7L2Y6
"""
import argparse
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation")]
import paths                                                       # noqa: E402

NAME = {"fetal_cmr_4d": "Fetal CMR 4D", "vggt_final518_diff1000_ep300": "VGGT diff1000",
        "vggt_final518_base_ep300": "VGGT base"}
CELL, PAD, MARGIN = 230, 4, 12


def font(sz):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--subjects", nargs="+", required=True, help="source:subject")
    ap.add_argument("--methods", nargs="+", default=["fetal_cmr_4d", "vggt_final518_diff1000_ep300"])
    ap.add_argument("--offset", type=int, default=2, help="slices above/below the middle one")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "figs", "rhythm24", f"{a.arm}_recon_compare")
    os.makedirs(outdir, exist_ok=True)
    f_big, f_small = font(15), font(12)

    for item in a.subjects:
        src, subj = item.split(":")
        ds = f"{src}_{a.arm}"
        vols = {"GT": np.asarray(nib.load(str(paths.cine_gt(ds, subj))).dataobj, np.float32)}
        for m in a.methods:
            vols[NAME.get(m, m)] = np.asarray(nib.load(os.path.join(str(paths.arm_dir(ds, subj, m)), "cine_breath.nii.gz")).dataobj, np.float32)
        shp = {k: v.shape for k, v in vols.items()}
        assert len(set(shp.values())) == 1, f"cines are not on one grid: {shp}"
        roi = np.asarray(nib.load(str(paths.heart_mask_pad(ds, subj))).dataobj) > 0.5
        zz = np.where(roi.any((0, 1)))[0]; zmid = int(round((zz[0] + zz[-1]) / 2))
        zs = [int(np.clip(zmid + d, zz[0], zz[-1])) for d in (-a.offset, 0, a.offset)]
        xs, ys = np.where(roi.any((1, 2)))[0], np.where(roi.any((0, 2)))[0]
        X, Y = vols["GT"].shape[:2]
        x0, x1 = max(xs[0] - MARGIN, 0), min(xs[-1] + MARGIN + 1, X)
        y0, y1 = max(ys[0] - MARGIN, 0), min(ys[-1] + MARGIN + 1, Y)
        T = vols["GT"].shape[3]
        # one window per column, from that column's own heart-ROI intensities over the whole clip
        vmax = {k: float(np.percentile(v[roi][v[roi] > 0], 99.5)) for k, v in vols.items()}

        keys = list(vols)
        W = 46 + len(keys) * (CELL + PAD); H = 30 + len(zs) * (CELL + PAD) + 22
        frames = []
        for t in range(T):
            cv = Image.new("RGB", (W, H), (14, 14, 16)); dr = ImageDraw.Draw(cv)
            for c, k in enumerate(keys):
                dr.text((46 + c * (CELL + PAD) + 6, 6), k, font=f_big, fill=(235, 235, 240))
                for r, z in enumerate(zs):
                    sl = np.clip(vols[k][x0:x1, y0:y1, z, t].T / vmax[k], 0, 1)
                    im = Image.fromarray((sl * 255).astype(np.uint8), "L").resize((CELL, CELL), Image.BILINEAR)
                    cv.paste(im.convert("RGB"), (46 + c * (CELL + PAD), 30 + r * (CELL + PAD)))
            for r, z in enumerate(zs):
                dr.text((4, 30 + r * (CELL + PAD) + CELL // 2 - 8), f"z={z}", font=f_small, fill=(255, 210, 90))
            dr.text((6, H - 18), f"{ds} / {subj}    frame {t:2d}/{T}    (scored cines, heart-ROI crop, "
                                 f"mid slice z={zmid} and ±{a.offset})", font=f_small, fill=(160, 160, 165))
            frames.append(cv)
        out = os.path.join(outdir, f"{subj}.gif")
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=170, loop=0)
        print(f"wrote {out}   slices {zs}  crop x{x0}:{x1} y{y0}:{y1}", flush=True)


if __name__ == "__main__":
    main()
