#!/usr/bin/env python
"""One GIF per subject from a BUILT 24-frame (`*24`) bundle: regular | hrv | af, side by side.

Reads what the builder actually wrote (unlike tools/render_af24_demo.py, which re-simulates
positions to argue the design), so this is what the engines will consume. Mid slice, 24 frames,
2.0 s of acquisition ~= 2 nominal beats.

Under `regular` frames f and f+12 are the same cardiac phase; under `af` the rhythm has drifted and
they are not. That is the whole point of going to 24 frames: Fetal CMR 4D averages each such pair
into one output bin.

Usage:
    PYTHONPATH=training:. python tools/render_af24_bundle_demo.py \
        --eval-root <root> --source cmrx2024 --subjects CMRx24_Test_P017
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation")]

ARMS = ("regular24", "hrv24", "af24")
OUTDIR = os.path.join(ROOT, "temp", "af24_builder_demo")
SIZE = 300
TRACE = 122        # height of the LV-volume trace under each panel


def font(sz=15):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def plane(d, f, z):
    """Mid-slice of input frame f, as (Y, X)."""
    p = os.path.join(d, "breath", f"stack_t{f:02d}.nii.gz")
    return np.asarray(nib.load(p).dataobj, dtype=np.float32)[:, :, z].T


def window(d, nf, z):
    """A FIXED intensity window across the whole clip, so brightness changes are real signal."""
    s = np.concatenate([plane(d, f, z).ravel() for f in range(0, nf, 4)])
    s = s[s > 0]
    return 0.0, float(np.percentile(s, 99.5)) if s.size else 1.0


def lv_stored(src_dir, ncp):
    """(ncp,) whole-LV voxel count per STORED cardiac phase, from the source cohort's seg cache.

    Read from the SOURCE, never from an arm's own seg_gt/: on an integral arm that directory is a
    gt_map-PERMUTED copy, so it would plot a permuted curve against unpermuted positions.
    """
    segs = sorted(glob.glob(os.path.join(src_dir, "seg_gt", "seg_t*.nii.gz")))
    if len(segs) != ncp:
        return None
    v = np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
    return v if v.max() > 0 else None


def render(subj, bundles, src_dir, out):
    nf = int(next(iter(bundles.values()))[1]["T"])
    fnt, fnt_s = font(), font(13)
    wins, mids, curves = {}, {}, {}
    for arm, (d, man) in bundles.items():
        mids[arm] = int(man["D"]) // 2
        wins[arm] = window(d, nf, mids[arm])
        ncp = int(man["n_cardphase"])
        stored = lv_stored(src_dir, ncp)
        pos = np.asarray(man["rhythm"]["pos_per_plane"], float)[mids[arm]] % ncp
        # LV volume this slice is actually showing at each frame. One shared y-range across arms
        # (the stored curve is the same heart), so the traces are directly comparable.
        curves[arm] = (None if stored is None
                       else np.interp(pos, np.arange(ncp + 1), np.r_[stored, stored[0]]))
    have = [c for c in curves.values() if c is not None]
    ymin = min(c.min() for c in have) if have else 0.0
    ymax = max(c.max() for c in have) if have else 1.0
    W, H = SIZE * len(bundles), SIZE + 52 + TRACE
    frames = []
    for f in range(nf):
        canvas = Image.new("RGB", (W, H), (12, 12, 14))
        dr = ImageDraw.Draw(canvas)
        for i, (arm, (d, man)) in enumerate(bundles.items()):
            lo, hi = wins[arm]
            a = plane(d, f, mids[arm])
            x = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
            canvas.paste(Image.fromarray((x * 255).astype(np.uint8), "L").convert("RGB")
                         .resize((SIZE, SIZE), Image.BILINEAR), (i * SIZE, 24))
            ncp = int(man["n_cardphase"])
            pos = np.asarray(man["rhythm"]["pos_per_plane"], float)[mids[arm], f] % ncp
            dr.text((i * SIZE + 8, 4), arm, font=fnt, fill=(235, 235, 240))
            dr.text((i * SIZE + 8, SIZE + 26), f"cardiac phase {pos:5.2f} / {ncp}",
                    font=fnt_s, fill=(255, 210, 90))
            lv = curves[arm]
            if lv is None:
                continue
            x0, y0, y1 = i * SIZE + 10, SIZE + 54, SIZE + 46 + TRACE - 30
            xs = [x0 + k * (SIZE - 20) / (nf - 1) for k in range(nf)]
            ys = [y1 - (lv[k] - ymin) / max(ymax - ymin, 1e-6) * (y1 - y0) for k in range(nf)]
            for b in range(1, (nf + ncp - 1) // ncp):          # nominal beat boundary
                bx = x0 + (b * ncp) * (SIZE - 20) / (nf - 1)
                dr.line([(bx, y0 - 4), (bx, y1 + 3)], fill=(90, 90, 96), width=1)
            dr.line(list(zip(xs, ys)), fill=(110, 175, 60), width=2)
            dr.ellipse([xs[f] - 4, ys[f] - 4, xs[f] + 4, ys[f] + 4], fill=(255, 210, 90))
        dr.text((W - 150, 6), f"frame {f:2d}/{nf}   beat {f // 12}", font=fnt_s,
                fill=(150, 150, 155))
        if have:
            dr.text((10, H - 14), "LV volume at the shown slice's cardiac position   "
                                  "(grey = nominal beat boundary)", font=fnt_s,
                    fill=(150, 150, 155))
        frames.append(canvas)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=170, loop=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=os.path.join(ROOT, "scratch/eval"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--subjects", required=True, help="comma-separated")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for subj in [s.strip() for s in a.subjects.split(",") if s.strip()]:
        bundles = {}
        for arm in ARMS:
            d = os.path.join(a.eval_root, f"{a.source}_{arm}", "out", subj)
            if os.path.isfile(os.path.join(d, "manifest.json")):
                bundles[arm] = (d, json.load(open(os.path.join(d, "manifest.json"))))
        if not bundles:
            print(f"  skip {subj}: nothing built")
            continue
        p = os.path.join(OUTDIR, f"{subj}.gif")
        render(subj, bundles, os.path.join(a.eval_root, a.source, "out", subj), p)
        print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    main()
