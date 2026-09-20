#!/usr/bin/env python
"""Show the GT target cine a subject actually gets under `af`, next to its clean counterpart.

For each subject: two rows of 12 frames (top = regular_frozen, bottom = af) at the mid slice,
with each frame's LV cavity volume (from the scorer's own seg_gt cache) printed under it, and
the resulting GT EF. That makes the two distinct failure modes visible and separable:

  frozen window   several frames are the IDENTICAL picture, so the cine never reaches ES and
                  EF is UNMEASURABLE -- an artifact of the simulator parking the heart.
  weak beats      every frame is different and the heart really is contracting, but it ejects
                  less because short beats never let it fill -- REAL simulated AF physiology,
                  which must NOT be excluded.

Usage: PYTHONPATH=training:. python tools/render_af_gt_check.py [SUBJ ...]
"""
import glob
import os
import sys

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import paths  # noqa: E402

T = 12
CELL = 118
SUBJS = sys.argv[1:] or ["CMRx24_Test_P004", "CMRx24_Train_P092", "CMRx24_Test_P034"]


def load(cohort, subj):
    d = paths.subject_dir(cohort, subj)
    gt = np.stack([np.asarray(nib.load(str(d / "gt" / f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32) for t in range(T)])          # (T,X,Y,Z)
    segs = sorted(glob.glob(str(d / "seg_gt" / "seg_t*.nii.gz")))
    vol = (np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
           if len(segs) == T else None)
    return gt, vol


def strip(gt, vol, vmax, label):
    z = gt.shape[3] // 2
    W = CELL * T
    img = Image.new("RGB", (W, CELL + 34), "black")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    prev = None
    for t in range(T):
        a = np.clip(gt[t, :, :, z] / vmax * 255, 0, 255).astype(np.uint8)
        pim = Image.fromarray(a.T[::-1]).resize((CELL, CELL), Image.NEAREST).convert("RGB")
        img.paste(pim, (t * CELL, 16))
        same = prev is not None and np.array_equal(gt[t], prev)
        d.text((t * CELL + 3, 2), f"f{t}" + ("  =prev" if same else ""),
               fill="red" if same else "white", font=font)
        if vol is not None:
            frac = vol[t] / max(vol.max(), 1e-9)
            d.text((t * CELL + 3, CELL + 18), f"{frac*100:4.0f}%", fill="yellow", font=font)
        prev = gt[t]
    d.text((4, CELL + 4), label, fill="cyan", font=font)
    return img


for s in SUBJS:
    panels, labels = [], []
    for c in ("cmrx2024_regular_frozen", "cmrx2024_af"):
        gt, vol = load(c, s)
        ef = (vol.max() - vol.min()) / vol.max() * 100 if vol is not None else float("nan")
        dup = sum(1 for i in range(1, T) if np.array_equal(gt[i], gt[i - 1]))
        panels.append((gt, vol, ef, dup))
        labels.append(c.replace("cmrx2024_", ""))
    vmax = float(np.percentile(panels[0][0], 99.5))
    out = Image.new("RGB", (CELL * T, (CELL + 34) * 2 + 20), "black")
    for i, ((gt, vol, ef, dup), lab) in enumerate(zip(panels, labels)):
        out.paste(strip(gt, vol, vmax, f"{lab}   GT EF {ef:.1f}%   identical-consecutive frames: {dup}"),
                  (0, i * (CELL + 34)))
    d = ImageDraw.Draw(out)
    d.text((4, (CELL + 34) * 2 + 4), f"{s}   yellow = LV cavity volume as % of that cine's max",
           fill="white")
    p = os.path.join(ROOT, "figs", f"af_gt_check_{s}.png")
    out.save(p)
    print(f"-> {p}   frozen: EF {panels[0][2]:.1f} -> af {panels[1][2]:.1f}, "
          f"identical frames {panels[0][3]} -> {panels[1][3]}")
