#!/usr/bin/env python
"""Render the raw GT target cine (not a reconstruction) for one or more af-cohort subjects, to
show the hold mechanism directly: several near-identical frames in a row, then a jump.

Usage: PYTHONPATH=training:. python tools/render_af_hold_gif.py [SUBJ1 SUBJ2 ...]
"""
import sys
import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

SUBJS = sys.argv[1:] or ["CMRx24_Test_P017", "CMRx24_Test_P004"]
COHORT = "cmrx2024_af"
ROOT = "/home/minsukc/vggt-afsim/scratch/eval"
T = 12
CELL = 220

data = {}
for s in SUBJS:
    gt = np.stack([np.asarray(nib.load(f"{ROOT}/{COHORT}/out/{s}/gt/gt_t{t:02d}.nii.gz").dataobj,
                              dtype=np.float32) for t in range(T)])
    D = gt.shape[3]
    z = D // 2
    data[s] = (gt[:, :, :, z], gt.max())
    print(f"{s}: D={D}, mid z={z}")

try:
    font = ImageFont.load_default()
except Exception:
    font = None

frames = []
for t in range(T):
    row = Image.new("RGB", (CELL * len(SUBJS), CELL + 30), "black")
    d = ImageDraw.Draw(row)
    for i, s in enumerate(SUBJS):
        vol, vmax = data[s]
        img = np.clip(vol[t] / vmax * 255, 0, 255).astype(np.uint8)
        pim = Image.fromarray(img.T[::-1]).resize((CELL, CELL), Image.NEAREST).convert("RGB")
        row.paste(pim, (i * CELL, 30))
        d.text((i * CELL + 5, 2), s, fill="white", font=font)
    d.text((CELL * len(SUBJS) - 60, 2), f"target f={t}", fill="yellow", font=font)
    frames.append(row)

out = "/home/minsukc/vggt-afsim/figs/af_hold_gt_cine.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=500, loop=0)
print(f"-> {out}")
