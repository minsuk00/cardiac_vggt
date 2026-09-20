#!/usr/bin/env python
"""What the model/baseline SEES at different frame spacings: the same 12-frame budget, sampled
1/12 s apart (one beat) vs 2x/3x/4x that (several beats).

Columns = stride. Each column shows its 12 frames as a cine, with the LV volume at each sampled
position underneath and the phase step between consecutive frames printed -- so the loss of
temporal redundancy at wide spacing is visible rather than argued about.

Usage: PYTHONPATH=training:. python tools/render_stride_demo.py [SUBJ ...]
"""
import glob
import json
import os
import sys

import numpy as np
import nibabel as nib
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import paths            # noqa: E402
from build_af_bundle import BURN, N_BEATS, T, rr_sequence, pos_physio, render_phase  # noqa: E402

DT0 = 1.0 / 12
STRIDES = [1, 2, 3, 4]
CELL = 205
TRACE = 96
OUTDIR = os.path.join(ROOT, "temp", "af_rule_gifs")


def lv(subj):
    fs = sorted(glob.glob(str(paths.subject_dir("cmrx2024", subj) / "seg_gt" / "seg_t*.nii.gz")))
    return np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in fs], float)


def panel(vol, pos, W, k, label):
    img = Image.new("RGB", (W, TRACE), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    vp = np.r_[vol, vol[0]]
    lo, hi = vp.min(), vp.max()
    X = lambda i: int(6 + i / (T - 1) * (W - 12))                          # noqa: E731
    Y = lambda v: int(TRACE - 16 - (v - lo) / max(hi - lo, 1e-9) * (TRACE - 42))  # noqa: E731
    pts = [(X(i), Y(np.interp(p % T, np.arange(T + 1), vp))) for i, p in enumerate(pos)]
    d.line(pts, fill=(60, 60, 60), width=1)
    for i, (px, py) in enumerate(pts):
        r = 4 if i == k else 2
        d.ellipse([px - r, py - r, px + r, py + r], fill=(200, 80, 0) if i == k else (150, 150, 150))
    step = np.abs(np.diff(pos % T))
    step = np.where(step > T / 2, T - step, step)
    d.text((5, 3), f"{label}   mean phase step {step.mean():.1f}/12", fill=(30, 30, 30), font=font)
    d.text((5, TRACE - 13), "LV volume at each sampled frame", fill=(120, 120, 120), font=font)
    return img


for subj in sys.argv[1:] or ["CMRx24_Test_P017", "CMRx24_Test_P034"]:
    src = paths.subject_dir("cmrx2024", subj)
    gt = np.stack([np.asarray(nib.load(str(src / "gt" / f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])
    g = torch.from_numpy(gt)
    vol = lv(subj)
    man = json.load(open(paths.manifest("cmrx2024_af", subj)))
    ref = int(man["rhythm"]["ref_plane"]); seed = int(man["seed"])
    roll = int(man["rolled"]["roll_per_plane"][ref])
    rr = rr_sequence("af", N_BEATS, np.random.default_rng([seed, ref]))
    t0 = rr[:BURN].sum() + ((-roll) % T) / T * rr[BURN]
    z = gt.shape[1] // 2
    vmax = float(np.percentile(gt, 99.5))

    POS = {}
    for st in STRIDES:
        POS[st] = pos_physio(t0 + np.arange(T) * DT0 * st, rr, vol)[0]

    frames = []
    for k in range(T):
        c = Image.new("RGB", (CELL * len(STRIDES), CELL + TRACE + 18), "white")
        for i, st in enumerate(STRIDES):
            v = render_phase(g, float(POS[st][k]), True)[z].numpy()
            a = np.clip(v / vmax * 255, 0, 255).astype(np.uint8)
            c.paste(Image.fromarray(a.T[::-1]).resize((CELL, CELL), Image.BILINEAR).convert("RGB"),
                    (i * CELL, 0))
            c.paste(panel(vol, POS[st], CELL, k, f"x{st}  ({T*st*DT0:.1f}s span)"),
                    (i * CELL, CELL + 6))
        d = ImageDraw.Draw(c)
        d.text((5, CELL + TRACE + 8),
               f"{subj}   same 12 frames, wider spacing ->   frame {k}", fill=(30, 30, 30))
        frames.append(c)
    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, f"{subj}_stride.gif")
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=240, loop=0)
    steps = {st: float(np.mean(np.minimum(np.abs(np.diff(POS[st] % T)),
                                          T - np.abs(np.diff(POS[st] % T))))) for st in STRIDES}
    print(f"-> {p}   mean phase step per stride: " +
          "  ".join(f"x{st}={steps[st]:.1f}" for st in STRIDES))
