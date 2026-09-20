#!/usr/bin/env python
"""One GIF per subject: the REGULAR cine next to the AF cine under the current rhythm rule,
with the LV volume curve underneath and a marker on the frame being shown.

Both rows are rendered from the same source cine through the same `render_phase`, so the only
difference is the cardiac position each frame is sampled at:
  regular : positions 0..11, one stored phase per frame
  af      : positions from `pos_physio` -- contraction at the stored rate, expansion stretched to
            end exactly at R-R for a long beat, cut off partway for a short one

Usage: PYTHONPATH=training:. python tools/render_af_rule_gif.py [SUBJ ...]
"""
import glob
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
from build_af_bundle import BURN, DT, N_BEATS, T, rr_sequence, pos_physio, render_phase  # noqa: E402

CELL = 230
CURVE_H = 104
SUBJS = sys.argv[1:] or ["CMRx24_Test_P004", "CMRx24_Test_P017", "CMRx24_Val_P026"]


def lv_curve(subj):
    fs = sorted(glob.glob(str(paths.subject_dir("cmrx2024", subj) / "seg_gt" / "seg_t*.nii.gz")))
    return np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in fs], float)


def curve_panel(vol, pos, W, marker, label, colour):
    """LV volume along the cine, with a dot at each frame's sampled position."""
    img = Image.new("RGB", (W, CURVE_H), "black")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    vp = np.r_[vol, vol[0]]
    lo, hi = vp.min(), vp.max()
    x = lambda p: int(4 + (p / T) * (W - 8))                              # noqa: E731
    y = lambda v: int(CURVE_H - 20 - (v - lo) / max(hi - lo, 1e-9) * (CURVE_H - 34))  # noqa: E731
    pts = [(x(p), y(np.interp(p, np.arange(T + 1), vp))) for p in np.linspace(0, T, 200)]
    d.line(pts, fill=(70, 70, 70), width=1)
    for f, p in enumerate(pos):
        px, py = x(p % T), y(np.interp(p % T, np.arange(T + 1), vp))
        r = 4 if f == marker else 2
        d.ellipse([px - r, py - r, px + r, py + r],
                  fill=colour if f == marker else (110, 110, 110))
    v_now = np.interp(pos[marker] % T, np.arange(T + 1), vp)
    d.text((4, 2), f"{label}   phase {pos[marker] % T:5.2f}   LV {v_now / hi * 100:3.0f}%",
           fill=colour, font=font)
    return img


for subj in SUBJS:
    src = paths.subject_dir("cmrx2024", subj)
    gt = np.stack([np.asarray(nib.load(str(src / "gt" / f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])  # (T,D,H,W)
    gt_t = torch.from_numpy(gt)
    vol = lv_curve(subj)
    man = __import__("json").load(open(paths.manifest("cmrx2024_af", subj)))
    ref = int(man["rhythm"]["ref_plane"]); seed = int(man["seed"])
    roll = man["rolled"]["roll_per_plane"][ref]

    rr = rr_sequence("af", N_BEATS, np.random.default_rng([seed, ref]))
    t0 = rr[:BURN].sum() + ((-roll) % T) / T * rr[BURN]
    pos_af, _ = pos_physio(t0 + np.arange(T) * DT, rr, vol)
    pos_reg = (np.arange(T) + (-roll) % T) % T          # regular: one stored phase per frame

    z = gt.shape[1] // 2
    vmax = float(np.percentile(gt, 99.5))
    frames = []
    for f in range(T):
        row = Image.new("RGB", (CELL * 2, CELL + CURVE_H + 20), "black")
        for i, (p, lab, col) in enumerate(((pos_reg[f], "REGULAR", (120, 220, 255)),
                                           (pos_af[f], "AF (new rule)", (255, 190, 90)))):
            v = render_phase(gt_t, float(p), True)[z].numpy()
            a = np.clip(v / vmax * 255, 0, 255).astype(np.uint8)
            row.paste(Image.fromarray(a.T[::-1]).resize((CELL, CELL), Image.BILINEAR).convert("RGB"),
                      (i * CELL, 0))
            row.paste(curve_panel(vol, pos_reg if i == 0 else pos_af, CELL, f, lab, col),
                      (i * CELL, CELL + 8))
        d = ImageDraw.Draw(row)
        d.text((4, CELL + CURVE_H + 10), f"{subj}   frame {f}/{T}   beat R-R ~ N(1, 0.25)s",
               fill="white")
        frames.append(row)
    out = os.path.join(ROOT, "figs", f"af_rule_{subj}.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=260, loop=0)
    span = float(np.ptp(np.unwrap(pos_af * 2 * np.pi / T) * T / (2 * np.pi)))
    print(f"-> {out}   af positions {np.round(pos_af, 2)}  span {span:.2f} phases")
