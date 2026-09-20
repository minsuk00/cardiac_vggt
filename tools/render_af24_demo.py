#!/usr/bin/env python
"""What 24 frames/slice at normal spacing would give the baseline, and why it hurts under AF.

Per subject, two GIFs in temp/af24_gifs/:

  <subj>_cine24.gif   the 24-frame acquisition (2.0 s, ~2 beats) as a cine, LV volume underneath
                      with beat boundaries -- regular vs af side by side.

  <subj>_bins.gif     THE POINT. The gate's ramp wraps twice, so frames f and f+12 land in the
                      SAME output bin. Left = frame f, right = frame f+12, and the engine averages
                      them. Under `regular` they are one beat apart = the same cardiac phase, so
                      averaging is pure noise reduction. Under `af` the rhythm has drifted, so they
                      are DIFFERENT phases and averaging is temporal blur. The phase gap is printed.

Usage: PYTHONPATH=training:. python tools/render_af24_demo.py [SUBJ ...]
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
from build_af_bundle import BURN, N_BEATS, T, rr_sequence, pos_physio, pos_compress, render_phase  # noqa: E402

NF = 24
DT0 = 1.0 / 12
CELL = 210
TRACE = 104
OUTDIR = os.path.join(ROOT, "temp", "af24_gifs")
ARMS = [("regular", (110, 175, 60)), ("hrv", (40, 130, 200)), ("af", (210, 110, 30))]


def lv(subj):
    fs = sorted(glob.glob(str(paths.subject_dir("cmrx2024", subj) / "seg_gt" / "seg_t*.nii.gz")))
    return np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in fs], float)


def font_():
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def trace(vol, pos, starts, times, W, k, col, label):
    img = Image.new("RGB", (W, TRACE), "white")
    d = ImageDraw.Draw(img); f = font_()
    vp = np.r_[vol, vol[0]]; lo, hi = vp.min(), vp.max()
    t0, t1 = times[0], times[-1]
    X = lambda t: int(5 + (t - t0) / max(t1 - t0, 1e-9) * (W - 10))              # noqa: E731
    Y = lambda v: int(TRACE - 16 - (v - lo) / max(hi - lo, 1e-9) * (TRACE - 42))  # noqa: E731
    for i, s in enumerate(starts[:-1]):
        e = starts[i + 1]
        if e < t0 or s > t1:
            continue
        if t0 <= s <= t1:
            d.line([(X(s), 18), (X(s), TRACE - 12)], fill=(175, 175, 175), width=1)
        if X(min(e, t1)) - X(max(s, t0)) > 26:
            d.text(((X(max(s, t0)) + X(min(e, t1))) // 2 - 11, TRACE - 12),
                   f"{e - s:.2f}s", fill=(95, 95, 95), font=f)
    pts = [(X(t), Y(np.interp(p % T, np.arange(T + 1), vp))) for t, p in zip(times, pos)]
    d.line(pts, fill=(60, 60, 60), width=1)
    d.ellipse([pts[k][0] - 4, pts[k][1] - 4, pts[k][0] + 4, pts[k][1] + 4], fill=col)
    d.text((5, 3), f"{label}   24 frames / 2.0 s", fill=col, font=f)
    return img


for subj in sys.argv[1:] or ["CMRx24_Test_P017", "CMRx24_Test_P034"]:
    src = paths.subject_dir("cmrx2024", subj)
    gt = np.stack([np.asarray(nib.load(str(src / "gt" / f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])
    g = torch.from_numpy(gt); vol = lv(subj)
    man = json.load(open(paths.manifest("cmrx2024_af", subj)))
    ref = int(man["rhythm"]["ref_plane"]); seed = int(man["seed"])
    roll = int(man["rolled"]["roll_per_plane"][ref])
    z = gt.shape[1] // 2
    vmax = float(np.percentile(gt, 99.5))

    D = {}
    for arm, col in ARMS:
        rr = (np.ones(N_BEATS) if arm == "regular"
              else rr_sequence(arm, N_BEATS, np.random.default_rng([seed, ref])))
        t0 = rr[:BURN].sum() + ((-roll) % T) / T * rr[BURN]
        tm = t0 + np.arange(NF) * DT0
        pos = (pos_physio(tm, rr, vol)[0] if arm == "af" else pos_compress(tm, rr)[0])
        D[arm] = (rr, np.concatenate([[0.0], np.cumsum(rr)]), tm, pos, col)
    os.makedirs(OUTDIR, exist_ok=True)

    # ---- 1. the 24-frame cine ----
    frames = []
    for k in range(NF):
        c = Image.new("RGB", (CELL * len(ARMS), CELL + TRACE + 18), "white")
        for i, (arm, col) in enumerate(ARMS):
            rr, st, tm, pos, _ = D[arm]
            v = render_phase(g, float(pos[k]), True)[z].numpy()
            a = np.clip(v / vmax * 255, 0, 255).astype(np.uint8)
            c.paste(Image.fromarray(a.T[::-1]).resize((CELL, CELL), Image.BILINEAR).convert("RGB"),
                    (i * CELL, 0))
            c.paste(trace(vol, pos, st, tm, CELL, k, col, arm), (i * CELL, CELL + 6))
        ImageDraw.Draw(c).text((5, CELL + TRACE + 8), f"{subj}   frame {k}/{NF}",
                               fill=(30, 30, 30))
        frames.append(c)
    p1 = os.path.join(OUTDIR, f"{subj}_cine24.gif")
    frames[0].save(p1, save_all=True, append_images=frames[1:], duration=130, loop=0)

    gaps = {a: float(np.mean([abs((D[a][3][k] % T - D[a][3][k + T] % T + T / 2) % T - T / 2)
                              for k in range(T)])) for a, _ in ARMS}
    print(f"-> {p1}   mean phase gap between the two frames sharing a bin: " +
          "  ".join(f"{a}={v:.2f}/12" for a, v in gaps.items()))
