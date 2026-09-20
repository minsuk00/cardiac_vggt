#!/usr/bin/env python
"""Rhythm demo: regular vs hrv vs af, over SEVERAL beats, plus the 12-frame window the model sees.

Two GIFs per subject, written to temp/af_rule_gifs/:

  <subj>_multibeat.gif  N_SEC of continuous acquisition at the real frame rate, three columns
                        (regular / hrv / af). Under each, LV volume vs TIME with beat boundaries
                        marked and a moving cursor -- so the rhythm itself is visible: even beat
                        spacing for regular, jittered for hrv, jittered + stretched expansion for af.

  <subj>_window12.gif   the 12 frames that actually become the model's input/target for that arm.

Both rows come from the same source cine through the same `render_phase`; only the cardiac
position per frame differs.

Usage: PYTHONPATH=training:. python tools/render_rhythm_demo.py [--sec 5] [SUBJ ...]
"""
import argparse
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
from build_af_bundle import (BURN, DT, N_BEATS, T, rr_sequence, pos_physio,  # noqa: E402
                             pos_compress, render_phase)

ARMS = [("regular", (120, 220, 255)), ("hrv", (160, 255, 160)), ("af", (255, 185, 80))]
CELL = 210
TRACE_H = 116
OUTDIR = os.path.join(ROOT, "temp", "af_rule_gifs")


def lv_curve(subj):
    fs = sorted(glob.glob(str(paths.subject_dir("cmrx2024", subj) / "seg_gt" / "seg_t*.nii.gz")))
    return np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in fs], float)


def positions(arm, rr, vol, times):
    if arm == "af":
        return pos_physio(times, rr, vol)
    return pos_compress(times, rr)


def _dark(c):
    """A colour dark enough to read on white."""
    return tuple(int(v * 0.55) for v in c)


def trace(vol, pos, times, starts, W, k, colour, label, rr):
    """LV volume vs TIME on white: beat boundaries as ticks, each beat's R-R printed in its
    span, cursor on the current frame."""
    img = Image.new("RGB", (W, TRACE_H), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    col = _dark(colour)
    vp = np.r_[vol, vol[0]]
    lo, hi = vp.min(), vp.max()
    t0, t1 = times[0], times[-1]
    X = lambda t: int(4 + (t - t0) / max(t1 - t0, 1e-9) * (W - 8))            # noqa: E731
    Y = lambda v: int(TRACE_H - 16 - (v - lo) / max(hi - lo, 1e-9) * (TRACE_H - 42))  # noqa: E731
    # beat boundaries + the R-R of each beat, printed inside its own span
    for i, s in enumerate(starts[:-1]):
        e = starts[i + 1]
        if e < t0 or s > t1:
            continue
        if t0 <= s <= t1:
            d.line([(X(s), 18), (X(s), TRACE_H - 12)], fill=(170, 170, 170), width=1)
        xm = (X(max(s, t0)) + X(min(e, t1))) // 2
        if X(min(e, t1)) - X(max(s, t0)) > 26:            # only if the span is wide enough to read
            d.text((xm - 11, TRACE_H - 12), f"{e - s:.2f}s", fill=(90, 90, 90), font=font)
    pts = [(X(t), Y(np.interp(p % T, np.arange(T + 1), vp))) for t, p in zip(times, pos)]
    d.line(pts, fill=(60, 60, 60), width=1)
    d.ellipse([pts[k][0] - 4, pts[k][1] - 4, pts[k][0] + 4, pts[k][1] + 4], fill=col)
    v_now = np.interp(pos[k] % T, np.arange(T + 1), vp)
    rr_txt = "R-R 1.00 fixed" if label == "regular" else f"R-R sd {rr.std():.2f}"
    d.text((4, 3), f"{label}   LV {v_now / hi * 100:3.0f}%   {rr_txt}", fill=col, font=font)
    return img


def build(subj, n_sec, fps_ms):
    src = paths.subject_dir("cmrx2024", subj)
    gt = np.stack([np.asarray(nib.load(str(src / "gt" / f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])
    gt_t = torch.from_numpy(gt)
    vol = lv_curve(subj)
    man = json.load(open(paths.manifest("cmrx2024_af", subj)))
    ref = int(man["rhythm"]["ref_plane"]); seed = int(man["seed"])
    roll = man["rolled"]["roll_per_plane"][ref]
    z = gt.shape[1] // 2
    vmax = float(np.percentile(gt, 99.5))
    n_long = int(round(n_sec / DT))

    data = {}
    for arm, col in ARMS:
        rr = (np.ones(N_BEATS) if arm == "regular"
              else rr_sequence(arm, N_BEATS, np.random.default_rng([seed, ref])))
        t0 = rr[:BURN].sum() + ((-roll) % T) / T * rr[BURN]
        starts = np.concatenate([[0.0], np.cumsum(rr)])
        tl = t0 + np.arange(n_long) * DT
        tw = t0 + np.arange(T) * DT
        data[arm] = (rr, starts, tl, positions(arm, rr, vol, tl)[0],
                     tw, positions(arm, rr, vol, tw)[0], col)

    def render(kind, n, key_t, key_p, dur):
        frames = []
        for k in range(n):
            canvas = Image.new("RGB", (CELL * len(ARMS), CELL + TRACE_H + 18), "white")
            for i, (arm, _) in enumerate(ARMS):
                rr, starts, tl, pl, tw, pw, col = data[arm]
                times, pos = (tl, pl) if kind == "multibeat" else (tw, pw)
                v = render_phase(gt_t, float(pos[k]), True)[z].numpy()
                a = np.clip(v / vmax * 255, 0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(a.T[::-1]).resize((CELL, CELL), Image.BILINEAR)
                             .convert("RGB"), (i * CELL, 0))
                canvas.paste(trace(vol, pos, times, starts, CELL, k, col, arm, rr),
                             (i * CELL, CELL + 6))
            d = ImageDraw.Draw(canvas)
            tag = (f"{n*DT:.1f}s continuous  |  vertical ticks = beat boundaries"
                   if kind == "multibeat" else
                   "the 12 frames the model actually receives")
            d.text((4, CELL + TRACE_H + 8), f"{subj}   {tag}   f{k}", fill=(40, 40, 40))
            frames.append(canvas)
        p = os.path.join(OUTDIR, f"{subj}_{kind}.gif")
        frames[0].save(p, save_all=True, append_images=frames[1:], duration=dur, loop=0)
        return p

    return render("multibeat", n_long, None, None, fps_ms), render("window12", T, None, None, 200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*",
                    default=["CMRx24_Test_P004", "CMRx24_Test_P017",
                             "CMRx24_Val_P026", "CMRx24_Test_P023"])
    ap.add_argument("--sec", type=float, default=5.0)
    ap.add_argument("--ms", type=int, default=70)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for s in a.subjects:
        for p in build(s, a.sec, a.ms):
            print(f"-> {p}")
