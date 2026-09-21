#!/usr/bin/env python
"""GIFs of the two pos_physio behaviours flagged in the 2026-09-20 review.

Everything here is rendered from the SOURCE gated cine at the simulated cardiac positions, with
breathing left off, so what moves is cardiac motion and nothing else.

  <subj>_freeze.gif    left  = the stored 12-phase cine, the ground truth this subject's heart
                               actually does (does it really keep contracting to frame 10?)
                       right = the simulated 24-frame acquisition at the worst plane
  <subj>_teleport.gif  left  = shipped, right = the resume-at-actual-phase fix, same plane,
                               same beats -- the only difference is what happens to a beat whose
                               R-R runs out mid-contraction

Usage: PYTHONPATH=training:. python tools/render_af24_bug_gifs.py
"""
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation"),
                os.path.join(ROOT, "tools")]
import paths                                                              # noqa: E402
from build_af_bundle import BURN, DT, N_BEATS, T, rr_sequence, render_phase   # noqa: E402

NF, SIZE, TRACE = 24, 300, 96
OUT = os.path.join(ROOT, "temp", "af24_bug_demo")
TELE = ("cmrx2025", "CMRx25_R1test_Center004_UIH_15T_umr680_P030")
FREEZE = ("cmrx2025", "CMRx25_R2val_Center004_UIH_15T_umr680_P017")


def font(sz=14):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load(ds, subj):
    """(T,D,H,W) source cine + (T,) LV volume curve."""
    gt = np.stack([np.asarray(nib.load(str(paths.subject_dir(ds, subj) / "gt" /
                                           f"gt_t{t:02d}.nii.gz")).dataobj,
                              dtype=np.float32).transpose(2, 1, 0) for t in range(T)])
    f = sorted(glob.glob(str(paths.subject_dir(ds, subj) / "seg_gt" / "seg_t*.nii.gz")))
    vol = np.array([(np.asarray(nib.load(p).dataobj) == 1).sum() for p in f], float)
    return gt, vol


def simulate(vol, es, rr, times, fixed):
    vol_p = np.r_[vol, vol[0]]
    sys_v = np.minimum.accumulate(vol[:es + 1])
    starts = np.concatenate([[0.0], np.cumsum(rr)])
    p_start = np.zeros(len(rr))
    for i in range(len(rr) - 1):
        p0 = p_start[i]
        t_c = max(es - p0, 0.0) * DT
        t_nom = t_c + (T - es) * DT
        cut = rr[i] <= t_c
        p_end = float(T) if rr[i] >= t_nom else (p0 + rr[i] / DT if cut
                                                 else es + (rr[i] - t_c) / DT)
        if fixed and cut:
            p_start[i + 1] = p_end
            continue
        p_end = max(min(p_end, float(T)), float(es))
        vv = np.interp(p_end, np.arange(T + 1), vol_p)
        p_start[i + 1] = (0.0 if vv >= sys_v[0]
                          else np.interp(vv, sys_v[::-1], np.arange(es + 1)[::-1]))
    b = np.searchsorted(starts, times, side="right") - 1
    el = times - starts[b]
    pos = np.empty(len(times))
    for k in range(len(times)):
        i = int(b[k]); p0 = p_start[i]; e = el[k]
        t_c = max(es - p0, 0.0) * DT
        t_nom = t_c + (T - es) * DT
        if e <= t_c:
            pos[k] = p0 + e / DT
        elif rr[i] >= t_nom:
            pos[k] = es + min((e - t_c) / max(rr[i] - t_c, 1e-9), 1.0) * (T - es)
        else:
            pos[k] = min(es + (e - t_c) / DT, float(T))
    return pos, b


def draw(gt, z, p, lo, hi):
    a = render_phase(__import__("torch").from_numpy(gt), float(p), True).numpy()[z]
    x = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((x * 255).astype(np.uint8), "L").convert("RGB") \
        .resize((SIZE, SIZE), Image.BILINEAR)


def trace(dr, x0, y0, y1, w, curve, k, col, marks=()):
    lo, hi = min(curve), max(curve)
    xs = [x0 + i * w / (len(curve) - 1) for i in range(len(curve))]
    ys = [y1 - (v - lo) / max(hi - lo, 1e-9) * (y1 - y0) for v in curve]
    for s, e, c in marks:
        dr.rectangle([xs[s], y0 - 4, xs[e], y1 + 4], fill=c)
    dr.line(list(zip(xs, ys)), fill=col, width=2)
    dr.ellipse([xs[k] - 4, ys[k] - 4, xs[k] + 4, ys[k] + 4], fill=(255, 210, 90))


def panel(img, lines, fnt, curve, k, marks=()):
    out = Image.new("RGB", (SIZE, 44 + SIZE + TRACE), (12, 12, 14))
    out.paste(img, (0, 44))
    dr = ImageDraw.Draw(out)
    for i, (t, c) in enumerate(lines):
        dr.text((7, 4 + 19 * i), t, font=fnt, fill=c)
    trace(dr, 10, 44 + SIZE + 16, 44 + SIZE + TRACE - 22, SIZE - 20, curve, k,
          (110, 175, 60), marks)
    return out


def save(frames, name):
    p = os.path.join(OUT, name)
    frames[0].save(p, save_all=True, append_images=frames[1:], duration=200, loop=0)
    print("wrote", p)


def worst_plane(ds, subj, vol, es, key):
    man = json.load(open(paths.manifest(f"{ds}_af24", subj)))
    roll = man["rolled"]["roll_per_plane"]; seed = int(man["seed"])
    best, out = -1, None
    for z in range(int(man["D"])):
        rng = np.random.default_rng([seed, z])
        rr = rr_sequence("af", N_BEATS, rng)
        t0 = rr[:BURN].sum() + ((-roll[z]) % T) / T * rr[BURN]
        pos, b = simulate(vol, es, rr, t0 + np.arange(NF) * DT, False)
        d = np.diff(pos); same = np.diff(b) == 0
        if key == "jump":
            v = max([d[k] for k in range(len(d)) if not same[k]] or [-1])
        else:
            run = v = 0
            for k in range(len(d)):
                run = run + 1 if (same[k] and abs(d[k]) < 0.25) else 0
                v = max(v, run)
        if v > best:
            best, out = v, (z, rr, t0)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fnt = font()

    # ---- FREEZE: is the stored cine really still contracting at frame 10? ----------------
    ds, subj = FREEZE
    gt, vol = load(ds, subj); es = int(np.argmin(vol))
    z, rr, t0 = worst_plane(ds, subj, vol, es, "static")
    pos, b = simulate(vol, es, rr, t0 + np.arange(NF) * DT, False)
    s = gt[:, z][gt[:, z] > 0]
    lo, hi = 0.0, float(np.percentile(s, 99.5))
    src_curve = list(vol / vol.max())
    sim_curve = list(np.interp(pos % T, np.arange(T + 1), np.r_[vol, vol[0]]) / vol.max())
    d = np.diff(pos); same = np.diff(b) == 0
    runs = []; run = st = 0
    for k in range(len(d)):
        if same[k] and abs(d[k]) < 0.25:
            st = k if run == 0 else st
            run += 1
        else:
            if run >= 6:
                runs.append((st, k, (70, 30, 30)))
            run = 0
    if run >= 6:
        runs.append((st, NF - 1, (70, 30, 30)))
    frames = []
    for f in range(NF):
        left = panel(draw(gt, z, f % T, lo, hi),
                     [("SOURCE cine (12 phases, looped)", (235, 235, 240)),
                      (f"stored phase {f % T}   ES={es}", (255, 210, 90))], fnt,
                     src_curve, f % T)
        right = panel(draw(gt, z, pos[f], lo, hi),
                      [("SIMULATED 24-frame acquisition", (235, 235, 240)),
                       (f"frame {f}  cardiac pos {pos[f] % T:5.2f}", (250, 130, 90))], fnt,
                      sim_curve, f, runs)
        c = Image.new("RGB", (SIZE * 2 + 8, left.height), (12, 12, 14))
        c.paste(left, (0, 0)); c.paste(right, (SIZE + 8, 0))
        frames.append(c)
    save(frames, f"{subj[:28]}_freeze.gif")

    # ---- TELEPORT: shipped vs fixed, same beats ------------------------------------------
    ds, subj = TELE
    gt, vol = load(ds, subj); es = int(np.argmin(vol))
    z, rr, t0 = worst_plane(ds, subj, vol, es, "jump")
    times = t0 + np.arange(NF) * DT
    pb, b = simulate(vol, es, rr, times, False)
    pf, _ = simulate(vol, es, rr, times, True)
    s = gt[:, z][gt[:, z] > 0]
    lo, hi = 0.0, float(np.percentile(s, 99.5))
    vp = np.r_[vol, vol[0]]
    cb = list(np.interp(pb % T, np.arange(T + 1), vp) / vol.max())
    cf = list(np.interp(pf % T, np.arange(T + 1), vp) / vol.max())
    jump = [k for k in range(NF - 1) if b[k] != b[k + 1] and pb[k + 1] - pb[k] > 1.5]
    frames = []
    for f in range(NF):
        hit = any(f == k + 1 for k in jump)
        left = panel(draw(gt, z, pb[f], lo, hi),
                     [("SHIPPED", (250, 130, 90)),
                      (f"frame {f}  pos {pb[f] % T:5.2f}" + ("   <- JUMPED" if hit else ""),
                       (250, 130, 90) if hit else (200, 200, 205))], fnt, cb, f)
        right = panel(draw(gt, z, pf[f], lo, hi),
                      [("FIXED", (110, 200, 120)),
                       (f"frame {f}  pos {pf[f] % T:5.2f}", (200, 200, 205))], fnt, cf, f)
        c = Image.new("RGB", (SIZE * 2 + 8, left.height), (12, 12, 14))
        c.paste(left, (0, 0)); c.paste(right, (SIZE + 8, 0))
        frames.append(c)
    save(frames, f"{subj[:28]}_teleport.gif")


if __name__ == "__main__":
    main()
