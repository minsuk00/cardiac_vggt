#!/usr/bin/env python
"""Demonstrate a BUILT 24-frame (`*24`) bundle, straight off disk.

Unlike tools/render_af24_demo.py -- which re-simulates positions to argue the design before it
existed -- this reads the bundle the builder actually wrote, so what you see is what the engines
will consume. Three outputs per subject in temp/af24_builder_demo/:

  <subj>_acq_<arm>.gif    the 24-frame acquisition, EVERY z-plane in a grid, animated over f.
                          This is the whole input Fetal CMR 4D receives. Header prints the
                          reference plane's cardiac position, so a hold/cut-off is visible.

  <subj>_gt_<arm>.gif     the 24 GT volumes (4 z-planes spread over the stack) with the LV volume
                          curve and a cursor. GT is a REAL-TIME 2-beat cine, not one averaged cycle.

  <subj>_binpairs.png     THE POINT, per arm. The gate's ramp wraps twice (modulus n_cardphase=12),
                          so frames k and k+12 land in the SAME output bin and get averaged. Each
                          row is one bin: frame k | frame k+12 | |difference|. Under `regular` the
                          two are one beat apart = the same cardiac phase, so the difference is
                          breathing only. Under `af` the rhythm has drifted and they are different
                          phases -- that difference is the temporal blur the baseline eats.

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
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation"),
                os.path.join(ROOT, "tools")]

ARMS = ("regular24", "hrv24", "af24")
OUTDIR = os.path.join(ROOT, "temp", "af24_builder_demo")
CELL = 150          # per-plane tile size in the acquisition grid
BIG = 200           # per-panel size in the GT gif / bin-pair sheet
TRACE_H = 110


def font(sz=13):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load_stack(d, sub, f):
    """One bundle volume as (Z, Y, X) display order."""
    p = os.path.join(d, sub, f"{'gt' if sub == 'gt' else 'stack'}_t{f:02d}.nii.gz")
    return np.asarray(nib.load(p).dataobj, dtype=np.float32).transpose(2, 1, 0)


def to_img(a, lo, hi, size):
    """One 2-D plane -> an 8-bit PIL image at `size`, using a FIXED window so brightness is
    comparable across frames and planes (a per-frame window would hide the very intensity
    differences these panels exist to show)."""
    x = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    im = Image.fromarray((x * 255).astype(np.uint8), mode="L").convert("RGB")
    return im.resize((size, size), Image.BILINEAR)


def window(d, sub, nf):
    """(lo, hi) intensity window from a few frames of this arm."""
    s = np.concatenate([load_stack(d, sub, f).ravel() for f in (0, nf // 3, 2 * nf // 3)])
    s = s[s > 0]
    return (0.0, float(np.percentile(s, 99.5))) if s.size else (0.0, 1.0)


def lv_curve(src_dir, n_card):
    """(n_card,) whole-LV voxel count per stored phase, from the SOURCE cohort's seg cache.

    Read from the SOURCE, never from the arm's own seg_gt/: on an integral arm that directory is a
    gt_map-PERMUTED copy, so using it would plot a permuted curve against unpermuted positions.
    """
    segs = sorted(glob.glob(os.path.join(src_dir, "seg_gt", "seg_t*.nii.gz")))
    if len(segs) != n_card:
        return None
    v = np.array([(np.asarray(nib.load(f).dataobj) == 1).sum() for f in segs], float)
    return v if v.max() > 0 else None


def grid(planes, cols):
    """Tile a list of same-size PIL images into `cols` columns."""
    rows = (len(planes) + cols - 1) // cols
    w, h = planes[0].size
    out = Image.new("RGB", (cols * w, rows * h), (12, 12, 14))
    for i, im in enumerate(planes):
        out.paste(im, ((i % cols) * w, (i // cols) * h))
    return out


def header(im, lines, fnt):
    """Prepend a text band to an image."""
    pad = 6 + 17 * len(lines)
    out = Image.new("RGB", (im.width, im.height + pad), (12, 12, 14))
    out.paste(im, (0, pad))
    dr = ImageDraw.Draw(out)
    for i, (txt, col) in enumerate(lines):
        dr.text((7, 3 + 17 * i), txt, fill=col, font=fnt)
    return out


# ───────────────────────── panels ─────────────────────────
def acq_gif(d, subj, arm, man, out):
    """Every z-plane of the 24-frame acquisition, animated over f."""
    nf, ref = int(man["T"]), int(man["scatter"]["ref_plane"])
    ncp = int(man["n_cardphase"])
    pos = np.asarray(man["rhythm"]["pos_per_plane"], float)          # (D, nf)
    lo, hi = window(d, "breath", nf)
    D = pos.shape[0]
    cols = min(6, D)
    fnt = font()
    frames = []
    for f in range(nf):
        vol = load_stack(d, "breath", f)
        tiles = []
        for z in range(D):
            im = to_img(vol[z], lo, hi, CELL)
            dr = ImageDraw.Draw(im)
            mark = z == ref
            dr.text((4, 3), f"z{z}" + ("  REF" if mark else ""), font=fnt,
                    fill=(255, 210, 90) if mark else (150, 150, 155))
            # each plane's OWN cardiac position at this frame -- the scatter is per plane
            dr.text((4, CELL - 18), f"p={pos[z, f] % ncp:5.2f}", font=fnt, fill=(120, 190, 250))
            if mark:
                dr.rectangle([0, 0, CELL - 1, CELL - 1], outline=(255, 210, 90), width=2)
            tiles.append(im)
        beat = f // ncp
        frames.append(header(grid(tiles, cols), [
            (f"{subj}   {arm}   24-frame acquisition (what Fetal CMR 4D receives)",
             (235, 235, 240)),
            (f"frame {f:2d}/{nf}   beat {beat}   ref-plane cardiac position "
             f"{pos[ref, f] % ncp:5.2f} / {ncp}", (255, 210, 90))], fnt))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=170, loop=0)


def gt_gif(d, subj, arm, man, src_dir, out):
    """The 24 GT volumes at 4 z-planes, with the reference plane's LV curve and a cursor."""
    nf, ref = int(man["T"]), int(man["scatter"]["ref_plane"])
    ncp = int(man["n_cardphase"])
    pos = np.asarray(man["rhythm"]["ref_pos"], float)
    vol_c = lv_curve(src_dir, ncp)
    lo, hi = window(d, "gt", nf)
    D = int(man["D"])
    zs = sorted(set(np.linspace(1, D - 2, 4).round().astype(int).tolist()))
    fnt = font()
    lv = None
    if vol_c is not None:
        lv = np.interp(pos % ncp, np.arange(ncp + 1), np.r_[vol_c, vol_c[0]])
    W = BIG * len(zs)
    frames = []
    for f in range(nf):
        v = load_stack(d, "gt", f)
        tiles = []
        for z in zs:
            im = to_img(v[z], lo, hi, BIG)
            ImageDraw.Draw(im).text((5, 4), f"z{z}", font=fnt, fill=(150, 150, 155))
            tiles.append(im)
        row = grid(tiles, len(zs))
        panel = Image.new("RGB", (W, row.height + TRACE_H), (12, 12, 14))
        panel.paste(row, (0, 0))
        if lv is not None:
            dr = ImageDraw.Draw(panel)
            y0, y1 = row.height + 14, row.height + TRACE_H - 20
            rng = max(lv.max() - lv.min(), 1e-6)
            pts = [(6 + i * (W - 12) / (nf - 1), y1 - (lv[i] - lv.min()) / rng * (y1 - y0))
                   for i in range(nf)]
            dr.line(pts, fill=(110, 175, 60), width=2)
            for b in range(1, (nf + ncp - 1) // ncp):          # nominal beat boundaries
                x = 6 + (b * ncp) * (W - 12) / (nf - 1)
                dr.line([(x, y0 - 6), (x, y1 + 4)], fill=(90, 90, 96), width=1)
            dr.ellipse([pts[f][0] - 4, pts[f][1] - 4, pts[f][0] + 4, pts[f][1] + 4],
                       fill=(255, 210, 90))
            dr.text((6, y1 + 5), "LV volume at the reference plane's cardiac position "
                                 "(grey = nominal beat boundary)", font=fnt, fill=(150, 150, 155))
        frames.append(header(panel, [
            (f"{subj}   {arm}   GT: {nf} volumes (real-time 2-beat cine, not one averaged cycle)",
             (235, 235, 240)),
            (f"gt_t{f:02d}   ref-plane cardiac position {pos[f] % ncp:5.2f} / {ncp}",
             (255, 210, 90))], fnt))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=170, loop=0)


def binpairs_png(bundles, subj, out):
    """Per arm: the two frames that share each output bin, and their difference."""
    fnt = font()
    cols = []
    for arm, (d, man) in bundles.items():
        nf, ref = int(man["T"]), int(man["scatter"]["ref_plane"])
        ncp = int(man["n_cardphase"])
        pos = np.asarray(man["rhythm"]["pos_per_plane"], float)[ref]
        lo, hi = window(d, "breath", nf)
        S = 120
        tiles, gaps = [], []
        for k in range(ncp):
            a = load_stack(d, "breath", k)[ref]
            b = load_stack(d, "breath", k + ncp)[ref]
            # circular phase gap, in stored phases, between the two frames sharing bin k
            g = abs(((pos[k] - pos[k + ncp]) % ncp + ncp / 2) % ncp - ncp / 2)
            gaps.append(g)
            row = Image.new("RGB", (3 * S, S), (12, 12, 14))
            row.paste(to_img(a, lo, hi, S), (0, 0))
            row.paste(to_img(b, lo, hi, S), (S, 0))
            row.paste(to_img(np.abs(a - b), lo, hi * 0.6, S), (2 * S, 0))
            dr = ImageDraw.Draw(row)
            dr.text((4, 3), f"bin {k:2d}: f{k} | f{k + ncp} | diff", font=fnt,
                    fill=(200, 200, 205))
            dr.text((2 * S + 4, S - 18), f"gap {g:4.2f} ph", font=fnt,
                    fill=(250, 130, 90) if g > 0.5 else (110, 175, 60))
            tiles.append(row)
        col = Image.new("RGB", (3 * S, ncp * S), (12, 12, 14))
        for i, t in enumerate(tiles):
            col.paste(t, (0, i * S))
        cols.append(header(col, [(f"{arm}", (235, 235, 240)),
                                 (f"mean phase gap {np.mean(gaps):4.2f} of {ncp}",
                                  (250, 130, 90) if np.mean(gaps) > 0.5 else (110, 175, 60))], fnt))
    W = sum(c.width for c in cols)
    sheet = Image.new("RGB", (W, max(c.height for c in cols) + 22), (12, 12, 14))
    x = 0
    for c in cols:
        sheet.paste(c, (x, 22)); x += c.width
    ImageDraw.Draw(sheet).text(
        (7, 4), f"{subj}  —  frames sharing one output bin (gate ramp wraps twice, modulus 12). "
                f"Fetal CMR 4D averages each pair.", font=fnt, fill=(235, 235, 240))
    sheet.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=os.path.join(ROOT, "scratch/eval"))
    ap.add_argument("--source", required=True)
    ap.add_argument("--subjects", required=True, help="comma-separated")
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    for subj in [s.strip() for s in a.subjects.split(",") if s.strip()]:
        src_dir = os.path.join(a.eval_root, a.source, "out", subj)
        bundles = {}
        for arm in ARMS:
            d = os.path.join(a.eval_root, f"{a.source}_{arm}", "out", subj)
            if not os.path.isfile(os.path.join(d, "manifest.json")):
                print(f"  skip {subj} {arm}: not built")
                continue
            man = json.load(open(os.path.join(d, "manifest.json")))
            bundles[arm] = (d, man)
            for tag, fn in (("acq", acq_gif), ("gt", gt_gif)):
                p = os.path.join(OUTDIR, f"{subj}_{tag}_{arm}.gif")
                fn(d, subj, arm, man, p) if tag == "acq" else fn(d, subj, arm, man, src_dir, p)
                print(f"  wrote {p}", flush=True)
        if bundles:
            p = os.path.join(OUTDIR, f"{subj}_binpairs.png")
            binpairs_png(bundles, subj, p)
            print(f"  wrote {p}", flush=True)


if __name__ == "__main__":
    main()
