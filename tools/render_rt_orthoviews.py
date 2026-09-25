#!/usr/bin/env python
"""Orthogonal-view GIF of a real-time (RT) 4D recon vs the raw RT stack (docs/87 outputs).

Columns: SAX (at the LV-centre plane) | horizontal pseudo-LAX (x-z cut) | vertical pseudo-LAX
(y-z cut), all through the LV centre. The "LAX" cuts are orthogonal to the SAX stack, not true
2ch/4ch planes. z is nearest-neighbour stretched by dz/1.4 so each slab shows at physical size.

Rows: raw RT stack (plane z at ITS OWN frame k — planes were acquired at different times, so this
is what naive slice-stacking gives) | each recon passed via --recon.

LV centre = centroid of the union-over-time LV label (1) of the RT heart_seg, mapped onto the
canonical grid with the same in-plane resample as run_vggt_rt.build_rt_bundle.

    PYTHONPATH=training:. python tools/render_rt_orthoviews.py \
        --rt-input <subj>/rt_input.nii.gz --seg <RT heart_seg.nii.gz> \
        --recon vggt_diff1000=<subj>/vggt_final518_diff1000_ep300/recon_rt.nii.gz --out <dir>
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
import run_vggt_rt as rt  # noqa: E402

HALF = 64          # crop half-width (px, 1.4 mm) around the LV centre
UP = 2             # display upsample


def lv_centre(seg_path):
    """(x, y, z) of the LV centre on the canonical (X, Y, Z) grid."""
    s = np.asarray(nib.load(seg_path).dataobj)                 # (X, Y, Z, T) native RT grid
    lv = (s == 1).any(axis=3).astype(np.float32)               # (X, Y, Z)
    area = lv.sum(axis=(0, 1))
    z = int(np.argmax(area))
    can = rt.to_canonical_inplane(lv[:, :, z].T)               # (H, W) = (Y, X), same as the bundle
    ys, xs = np.nonzero(can > 0.5)
    return int(round(xs.mean())), int(round(ys.mean())), z


def views(vol, c, zs, sax_z):
    """vol (X, Y, Z) -> one SAX per plane in sax_z + two through-plane cuts, cropped around
    c=(x, y, z)."""
    x, y, _ = c
    xa, xb, ya, yb = x - HALF, x + HALF, y - HALF, y + HALF
    sax = [vol[xa:xb, ya:yb, z].T for z in sax_z]              # rows y, cols x
    hlax = np.repeat(vol[xa:xb, y, :].T[::-1], zs, axis=0)     # rows z (base up), cols x
    vlax = np.repeat(vol[x, ya:yb, :].T[::-1], zs, axis=0)     # rows z (base up), cols y
    return sax + [hlax, vlax]


def tile(imgs, vmax):
    """Concatenate one row of views with 4 px gaps, scale to uint8."""
    h = max(i.shape[0] for i in imgs)
    cols = []
    for i in imgs:
        pad = np.zeros((h, i.shape[1]), np.float32)
        pad[(h - i.shape[0]) // 2:(h - i.shape[0]) // 2 + i.shape[0]] = i
        cols += [pad, np.zeros((h, 4), np.float32)]
    row = np.concatenate(cols[:-1], axis=1)
    return (np.clip(row / vmax, 0, 1) ** 0.8 * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rt-input", required=True)
    ap.add_argument("--seg", required=True, help="RT heart_seg.nii.gz (native RT grid)")
    ap.add_argument("--recon", nargs="+", required=True, help="name=path, one per recon row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sax-z", type=int, nargs="+", default=None,
                    help="SAX planes to show (default: the LV-centre plane)")
    ap.add_argument("--ref-z", type=int, default=None, help="reference plane, labelled in titles")
    ap.add_argument("--input-frames", default=None,
                    help="json {ref_z, frame_of_z}: top row = the model input naively stacked "
                         "instead of the raw RT stack")
    ap.add_argument("--ms-per-frame", type=int, default=50, help="GIF delay; 25 ms = true rate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw_img = nib.load(args.rt_input)
    dz = float(raw_img.header.get_zooms()[2])
    zs = int(round(dz / rt.rv.INPLANE_MM))
    raw = np.asarray(raw_img.dataobj, np.float32)
    if args.input_frames:
        # The model's actual input, naively stacked: every plane frozen at its one fixed frame,
        # the reference plane at frame t. No motion correction.
        sf = json.load(open(args.input_frames))
        fz = [sf["frame_of_z"][str(z)] for z in range(raw.shape[2])]
        stacked = np.repeat(raw[:, :, np.arange(raw.shape[2]), fz][..., None], raw.shape[3], axis=3)
        stacked[:, :, sf["ref_z"], :] = raw[:, :, sf["ref_z"], :]
        rows = [("just stacked (model input, no correction)", stacked)]
    else:
        rows = [("raw RT stack (planes not simultaneous)", raw)]
    for spec in args.recon:
        name, path = spec.split("=", 1)
        rows.append((name, np.asarray(nib.load(path).dataobj, np.float32)))
    T = min(r[1].shape[3] for r in rows)
    c = lv_centre(args.seg)
    sax_z = args.sax_z if args.sax_z is not None else [c[2]]
    titles = [f"SAX z{z}" + (" (reference)" if z == args.ref_z else "") for z in sax_z]
    titles += ["x-z cut, base at top", "y-z cut, base at top"]
    print(f"LV centre (x, y, z) = {c}, dz={dz} mm -> z stretch x{zs}, T={T}")
    vmax = float(np.percentile(rows[0][1][..., ::10][rows[0][1][..., ::10] > 0], 99.5))

    frames = []
    for t in range(T):
        strips = []
        for name, v in rows:
            strip = tile(views(v[..., t], c, zs, sax_z), vmax)
            lab = np.zeros((14, strip.shape[1]), np.uint8)
            strips += [lab, strip]
        strip_h = strips[1].shape[0]
        strips.insert(0, np.zeros((14, strips[1].shape[1]), np.uint8))       # column-title strip
        img = Image.fromarray(np.concatenate(strips, axis=0))
        img = img.resize((img.width * UP, img.height * UP), Image.NEAREST).convert("RGB")
        d = ImageDraw.Draw(img)
        for k, title in enumerate(titles):
            d.text((k * (2 * HALF + 4) * UP + 4, 4), title, fill=(120, 200, 255))
        y = 14
        for name, v in rows:
            d.text((4, y * UP + 2), name, fill=(255, 220, 0))
            y += 14 + strip_h
        d.text((img.width - 150, 30), f"frame {t:3d}  t={t * 25:4d} ms", fill=(255, 220, 0))
        frames.append(img)

    heads = " | ".join(titles)
    gif = os.path.join(args.out, "orthoviews.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=args.ms_per_frame, loop=0)
    for t in (0, T // 3, 2 * T // 3):
        frames[t].save(os.path.join(args.out, f"orthoviews_f{t:03d}.png"))
    print(f"{heads}\n-> {gif}")


if __name__ == "__main__":
    main()
