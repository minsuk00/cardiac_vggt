#!/usr/bin/env python
"""LAX (long-axis) views of every arm's reconstruction -- is fetal_cmr_4d steppy through-plane?

Renders the two long-axis planes (XZ and YZ through the heart centroid) at TRUE physical aspect,
with NEAREST display sampling so through-plane stepping is shown honestly rather than smoothed
away by the viewer.

TWO SPACES, because they answer different questions and only showing one is misleading:

  scored : every arm resampled onto GT's canonical grid (in-plane 1.4 mm, z = the subject's native
           pitch, 12 mm on cmrx2024) through image_metrics' own load_blurred -> fit_rigid ->
           resample. This is EXACTLY what the metrics see, so it is the fair comparison -- but it
           forces every arm onto a 12 mm z grid, which HIDES what each engine natively produces.

  native : each arm's own reconstruction grid, untouched. fetal_cmr_4d reconstructs at 1.4 mm
           ISOTROPIC (run_fetal4d.sh RES=1.4), so it has fine z sampling of its own; VGGT splats
           onto the canonical grid, so its z sampling IS the 12 mm pitch. Whether fetal's fine z
           grid carries real through-plane detail or just interpolates between 12 mm-pitch inputs
           is the actual question, and only this view can show it.

Usage:
    PYTHONPATH=training:. python tools/render_lax_comparison.py [SUBJECT] [--cohort C] [--phases 0,6]
"""
import argparse
import json
import os
import sys

import numpy as np
import nibabel as nib
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "score"))
import paths            # noqa: E402
import pose_psf         # noqa: E402
import image_metrics as im   # noqa: E402

T = 12
ARMS = [("fetal_cmr_4d", "fetal self-gated"),
        ("fetal_cmr_4d_oracle_balanced", "fetal oracle-BAL"),
        ("vggt_final518_diff1000_ep300", "VGGT")]


def to_img(plane, zoom_y, cell_w, vmax):
    """(H, Zsmall) -> a uint8 image stretched to true physical aspect, NEAREST (no smoothing)."""
    a = np.clip(plane / max(vmax, 1e-8) * 255, 0, 255).astype(np.uint8)
    pim = Image.fromarray(a.T[::-1])                      # rows = z (flip so apex is at bottom)
    w, h = pim.size
    return pim.resize((cell_w, max(1, int(round(h * zoom_y * cell_w / w)))), Image.NEAREST)


def panel(cols, titles, cell_w, note):
    hmax = max(c.height for c in cols)
    out = Image.new("RGB", (cell_w * len(cols), hmax + 34), "black")
    d = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for i, (c, t) in enumerate(zip(cols, titles)):
        out.paste(c.convert("RGB"), (i * cell_w, 30))
        d.text((i * cell_w + 4, 2), t[:26], fill="white", font=font)
    d.text((4, hmax + 32 - 12), note, fill="yellow", font=font)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", nargs="?", default="CMRx24_Test_P017")
    ap.add_argument("--cohort", default="cmrx2024_af")
    ap.add_argument("--phases", default="0,6", help="comma-separated frame indices to render")
    ap.add_argument("--cell", type=int, default=190)
    args = ap.parse_args()
    S, C = args.subject, args.cohort
    phases = [int(x) for x in args.phases.split(",")]

    shape_xyz, aff = im.subject_grid(C, S)
    inplane = float(abs(aff[0, 0]))
    dz = float(abs(aff[2, 2]))
    man = json.load(open(paths.manifest(C, S)))
    content = im.load_canon(str(paths.fov_mask(C, S)), shape_xyz, aff) > 0.5
    heart = im.load_canon(str(paths.heart_mask_pad(C, S)), shape_xyz, aff) > 0.5
    mask = heart & content
    gt = np.stack([im.load_canon(str(paths.bundle_stack(C, S, "gt", t)), shape_xyz, aff)
                   for t in range(T)])
    cx, cy = [int(round(v)) for v in np.array(np.nonzero(heart)[:2]).mean(1)]
    vmax = float(np.percentile(gt, 99.5))
    print(f"{S}  grid={shape_xyz}  in-plane {inplane:.2f} mm  dz {dz:.1f} mm  "
          f"heart centre (x,y)=({cx},{cy})  z-stretch x{dz/inplane:.1f}")

    # ---------- scored space ----------
    scored = {"GT": gt}
    for arm, label in ARMS:
        # Not every arm exists in every cohort (oracle_balanced is a provable no-op on the
        # periodic cohorts and was skipped there) -- skip instead of crashing.
        if not os.path.isfile(str(paths.recon(C, S, arm, "breath", 0))):
            print(f"  scored: {label} -- absent in {C}, skipped")
            continue
        thick = pose_psf.stamp_thickness(C, S, arm, "breath") if pose_psf.needs_psf(arm) else None
        rec = []
        for t in range(T):
            v, va = pose_psf.load_blurred(paths.recon(C, S, arm, "breath", t), thick, inplane)
            rec.append(pose_psf.resample(v, va, shape_xyz, aff,
                                         pose_psf.fit_rigid(v, va, gt[t], mask, aff)))
        rec = np.stack(rec)
        rec = im.prep_recon(rec, arm, content, stacks=None,
                            nmap=im.norm_map(rec, arm, content, None)) * content[None]
        scored[label] = rec
        print(f"  scored: {label}")

    os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
    zoom = dz / inplane
    for t in phases:
        for axis, cut, nm in (("XZ", cy, f"coronal-ish, y={cy}"), ("YZ", cx, f"sagittal-ish, x={cx}")):
            cols, titles = [], []
            for k, v in scored.items():
                pl = v[t][:, cut, :] if axis == "XZ" else v[t][cut, :, :]
                cols.append(to_img(pl, zoom, args.cell, vmax))
                titles.append(k)
            p = panel(cols, titles, args.cell,
                      f"SCORED space (all on GT grid, z={dz:.0f}mm) {nm} f={t}")
            # The COHORT must be in the filename: the same subject exists in every rhythm cohort,
            # and without it a second run silently overwrites the first one's figures.
            out = os.path.join(ROOT, "figs", f"lax_scored_{axis}_{C}_{S}_f{t}.png")
            p.save(out)
            print(f"-> {out}")

    # ---------- native space ----------
    for t in phases:
        cols, titles = [], []
        g = nib.load(str(paths.bundle_stack(C, S, "gt", t)))
        ga = np.asarray(g.dataobj, dtype=np.float32)
        gz = np.abs(np.diag(g.affine)[:3])
        cxg, cyg = int(ga.shape[0] // 2), int(ga.shape[1] // 2)
        cols.append(to_img(ga[:, cyg, :], gz[2] / gz[0], args.cell, vmax))
        titles.append(f"GT ({gz[0]:.1f}x{gz[2]:.0f}mm)")
        for arm, label in ARMS:
            p = paths.recon(C, S, arm, "breath", t)
            if not os.path.isfile(str(p)):
                continue
            img = nib.load(str(p))
            a = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
            a = np.clip(a, 0, None)
            z = np.abs(np.diag(img.affine)[:3])
            cut = int(a.shape[1] // 2)
            v = float(np.percentile(a[a > 0], 99.5)) if (a > 0).any() else 1.0
            cols.append(to_img(a[:, cut, :], z[2] / z[0], args.cell, v))
            titles.append(f"{label} ({z[0]:.1f}x{z[2]:.1f}mm)")
        out = os.path.join(ROOT, "figs", f"lax_native_{C}_{S}_f{t}.png")
        panel(cols, titles, args.cell,
              f"NATIVE grids, each arm as its engine wrote it, f={t}").save(out)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
