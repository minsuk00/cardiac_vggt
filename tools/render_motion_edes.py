"""Learned motion at target ED vs target ES, under AF and HRV (Fig. 6(b) style), one full-width row.

Panels: [AF input | AF ED target | AF ES target | HRV input | HRV ED target | HRV ES target]. Each rhythm has
its own subject/plane (same subject under both rhythms gives near-duplicate halves). Within a rhythm the input
and both target panels show the SAME input slice (identical pixels); only the reference frame differs. Target
panels: input slice + in-plane displacement magnitude (heart ROI, rigid heart-ROI mean shift removed) + arrows
at true length; the reference frame that defines the target is an outlined inset.
ED/ES = reference frame with max/min GT LV volume.

Motion fields: tools/dump_vggt_motion.py (<dump>/<rhythm>/<subject>/) or tools/sweep_input_frame.py
(<dump>/<rhythm>/<subject>_z{z}f{f}/, input frame of plane z overridden).

Usage: PYTHONPATH=tools micromamba run -n svr python tools/render_motion_edes.py --dump scratch/motion_edes/sweep \
    --af cmrx2024/CMRx24_Train_P055,3,CMRx24_Train_P055_z3f9 --hrv mnms/MNMs_O5U2U7,4,MNMs_O5U2U7_z4f2 --out x.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_opening, rotate, zoom

from render_runtime_accuracy import TEXTW, paper_rc

EV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
VOL, RES = f"{EV}/volumes", f"{EV}/metric_results/test"
ARM = "vggt_final518_diff1000_ep300"
PX_PER_MM = 518 / (256 * 1.4)
MARGIN, PAD = 12, 0.015
INS = 0.30     # inset side, fraction of the panel; always bottom-right
REF = "#6D8EAE"   # outline colour of the reference-frame inset


def up(a2d, order):
    return zoom(a2d, 518 / 256, order=order)


def sax_standard(img, seg):
    """Standard cardiology SAX view of one (y, x) slice: rotate so the RV centroid (label 3) is left of the LV
    centroid (label 1), then flip vertically if needed so the body surface nearest the heart (anterior chest
    wall) is at the top. The bundle affines are placeholders, so anterior is read from anatomy, not metadata.
    Returns (angle for scipy.ndimage.rotate, flip)."""
    d = np.argwhere(seg == 1).mean(0) - np.argwhere(seg == 3).mean(0)          # RV -> LV, (dy, dx)
    ang = float(np.degrees(np.arctan2(d[0], d[1])))
    im, s = rotate(img, ang, reshape=False, order=1), rotate(seg, ang, reshape=False, order=0)
    body = binary_fill_holes(binary_opening(im > np.percentile(img[img > 0], 20), iterations=2))
    edge = np.argwhere(body & ~binary_erosion(body))
    edge = edge[(edge > 2).all(1) & (edge < im.shape[0] - 3).all(1)]           # drop the FOV border
    dist = np.hypot(*(edge - np.argwhere(s > 0).mean(0)).T)
    return ang, bool((edge[dist < dist.min() + 15, 0] - np.argwhere(s > 0).mean(0)[0]).mean() > 0)


def orient(a, ang, flip, order):
    r = rotate(a, ang, reshape=False, order=order)
    return r[::-1] if flip else r


def orient_field(f, ang, flip):
    """Resample a (y, x, [dx, dy]) field like the image and rotate its vectors with it."""
    t = np.radians(ang)
    fx, fy = (rotate(f[..., k], ang, reshape=False, order=1) for k in (0, 1))
    g = np.stack([np.cos(t) * fx + np.sin(t) * fy, -np.sin(t) * fx + np.cos(t) * fy], -1)
    return np.stack([g[::-1, :, 0], -g[::-1, :, 1]], -1) if flip else g


def load_rhythm(spec, rh, dump):
    """spec 'source/subject,z[,dump name]' -> dict with input, ED/ES fields, reference crops, view box."""
    parts = spec.split(",")
    src, sid = parts[0].split("/")
    z = int(parts[1])
    dd = f"{dump}/{rh}/{parts[2] if len(parts) > 2 else sid}"
    sd = f"{VOL}/{src}_{rh}/out/{sid}"
    curve = np.array(next(r for r in json.load(open(f"{RES}/{src}_{rh}/ef/{ARM}.json"))["per_subject"]
                          if r["subject"] == sid)["lv_curve_gt"])
    meta = json.load(open(f"{dd}/motion_meta.json"))
    heart = nib.load(f"{sd}/mask_heart.nii.gz").get_fdata() > 0
    m = up(heart[:, :, z].T.astype(np.uint8), 0) > 0
    out = {"m": m, "targets": {}, "label": f"{sid} z{z}"}
    frames = set()
    for kind, t in (("ED", int(np.argmax(curve))), ("ES", int(np.argmin(curve)))):
        tg = meta["targets"][str(t)]
        s = tg["slot_z"].index(z)
        assert s > 0, "chosen plane is the reference slice"
        frames.add(tg["slot_t"][s])
        zr = tg["slot_z"][0]
        ref = nib.load(f"{sd}/breath/stack_t{tg['slot_t'][0]:02d}.nii.gz").get_fdata()[:, :, zr].T
        f = np.asarray(nib.load(f"{dd}/motion_t{t:02d}.nii.gz").dataobj[:, :, z, :2],
                       np.float32).transpose(1, 0, 2)                      # (y, x, [dx, dy]) mm
        out["targets"][kind] = dict(t=t, f=f - f[m].mean(0), ref=up(ref, 1),
                                    mr=up(heart[:, :, zr].T.astype(np.uint8), 0) > 0)
    assert len(frames) == 1, "ED and ES panels must show the same input frame"
    j = frames.pop()
    out["frame"] = j
    inp = nib.load(f"{sd}/breath/stack_t{j:02d}.nii.gz").get_fdata()[:, :, z].T
    out["inp"] = up(inp, 1)

    # standard SAX orientation (RV left, anterior up), from the GT seg of this plane; same plane for the refs
    ang, flip = sax_standard(inp, nib.load(f"{sd}/seg_gt_3d_fullres/seg_t00.nii.gz").get_fdata()[:, :, z].T)
    out["label"] += f" rot {ang:.0f}{' flip' if flip else ''}"
    m = out["m"] = orient(m, ang, flip, 0)
    out["inp"] = orient(out["inp"], ang, flip, 1)
    for d in out["targets"].values():
        d["f"] = orient_field(d["f"], ang, flip)
        d["ref"], d["mr"] = orient(d["ref"], ang, flip, 1), orient(d["mr"], ang, flip, 0)

    # tight square crop centred on the heart ROI; the inset sits bottom-right (may overlap the ROI edge)
    ys, xs = np.where(m)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    w = max(ys.max() - ys.min(), xs.max() - xs.min()) + 2 * MARGIN
    y0, x0 = cy - w // 2, cx - w // 2
    out["box"] = (y0, y0 + w, x0, x0 + w)
    out["inset"] = (1 - INS - PAD, PAD, INS, INS)
    return out


def view(a2, box):
    y0, y1, x0, x1 = box
    a2.set_xlim(x0, x1); a2.set_ylim(y1, y0)
    a2.set_aspect("equal")
    a2.set_xticks([]); a2.set_yticks([])
    for sp in a2.spines.values():
        sp.set_visible(False)


def gray(a2, img):
    a2.imshow(np.clip(img / np.percentile(img, 99.5), 0, 1), cmap="gray", vmin=0, vmax=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--af", required=True, help="source/subject,z[,dump name]")
    ap.add_argument("--hrv", required=True, help="source/subject,z[,dump name]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-map", action="store_true", help="arrows only: no magnitude overlay, no colour bar")
    ap.add_argument("--arrow-color", default="white")
    a = ap.parse_args()
    R = [("AF", load_rhythm(a.af, "af12", a.dump)), ("HRV", load_rhythm(a.hrv, "hrv12", a.dump))]
    vmax = np.percentile(np.concatenate([np.hypot(d["f"][..., 0], d["f"][..., 1])[r["m"]]
                                         for _, r in R for d in r["targets"].values()]), 98)

    paper_rc()
    fig, ax = plt.subplots(1, 6, figsize=(TEXTW, 1.17), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.01, hspace=0.01)
    for g, (_, r) in enumerate(R):
        a0 = ax[3 * g]
        gray(a0, r["inp"])
        view(a0, r["box"])
        a0.set_title("Input", fontsize=7, pad=2)
        y0, y1, x0, x1 = r["box"]
        for k, kind in enumerate(("ED", "ES")):
            d = r["targets"][kind]
            a2, f, m = ax[3 * g + 1 + k], d["f"], r["m"]
            gray(a2, r["inp"])
            if not a.no_map:
                im = a2.imshow(np.where(m, np.hypot(f[..., 0], f[..., 1]), np.nan), cmap="magma", vmin=0,
                               vmax=vmax, alpha=0.6)
            st = 11
            yy, xx = np.mgrid[y0 + st // 2:y1:st, x0 + st // 2:x1:st]
            keep = m[yy, xx]
            a2.quiver(xx[keep], yy[keep], f[yy, xx, 0][keep], f[yy, xx, 1][keep], color=a.arrow_color,
                      angles="xy", scale_units="xy", scale=1 / PX_PER_MM, width=0.009,
                      headwidth=3.5, headlength=3.5, headaxislength=3)
            view(a2, r["box"])
            a2.set_title(f"{kind} target", fontsize=7, pad=2)
            # reference frame (defines the target), cropped on its own heart ROI
            ia = a2.inset_axes(r["inset"])
            for sp in ia.spines.values():   # paper_rc hides top/right spines; the outline needs all four
                sp.set_visible(True); sp.set_edgecolor(REF); sp.set_linewidth(1.0)
            gray(ia, d["ref"])
            ry, rx = np.where(d["mr"])
            rc = ((ry.min() + ry.max()) // 2, (rx.min() + rx.max()) // 2)
            rh = max(ry.max() - ry.min(), rx.max() - rx.min()) // 2 + 2
            ia.set_xlim(rc[1] - rh, rc[1] + rh); ia.set_ylim(rc[0] + rh, rc[0] - rh)
            ia.set_xticks([]); ia.set_yticks([])
    if not a.no_map:
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label("mm", fontsize=6.5, labelpad=1)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)
    # rhythm header over each group of panels, with a rule underneath (placed once the layout is final)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    fig.set_layout_engine("none")
    if not a.no_map:
        p, q = ax[0].get_position(), cb.ax.get_position()   # colorbar exactly as tall as the image panels
        cb.ax.set_position([q.x0, p.y0, q.width, p.height])
    for g, (lab, _) in enumerate(R):
        l = to_fig.transform(ax[3 * g].get_window_extent(rend).p0)[0]
        r_ = to_fig.transform(ax[3 * g + 2].get_window_extent(rend).p1)[0]
        top = to_fig.transform((0, ax[3 * g].title.get_window_extent(rend).y1))[1] + 0.02
        fig.add_artist(plt.Line2D([l + 0.01, r_ - 0.01], [top, top], color="k", lw=0.6))
        fig.text((l + r_) / 2, top + 0.015, lab, ha="center", va="bottom", fontsize=8, fontweight="bold")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", a.out, "|", "; ".join(f"{lab} {r['label']} frame {r['frame']} targets "
                                        f"ED t{r['targets']['ED']['t']} ES t{r['targets']['ES']['t']}"
                                        for lab, r in R))


if __name__ == "__main__":
    main()
