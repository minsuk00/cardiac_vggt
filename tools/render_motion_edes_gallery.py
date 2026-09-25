#!/usr/bin/env python
"""Candidate gallery for the reference-conditioning figure (tools/render_motion_edes.py): one row per
candidate, [Input | ED target | ES target], one PNG per rhythm, each row on its own colour scale. Used to pick
the AF and HRV examples by eye (docs/127).

specs: tools/rank_input_frame_sweep.py --specs output; the fields of every listed frame must be saved first
(tools/sweep_input_frame.py --subjects <source>:<subject>:<z>:<frame> ...).

Usage: PYTHONPATH=tools micromamba run -n svr python tools/render_motion_edes_gallery.py \
    temp/gallery_specs.txt scratch/motion_edes/sweep80_save figs/motion_edes
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from render_motion_edes import PX_PER_MM, REF, gray, load_rhythm, view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs")
    ap.add_argument("dump")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    rows = [l.split() for l in open(a.specs) if l.strip()]
    for rh, lab in (("af12", "AF"), ("hrv12", "HRV")):
        rr = [r for r in rows if r[0] == rh]
        fig, ax = plt.subplots(len(rr), 3, figsize=(6.4, 2.2 * len(rr)), squeeze=False)
        for i, (_, spec, sc, ed, es) in enumerate(rr):
            src, sid, z, f = spec.split(":")
            r = load_rhythm(f"{src}/{sid},{z},{sid}_z{z}f{f}", rh, a.dump)
            vmax = np.percentile(np.concatenate([np.hypot(d["f"][..., 0], d["f"][..., 1])[r["m"]]
                                                 for d in r["targets"].values()]), 98)
            y0, y1, x0, x1 = r["box"]
            gray(ax[i, 0], r["inp"]); view(ax[i, 0], r["box"])
            ax[i, 0].set_ylabel(f"#{i + 1} {sid.replace('_Siemens_15T_Aera', '')}\nz{z} frame {f}\n"
                                f"min {sc} mm", fontsize=7)
            for k, kind in enumerate(("ED", "ES")):
                d, a2, m = r["targets"][kind], ax[i, 1 + k], r["m"]
                fl = d["f"]
                gray(a2, r["inp"])
                im = a2.imshow(np.where(m, np.hypot(fl[..., 0], fl[..., 1]), np.nan), cmap="magma", vmin=0,
                               vmax=vmax, alpha=0.6)
                st = 11
                yy, xx = np.mgrid[y0 + st // 2:y1:st, x0 + st // 2:x1:st]
                keep = m[yy, xx]
                a2.quiver(xx[keep], yy[keep], fl[yy, xx, 0][keep], fl[yy, xx, 1][keep], color="white",
                          angles="xy", scale_units="xy", scale=1 / PX_PER_MM, width=0.009,
                          headwidth=3.5, headlength=3.5, headaxislength=3)
                view(a2, r["box"])
                a2.set_title(f"{kind} target (t{d['t']}) mean {ed if kind == 'ED' else es} mm", fontsize=7, pad=2)
                ia = a2.inset_axes(r["inset"])
                for sp in ia.spines.values():
                    sp.set_visible(True); sp.set_edgecolor(REF); sp.set_linewidth(1.0)
                gray(ia, d["ref"])
                ry, rx = np.where(d["mr"])
                rc = ((ry.min() + ry.max()) // 2, (rx.min() + rx.max()) // 2)
                rhh = max(ry.max() - ry.min(), rx.max() - rx.min()) // 2 + 2
                ia.set_xlim(rc[1] - rhh, rc[1] + rhh); ia.set_ylim(rc[0] + rhh, rc[0] - rhh)
                ia.set_xticks([]); ia.set_yticks([])
            cb = fig.colorbar(im, ax=ax[i, :].tolist(), fraction=0.02, pad=0.01)
            cb.ax.tick_params(labelsize=6); cb.set_label("mm", fontsize=6)
        fig.suptitle(f"{lab} candidates (per-row colour scale)", fontsize=10, fontweight="bold")
        fig.savefig(f"{a.out_dir}/gallery_{lab}.png", dpi=130, bbox_inches="tight")
        print("wrote", f"{a.out_dir}/gallery_{lab}.png")


if __name__ == "__main__":
    main()
