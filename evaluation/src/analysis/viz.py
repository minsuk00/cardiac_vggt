"""GT-vs-pred montage GIFs from the scored cine files (all z-planes, animated over the cycle).

The rendering half of _archive/assemble_and_gif.py (render_gif / gamma display / vmax lifted
verbatim), decoupled from scoring: it reads the 4D cines that score/image_metrics.py writes
(<subj>/cine_gt.nii.gz + <subj>/<arm>/cine_{clean,breath}.nii.gz — already on the subject grid,
post gauge/pose/PSF), so pictures always show exactly what was scored.

VGGT arms: the reference slice (slot 0 = target-phase input at z_mid, read from ed_dvf.npz's
slot_z[0]) gets a red starred z-label — that column's content was GIVEN to the model at the
queried phase. Baselines have no ed_dvf.npz -> no marker.

Run: EVAL_DATASET=<ds> micromamba run -n svr python evaluation/src/analysis/viz.py <subject> <method>
"""
import json
import os
import re
import sys

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

# Same curve as training/trainer_viz.py:_display_gamma — mid-tones brightened for readability.
GAMMA_ON = os.environ.get("GAMMA", "1") != "0"
GAMMA_TAG = "  [gamma 0.7]" if GAMMA_ON else ""


def display_gamma(image, vmax, gamma=0.7):
    return np.clip(image / max(float(vmax), 1e-8), 0.0, 1.0) ** gamma


def render_gif(out_path, rows, planes, T, vmax, titles, fps=6, plane_disp=None, ref_z=None):
    """rows: list of (label, cine[T,X,Y,Z]); one animation frame per cardiac phase t,
    each frame = len(rows) x len(planes) montage. plane_disp: optional per-z applied breathing
    |disp| (mm) under each z-label. ref_z: reference-slice plane -> red starred z-label."""
    nrow, ncol = len(rows), len(planes)
    H = nrow * 1.15 + 0.8            # fixed strip at top for the title + z/disp labels
    top = 1.0 - 0.68 / H
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.15, H))
        axes = np.atleast_2d(axes)
        for ri, (label, cine) in enumerate(rows):
            for ci, z in enumerate(planes):
                ax = axes[ri, ci]
                img = cine[t, :, :, z].T
                if GAMMA_ON:
                    ax.imshow(display_gamma(img, vmax), cmap="gray", vmin=0, vmax=1,
                              origin="lower", interpolation="nearest")
                else:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax,
                              origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    v = None if plane_disp is None else plane_disp[z]
                    lbl = f"z{z}" if v is None else f"z{z}\n{v:.1f}mm"
                    if ref_z is not None and z == ref_z:
                        ax.set_title(f"★{lbl}", fontsize=6.5, color="red")   # the given slice
                    else:
                        ax.set_title(lbl, fontsize=6.5)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=8)
        fig.suptitle(titles.format(t=t) + GAMMA_TAG, fontsize=9, y=0.985, va="top")
        fig.subplots_adjust(left=0.06, right=0.99, top=top, bottom=0.01,
                            wspace=0.03, hspace=0.06)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(buf[..., :3].copy())
        plt.close(fig)
    # imageio 2.x GIF duration is MILLISECONDS.
    imageio.mimsave(out_path, frames, duration=1000.0 / fps, loop=0)
    print(f"  -> {out_path}")


def load_cine(path):
    """(X,Y,Z,T) on disk -> (T,X,Y,Z)."""
    return np.moveaxis(np.asarray(nib.load(str(path)).dataobj, dtype=np.float32), -1, 0)


def main():
    ds = os.environ.get("EVAL_DATASET", "cmrx2024")
    if len(sys.argv) < 3:
        sys.exit("usage: EVAL_DATASET=<ds> python viz.py <subject> <method>")
    subj, method = sys.argv[1], sys.argv[2]
    md = paths.arm_dir(ds, subj, method)
    arm_label = re.sub(r"_ep\d+.*$", "", re.sub(r"^vggt_\d+_1f_", "", method))

    manifest = json.load(open(paths.manifest(ds, subj)))
    T = manifest["T"]
    disp = np.asarray(manifest["breath"]["disp_dhw_mm"], dtype=np.float64)
    disp_mag = np.linalg.norm(disp, axis=1)

    gt = load_cine(paths.cine_gt(ds, subj))
    D = gt.shape[3]
    planes = list(range(D))
    present = [v for v in ("clean", "breath") if paths.cine(ds, subj, method, v).exists()]
    if not present:
        sys.exit(f"{subj} [{method}]: no cine_*.nii.gz — run score/image_metrics.py first")
    cines = {v: load_cine(paths.cine(ds, subj, method, v)) for v in present}

    # Reference slice (VGGT arms only): slot 0's z index from ed_dvf.npz.
    ref_z = None
    dvf = md / "ed_dvf.npz"
    if dvf.exists():
        ref_z = int(round(float(np.load(dvf)["slot_z"][0])))

    # Shared display window across all rows, ONCE over all phases — p99.9 over the SCORING ROI
    # (heart∩FOV), same as the archived scorer's gifs, so hearts render at the same brightness
    # as the historical record (a full-FOV window includes bright chest wall and darkens them).
    fov = np.asarray(nib.load(str(paths.fov_mask(ds, subj))).dataobj) > 0.5
    heart_p = paths.heart_mask(ds, subj)
    heart = np.asarray(nib.load(str(heart_p)).dataobj) > 0.5 if os.path.exists(heart_p) else fov
    if fov.shape != gt.shape[1:] or heart.shape != gt.shape[1:]:   # masks must already be on the
        sys.exit(f"{subj}: mask grid {fov.shape}/{heart.shape} != GT {gt.shape[1:]} — "
                 f"off-grid bundle? (score/image_metrics.load_canon resamples; this renderer does not)")
    roi = heart & fov
    _in = roi[None].repeat(T, axis=0)
    vals = np.concatenate([np.nan_to_num(gt)[_in]] + [cines[v][_in] for v in present])
    vmax = float(np.percentile(vals, 99.9)) if vals.size else 1.0

    aligned = len(disp_mag) == D
    if not aligned:
        print(f"  WARNING: manifest disp has {len(disp_mag)} planes but D={D}; "
              f"per-plane labels suppressed (stale bundle?)")
    pd = [float(disp_mag[z]) if (aligned and fov[:, :, z].any()) else None for z in range(D)]
    breath_tag = f"breathing |disp| mean {disp_mag.mean():.1f} / max {disp_mag.max():.1f} mm"
    ref_tag = "  (★red = reference slice given to the model)" if ref_z is not None else ""
    if "clean" in present:
        render_gif(md / "gif_clean.gif",
                   [("GT", gt), (f"{arm_label}\n(no breath)", cines["clean"])], planes, T, vmax,
                   f"{subj} [{method}]  —  clean input{ref_tag}   phase t={{t}}",
                   plane_disp=pd, ref_z=ref_z)
    if "breath" in present:
        render_gif(md / "gif_breath.gif",
                   [("GT", gt), (f"{arm_label}\n(breathing)", cines["breath"])], planes, T, vmax,
                   f"{subj} [{method}]  —  breathing input ({breath_tag}){ref_tag}   phase t={{t}}",
                   plane_disp=pd, ref_z=ref_z)


if __name__ == "__main__":
    main()
