"""The qualitative counterpart to the C1 finding: GT vs gather05 vs no_gather, ALL z-planes, over the cycle.

The report's headline is that the two models reconstruct nearly the same volume (they agree with each
other at 26.9 dB while agreeing with GT at only 20.3) despite one having 2x the through-plane error and
ignoring 23/100 real breaths. This renders that so a reader can see it rather than take it on trust:
three rows, every canonical z-plane, animated over the cardiac cycle, breathing input.

Layout deliberately matches scratch/eval/engine/assemble_and_gif.py (same ROI-derived window, same
per-z applied-mm labels) so it reads as the same family of figure.
"""
import json
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ROOT = "/home/minsukc/vggt/scratch/eval/cmrxrecon/out"
OUT = "/home/minsukc/vggt/result/1frame_series/gifs"
A, B = "vggt_20260715_1f_gather05", "vggt_20260715_1f_no_gather"


def load(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float32)


def cine(subj, meth, var, T):
    return np.stack([load(f"{ROOT}/{subj}/{meth}/recon_{var}/vol_t{t:02d}.nii.gz") for t in range(T)])


def render(subj, var="breath", fps=3):
    man = json.load(open(f"{ROOT}/{subj}/manifest.json"))
    T, D = man["T"], man["D"]
    gt = np.stack([load(f"{ROOT}/{subj}/gt/gt_t{t:02d}.nii.gz") for t in range(T)])
    ca, cb = cine(subj, A, var, T), cine(subj, B, var, T)
    heart = load(f"{ROOT}/{subj}/mask_heart.nii.gz") > 0.5
    fov = load(f"{ROOT}/{subj}/mask.nii.gz") > 0.5
    roi = heart & fov
    content = fov
    disp = np.abs(np.array(man["breath"]["disp_dhw_mm"]))[:, 0]
    m = roi[None].repeat(T, axis=0)
    vmax = float(np.percentile(np.concatenate([gt[m], ca[m], cb[m]]), 99.9))
    rows = [("GT", gt), ("gather05\nEPE 1.4 mm", ca), ("no_gather\nEPE 2.4 mm", cb)]
    planes = list(range(D))
    frames = []
    for t in range(T):
        fig, axes = plt.subplots(len(rows), len(planes),
                                 figsize=(len(planes) * 1.15, len(rows) * 1.15 + 0.8))
        axes = np.atleast_2d(axes)
        for ri, (lab, c) in enumerate(rows):
            for ci, z in enumerate(planes):
                ax = axes[ri, ci]
                ax.imshow(c[t, :, :, z].T, cmap="gray", vmin=0, vmax=vmax,
                          origin="lower", interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    # d_D (SI / through-plane) ONLY -- this figure is about through-plane estimation,
                    # and d_D is exactly what the breathing metric regresses against. NOTE the sibling
                    # assemble_and_gif.py GIFs label the 3D vector magnitude |v| instead, which is larger
                    # (per-subject tilt puts much of the breath in-plane: Test_P023 z10 is 12.3 mm SI but
                    # 27.4 mm |v|). Different quantity -- hence the explicit "SI" in the label.
                    v = disp[z] if (z < len(disp) and content[:, :, z].any()) else None
                    ax.set_title(f"z{z}" if v is None else f"z{z}\n{v:.1f} SI", fontsize=6.5)
                if ci == 0:
                    ax.set_ylabel(lab, fontsize=7.5)
        fig.suptitle(f"{subj} — breathing input, ALL z-planes (mm under z = applied THROUGH-PLANE/SI shift). "
                     f"The two model rows differ by 2x in through-plane error and look the same: they agree "
                     f"with each other at 26.9 dB while agreeing with GT at only 20.3 dB.   phase t={t}",
                     fontsize=8.5, y=0.985, va="top")
        H = len(rows) * 1.15 + 0.8
        fig.subplots_adjust(left=0.055, right=0.995, top=1.0 - 0.72 / H, bottom=0.01,
                            wspace=0.03, hspace=0.06)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/{subj}__C1_gather_vs_nogather_{var}.gif"
    imageio.mimsave(p, frames, duration=1.0 / fps, loop=0)
    print(f"  -> {p}  ({os.path.getsize(p)/1e6:.1f} MB)")


if __name__ == "__main__":
    import sys
    for s in (sys.argv[1:] or ["Test_P023"]):
        render(s)
