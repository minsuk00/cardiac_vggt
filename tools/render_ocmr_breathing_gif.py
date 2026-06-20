"""Render a GIF of OCMR raw reconstructed frames so the REAL breathing is visible by eye.

Left: the slice's real-time frames with a FIXED crosshair at the heart's reference position —
the heart drifts relative to it = respiratory motion (plus fast cardiac beating). Right: the
measured in-plane displacement trace (PC1, mm) with a cursor at the current frame.

Run: micromamba run -n svr python tools/render_ocmr_breathing_gif.py
"""
import json
import os

import numpy as np
import SimpleITK as sitk
from skimage.registration import phase_cross_correlation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
RECON = os.path.join(ROOT, "scratch/data/ocmr/recon")
OUT = os.path.join(ROOT, "result", "ocmr_cleaner")
SUBJECTS = [("us_0084_1_5T", "volunteer · 10 br/min · SI 6.4mm"),
            ("us_0183_pt_1_5T", "patient · 20 br/min · SI 1.6mm")]


def disp_trace(clip, ix, iy):
    ref = np.median(clip, 0); F = len(clip); dy = np.zeros(F); dx = np.zeros(F)
    for f in range(F):
        (a, b), _, _ = phase_cross_correlation(ref, clip[f], upsample_factor=10, normalization=None)
        dy[f] = a * iy; dx[f] = b * ix
    D = np.stack([dy, dx], 1); D -= D.mean(0)
    _, _, vt = np.linalg.svd(D, full_matrices=False)
    return D @ vt[0], D, vt[0]


def main():
    for name, label in SUBJECTS:
        c = sitk.GetArrayFromImage(sitk.ReadImage(f"{RECON}/{name}/sax_cine.nii.gz")).astype(np.float32)
        m = json.load(open(f"{RECON}/{name}/meta.json")); ix, iy = m["inplane_mm"]; tr = m["TRes_ms"]
        F, S, H, W = c.shape
        s_h = int(np.argmax(c.reshape(F, S, -1).mean(-1).mean(0)))
        hy0, hy1, hx0, hx1 = H // 3, H - H // 3, W // 3, W - W // 3
        heart = c[:, s_h, hy0:hy1, hx0:hx1]
        sig, D2, pc = disp_trace(heart, ix, iy)
        # heart reference centroid (median frame, bright blob) in full-image coords
        med = np.median(heart, 0); thr = med > np.percentile(med, 85)
        cy, cx = np.array(np.nonzero(thr)).mean(1)
        cy += hy0; cx += hx0
        t = np.arange(F) * tr / 1000.0
        vmax = float(np.percentile(c[:, s_h], 99))
        frames = []
        for f in range(F):
            fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})
            ax[0].imshow(c[f, s_h], cmap="gray", vmin=0, vmax=vmax)
            ax[0].axhline(cy, color="lime", lw=0.7, alpha=0.8)
            ax[0].axvline(cx, color="lime", lw=0.7, alpha=0.8)
            ax[0].set_title(f"{name} slice {s_h} — frame {f}/{F}", fontsize=8); ax[0].axis("off")
            ax[1].plot(t, sig, lw=1.0, color="steelblue")
            ax[1].axvline(t[f], color="r", lw=1.0)
            ax[1].scatter([t[f]], [sig[f]], color="r", s=18, zorder=5)
            ax[1].set_xlabel("time (s)", fontsize=8); ax[1].set_ylabel("in-plane displacement (mm)", fontsize=8)
            ax[1].set_title(f"{label}", fontsize=8); ax[1].grid(alpha=0.25)
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
            frames.append(Image.fromarray(img)); plt.close(fig)
        out = f"{OUT}/breathing_{name}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0)
        print(f"wrote {out}  ({F} frames, {os.path.getsize(out)/1e6:.1f} MB)", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
