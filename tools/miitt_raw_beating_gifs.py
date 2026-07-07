#!/usr/bin/env python
"""Ground-truth beating-heart GIFs from the raw MIITT cine (NO model) — mid-z slice cycled
over time. Gated = 30 ECG cardiac phases (clean beat). Realtime = 180 free-breathing frames
(beat + respiratory drift, 25 ms/frame). One GIF per subject per arm. CPU-only."""
import glob
import os

import numpy as np
import nibabel as nib
from PIL import Image

ROOT = "/home/minsukc/MIITT/nifti"
OUT = "/home/minsukc/vggt/result/miitt_raw_gifs"
ARMS = {
    # arm: (rel path, per-frame ms). realtime is real-time (25 ms) but GIF viewers clamp ~ so use 40.
    "gated":    ("gated/sax/4d_recon.nii.gz", 60),
    "realtime": ("realtime/sax/4d_recon.nii.gz", 40),
}


def to_uint8(sl, vmax):
    return np.clip(sl / vmax * 255.0, 0, 255).astype(np.uint8)


def make_gif(nii_path, out_path, dur_ms):
    a = nib.load(nii_path).get_fdata().astype(np.float32)   # (X, Y, Z, T)
    Z, T = a.shape[2], a.shape[3]
    midz = Z // 2
    stack = a[:, :, midz, :]                                # (X, Y, T)
    vmax = float(np.percentile(stack[stack > 0], 99.5)) or 1e-3
    # (X,Y) -> display (row=Y, col=X); transpose so anatomy is upright-ish
    frames = [Image.fromarray(to_uint8(stack[..., t].T, vmax)) for t in range(T)]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=dur_ms, loop=0)
    return Z, T, midz


def main():
    os.makedirs(OUT, exist_ok=True)
    subjects = sorted(os.path.basename(d) for d in glob.glob(os.path.join(ROOT, "*"))
                      if os.path.isdir(d))
    for name in subjects:
        for arm, (rel, dur) in ARMS.items():
            p = os.path.join(ROOT, name, rel)
            if not os.path.exists(p):
                print(f"  [{name}/{arm}] MISSING {p}", flush=True)
                continue
            out_path = os.path.join(OUT, f"{name}__{arm}.gif")
            Z, T, midz = make_gif(p, out_path, dur)
            print(f"  [{name}/{arm}] Z={Z} T={T} midz={midz} -> {os.path.basename(out_path)}", flush=True)
    print("DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
