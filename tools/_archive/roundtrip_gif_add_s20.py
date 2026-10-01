#!/usr/bin/env python
"""Add s20contz as a 4th column to the beating-heart GIF, fed its OWN regime (multiframe +
continuous-z), then rebuild input | control0 | gather05 | s20contz from the cached
gif_zref.npz (control0/gather05/input) + the new s20contz sweep. Same evenly-spaced 30-frame
reference sweep; reference plane driven by slot 0 (z_mid companions stripped) for all models.
"""
import os
import sys
from io import BytesIO

import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from inference.adapters import MIITTAdapter
from inference.inference import reference_sweep
from inference.adapters.base import GRID_SHAPE
from tools.roundtrip_gif import fast_load, drop_zmid_companions

DEVICE = "cuda"
NII = "scratch/data/MIITT/nifti/Volunteer1/realtime/sax/4d_recon.nii.gz"
OUT = "result/roundtrip_miitt"
S20 = "scratch/logs/216949414_mri_volume_diffusion_s20contz_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"


@torch.no_grad()
def sweep_s20(sweep_frames):
    model = fast_load(S20)
    adapter = MIITTAdapter(NII)
    # s20 regime: multiframe (frames_per_slice=5) + continuous z.
    batch, S, picks, ref_ctx = adapter.build_batch_multiframe(
        DEVICE, frames_per_slice=5, frames_for_reference=1, continuous_z=True)
    z_ref = int(round(float(ref_ctx["z_canon"])))
    batch, picks = drop_zmid_companions(batch, picks, z_ref)
    ref_ctx["frame_indices"] = list(sweep_frames)
    vols, _, _ = reference_sweep(model, batch, ref_ctx, device=DEVICE)
    del model; torch.cuda.empty_cache()
    print(f"  s20contz: swept {len(vols)} frames, S={len(picks)}, z_ref={z_ref}", flush=True)
    return vols, z_ref


def main():
    d = np.load(os.path.join(OUT, "gif_zref.npz"))
    inp = d["inp"].astype(np.float32); ctrl = d["ctrl"].astype(np.float32)
    g05 = d["g05"].astype(np.float32); frames = d["frames"]; z_c = int(d["z_ref"])

    vols, z_s = sweep_s20(frames)
    # Display all columns at the SAME plane (z_c, the control models' reference plane) for a fair
    # side-by-side; note s20contz's own reference plane index (z_s) in the title.
    s20 = np.stack([v[z_c] for v in vols]).astype(np.float32)
    np.savez_compressed(os.path.join(OUT, "gif_zref_s20.npz"), s20=s20, z_s=z_s)

    vmax = max(inp.max(), ctrl.max(), g05.max(), float(s20.max()), 1e-3)
    cols = [("real-time input", inp), (f"control0 z={z_c}", ctrl),
            (f"gather05 z={z_c}", g05), (f"s20contz z={z_c}", s20)]
    N = len(frames); imgs = []
    for i in range(N):
        fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.6), dpi=100)
        for c, (t, seq) in enumerate(cols):
            ax[c].imshow(seq[i], cmap="gray", vmin=0, vmax=vmax)
            ax[c].set_xticks([]); ax[c].set_yticks([]); ax[c].set_title(t, fontsize=8)
        fig.suptitle(f"MIITT Volunteer1 — frame {int(frames[i])} ({i+1}/{N})", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB")); plt.close(fig)
    imgs[0].save(os.path.join(OUT, "input_vs_control_vs_gather_vs_s20.gif"),
                 save_all=True, append_images=imgs[1:], duration=150, loop=0)
    picks = [0, N // 4, N // 2, 3 * N // 4]; w, h = imgs[0].size
    mont = Image.new("RGB", (w, h * len(picks)), "white")
    for i, p in enumerate(picks):
        mont.paste(imgs[p], (0, i * h))
    mont.save(os.path.join(OUT, "gif_montage_4col.png"))
    print("wrote input_vs_control_vs_gather_vs_s20.gif + gif_montage_4col.png", flush=True)


if __name__ == "__main__":
    main()
