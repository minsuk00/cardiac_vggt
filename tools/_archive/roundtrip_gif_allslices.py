#!/usr/bin/env python
"""Per-slice beating GIF: rows = all 12 canonical z-planes, cols = input | control0 | gather05 |
s20contz, animated over the 30-frame reference sweep.

  • input column = each canonical plane's OWN real-time cine (real motion at every plane), so it's
    the ground-truth "what the heart does at plane z".
  • recon columns = each model's V_canon[z] over the sweep — reveals which planes actually move
    (only the reference-landing plane) vs frozen, visually confirming the per-z aliveness probe.

Note: the models are fed ONE frame per non-reference plane and only the reference is swept, so a
recon plane can move only if the swept reference's content lands there — the input column is NOT
what the model saw at those planes (it saw a single frame), it's the real cine for reference.

  micromamba run -n svr python tools/roundtrip_gif_allslices.py [Volunteer1]
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
from inference.adapters.base import assign_canonical_z, percentile_scale, to_canonical_inplane
from inference.inference import reference_sweep
from tools.roundtrip_gif import fast_load, drop_zmid_companions

DEVICE = "cuda"
OUT = "result/roundtrip_miitt"
N_SWEEP = 30
MODELS = [
    ("control0", "scratch/logs/216539845_mri_volume_diffusion_ftctrl_gather0_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 1, False),
    ("gather05", "scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 1, False),
    ("s20contz", "scratch/logs/216949414_mri_volume_diffusion_s20contz_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 5, True),
]


@torch.no_grad()
def sweep_full(ckpt, nii, fps, contz, sweep_frames):
    model = fast_load(ckpt)
    ad = MIITTAdapter(nii)
    b, S, picks, rc = ad.build_batch_multiframe(DEVICE, frames_per_slice=fps, frames_for_reference=1, continuous_z=contz)
    zr = int(round(float(rc["z_canon"])))
    b, picks = drop_zmid_companions(b, picks, zr)
    rc["frame_indices"] = sweep_frames
    vols, _, _ = reference_sweep(model, b, rc, device=DEVICE)
    del model; torch.cuda.empty_cache()
    return np.stack(vols).astype(np.float32), zr        # (F, 12, 256, 256)


def input_per_plane(nii, sweep_frames, D=12):
    ad = MIITTAdapter(nii); cine = ad.load(); vmin, vmax = percentile_scale(cine)
    inpl = ad.inplane_mm()
    zmap = {int(zc): si for zc, si in assign_canonical_z(ad.slice_positions_mm(), continuous_z=False)}
    out = np.zeros((len(sweep_frames), D, 256, 256), np.float32)
    for z, si in zmap.items():
        if 0 <= z < D:
            for i, f in enumerate(sweep_frames):
                out[i, z] = to_canonical_inplane(np.clip((cine[f, si] - vmin) / (vmax - vmin), 0, 1), inpl).numpy()
    return out


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 else "Volunteer1"
    nii = f"scratch/data/MIITT/nifti/{subject}/realtime/sax/4d_recon.nii.gz"
    odir = os.path.join(OUT, subject); os.makedirs(odir, exist_ok=True)
    n = MIITTAdapter(nii).load().shape[0]
    sweep = np.linspace(0, n - 1, N_SWEEP).round().astype(int).tolist()
    print(f"[{subject}] frames {sweep}", flush=True)

    inp = input_per_plane(nii, sweep)
    Vs, zrefs = {}, {}
    for tag, ckpt, fps, contz in MODELS:
        try:
            V, zr = sweep_full(ckpt, nii, fps, contz, sweep)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print(f"[{subject}] {tag}: OOM skip", flush=True); continue
        Vs[tag] = V; zrefs[tag] = zr
        print(f"[{subject}] {tag}: done, ref z={zr}", flush=True)

    np.savez_compressed(os.path.join(odir, "allslices.npz"),
                        inp=inp.astype(np.float16), frames=np.array(sweep),
                        **{k: v.astype(np.float16) for k, v in Vs.items()})

    cols = [("input", inp)] + [(f"{t} (ref z={zrefs[t]})", Vs[t]) for t, *_ in MODELS if t in Vs]
    D = 12
    vmax = max([float(np.percentile(c[1], 99.5)) for c in cols] + [1e-3])
    frames = []
    for i in range(N_SWEEP):
        fig, axes = plt.subplots(D, len(cols), figsize=(1.5 * len(cols), 1.5 * D), dpi=62)
        for z in range(D):
            for c, (t, vol) in enumerate(cols):
                ax = axes[z][c]
                ax.imshow(vol[i, z], cmap="gray", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                if z == 0:
                    ax.set_title(t, fontsize=8)
                if c == 0:
                    ax.set_ylabel(f"z={z}", fontsize=8)
        fig.suptitle(f"MIITT {subject} — per-slice, frame {sweep[i]} ({i+1}/{N_SWEEP})", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("RGB")); plt.close(fig)
    gif = os.path.join(odir, "allslices.gif")
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=150, loop=0)
    # also a static montage at 2 timepoints (min/max of the sweep) for quick inspection
    Image.fromarray(np.concatenate([np.array(frames[0]), np.array(frames[N_SWEEP // 2])], axis=1)
                    ).save(os.path.join(odir, "allslices_2tp.png"))
    print(f"[{subject}] wrote {gif}", flush=True)


if __name__ == "__main__":
    main()
