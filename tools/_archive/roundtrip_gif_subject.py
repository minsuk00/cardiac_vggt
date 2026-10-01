#!/usr/bin/env python
"""Beating-heart GIF for one MIITT subject: real-time INPUT | control0 | gather05 | s20contz,
at the reference plane, sweeping the reference over 30 evenly-spaced frames.

Each model is fed ITS OWN training regime:
  • control0, gather05 : 1-frame-per-slice (frames_per_slice=1), snap-z
  • s20contz           : multiframe (frames_per_slice=5) + continuous-z
For all models the z_mid companions are stripped so the swept reference alone drives the
reference plane (else the splat coverage-average of constant companions freezes the recon).

Usage:  micromamba run -n svr python tools/roundtrip_gif_subject.py [Volunteer1 Volunteer2 ...]
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
from tools.roundtrip_gif import fast_load, drop_zmid_companions

DEVICE = "cuda"
OUT = "result/roundtrip_miitt"
N_SWEEP = 30
# (label, ckpt, frames_per_slice, continuous_z)
MODELS = [
    ("control0", "scratch/logs/216539845_mri_volume_diffusion_ftctrl_gather0_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 1, False),
    ("gather05", "scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 1, False),
    ("s20contz", "scratch/logs/216949414_mri_volume_diffusion_s20contz_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", 5, True),
]


def nii_for(subject):
    return f"scratch/data/MIITT/nifti/{subject}/realtime/sax/4d_recon.nii.gz"


@torch.no_grad()
def sweep_one(ckpt, nii, fps, contz, sweep_frames):
    model = fast_load(ckpt)
    adapter = MIITTAdapter(nii)
    batch, S, picks, ref_ctx = adapter.build_batch_multiframe(
        DEVICE, frames_per_slice=fps, frames_for_reference=1, continuous_z=contz)
    z_ref = int(round(float(ref_ctx["z_canon"])))
    batch, picks = drop_zmid_companions(batch, picks, z_ref)
    ref_ctx["frame_indices"] = list(sweep_frames)
    vols, _, _ = reference_sweep(model, batch, ref_ctx, device=DEVICE)
    del model; torch.cuda.empty_cache()
    return vols, z_ref, ref_ctx


def input_frames(ref_ctx, sweep_frames):
    from inference.adapters.base import to_canonical_inplane
    vmin, vmax = ref_ctx["scale"]; inplane = ref_ctx["inplane"]; cine = ref_ctx["cine_slice"]
    return [to_canonical_inplane(np.clip((cine[f] - vmin) / (vmax - vmin), 0, 1), inplane).numpy()
            for f in sweep_frames]


def best_plane(vols):
    """Plane-following: the z with the most temporal motion (max aliveness over the sweep).
    Corrects the continuous-z model's constant through-plane RELOCATION — its dynamic content
    lands ~1 plane off the reference/read plane, so a fixed-z slice reads as frozen."""
    V = np.stack([v.astype(np.float32) for v in vols])   # (F, D, H, W)
    best, ba = 0, -1.0
    for z in range(V.shape[1]):
        a = V[:, z]; msk = a.mean(0) > 0.05
        al = float(a.std(0)[msk].mean()) if msk.any() else 0.0
        if al > ba:
            ba, best = al, z
    return best, ba


def run_subject(subject):
    nii = nii_for(subject)
    if not os.path.exists(nii):
        print(f"[{subject}] MISSING {nii} — skipping", flush=True); return
    odir = os.path.join(OUT, subject); os.makedirs(odir, exist_ok=True)
    n_frames = MIITTAdapter(nii).load().shape[0]
    sweep_frames = np.linspace(0, n_frames - 1, N_SWEEP).round().astype(int).tolist()
    print(f"[{subject}] frames over {n_frames}: {sweep_frames}", flush=True)

    seqs, planes, ref_ctx = {}, {}, None
    for label, ckpt, fps, contz in MODELS:
        try:
            vols, z_ref, rc = sweep_one(ckpt, nii, fps, contz, sweep_frames)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[{subject}] {label}: OOM — SKIPPED", flush=True); continue
        if ref_ctx is None:
            ref_ctx = rc
        zb, al = best_plane(vols)                 # plane-follow (relocation-robust)
        planes[label] = zb
        seqs[label] = np.stack([v[zb] for v in vols]).astype(np.float32)
        print(f"[{subject}] {label}: ref z={z_ref}, MOTION plane z={zb} (aliveness {al:.4f})", flush=True)

    if not seqs:
        print(f"[{subject}] no model succeeded — skipping outputs", flush=True); return
    inp = np.stack(input_frames(ref_ctx, sweep_frames)).astype(np.float32)
    np.savez_compressed(os.path.join(odir, "gif_zref.npz"),
                        inp=inp.astype(np.float16), frames=np.array(sweep_frames),
                        planes=np.array([planes.get(l, -1) for l, *_ in MODELS]),
                        **{k: v.astype(np.float16) for k, v in seqs.items()})

    cols = [("real-time input", inp)] + [(f"{lbl} z={planes[lbl]}", seqs[lbl]) for lbl, *_ in MODELS if lbl in seqs]
    vmax = max([float(c[1].max()) for c in cols] + [1e-3])
    imgs = []
    for i in range(N_SWEEP):
        fig, ax = plt.subplots(1, len(cols), figsize=(2.4 * len(cols), 2.6), dpi=100)
        for c, (t, seq) in enumerate(cols):
            ax[c].imshow(seq[i], cmap="gray", vmin=0, vmax=vmax)
            ax[c].set_xticks([]); ax[c].set_yticks([]); ax[c].set_title(t, fontsize=8)
        fig.suptitle(f"MIITT {subject} — frame {sweep_frames[i]} ({i+1}/{N_SWEEP})", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        imgs.append(Image.open(buf).convert("RGB")); plt.close(fig)
    gif = os.path.join(odir, "input_vs_models.gif")
    imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=150, loop=0)
    picks = [0, N_SWEEP // 4, N_SWEEP // 2, 3 * N_SWEEP // 4]; w, h = imgs[0].size
    mont = Image.new("RGB", (w, h * len(picks)), "white")
    for i, p in enumerate(picks):
        mont.paste(imgs[p], (0, i * h))
    mont.save(os.path.join(odir, "gif_montage.png"))
    print(f"[{subject}] wrote {gif} + gif_montage.png", flush=True)


def main():
    subjects = sys.argv[1:] or ["Volunteer1"]
    for s in subjects:
        run_subject(s)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
