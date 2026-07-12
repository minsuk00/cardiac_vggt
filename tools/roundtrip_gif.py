#!/usr/bin/env python
"""Beating-heart GIF on MIITT Volunteer1 real-time input: real-time INPUT | control0 | gather05,
at the reference plane, sweeping the reference over EVENLY-SPACED frames across the full cine.

Design fixes over v1:
  • EVENLY-SPACED sweep frames (linspace over the whole recording) so each step is a genuinely
    different cardiac phase — not a consecutive burst.
  • z_mid COMPANIONS REMOVED: build one frame per plane, then drop every non-slot-0 frame at the
    reference plane, so slot 0 (the swept reference) is the ONLY contributor there and actually
    DRIVES the reconstruction (v1 diluted it 1/31 with constant companions → frozen recon). Also
    matches these models' 1-frame-per-plane training regime (control0/gather05 trained at S=12).
  • ALIVENESS measured: temporal std over the 30 frames at z_ref (mean over content pixels) for
    each column — quantifies motion instead of eyeballing.

  micromamba run -n svr python tools/roundtrip_gif.py
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
from inference.adapters.base import GRID_SHAPE, to_canonical_inplane
from vggt.models.vggt import VGGT

DEVICE = "cuda"
NII = "scratch/data/MIITT/nifti/Volunteer1/realtime/sax/4d_recon.nii.gz"
OUT = "result/roundtrip_miitt"
N_SWEEP = 30
MODELS = {
    "control0": "scratch/logs/216539845_mri_volume_diffusion_ftctrl_gather0_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt",
    "gather05": "scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt",
}


def fast_load(ckpt):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, enable_refiner=False, grid_shape=GRID_SHAPE)
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    return m.to(DEVICE).eval()


def drop_zmid_companions(batch, picks, z_ref):
    """Keep slot 0; drop every OTHER slot whose canonical z rounds to the reference plane, so the
    swept reference alone drives z_ref. Returns the reduced batch + kept picks."""
    keep = [0] + [i for i in range(1, len(picks)) if round(float(picks[i][0])) != z_ref]
    idx = torch.tensor(keep, device=batch["images"].device)
    out = dict(batch)
    for k in ("images", "scanner_coords", "z_indices"):
        out[k] = batch[k][:, idx].contiguous()
    return out, [picks[i] for i in keep]


@torch.no_grad()
def sweep_model(ckpt, sweep_frames):
    model = fast_load(ckpt)
    adapter = MIITTAdapter(NII)
    batch, S, picks, ref_ctx = adapter.build_batch_multiframe(
        DEVICE, frames_per_slice=1, frames_for_reference=1, continuous_z=False)
    z_ref = int(round(float(ref_ctx["z_canon"])))
    batch, picks = drop_zmid_companions(batch, picks, z_ref)
    ref_ctx["frame_indices"] = list(sweep_frames)   # override: evenly-spaced, decoupled from companions
    vols, _, _ = reference_sweep(model, batch, ref_ctx, device=DEVICE)
    del model; torch.cuda.empty_cache()
    print(f"  swept {len(vols)} frames, S={len(picks)} (companions at z={z_ref} removed)", flush=True)
    return vols, ref_ctx, z_ref


def input_frames(ref_ctx, sweep_frames):
    vmin, vmax = ref_ctx["scale"]; inplane = ref_ctx["inplane"]; cine = ref_ctx["cine_slice"]
    return [to_canonical_inplane(np.clip((cine[f] - vmin) / (vmax - vmin), 0, 1), inplane).numpy()
            for f in sweep_frames]


def aliveness(seq):
    """Temporal std over frames, averaged over pixels that carry content (mean>0.05)."""
    a = np.stack(seq)                      # (F,H,W)
    std = a.std(0)
    m = a.mean(0) > 0.05
    return float(std[m].mean()) if m.any() else 0.0


def main():
    n_frames = MIITTAdapter(NII).load().shape[0]
    sweep_frames = np.linspace(0, n_frames - 1, N_SWEEP).round().astype(int).tolist()
    print(f"sweep frames (evenly spaced over {n_frames}): {sweep_frames}", flush=True)

    print("control0 ...", flush=True); vc, ctx, z_ref = sweep_model(MODELS["control0"], sweep_frames)
    print("gather05 ...", flush=True); vg, _, _ = sweep_model(MODELS["gather05"], sweep_frames)
    inp = input_frames(ctx, sweep_frames)
    ctrl = [v[z_ref] for v in vc]
    g05 = [v[z_ref] for v in vg]

    al = {"input": aliveness(inp), "control0": aliveness(ctrl), "gather05": aliveness(g05)}
    print(f"ALIVENESS (temporal std @ z={z_ref}): "
          f"input={al['input']:.4f}  control0={al['control0']:.4f}  gather05={al['gather05']:.4f}", flush=True)

    np.savez_compressed(os.path.join(OUT, "gif_zref.npz"),
                        inp=np.stack(inp).astype(np.float16), ctrl=np.stack(ctrl).astype(np.float16),
                        g05=np.stack(g05).astype(np.float16), z_ref=z_ref, frames=np.array(sweep_frames))

    vmax = max(np.max(inp), np.max(ctrl), np.max(g05), 1e-3)
    cols = [("real-time input", inp, al["input"]),
            (f"control0 recon z={z_ref}", ctrl, al["control0"]),
            (f"gather05 recon z={z_ref}", g05, al["gather05"])]
    frames = []
    for i in range(N_SWEEP):
        fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=100)
        for c, (title, seq, a) in enumerate(cols):
            ax[c].imshow(seq[i], cmap="gray", vmin=0, vmax=vmax)
            ax[c].set_xticks([]); ax[c].set_yticks([])
            ax[c].set_title(f"{title}\naliveness={a:.3f}", fontsize=8)
        fig.suptitle(f"MIITT Volunteer1 — frame {ctx['frame_indices'][i]} ({i+1}/{N_SWEEP})", fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        frames.append(Image.open(buf).convert("RGB")); plt.close(fig)
    path = os.path.join(OUT, "input_vs_control_vs_gather.gif")
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=150, loop=0)
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
