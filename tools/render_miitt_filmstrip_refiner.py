"""Cardiac-cycle filmstrip on MIITT for the SSIM refiner (frozen-backbone, target_t) model
217891050_mri_refiner_frozen_ssim_newseed.

This model is the LEGACY target_t contract (use_target_t_pose_embedding=True, NO reference slot)
and predates the num_freqs 6->3 change (commit 1856a82), so its z/target_t embedders are the
num_freqs=6 (13-dim) basis. We override ONLY this script's model instance to num_freqs=6 at
runtime — vggt/models/aggregator.py is NOT modified, so current/future runs are unaffected.

Per MIITT subject: build the scattered RT batch once, sweep the query target_t = 0..11, take the
refiner's V_refined at each phase, and render a beating-heart filmstrip (z x phase grid) + GIF.

Run: micromamba run -n svr python tools/render_miitt_filmstrip_refiner.py
"""
import os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "training"))

from inference.adapters.miitt import MIITTAdapter
from vggt.models.vggt import VGGT
from vggt.models.aggregator import ZIndexEmbedder, TIndexEmbedder

DEV = torch.device("cuda")
CKPT = os.path.join(_ROOT, "scratch/logs/217891050_mri_refiner_frozen_ssim_newseed/ckpts/checkpoint_last.pt")
MIITT_RECON = os.path.join(_ROOT, "scratch/data/MIITT/nifti")
SUBJECTS = ["Volunteer1", "Volunteer2", "Volunteer3", "Volunteer4", "Volunteer5"]
OUT = os.path.join(_ROOT, "result", "miitt_filmstrip_ssim_refiner")
GRID = (12, 256, 256)
T = 12
Z_ROWS = [4, 6, 8]  # apex-ish / mid-ventricular / base-ish canonical planes


def build_model():
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False, use_target_t_pose_embedding=True,
             train_on_residual_dvf=True, enable_refiner=True, refiner_use_coverage=True,
             grid_shape=GRID).to(DEV).eval()
    # runtime-only num_freqs=6 override (matches this legacy ckpt; aggregator.py default stays 3)
    m.aggregator.z_embedder = ZIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    m.aggregator.target_t_embedder = TIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    ck = torch.load(CKPT, map_location=DEV, weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:5]} unexpected={unexp[:5]}"
    print(f"  loaded 217891050 (refiner, target_t, num_freqs=6): clean", flush=True)
    return m


def v_refined(m, batch, target_t):
    S = batch["images"].shape[1]
    batch["target_t_indices"] = torch.full((1, S, 1), target_t, dtype=torch.float32, device=DEV)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = m(batch["images"], batch=batch)
    return preds["V_refined"][0].float().cpu().numpy()  # (12,256,256)


def win(vols):
    """consistent intensity window across all phases (so contraction is visible, not re-normalized)."""
    allv = np.concatenate([v[v > 0].ravel() for v in vols if (v > 0).any()])
    hi = float(np.percentile(allv, 99.5)); lo = float(np.percentile(allv, 1.0))
    return lo, hi


def render_grid(vols, subj, path):
    lo, hi = win(vols)
    nr, nc = len(Z_ROWS), T
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 1.5, nr * 1.5))
    for r, z in enumerate(Z_ROWS):
        for k in range(T):
            ax = axes[r, k]
            ax.imshow(np.clip((vols[k][z] - lo) / (hi - lo + 1e-9), 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t{k}", fontsize=9)
            if k == 0:
                ax.set_ylabel(f"z={z}", fontsize=9)
    fig.suptitle(f"MIITT {subj} — SSIM refiner (target_t sweep): V_refined across the cardiac cycle",
                 fontsize=12)
    fig.savefig(path, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def render_gif(vols, subj, path):
    lo, hi = win(vols)
    frames = []
    for k in range(T):
        montage = np.concatenate([vols[k][z] for z in Z_ROWS], axis=1)   # 3 z side by side
        g = np.clip((montage - lo) / (hi - lo + 1e-9), 0, 1)
        frames.append(Image.fromarray((g * 255).astype(np.uint8)))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=120, loop=0)
    print(f"  wrote {path}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = build_model()
    for subj in SUBJECTS:
        nii = os.path.join(MIITT_RECON, subj, "realtime", "sax", "4d_recon.nii.gz")
        if not os.path.exists(nii):
            print(f"  skip {subj}: no recon"); continue
        batch = MIITTAdapter(nii).build_batch(np.random.default_rng(0), DEV)[0]
        vols = [v_refined(m, batch, k / T * 2.0 - 1.0) for k in range(T)]
        render_grid(vols, subj, os.path.join(OUT, f"{subj}_filmstrip.png"))
        render_gif(vols, subj, os.path.join(OUT, f"{subj}_beating.gif"))
    print(f"done -> {OUT}")


if __name__ == "__main__":
    main()
