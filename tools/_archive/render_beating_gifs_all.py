"""Beating-heart GIFs for the 3 reference-conditioned models (reference/diffusion/bspline),
covering BOTH the in-distribution val set and the OOD real-time datasets — the animated
companions to the static ED PNGs in result/ed_val_many_io and result/reference_models_io.

Each model is loaded ONCE and used for both parts:

  PART A — val (gated, GT available): sweep the queried cardiac phase t = 0..11 (slot 0 = the
    target-phase reference at the mid-ventricular plane), respiration ON. GIF frame = 5-z montage,
    GT row over pred row → a real corrupted->clean beating heart with ground truth.

  PART B — OOD (real-time, NO GT): the free-breathing cine is ungated, so there is no 12-phase
    bundle. Instead sweep the slot-0 reference over a consecutive window of real-time frames at
    the mid-ventricular plane; the model reconstructs the whole volume at each reference phase →
    a beating heart driven purely by the reference content. GIF frame = 5-z montage (pred only).

Run: micromamba run -n svr python tools/render_beating_gifs_all.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.render_reference_5dataset_io import MODELS, forward
from tools.render_cardiac_filmstrip_multislice import reconstruct_cycle, pick_zslices
from tools.five_row_compare import (
    OCMR_SUBJECTS, GOTT_SUBJECTS, MIITT_SUBJECTS, GOTT_RECON, MIITT_RECON,
    build_val_dataset,
)
from inference.adapters.base import (
    percentile_scale, assign_canonical_z, to_canonical_inplane, GRID_SHAPE,
)
from inference.adapters.ocmr import OCMRAdapter
from inference.adapters.goettingen import GoettingenAdapter
from inference.adapters.miitt import MIITTAdapter
from vggt.models.vggt import VGGT

DEV = torch.device("cuda")
D = GRID_SHAPE[0]
OUT = os.path.join(_ROOT, "result", "beating_gifs")
VAL_SUBJECTS = [0, 1, 2, 3, 7, 12]   # a spread of val subjects (GT available)
OOD_SWEEP = 20                        # consecutive RT frames swept for the OOD beat


def build_model(ckpt, head):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
    ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:5]} unexpected={unexp[:5]}"
    return m


def gif_montage(frames_volumes, zs, path, vmax, gt_volumes=None, dur=130):
    """frames_volumes: list of (D,H,W) pred per swept phase. Optional gt_volumes same length.
    Writes a GIF where each frame is a 5-z montage (pred; with GT row on top if provided)."""
    out = []
    for i, V in enumerate(frames_volumes):
        pred_row = np.concatenate([V[z] for z in zs], axis=1)
        if gt_volumes is not None:
            gt_row = np.concatenate([gt_volumes[i][z] for z in zs], axis=1)
            montage = np.concatenate([gt_row, pred_row], axis=0)
        else:
            montage = pred_row
        g = np.clip(montage / vmax, 0, 1)
        out.append(Image.fromarray((g * 255).astype(np.uint8)))
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0)
    print(f"    wrote {os.path.relpath(path, _ROOT)}", flush=True)


def ood_beating(model, adapter):
    """Sweep slot-0 reference over RT frames at the mid-ventricular plane → list of pred volumes."""
    cine = adapter.load()                      # (F, nS, H, W)
    F_total = cine.shape[0]
    vmin, vmax = percentile_scale(cine)
    inplane = adapter.inplane_mm()
    zmap = assign_canonical_z(adapter.slice_positions_mm())
    batch, S, picks = adapter.build_batch(np.random.default_rng(0), DEV)

    zc = [p[0] for p in picks]
    ref_slot = int(np.argmin([abs(z - (D - 1) / 2.0) for z in zc]))
    ref_slice_idx = picks[ref_slot][1]
    # move the mid-ventricular slot to position 0 (the reference anchor)
    for key in ("images", "scanner_coords", "z_indices"):
        idx = list(range(S)); idx[0], idx[ref_slot] = idx[ref_slot], idx[0]
        batch[key] = batch[key][:, idx].contiguous()

    win = min(OOD_SWEEP, F_total)
    start = max(0, F_total // 2 - win // 2)
    frames = list(range(start, start + win))

    vols = []
    for f in frames:
        norm = np.clip((cine[f, ref_slice_idx] - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)                       # (256,256)
        up = F.interpolate(canon[None, None], size=(518, 518),
                           mode="bilinear", align_corners=True)[0, 0]
        batch["images"][0, 0] = up.repeat(3, 1, 1).to(DEV)
        V_can, _ = forward(model, batch)
        vols.append(V_can)
    zlo, zhi = min(zc), max(zc)
    zs = sorted(set(int(round(z)) for z in np.linspace(zlo, zhi, 5)))
    return vols, zs


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.t_target_fixed = None  # multi-phase sweep for the gated beat

    ood_jobs = ([("OCMR", s, OCMRAdapter(os.path.join(_ROOT, "scratch/data/ocmr/recon", s)))
                 for s in OCMR_SUBJECTS]
                + [("Goett", s, GoettingenAdapter(os.path.join(GOTT_RECON, s, s + ".nii.gz")))
                   for s in GOTT_SUBJECTS]
                + [("MIITT", s, MIITTAdapter(os.path.join(MIITT_RECON, s, "realtime", "sax",
                                                          "4d_recon.nii.gz")))
                   for s in MIITT_SUBJECTS])

    for name, ckpt, head in MODELS:
        print(f"=== {name} ({head}) ===", flush=True)
        model = build_model(ckpt, head)
        vdir = os.path.join(OUT, "val", name); os.makedirs(vdir, exist_ok=True)
        odir = os.path.join(OUT, "ood", name); os.makedirs(odir, exist_ok=True)

        # PART A — gated val beating GIFs (GT + pred), respiration ON
        for subj in VAL_SUBJECTS:
            try:
                canon, gt, bb = reconstruct_cycle(model, val_ds, subj,
                                                  do_resp=True, reference_slot=True, resp_cfg=rcfg)
            except Exception as e:
                print(f"    skip val seq{subj}: {e}", flush=True); continue
            zs = pick_zslices(bb)
            vmax = float(max(max(v.max() for v in canon), max(v.max() for v in gt), 1e-3))
            gif_montage(canon, zs, os.path.join(vdir, f"val_seq{subj:02d}_beating.gif"),
                        vmax, gt_volumes=gt)

        # PART B — OOD beating GIFs (pred only), real-time reference sweep
        for ds_name, sub, adapter in ood_jobs:
            try:
                vols, zs = ood_beating(model, adapter)
            except Exception as e:
                print(f"    skip {ds_name}_{sub}: {e}", flush=True); continue
            vmax = float(max(max(v.max() for v in vols), 1e-3))
            gif_montage(vols, zs, os.path.join(odir, f"{ds_name}_{sub}_beating.gif"), vmax)
        del model
        torch.cuda.empty_cache()
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
