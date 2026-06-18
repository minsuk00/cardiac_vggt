#!/usr/bin/env python
"""Qualitative inference of the trained VGGT-MRI model on REAL OCMR real-time
free-breathing SAX cines (the gated->real-time transfer test; see docs/06).

There is NO ground-truth volume for OCMR (prospectively undersampled), so this is a
qualitative beating-heart check, not a metric. We:
  1. load a reconstructed OCMR cine (`recon/<subj>/sax_cine.nii.gz` + `meta.json`),
  2. adapt it into the model's canonical input contract (in-plane resample to 1.4 mm +
     256x256 crop/pad; z-spacing read from meta `slice_positions_mm`, NOT the 8 mm
     thickness in the header; one random frame per slice = the scattered single-frame
     regime; per-subject percentile intensity normalization matching preprocess.py),
  3. run the z-only model (use_t_pose_embedding=False -> input cardiac phase not needed),
  4. splat V_canon over a 12-phase target_t sweep and save a beating-heart GIF.

Repeats with several random input draws (`--draws`) so we can see whether *which*
cardiac/respiratory states get sampled changes the reconstruction.

Usage:
  micromamba run -n svr python tools/eval_ocmr_inference.py \
      [--ckpt PATH] [--subjects us_0084_1_5T ...] [--draws 3] [--out result/ocmr_eval]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
from PIL import Image

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))
from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions

INPUT_IMG_SIZE = 518
TARGET_INPLANE_MM = 1.4
GRID_SHAPE = (12, 256, 256)          # (D, H, W) canonical splat grid
D_CANON = GRID_SHAPE[0]
CANON_Z_SPACING_MM = 8.0             # canonical plane spacing
DEFAULT_CKPT = ("scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/"
                "ckpts/checkpoint_last.pt")
PCT_LO, PCT_HI = 0.5, 99.9           # matches ScaleIntensityByT0PercentilesD


# ───────────────────────── model ─────────────────────────
def load_model(ckpt_path, device):
    """Construct the z-only VGGT-MRI model and load weights (offline; full state dict)."""
    model = VGGT(
        img_size=INPUT_IMG_SIZE, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False,
        use_target_t_pose_embedding=True, train_on_residual_dvf=True,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    # frozen depth/track/camera heads are absent by config; tolerate only those.
    bad = [k for k in missing if "aggregator" in k or "point_head" in k]
    if bad:
        raise RuntimeError(f"missing critical weights: {bad[:5]} ...")
    print(f"  loaded {ckpt_path}  (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    return model.to(device).eval()


# ───────────────────────── data adapter ─────────────────────────
def load_cine(subj_dir):
    """-> cine[frame, slice, H, W] float32, meta dict."""
    img = sitk.ReadImage(os.path.join(subj_dir, "sax_cine.nii.gz"))
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # (frame, slice, H, W)
    meta = json.load(open(os.path.join(subj_dir, "meta.json")))
    return arr, meta


def percentile_scale(cine):
    """Single per-subject (vmin, vmax) over ALL nonzero voxels of the whole cine.
    Frame-selection-invariant so different random draws share one intensity scale.
    Mirrors preprocess.py's clip-and-rescale to [0, 1]."""
    nz = cine[cine > 0]
    vmin = np.percentile(nz, PCT_LO)
    vmax = np.percentile(nz, PCT_HI)
    return float(vmin), float(max(vmax, vmin + 1e-6))


def assign_canonical_z(positions):
    """Map each physical slice to a canonical z-index using the TRUE slice spacing
    (center-to-center along the stack axis, ~10 mm), not the 8 mm thickness.
    Returns list of (slice_idx, z_canon_idx) for slices landing in [0, D-1];
    on collision keeps the slice closest to that plane center. No through-plane interp."""
    pos = np.asarray(positions, dtype=np.float64)       # (nS, 3) scanner mm
    axis = pos[-1] - pos[0]
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    d = (pos - pos[0]) @ axis                            # signed depth along stack (mm)
    d = d - d.mean()                                     # center the stack
    cont = d / CANON_Z_SPACING_MM + (D_CANON - 1) / 2.0  # continuous canonical index
    idx = np.round(cont).astype(int)
    best = {}                                            # z_canon -> (slice, |residual|)
    for s, (k, c) in enumerate(zip(idx, cont)):
        if 0 <= k <= D_CANON - 1:
            res = abs(c - k)
            if k not in best or res < best[k][1]:
                best[k] = (s, res)
    return sorted((k, s) for k, (s, _) in best.items())  # [(z_canon, slice_idx), ...]


def to_canonical_inplane(slice2d, inplane_mm):
    """(H, W) at native in-plane mm -> (256, 256) at 1.4 mm (bilinear resample + center
    crop/pad), matching Spacingd + ResizeWithPadOrCropd."""
    H, W = slice2d.shape
    sh = int(round(H * inplane_mm[1] / TARGET_INPLANE_MM))
    sw = int(round(W * inplane_mm[0] / TARGET_INPLANE_MM))
    t = torch.from_numpy(slice2d)[None, None].float()
    r = F.interpolate(t, size=(sh, sw), mode="bilinear", align_corners=True)[0, 0]
    out = torch.zeros(256, 256)
    # center crop/pad
    y0s, x0s = max(0, (sh - 256) // 2), max(0, (sw - 256) // 2)
    y0d, x0d = max(0, (256 - sh) // 2), max(0, (256 - sw) // 2)
    hh, ww = min(sh, 256), min(sw, 256)
    out[y0d:y0d + hh, x0d:x0d + ww] = r[y0s:y0s + hh, x0s:x0s + ww]
    return out  # (256, 256), values already normalized [0,1]


def build_batch(cine, meta, scale, z_map, rng, device):
    """One random frame per slice -> model batch (images[0,1], scanner_coords, z_indices)."""
    vmin, vmax = scale
    n_frames = cine.shape[0]
    inplane = meta["inplane_mm"]
    py, px = np.meshgrid(np.arange(INPUT_IMG_SIZE), np.arange(INPUT_IMG_SIZE), indexing="ij")
    x_norm = (px / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)
    y_norm = (py / (INPUT_IMG_SIZE - 1) * 2.0 - 1.0).astype(np.float32)

    imgs, coords, z_idx, picks = [], [], [], []
    for z_canon, slice_idx in z_map:
        f = int(rng.integers(n_frames))
        raw = cine[f, slice_idx]
        norm = np.clip((raw - vmin) / (vmax - vmin), 0.0, 1.0)
        canon = to_canonical_inplane(norm, inplane)                  # (256,256) [0,1]
        up = F.interpolate(canon[None, None], size=(INPUT_IMG_SIZE, INPUT_IMG_SIZE),
                           mode="bilinear", align_corners=True)[0, 0].numpy()
        imgs.append(np.repeat(up[None], 3, axis=0))                  # (3,518,518)
        z_val = z_canon / max(1, D_CANON - 1) * 2.0 - 1.0
        coords.append(np.stack([x_norm, y_norm, np.full_like(x_norm, z_val)], -1))
        z_idx.append([z_val])
        picks.append((z_canon, slice_idx, f, up))
    S = len(imgs)
    batch = {
        "images": torch.from_numpy(np.stack(imgs)).float()[None].to(device),          # (1,S,3,518,518) [0,1]
        "scanner_coords": torch.from_numpy(np.stack(coords)).float()[None].to(device),  # (1,S,518,518,3)
        "z_indices": torch.tensor(z_idx, dtype=torch.float32)[None].to(device),        # (1,S,1)
    }
    return batch, S, picks


# ───────────────────────── inference + render ─────────────────────────
@torch.no_grad()
def reconstruct_cycle(model, batch, S, device):
    """Sweep target_t over 12 phases -> V_canon[t] (D,H,W). Same inputs, varying query."""
    vols = []
    wp_by_t = []  # predicted world_points per phase (for the DVF figure)
    for t in range(D_CANON):
        t_norm = t / max(1, D_CANON) * 2.0 - 1.0
        batch["target_t_indices"] = torch.full((1, S, 1), t_norm, dtype=torch.float32, device=device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        wp = preds["world_points"].float()
        V_canon, _ = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
        vols.append(V_canon[0].float().cpu().numpy())
        wp_by_t.append(wp[0].cpu().numpy())  # (S, 518, 518, 3)
    return vols, wp_by_t  # list of (D,H,W), list of (S,518,518,3)


# in-plane: (256-1)/2 * 1.4 mm ; through-plane: (12-1)/2 * 8.0 mm  (norm[-1,1] -> mm)
MM_PER_NORM = (178.5, 178.5, 44.0)


def save_dvf_png(world_points, coords, picks, path, t=0):
    """Per-slot predicted displacement Δ = world_points - scanner_coords, in mm
    (Δx/Δy/Δz rows), masked to anatomy. Mirrors training's _DVF figure. No GT needed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    delta = (world_points - coords)                    # (S,518,518,3) normalized
    S = delta.shape[0]
    labels = ["Δx (mm)", "Δy (mm)", "Δz (mm)"]
    fig, axes = plt.subplots(4, S, figsize=(1.7 * S, 7.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for s in range(S):
        z_canon, slice_idx, f, up = picks[s]
        mask = up > 0.03
        axes[0][s].imshow(up, cmap="gray")
        axes[0][s].set_title(f"z{z_canon} s{slice_idx} f{f}", fontsize=7)
        for c in range(3):
            dm = delta[s, ..., c] * MM_PER_NORM[c]
            vlim = float(np.percentile(np.abs(dm[mask]), 99)) if mask.any() else 1.0
            vlim = max(vlim, 1e-3)
            disp = np.where(mask, dm, np.nan)
            im = axes[c + 1][s].imshow(disp, cmap="bwr", vmin=-vlim, vmax=vlim)
            if s == 0:
                axes[c + 1][s].set_ylabel(labels[c], fontsize=9)
            if s == S - 1:
                fig.colorbar(im, ax=axes[c + 1][s], fraction=0.046, pad=0.02)
    axes[0][0].set_ylabel("input slice", fontsize=9)
    fig.suptitle(f"{os.path.basename(path)}  predicted DVF (target t={t}, anatomy-masked)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def _u8(a, vmax):
    return np.clip(a / vmax * 255.0, 0, 255).astype(np.uint8)


def save_cycle_gif(vols, path, mid_d=None):
    """Mid-z V_canon across the 12 target phases -> beating-heart GIF."""
    if mid_d is None:
        mid_d = vols[0].shape[0] // 2
    vmax = max(float(v[mid_d].max()) for v in vols) or 1e-3
    frames = [Image.fromarray(_u8(v[mid_d], vmax)) for v in vols]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def save_inputs_png(picks, path):
    """Contact sheet of the S input slices actually fed (orientation/quality sanity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(picks)
    cols = min(n, 6); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (z_canon, slice_idx, f, up) in enumerate(picks):
        ax = axes[i // cols][i % cols]
        ax.imshow(up, cmap="gray"); ax.axis("off")
        ax.set_title(f"z{z_canon} s{slice_idx} f{f}", fontsize=7)
    fig.suptitle(os.path.basename(path), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_volume_png(vols, path, t=0):
    """All 12 canonical z-planes of V_canon at one target phase (volume coverage check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = vols[t]; D = V.shape[0]; vmax = float(V.max()) or 1e-3
    fig, axes = plt.subplots(2, (D + 1) // 2, figsize=(1.6 * ((D + 1) // 2), 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for z in range(D):
        ax = axes[z // ((D + 1) // 2)][z % ((D + 1) // 2)]
        ax.imshow(V[z], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(f"z={z}", fontsize=7)
    fig.suptitle(f"{os.path.basename(path)}  (target t={t})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--recon_dir", default="scratch/data/ocmr/recon")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all in recon_dir")
    ap.add_argument("--draws", type=int, default=3, help="random input draws per subject")
    ap.add_argument("--out", default="result/ocmr_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_model(args.ckpt, device)

    subj_dirs = ([os.path.join(args.recon_dir, s) for s in args.subjects] if args.subjects
                 else sorted(d for d in glob.glob(os.path.join(args.recon_dir, "*")) if os.path.isdir(d)))
    os.makedirs(args.out, exist_ok=True)

    for sd in subj_dirs:
        name = os.path.basename(sd)
        if not os.path.exists(os.path.join(sd, "sax_cine.nii.gz")):
            continue
        cine, meta = load_cine(sd)
        scale = percentile_scale(cine)
        z_map = assign_canonical_z(meta["slice_positions_mm"])
        sp = np.median(np.linalg.norm(np.diff(np.asarray(meta["slice_positions_mm"]), axis=0), axis=1))
        print(f"[{name}] {cine.shape[1]} slices, {cine.shape[0]} frames, "
              f"spacing~{sp:.1f}mm -> {len(z_map)} canonical planes {[z for z,_ in z_map]}", flush=True)
        odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
        for d in range(args.draws):
            rng = np.random.default_rng(args.seed + d)
            batch, S, picks = build_batch(cine, meta, scale, z_map, rng, device)
            coords0 = batch["scanner_coords"][0].cpu().numpy()  # (S,518,518,3)
            vols, wp_by_t = reconstruct_cycle(model, batch, S, device)
            save_cycle_gif(vols, os.path.join(odir, f"draw{d}_cycle.gif"))
            save_inputs_png(picks, os.path.join(odir, f"draw{d}_inputs.png"))
            if d == 0:
                save_volume_png(vols, os.path.join(odir, "draw0_volume_t0.png"))
                save_dvf_png(wp_by_t[0], coords0, picks, os.path.join(odir, "draw0_dvf_t0.png"), t=0)
            print(f"  draw {d}: S={S} -> {odir}/draw{d}_cycle.gif", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
