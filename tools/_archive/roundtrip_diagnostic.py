#!/usr/bin/env python
"""Round-trip / analysis-by-synthesis diagnostic for the splat renderer (loss-of-detail probe).

Idea (user's): after the model predicts per-pixel world_points p = scanner_coords + Δ and we
splat the input intensities into V_canon, we can SAMPLE V_canon back out at those same p and
compare to the original input slice I:

    resampled[s] = sample_volume(V_canon, world_points[s])          # (S, H, W)
    round_trip_err = | I - resampled |

If a pixel were the ONLY contributor near p, splat-then-sample is the identity (you get I back).
So any gap is the renderer's coverage-division AVERAGING — other input frames (at DIFFERENT cardiac
phases, same plane) landing in the same voxel and being averaged in. That is the per-pixel,
exactly-localized measurement of the "loss of detail" the splat causes (docs 08/10/13 measured it
only in aggregate, ~75% of the blur).

On MIITT real-time input there is NO ground-truth volume (prospectively acquired), so the V_gt(p)
arm (placement vs appearance-wall) is N/A here — that requires the in-distribution CMRxRecon val
path. This script does the GT-free 2-way (I vs V_canon(p)) on MIITT Volunteer1 for three models:
  gather05  — gather_weight=0.5 placement aux (docs/37), S=12 1-frame training, snap-z
  control0  — gather_weight=0.0 (clean A/B twin),               S=12 1-frame training, snap-z
  s20contz  — S=20 multiframe + continuous-z training,           fractional-z

Read-only; run inline on the GPU node:
  micromamba run -n svr python tools/roundtrip_diagnostic.py
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # ckpt paths in base.py are repo-relative

from math import ceil

from inference.adapters import MIITTAdapter
from inference.inference import forward
from vggt.models.vggt import VGGT
from vggt.utils.splat import sample_volume
from inference.adapters.base import GRID_SHAPE

ROWS_PER_PAGE = 20  # paginate the all-slots figure so each PNG stays openable


def fast_load_reference(ckpt, device):
    """Reference-slot VGGT-MRI model, loaded with mmap so only the `model` tensors are
    read from the 8.8 GB checkpoint (the optimizer half is never touched) — cuts the
    GPFS load from ~17 min. Same construction as inference.inference.load_rtfb_model_reference."""
    model = VGGT(
        img_size=518, patch_size=14, embed_dim=1024,
        enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
        use_z_pose_embedding=True, use_t_pose_embedding=False,
        use_target_t_pose_embedding=False, use_reference_token=True, train_on_residual_dvf=True,
        enable_refiner=False, refiner_use_coverage=False, grid_shape=GRID_SHAPE,
    )
    try:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    missing, _ = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if any(s in k for s in ("aggregator", "point_head", "z_embedder"))]
    if bad:
        raise RuntimeError(f"missing critical weights: {bad[:5]} ...")
    return model.to(device).eval()

DEVICE = "cuda"
SUBJECT = "Volunteer1"
NII = f"scratch/data/MIITT/nifti/{SUBJECT}/realtime/sax/4d_recon.nii.gz"
OUT = "result/roundtrip_miitt"
os.makedirs(OUT, exist_ok=True)

MODELS = [
    ("control0", "scratch/logs/216539845_mri_volume_diffusion_ftctrl_gather0_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", False),
    ("gather05", "scratch/logs/216539845_mri_volume_diffusion_ftgather05_1frame_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", False),
    ("s20contz", "scratch/logs/216949414_mri_volume_diffusion_s20contz_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", True),
]


@torch.no_grad()
def run_model(name, ckpt, continuous_z):
    t0 = time.time()
    print(f"\n===== {name}  (continuous_z={continuous_z}) =====", flush=True)
    model = fast_load_reference(ckpt, DEVICE)
    print(f"  [t+{time.time()-t0:.0f}s] model loaded", flush=True)
    adapter = MIITTAdapter(NII)
    batch, S, picks, ref_ctx = adapter.build_batch_multiframe(
        DEVICE, frames_per_slice=5, frames_for_reference=30, continuous_z=continuous_z)

    out = forward(model, batch, want=("V_canon", "world_points", "coverage"), device=DEVICE)
    V_canon = torch.from_numpy(out["V_canon"]).to(DEVICE)          # (D,H,W)
    wp = torch.from_numpy(out["world_points"]).to(DEVICE)          # (S,518,518,3)
    coverage = out["coverage"]                                     # (D,H,W) np

    # Input intensity actually splatted (mean over RGB), same as splat_predictions.
    I = batch["images"][0].mean(dim=1)                             # (S,518,518) in [0,1]

    # Round-trip: sample V_canon at the predicted coords.
    Sn, H, W = I.shape
    pos = wp.reshape(1, Sn * H * W, 3)
    resampled = sample_volume(V_canon.unsqueeze(0), pos).reshape(Sn, H, W)  # (S,518,518)

    I_np = I.cpu().numpy()
    Rp = resampled.cpu().numpy()                                  # V_canon @ pred coords (+Δ), round-trip
    # V_canon @ the slice's OWN home plane (Δ=0): the target-phase reconstruction AT this
    # input's nominal location. For the reference slot this is the recon at z_mid vs the
    # reference input — the "is the observed reference preserved?" panel.
    scan = batch["scanner_coords"][0]                             # (S,518,518,3)
    Rh = sample_volume(V_canon.unsqueeze(0),
                       scan.reshape(1, Sn * H * W, 3)).reshape(Sn, H, W).cpu().numpy()
    gate = I_np > 1e-3                                            # acquired (non-background) pixels
    err_h = np.abs(I_np - Rh)                                     # motion @ plane (recon vs input phase)
    err_p = np.abs(I_np - Rp)                                     # renderer round-trip
    mae = float(err_p[gate].mean()) if gate.any() else float("nan")
    mae_h = float(err_h[gate].mean()) if gate.any() else float("nan")
    rng = float(np.percentile(I_np[gate], 99) - np.percentile(I_np[gate], 1)) if gate.any() else 1.0
    print(f"  S={Sn}  MAE round-trip(pred)={mae:.4f}  MAE motion@plane(home)={mae_h:.4f}  "
          f"(input p1-99 range {rng:.3f})", flush=True)
    print(f"  coverage>1e-3 frac = {float((coverage>1e-3).mean()):.3f}", flush=True)

    # Dump arrays (float16) so any future figure change re-plots instantly (no reload).
    zf = np.array([float(p[0]) for p in picks]); slf = np.array([int(p[1]) for p in picks])
    ff = np.array([int(p[2]) for p in picks])
    np.savez_compressed(os.path.join(OUT, f"{name}_arrays.npz"),
                        I=I_np.astype(np.float16), Rh=Rh.astype(np.float16), Rp=Rp.astype(np.float16),
                        z=zf, slice_idx=slf, frame=ff, mae=mae, mae_h=mae_h,
                        cov=float((coverage > 1e-3).mean()))

    # ── ALL slots, 5 columns, paginated ───────────────────────────────────────────
    COLS = ["input I", "V_canon @ home (Δ=0)", "V_canon @ pred (+Δ)",
            "|I - home|  motion@plane (0-0.3)", "|I - pred|  round-trip (0-0.3)"]
    n_pages = ceil(Sn / ROWS_PER_PAGE)
    for pg in range(n_pages):
        s0, s1 = pg * ROWS_PER_PAGE, min(Sn, (pg + 1) * ROWS_PER_PAGE)
        nr = s1 - s0
        fig, ax = plt.subplots(nr, 5, figsize=(13.0, 2.55 * nr), dpi=80)
        ax = np.atleast_2d(ax)
        for r, s in enumerate(range(s0, s1)):
            a = ax[r]
            a[0].imshow(I_np[s], cmap="gray", vmin=0, vmax=1)
            a[0].set_ylabel(f"slot{s}\nz={zf[s]:.1f} f={ff[s]}" + ("\nREF" if s == 0 else ""), fontsize=6)
            a[1].imshow(Rh[s], cmap="gray", vmin=0, vmax=1)
            a[2].imshow(Rp[s], cmap="gray", vmin=0, vmax=1)
            a[3].imshow(err_h[s], cmap="magma", vmin=0, vmax=0.3)
            a[4].imshow(err_p[s], cmap="magma", vmin=0, vmax=0.3)
            for c, aa in enumerate(a):
                aa.set_xticks([]); aa.set_yticks([])
                if r == 0:
                    aa.set_title(COLS[c], fontsize=7)
        fig.suptitle(f"{SUBJECT} — {name}  slots {s0}-{s1-1}/{Sn}  |  "
                     f"MAE round-trip={mae:.4f}  motion@plane={mae_h:.4f}  cov={float((coverage>1e-3).mean()):.2f}",
                     fontsize=9)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        p = os.path.join(OUT, f"{name}_all_p{pg:02d}.png")
        fig.savefig(p); plt.close(fig)
        print(f"  wrote {p}  ({nr} slots)", flush=True)

    del model
    torch.cuda.empty_cache()
    return dict(name=name, S=Sn, mae=mae, mae_h=mae_h, cov=float((coverage > 1e-3).mean()))


def main():
    rows = []
    for name, ckpt, cz in MODELS:
        rows.append(run_model(name, ckpt, cz))
    print("\n================ SUMMARY ================")
    print(f"{'model':10s} {'S':>4s} {'RT-MAE':>8s} {'motion@pl':>10s} {'cov':>6s}")
    for r in rows:
        print(f"{r['name']:10s} {r['S']:>4d} {r['mae']:>8.4f} {r['mae_h']:>10.4f} {r['cov']:>6.3f}")


if __name__ == "__main__":
    main()
