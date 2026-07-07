"""DVF (Δx/Δy/Δz mm) WITH the ground-truth respiratory displacement per slot, for the 3
reference-conditioned models on the CMRxRecon val set at ED, respiration ON.

Purpose: check whether the model actually LEARNS the breathing motion. The respiratory sim
applies a known per-slot rigid shift d (mm); gpu_augment surfaces it as batch["resp_disp_mm"]
(S,3) = (d_D, d_H, d_W) canonical axes. Sign convention (respiratory.py): the breath reslices
the input by +d, so a model that CORRECTS it should predict Δ ≈ +d (same vector). Through-plane
(Δz vs SI=d_D) is the clean test — cardiac motion is mostly in-plane, so the z axis is dominated
by breathing; in-plane axes mix breath-AP with cardiac motion.

Per (model, subject) panel:
  rows 0-3 : input intensity / Δx / Δy / Δz maps over the S input slots (slot 0 = reference).
  rows 4-6 : per-slot bar comparison, one axis each (x, y, z) — GT applied breath vs the model's
             mean predicted Δ over that slot's anatomy. If the model learns motion, the z bars
             (pred Δz vs applied SI) line up.

Run: micromamba run -n svr python tools/render_ed_val_dvf_gt.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.render_reference_5dataset_io import MODELS, forward, IN_PLANE_MM, THROUGH_MM, IN_PLANE_R, THROUGH_R
from tools.five_row_compare import DEV, val_batch, build_val_dataset
from vggt.models.vggt import VGGT

OUT = os.path.join(_ROOT, "result", "ed_val_dvf_gt")
N_SUBJECTS = 30
ANAT_THR = 0.05  # input-intensity threshold for the "anatomy" mask used in the pred mean


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


def render(batch, pred_dvf, applied_dhw, title, path):
    """pred_dvf (S,H,W,3) normalized; applied_dhw (S,3) mm = (d_D,d_H,d_W)."""
    imgs = batch["images"][0].detach().float().cpu().clamp(0, 1).mean(dim=1).numpy()  # (S,H,W)
    S = imgs.shape[0]
    t_picks = batch["timesteps"][0].cpu().numpy()
    z_picks = batch["slice_indices"][0].cpu().numpy()

    # GT applied breath in (x,y,z)=(d_W,d_H,d_D) mm; model Δ scaled to mm per axis.
    gt_xyz = np.stack([applied_dhw[:, 2], applied_dhw[:, 1], applied_dhw[:, 0]], axis=1)  # (S,3)
    dvf_mm = pred_dvf * np.array([IN_PLANE_MM, IN_PLANE_MM, THROUGH_MM], np.float32)       # (S,H,W,3)
    pred_mean = np.zeros((S, 3), np.float32)
    for s in range(S):
        m = imgs[s] > ANAT_THR
        if m.any():
            pred_mean[s] = dvf_mm[s][m].mean(axis=0)

    p50, p95, p99 = (float(np.percentile(np.abs(pred_dvf), q)) for q in (50, 95, 99))
    fig = plt.figure(figsize=(1.9 * S + 1.6, 14.5), dpi=140)
    gs = gridspec.GridSpec(7, S + 1, width_ratios=[1.0] * S + [0.05],
                           height_ratios=[1, 1, 1, 1, 1.1, 1.1, 1.1], wspace=0.05, hspace=0.30)
    fig.suptitle(f"{title}    |Δ|(norm) p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}", fontsize=11)

    # ── rows 0-3: map rows ────────────────────────────────────────────────
    map_rows = [
        ("input", imgs,                  "gray",   0,           1.0,        True),
        ("Δx (mm)", dvf_mm[..., 0],      "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
        ("Δy (mm)", dvf_mm[..., 1],      "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
        ("Δz (mm)", dvf_mm[..., 2],      "RdBu_r", -THROUGH_R,  THROUGH_R,  False),
    ]
    for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(map_rows):
        last = None
        for s in range(S):
            ax = fig.add_subplot(gs[r, s])
            last = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if is_top:
                ax.set_title(f"t{int(t_picks[s])} z{int(z_picks[s])}"
                             + ("\n[ref]" if s == 0 else ""), fontsize=8)
            if s == 0:
                ax.set_ylabel(lbl, fontsize=9)
        plt.colorbar(last, cax=fig.add_subplot(gs[r, S]))

    # ── rows 4-6: per-slot applied-vs-predicted bar comparison, one axis each ──
    axis_names = ["x (in-plane)", "y (in-plane / AP)", "z (through-plane / SI)"]
    x = np.arange(S)
    for a, aname in enumerate(axis_names):
        ax = fig.add_subplot(gs[4 + a, :S])
        ax.bar(x - 0.2, gt_xyz[:, a], width=0.4, label="GT applied breath", color="#444")
        ax.bar(x + 0.2, pred_mean[:, a], width=0.4, label="pred mean Δ (anatomy)", color="#d62728")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(f"Δ{aname}\n(mm)", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t{int(t_picks[s])}z{int(z_picks[s])}" + ("*" if s == 0 else "")
                            for s in range(S)], fontsize=7, rotation=45)
        if a == 0:
            ax.legend(fontsize=8, ncol=2, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
    fig.text(0.5, 0.075, "slot 0 (*) = reference;  GT applied is the rigid breath shift, "
             "pred mean Δ averages the model's displacement over the slot's anatomy. "
             "Through-plane z (SI) is the clean breathing test.", ha="center", fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _ROOT)}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.t_target_fixed = 0  # ED

    for name, ckpt, head in MODELS:
        print(f"=== {name} ({head}) ===", flush=True)
        model = build_model(ckpt, head)
        mdir = os.path.join(OUT, name); os.makedirs(mdir, exist_ok=True)
        for seq in range(N_SUBJECTS):
            try:
                batch = val_batch(val_ds, rcfg, seq, breathing=True)
            except Exception as e:
                print(f"  skip seq{seq}: {e}", flush=True); continue
            applied = batch["resp_disp_mm"][0].detach().float().cpu().numpy()  # (S,3) mm (D,H,W)
            _V, dvf = forward(model, batch)
            render(batch, dvf, applied,
                   f"DVF + GT breath — {name} · val seq{seq} (ED, resp ON)",
                   os.path.join(mdir, f"val_seq{seq:02d}_ED_resp_dvf_gt.png"))
        del model
        torch.cuda.empty_cache()
    print(f"\ndone -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
