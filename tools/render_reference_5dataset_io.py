"""5-dataset IO + DVF panels for the reference-conditioned models (reference/diffusion/bspline).

Same structure as result/4way_refiner/refined_io_slices: 5 examples each from
  val (breath ON), val (breath OFF), OCMR, Göttingen, MIITT.
Per example, per model, writes to result/reference_models_io/{model}/:
  {mode}_{label}_io.png   — INPUT slices (left) | red line | V_canon reconstruction (right), 3x4 each.
  {mode}_{label}_dvf.png  — per-slot input intensity + Δx/Δy/Δz (mm), trainer-style.

val is in-contract (slot 0 = target-phase reference, from the mri_volume config's reference_slot).
OCMR/Göttingen/MIITT are OOD real-time free-breathing: slot 0 = a real acquired slice acting as the
anchor (the realistic "use a real frame as the reference" inference, brief §6 / docs/25).

Run: micromamba run -n svr python tools/render_reference_5dataset_io.py
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.five_row_compare import (
    DEV, VAL_SEQS, OCMR_SUBJECTS, GOTT_SUBJECTS, MIITT_SUBJECTS, MIITT_RECON, GRID_SHAPE,
    val_batch, ocmr_batch, goettingen_batch, miitt_batch, build_val_dataset, build_gott_dataset,
)
from vggt.models.vggt import VGGT
from vggt.utils.splat import splat_predictions

D = GRID_SHAPE[0]
OUT_ROOT = os.path.join(_ROOT, "result", "reference_models_io")
MODELS = [
    ("reference", "scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
    ("diffusion", "scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "dpt"),
    ("bspline",   "scratch/logs/217719798_mri_volume_bspline_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt", "bspline"),
]
IN_PLANE_MM = (256 - 1) / 2.0 * 1.4
THROUGH_MM = (12 - 1) / 2.0 * 12.0
IN_PLANE_R, THROUGH_R = 15.0, 25.0


def input_volume(batch):
    imgs = batch["images"][0]                       # (S,3,518,518) in [0,1]
    z = batch["z_indices"][0, :, 0].float().cpu().numpy()
    V = np.zeros((D, 256, 256), np.float32)
    for s in range(imgs.shape[0]):
        zi = int(round((z[s] + 1) / 2 * (D - 1)))
        if not (0 <= zi <= D - 1):
            continue
        sl = imgs[s, 0].float().cpu()
        V[zi] = F.interpolate(sl[None, None], size=(256, 256), mode="bilinear",
                              align_corners=True)[0, 0].numpy()
    return V


def window_pct(V):
    nz = V[V > 0]; ref = nz if nz.size else V
    hi = float(np.percentile(ref, 99.5)); lo = float(np.percentile(ref, 1.0))
    return np.clip((V - lo) / (hi - lo + 1e-9), 0, 1)


def render_io(Vin, Vout, title, path):
    Vin_w, Vout_w = window_pct(Vin), window_pct(Vout)
    fig = plt.figure(figsize=(8 * 2.6 + 0.5, 3 * 2.6))
    gs = gridspec.GridSpec(3, 9, figure=fig,
                           width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1], wspace=0.05, hspace=0.12)
    for Vw, c0, tag in [(Vin_w, 0, "in"), (Vout_w, 5, "V_canon")]:
        for k in range(D):
            r, c = k // 4, c0 + (k % 4)
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(Vw[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{tag} z={k}", fontsize=8); ax.axis("off")
    sep = fig.add_subplot(gs[:, 4]); sep.set_xlim(0, 1); sep.set_ylim(0, 1)
    sep.axvline(0.5, color="red", lw=3); sep.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def render_dvf(batch, pred_dvf, title, path):
    imgs = batch["images"][0].detach().float().cpu().clamp(0, 1).mean(dim=1).numpy()
    S = imgs.shape[0]
    t_picks = batch["timesteps"][0].cpu().numpy() if "timesteps" in batch else None
    z_picks = batch["slice_indices"][0].cpu().numpy() if "slice_indices" in batch else None
    p50, p95, p99 = (float(np.percentile(np.abs(pred_dvf), q)) for q in (50, 95, 99))
    fig = plt.figure(figsize=(1.9 * S + 1.6, 8.5), dpi=160)
    gs = gridspec.GridSpec(4, S + 1, width_ratios=[1.0] * S + [0.05], wspace=0.04, hspace=0.18)
    fig.suptitle(f"{title}    |Δ|(norm) p50={p50:.3f} p95={p95:.3f} p99={p99:.3f}", fontsize=10)
    rows = [
        ("input intensity", imgs,                          "gray",   0,          1.0,       True),
        ("Δx (mm)",         pred_dvf[..., 0] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
        ("Δy (mm)",         pred_dvf[..., 1] * IN_PLANE_MM, "RdBu_r", -IN_PLANE_R, IN_PLANE_R, False),
        ("Δz (mm)",         pred_dvf[..., 2] * THROUGH_MM,  "RdBu_r", -THROUGH_R,  THROUGH_R,  False),
    ]
    for r, (lbl, data, cmap, vmin, vmax, is_top) in enumerate(rows):
        last_im = None
        for s in range(S):
            ax = fig.add_subplot(gs[r, s])
            last_im = ax.imshow(data[s], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if is_top:
                ttl = (f"t={int(t_picks[s])}, z={int(z_picks[s])}"
                       if t_picks is not None and z_picks is not None else f"slot {s}")
                ax.set_title(ttl + ("  [ref]" if s == 0 else ""), fontsize=8)
            if s == 0:
                ax.set_ylabel(lbl, fontsize=9)
        plt.colorbar(last_im, cax=fig.add_subplot(gs[r, S]))
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    print(f"  wrote {path}", flush=True)


def forward(model, batch):
    S = batch["images"].shape[1]
    batch.setdefault("target_t_indices", torch.full((1, S, 1), -1.0, dtype=torch.float32, device=DEV))
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        preds = model(batch["images"], batch=batch)
    wp = preds["world_points"].float()
    V_can = preds.get("V_canon")
    if V_can is None:
        V_can, _ = splat_predictions({"world_points": wp}, batch, GRID_SHAPE)
    V_can = V_can[0].float().cpu().numpy()
    dvf = (wp[0] - batch["scanner_coords"][0]).detach().float().cpu().numpy()
    return V_can, dvf


def main():
    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    # (mode, label, build_fn) — 5 examples per dataset
    jobs = []
    for i in VAL_SEQS:
        jobs.append(("val_ON", f"seq{i}", lambda i=i: val_batch(val_ds, rcfg, i, True)))
    for i in VAL_SEQS:
        jobs.append(("val_OFF", f"seq{i}", lambda i=i: val_batch(val_ds, rcfg, i, False)))
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        jobs.append(("OCMR", sub, lambda sd=sd: ocmr_batch(sd)))
    for sub in GOTT_SUBJECTS:
        jobs.append(("Goett", sub, lambda sub=sub: goettingen_batch(gott_ds, sub)))
    for sub in MIITT_SUBJECTS:
        nii = os.path.join(MIITT_RECON, sub, "realtime", "sax", "4d_recon.nii.gz")
        if os.path.exists(nii):
            jobs.append(("MIITT", sub, lambda sub=sub: miitt_batch(sub)))

    for name, ckpt, head in MODELS:
        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== {name} ({head}) ===", flush=True)
        model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                     enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                     use_z_pose_embedding=True, use_t_pose_embedding=False,
                     use_target_t_pose_embedding=False, use_reference_token=True,
                     train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
        ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
        miss, unexp = model.load_state_dict(ck["model"], strict=False)
        assert not miss and not unexp, f"{name}: missing={miss[:5]} unexpected={unexp[:5]}"

        for mode, lbl, build in jobs:
            try:
                batch = build()
            except Exception as e:
                print(f"  skip {mode}_{lbl}: {e}"); continue
            V_can, dvf = forward(model, batch)
            Vin = input_volume(batch)
            render_io(Vin, V_can, f"{name} · {mode}_{lbl}  —  INPUT (left) | V_canon (right)",
                      os.path.join(out_dir, f"{mode}_{lbl}_io.png"))
            render_dvf(batch, dvf, f"DVF — {name} · {mode}_{lbl}",
                       os.path.join(out_dir, f"{mode}_{lbl}_dvf.png"))

    print(f"done -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
