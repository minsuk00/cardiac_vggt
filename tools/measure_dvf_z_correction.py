"""Quantify through-plane (SI) breathing correction for the 3 reference-conditioned models,
directly comparable to docs/07 for the target_t run t59w6nqy (corr=0.87, slope=0.42).

Per model, over all 30 val subjects (ED, resp ON): for each input slot, pair the APPLIED SI
breath d_D (mm, from batch["resp_disp_mm"]) against the model's MEAN predicted Δz over the slot's
anatomy (mm). Report Pearson corr(applied SI, pred Δz) and the linear-fit slope (slope≈1 = full
rigid correction; 0 = ignores breathing). Writes a scatter PNG per model + a JSON summary.

Run: micromamba run -n svr python tools/measure_dvf_z_correction.py
"""
import os
import sys
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.render_reference_5dataset_io import MODELS, forward, THROUGH_MM
from tools.five_row_compare import DEV, val_batch, build_val_dataset
from vggt.models.vggt import VGGT

OUT = os.path.join(_ROOT, "result", "dvf_z_correction")
N_SUBJECTS = 30
ANAT_THR = 0.05


def build_model(ckpt, head):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type=head).to(DEV).eval()
    ck = torch.load(os.path.join(_ROOT, ckpt), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.t_target_fixed = 0
    summary = {}

    for name, ckpt, head in MODELS:
        print(f"=== {name} ===", flush=True)
        model = build_model(ckpt, head)
        applied_si, pred_dz, is_ref = [], [], []
        for seq in range(N_SUBJECTS):
            try:
                batch = val_batch(val_ds, rcfg, seq, breathing=True)
            except Exception as e:
                print(f"  skip seq{seq}: {e}", flush=True); continue
            disp = batch["resp_disp_mm"][0].cpu().numpy()           # (S,3) mm (d_D,d_H,d_W)
            imgs = batch["images"][0].float().cpu().clamp(0, 1).mean(1).numpy()  # (S,H,W)
            _V, dvf = forward(model, batch)                          # dvf (S,H,W,3) norm
            S = imgs.shape[0]
            for s in range(S):
                m = imgs[s] > ANAT_THR
                if not m.any():
                    continue
                applied_si.append(float(disp[s, 0]))                 # SI = d_D mm
                pred_dz.append(float(dvf[s][m][:, 2].mean() * THROUGH_MM))
                is_ref.append(s == 0)
        x = np.array(applied_si); y = np.array(pred_dz); ref = np.array(is_ref)
        corr = float(np.corrcoef(np.abs(x), y)[0, 1])
        slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        summary[name] = {"n_slots": int(x.size), "corr_absSI_predZ": round(corr, 3),
                         "slope": round(slope, 3), "intercept_mm": round(intercept, 2),
                         "applied_SI_p95_mm": round(float(np.percentile(np.abs(x), 95)), 1),
                         "pred_Z_p95_mm": round(float(np.percentile(np.abs(y), 95)), 1),
                         "pred_Z_at_SI>=12mm_mean": round(float(y[np.abs(x) >= 12].mean())
                                                          if (np.abs(x) >= 12).any() else 0.0, 2)}
        print(f"  corr={corr:.3f} slope={slope:.3f} n={x.size}", flush=True)

        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        ax.scatter(x[~ref], y[~ref], s=10, alpha=0.4, label="scattered slot")
        ax.scatter(x[ref], y[ref], s=18, alpha=0.7, color="r", label="reference slot")
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, slope * xx + intercept, "k-", lw=2, label=f"fit slope={slope:.2f}")
        ax.plot(xx, xx, "g--", lw=1, label="full correction (slope 1)")
        ax.set_xlabel("applied SI breath  d_D (mm)"); ax.set_ylabel("pred mean Δz over anatomy (mm)")
        ax.set_title(f"{name}: corr={corr:.2f}, slope={slope:.2f}  (t59w6nqy: 0.87/0.42)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.savefig(os.path.join(OUT, f"{name}_z_correction.png"), bbox_inches="tight")
        plt.close(fig)
        del model; torch.cuda.empty_cache()

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== SUMMARY (vs t59w6nqy target_t: corr=0.87, slope=0.42) ===")
    print(json.dumps(summary, indent=2))
    print(f"done -> {os.path.relpath(OUT, _ROOT)}", flush=True)


if __name__ == "__main__":
    main()
