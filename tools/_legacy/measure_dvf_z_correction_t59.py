"""Run the OLD target_t model t59w6nqy (218747856_mri_volume_resp_allphases_aggft_z_no_t) through
the SAME DVF z-correction measurement + panel code as the reference models, so the comparison is
apples-to-apples (identical masking / per-slot mean Δz / ED / breathing-ON protocol) rather than
citing docs/07 (which used a slightly different metric).

t59w6nqy specifics: use_target_t=True, use_reference_token=False (NO reference slot), and the
PRE-num_freqs-change embedders (num_freqs=6, 13-dim proj) → overridden at runtime. The dataset is
run with reference_slot=False (slot 0 is a normal scattered slot, not an anchor); target phase =
ED via target_t_indices=-1.

Writes: result/dvf_z_correction/t59w6nqy_z_correction.png (+ appends t59w6nqy_summary.json) and
result/ed_val_dvf_gt/t59w6nqy/val_seq*.png panels.

Run: micromamba run -n svr python tools/measure_dvf_z_correction_t59.py
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

from tools.render_reference_5dataset_io import forward, THROUGH_MM
from tools.render_ed_val_dvf_gt import render
from tools.five_row_compare import DEV, val_batch, build_val_dataset
from vggt.models.vggt import VGGT
from vggt.models.aggregator import ZIndexEmbedder, TIndexEmbedder

CKPT = "scratch/logs/218747856_mri_volume_resp_allphases_aggft_z_no_t/ckpts/checkpoint_last.pt"
OUTZ = os.path.join(_ROOT, "result", "dvf_z_correction")
OUTP = os.path.join(_ROOT, "result", "ed_val_dvf_gt", "t59w6nqy")
N_SUBJECTS = 30
ANAT_THR = 0.05


def build_t59():
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024,
             enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
             use_z_pose_embedding=True, use_t_pose_embedding=False,
             use_target_t_pose_embedding=True, use_reference_token=False,
             train_on_residual_dvf=True, warp_head_type="dpt").to(DEV).eval()
    # runtime num_freqs=6 override (this ckpt predates the 6->3 change; 13-dim proj)
    m.aggregator.z_embedder = ZIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    m.aggregator.target_t_embedder = TIndexEmbedder(embed_dim=1024, num_freqs=6).to(DEV)
    ck = torch.load(os.path.join(_ROOT, CKPT), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    real_miss = [k for k in miss if "patch_embed" not in k]
    assert not real_miss and not unexp, f"missing={real_miss[:6]} unexpected={unexp[:6]}"
    print("loaded t59w6nqy (target_t, num_freqs=6) clean", flush=True)
    return m


def main():
    os.makedirs(OUTZ, exist_ok=True); os.makedirs(OUTP, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    val_ds.reference_slot = False   # t59w6nqy has NO reference slot
    val_ds.t_target_fixed = 0       # ED (target_t=-1 in forward)
    model = build_t59()

    applied_si, pred_dz, is_ref = [], [], []
    for seq in range(N_SUBJECTS):
        try:
            batch = val_batch(val_ds, rcfg, seq, breathing=True)
        except Exception as e:
            print(f"  skip seq{seq}: {e}", flush=True); continue
        disp = batch["resp_disp_mm"][0].cpu().numpy()
        imgs = batch["images"][0].float().cpu().clamp(0, 1).mean(1).numpy()
        _V, dvf = forward(model, batch)
        for s in range(imgs.shape[0]):
            m = imgs[s] > ANAT_THR
            if not m.any():
                continue
            applied_si.append(float(disp[s, 0]))
            pred_dz.append(float(dvf[s][m][:, 2].mean() * THROUGH_MM))
            is_ref.append(s == 0)
        render(batch, dvf, disp,
               f"DVF + GT breath — t59w6nqy (target_t) · val seq{seq} (ED, resp ON)",
               os.path.join(OUTP, f"val_seq{seq:02d}_ED_resp_dvf_gt.png"))

    x = np.array(applied_si); y = np.array(pred_dz); ref = np.array(is_ref)
    corr = float(np.corrcoef(np.abs(x), y)[0, 1])
    slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
    summ = {"n_slots": int(x.size), "corr_absSI_predZ": round(corr, 3), "slope": round(slope, 3),
            "intercept_mm": round(intercept, 2),
            "pred_Z_at_SI>=12mm_mean": round(float(y[np.abs(x) >= 12].mean())
                                             if (np.abs(x) >= 12).any() else 0.0, 2)}
    print(f"t59w6nqy: corr={corr:.3f} slope={slope:.3f} n={x.size}", flush=True)
    with open(os.path.join(OUTZ, "t59w6nqy_summary.json"), "w") as f:
        json.dump({"t59w6nqy": summ}, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    ax.scatter(x[~ref], y[~ref], s=10, alpha=0.4, label="scattered slot")
    ax.scatter(x[ref], y[ref], s=18, alpha=0.7, color="r", label="slot 0 (not a ref here)")
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, slope * xx + intercept, "k-", lw=2, label=f"fit slope={slope:.2f}")
    ax.plot(xx, xx, "g--", lw=1, label="full correction (slope 1)")
    ax.set_xlabel("applied SI breath d_D (mm)"); ax.set_ylabel("pred mean Δz over anatomy (mm)")
    ax.set_title(f"t59w6nqy (target_t): corr={corr:.2f}, slope={slope:.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUTZ, "t59w6nqy_z_correction.png"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summ, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
