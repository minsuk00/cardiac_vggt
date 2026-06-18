"""Evaluate a trained refiner checkpoint: V_canon vs V_refined — PSNR AND sharpness.

The PSNR gain (from the logs) shows the refiner helps; this confirms it's a genuine DEBLUR
(V_refined recovers high-frequency detail toward GT) rather than smoothing. Also renders
qualitative panels (V_gt / V_canon / V_refined / diff) for a hallucination check.

Run: micromamba run -n svr python tools/eval_refiner.py --ckpt /tmp/frozen_refiner_ckpt.pt --n 60
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_variants_matrix import build_dataset, build_batch, PROTOCOLS, GRID_SHAPE, NUM_SLICES  # noqa
from measure_sharpness import inplane_grad_energy  # noqa
sys.path.insert(0, os.path.join(REPO, "training"))
from data.gpu_aug import gpu_augment_batch          # noqa
from loss import compute_volume_intensity_loss       # noqa
from vggt.models.vggt import VGGT                     # noqa

OUT = os.path.join(REPO, "result", "refiner_eval")


def make_refiner_model(ckpt, device, use_t=False):
    model = VGGT(img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=False, enable_depth=False, enable_point=True, enable_track=False,
                 use_z_pose_embedding=True, use_t_pose_embedding=use_t, use_target_t_pose_embedding=True,
                 train_on_residual_dvf=True, enable_refiner=True, grid_shape=(12, 256, 256),
                 refiner_base_channels=16, refiner_levels=2, refiner_use_coverage=True).to(device)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    res = model.load_state_dict(ck["model"] if "model" in ck else ck, strict=False)
    model.eval()
    return model, dict(missing=len(res.missing_keys), unexpected=len(res.unexpected_keys),
                       has_refiner=any("refiner" in k for k in (ck.get("model", {})).keys()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/frozen_refiner_ckpt.pt")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--panels", default="0,7")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "panels"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset()
    model, info = make_refiner_model(args.ckpt, device)
    print("load:", info)
    cfg = PROTOCOLS["breathing"]

    recs = []
    panel_seqs = [int(s) for s in args.panels.split(",")]
    for seq in range(args.n):
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
        bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        Vc = preds["V_canon"][0].float().cpu().numpy()
        Vr = preds["V_refined"][0].float().cpu().numpy()
        Vg = out["V_gt"][0].float().cpu().numpy()
        g_gt = inplane_grad_energy(Vg, bbox, Vg)
        rec = dict(seq=seq, t_target=int(np.asarray(data["t_target"]).flatten()[0]),
                   psnr_bbox_canon=float(out["metric_psnr_3d_bbox"]),
                   psnr_bbox_refined=float(out["metric_psnr_3d_bbox_refined"]),
                   psnr_motion_canon=float(out.get("metric_psnr_3d_motion", float("nan"))),
                   psnr_motion_refined=float(out.get("metric_psnr_3d_motion_refined", float("nan"))),
                   sharp_gt=g_gt,
                   sharp_canon=inplane_grad_energy(Vc, bbox, Vg),
                   sharp_refined=inplane_grad_energy(Vr, bbox, Vg))
        rec["rel_canon"] = rec["sharp_canon"] / g_gt
        rec["rel_refined"] = rec["sharp_refined"] / g_gt
        recs.append(rec)

        if seq in panel_seqs:
            z0, z1 = bbox[0], bbox[1]; zs = list(range(z0, z1))
            vmax = float(max(Vg.max(), Vc.max(), Vr.max(), 1e-3))
            rows = [("V_gt", Vg, "gray", 0, vmax), ("V_canon (splat)", Vc, "gray", 0, vmax),
                    ("V_refined", Vr, "gray", 0, vmax), ("V_refined - V_gt", Vr - Vg, "RdBu_r", -0.1, 0.1)]
            fig = plt.figure(figsize=(1.5 * len(zs) + 1, 1.5 * len(rows) + 0.6), dpi=130)
            gs = gridspec.GridSpec(len(rows), len(zs), wspace=0.04, hspace=0.07)
            for r, (lab, vol, cm, vmn, vmx) in enumerate(rows):
                for c, z in enumerate(zs):
                    ax = fig.add_subplot(gs[r, c]); ax.imshow(vol[z], cmap=cm, vmin=vmn, vmax=vmx)
                    ax.set_xticks([]); ax.set_yticks([])
                    if r == 0: ax.set_title(f"z={z}", fontsize=8)
                    if c == 0: ax.set_ylabel(lab, fontsize=9)
            fig.suptitle(f"refiner — subj seq{seq} t={rec['t_target']} (breathing val) | "
                         f"bbox PSNR {rec['psnr_bbox_canon']:.2f}→{rec['psnr_bbox_refined']:.2f}, "
                         f"sharp {rec['rel_canon']:.2f}→{rec['rel_refined']:.2f}× GT", fontsize=10, y=1.0)
            p = os.path.join(OUT, "panels", f"refiner_seq{seq}_t{rec['t_target']}.png")
            plt.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
            print("saved", p)

    def mean(k):
        v = [r[k] for r in recs if np.isfinite(r[k])]
        return float(np.mean(v)) if v else None
    summary = {k: mean(k) for k in ["psnr_bbox_canon", "psnr_bbox_refined", "psnr_motion_canon",
                                    "psnr_motion_refined", "rel_canon", "rel_refined"]}
    summary["n"] = len(recs)
    json.dump({"summary": summary, "records": recs, "load": info},
              open(os.path.join(OUT, "refiner_eval.json"), "w"), indent=1)
    print("\n=== SUMMARY (breathing val, n=%d) ===" % len(recs))
    print(f"  bbox PSNR:   V_canon {summary['psnr_bbox_canon']:.2f}  →  V_refined {summary['psnr_bbox_refined']:.2f}  "
          f"(Δ {summary['psnr_bbox_refined']-summary['psnr_bbox_canon']:+.2f})")
    print(f"  motion PSNR: V_canon {summary['psnr_motion_canon']:.2f}  →  V_refined {summary['psnr_motion_refined']:.2f}  "
          f"(Δ {summary['psnr_motion_refined']-summary['psnr_motion_canon']:+.2f})")
    print(f"  sharpness/GT: V_canon {summary['rel_canon']:.3f}  →  V_refined {summary['rel_refined']:.3f}  "
          f"(Δ {summary['rel_refined']-summary['rel_canon']:+.3f})   [1.0 = as sharp as GT]")
    print("Wrote", os.path.join(OUT, "refiner_eval.json"))


if __name__ == "__main__":
    main()
