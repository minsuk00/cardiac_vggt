"""Is the breathing reconstruction actually BLURRY (lost high-freq) or just shifted/wrong?

Decisive test: relative sharpness = mean|grad(V_canon)| / mean|grad(V_gt)| over the anatomy
bbox (in-plane gradients, per z-plane). ~1 → as sharp as GT (not blur; error is displacement).
<1 → high-freq lost → genuinely blurry. Run the resp model (v2) under clean vs breathing on a
handful of val subjects; identity for reference.

Run: micromamba run -n svr python tools/measure_sharpness.py
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_variants_matrix import (build_dataset, build_batch, make_model, RUNS, LOGS,  # noqa: E402
                                  PROTOCOLS, GRID_SHAPE, NUM_SLICES)
sys.path.insert(0, os.path.join(REPO, "training"))
from data.gpu_aug import gpu_augment_batch         # noqa: E402
from loss import compute_volume_intensity_loss      # noqa: E402

SEQS = list(range(12))   # 12 val samples (subjects 0..11, phases 0..11)
EVAL_VARS = [2, 4]       # 2 = resp z-only, 4 = no-resp z-only (the headline pair)


def inplane_grad_energy(vol, bbox, gt_for_mask=None):
    """Mean in-plane gradient magnitude over the bbox region (anatomy voxels only)."""
    z0, z1, y0, y1, x0, x1 = bbox
    v = vol[z0:z1, y0:y1, x0:x1]
    gy = np.abs(np.diff(v, axis=1))   # H-gradient
    gx = np.abs(np.diff(v, axis=2))   # W-gradient
    # crop to common shape and combine
    g = gy[:, :, :-1] + gx[:, :-1, :]
    if gt_for_mask is not None:
        m = gt_for_mask[z0:z1, y0:y1, x0:x1][:, :-1, :-1] > 1e-2
        if m.sum() < 10:
            return float(g.mean())
        return float(g[m].mean())
    return float(g.mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset()
    out_json = {"sharp_GT": None, "results": {}}
    rows = []
    for var in EVAL_VARS:
        run = next(r for r in RUNS if r["var"] == var)
        ckpt = os.path.join(LOGS, run["exp_dir"], "ckpts", "checkpoint_last.pt")
        model, _ = make_model(run["use_t"], ckpt, device)
        print(f"\n=== model = var{var} ({run['name']}) ===")
        for proto in ["clean", "breathing"]:
            cfg = PROTOCOLS[proto]
            s_gt, s_model, s_ident, rel_model, rel_ident, psnrs, cov = [], [], [], [], [], [], []
            for seq in SEQS:
                data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
                batch = build_batch(data, device, seq_index=seq)
                batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
                bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
                outi = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]},
                                                     batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
                Vgt = outi["V_gt"][0].float().cpu().numpy()
                Vid = outi["V_canon"][0].float().cpu().numpy()
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    preds = model(batch["images"], batch=batch)
                outm = compute_volume_intensity_loss(preds, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
                Vm = outm["V_canon"][0].float().cpu().numpy()
                g_gt = inplane_grad_energy(Vgt, bbox, Vgt)
                g_m = inplane_grad_energy(Vm, bbox, Vgt)
                g_i = inplane_grad_energy(Vid, bbox, Vgt)
                s_gt.append(g_gt); s_model.append(g_m); s_ident.append(g_i)
                rel_model.append(g_m / g_gt); rel_ident.append(g_i / g_gt)
                psnrs.append(float(outm["metric_psnr_3d_bbox"]))
                cov.append(float(outm["metric_coverage_frac"]))
            rec = dict(var=var, name=run["name"], protocol=proto,
                       sharp_GT=float(np.mean(s_gt)), sharp_model=float(np.mean(s_model)),
                       sharp_ident=float(np.mean(s_ident)), rel_model=float(np.mean(rel_model)),
                       rel_ident=float(np.mean(rel_ident)), bbox_psnr=float(np.mean(psnrs)),
                       coverage_frac=float(np.mean(cov)), n=len(SEQS))
            rows.append(rec)
            out_json["results"][f"var{var}_{proto}"] = rec
        del model
        torch.cuda.empty_cache()

    out_json["sharp_GT"] = rows[0]["sharp_GT"]
    import json
    with open(os.path.join(REPO, "result", "variants_eval", "sharpness.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    print(f"\n{'var/proto':18s} {'sharp_GT':>9s} {'sharp_rec':>9s} {'rel_rec':>8s} "
          f"{'rel_ident':>9s} {'cov_frac':>8s} {'bbox_PSNR':>10s}")
    for r in rows:
        print(f"var{r['var']} {r['protocol']:11s} {r['sharp_GT']:9.4f} {r['sharp_model']:9.4f} "
              f"{r['rel_model']:8.3f} {r['rel_ident']:9.3f} {r['coverage_frac']:8.3f} {r['bbox_psnr']:10.2f}")
    print("\nrel_rec = sharpness(recon)/sharpness(GT). <1 => high-freq lost => blur.")
    print("Wrote result/variants_eval/sharpness.json")


if __name__ == "__main__":
    main()
