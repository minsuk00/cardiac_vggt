"""Test two candidate improvements with the TRAINED model's own geometry (training-free):

A. NATIVE-RES SPLAT. The decomposition proved the sharpness ceiling is the 256->518->256
   resize, not the trilinear kernel. Here we keep the model's predicted Δ but splat the
   NATIVE-256 intensity at native-256-downsampled world_points (instead of the 518 intensity).
   Does the model's *own* (imperfect) geometry + native-res splat recover sharpness?
   (No retraining — an upper-bound-ish probe of the fix, since the head was trained vs 518.)

B. MULTI-DRAW ENSEMBLE. Average V_canon / V_refined over K independent scattered draws of
   the SAME (subject, target phase). seq and seq+60 share subject (seq%30) and target phase
   (seq%12) but get different val RNG → independent draws. Tests whether the gap is
   coverage/variance (averaging helps) or genuine motion error (averaging doesn't).

Run: micromamba run -n svr python tools/improvements_test.py --n 30 --K 8
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from eval_refiner import make_refiner_model
from measure_sharpness import inplane_grad_energy
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume

JOINT = os.path.join(REPO, "scratch", "logs", "218349151_mri_refiner_joint", "ckpts", "checkpoint_last.pt")
OUT = os.path.join(REPO, "result", "limits_eval")
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35,
            ap_axis="H", per_slot=True, direction_jitter_deg=30.0)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def metrics(V, Vg, bbox, mm):
    z0, z1, y0, y1, x0, x1 = bbox
    sgt = inplane_grad_energy(Vg, bbox, Vg)
    return dict(psnr_motion=psnr(V[mm], Vg[mm]) if mm.any() else float("nan"),
                psnr_bbox=psnr(V[z0:z1, y0:y1, x0:x1], Vg[z0:z1, y0:y1, x0:x1]),
                sharp_rel=inplane_grad_energy(V, bbox, Vg) / sgt)


def native_splat(world_points, images, grid=(12, 256, 256)):
    """Splat NATIVE-256 intensity at 256-downsampled predicted positions."""
    B, S, H, W, _ = world_points.shape
    wp = world_points.permute(0, 1, 4, 2, 3).reshape(B * S, 3, H, W)
    wp256 = F.interpolate(wp, size=(256, 256), mode="bilinear", align_corners=False)
    wp256 = wp256.reshape(B, S, 3, 256, 256).permute(0, 1, 3, 4, 2)
    inten = images.float().mean(dim=2)
    inten256 = F.interpolate(inten, size=(256, 256), mode="bilinear", align_corners=False)
    pos = wp256.reshape(B, S * 256 * 256, 3)
    it = inten256.reshape(B, S * 256 * 256)
    w = (it > 1e-3).float()
    V, _ = splat_to_volume(pos, it, grid, weight=w)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--K", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    ds = build_dataset()
    n_subj = len(ds.subjects)
    model, info = make_refiner_model(JOINT, device)
    print("model load:", info)
    cfg = RespiratoryConfig(enable=True, **RESP)

    A = {"canon_518": [], "canon_native": [], "refined_518": []}    # native-splat test
    K = args.K
    Kgrid = sorted(set([1, 2, 4, K]))
    B = {f"canon_K{k}": [] for k in Kgrid}
    B.update({f"refined_K{k}": [] for k in Kgrid})

    for subj in range(min(args.n, n_subj)):
        # K draws of the SAME (subject, target phase): seq = subj + 60*j
        Vc_draws, Vr_draws = [], []
        base = None
        for j in range(K):
            seq = subj + 60 * j
            data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
            batch = build_batch(data, device, seq_index=seq)
            batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                preds = model(batch["images"], batch=batch)
            Vc_draws.append(preds["V_canon"][0].float())
            Vr_draws.append(preds["V_refined"][0].float())
            if j == 0:
                out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]},
                                                    batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
                Vg = out["V_gt"][0].float().cpu().numpy()
                bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
                mm = compute_motion_mask(batch["phases"])[0].cpu().numpy()
                # A: native-res splat with the model's own geometry (draw 0)
                Vc518 = preds["V_canon"][0].float().cpu().numpy()
                Vnat = native_splat(preds["world_points"], batch["images"])[0].cpu().numpy()
                A["canon_518"].append(metrics(Vc518, Vg, bbox, mm))
                A["canon_native"].append(metrics(Vnat, Vg, bbox, mm))
                A["refined_518"].append(metrics(preds["V_refined"][0].float().cpu().numpy(), Vg, bbox, mm))

        # B: ensemble over first k draws
        for k in Kgrid:
            Vc = torch.stack(Vc_draws[:k]).mean(0).cpu().numpy()
            Vr = torch.stack(Vr_draws[:k]).mean(0).cpu().numpy()
            B[f"canon_K{k}"].append(metrics(Vc, Vg, bbox, mm))
            B[f"refined_K{k}"].append(metrics(Vr, Vg, bbox, mm))
        if subj % 5 == 0:
            print(f"  subj {subj}: canon518 sharp={A['canon_518'][-1]['sharp_rel']:.3f} "
                  f"native sharp={A['canon_native'][-1]['sharp_rel']:.3f} | "
                  f"refined K1={B['refined_K1'][-1]['psnr_motion']:.2f} "
                  f"K{K}={B[f'refined_K{K}'][-1]['psnr_motion']:.2f}")

    def mean(lst, key):
        v = [x[key] for x in lst if np.isfinite(x[key])]
        return float(np.mean(v)) if v else float("nan")

    res = {"native_splat": {}, "multidraw": {}, "n": args.n, "K": K}
    for k, lst in A.items():
        res["native_splat"][k] = {m: mean(lst, m) for m in ["psnr_motion", "psnr_bbox", "sharp_rel"]}
    for k, lst in B.items():
        res["multidraw"][k] = {m: mean(lst, m) for m in ["psnr_motion", "psnr_bbox", "sharp_rel"]}
    json.dump(res, open(os.path.join(OUT, "improvements.json"), "w"), indent=2)

    print("\n=== A. NATIVE-RES SPLAT (model's own geometry, no retrain) ===")
    print(f"{'variant':16s} {'motion':>8s} {'bbox':>8s} {'sharp/GT':>9s}")
    for k in ["canon_518", "canon_native", "refined_518"]:
        s = res["native_splat"][k]
        print(f"{k:16s} {s['psnr_motion']:8.2f} {s['psnr_bbox']:8.2f} {s['sharp_rel']:9.3f}")
    print("\n=== B. MULTI-DRAW ENSEMBLE (K independent scattered draws, averaged) ===")
    print(f"{'variant':14s} {'motion':>8s} {'bbox':>8s} {'sharp/GT':>9s}")
    for k in Kgrid:
        s = res["multidraw"][f"refined_K{k}"]
        print(f"refined_K{k:<8d} {s['psnr_motion']:8.2f} {s['psnr_bbox']:8.2f} {s['sharp_rel']:9.3f}")
    print("\nWrote", os.path.join(OUT, "improvements.json"))


if __name__ == "__main__":
    main()
