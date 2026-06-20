"""Probe: confirm batch shapes + verify oracle splat (target-phase planes, Δ=0) ≈ V_gt.

De-risks the ceiling-decomposition experiment. Run on 1 val subject.
"""
import os, sys
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, PROTOCOLS, GRID_SHAPE, NUM_SLICES
from data.gpu_aug import gpu_augment_batch
from loss import compute_volume_intensity_loss
from vggt.utils.splat import splat_to_volume, splat_predictions
from measure_sharpness import inplane_grad_energy
import torch.nn.functional as F


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10 * np.log10(1.0 / mse)


def main():
    device = "cuda"
    ds = build_dataset()
    seq = 7
    data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
    batch = build_batch(data, device, seq_index=seq)
    batch = gpu_augment_batch(batch, None, device, respiratory_cfg=PROTOCOLS["breathing"], train=False)

    print("=== shapes ===")
    for k in ["images", "scanner_coords", "z_indices", "t_indices", "target_t_indices",
              "gt_target_volume", "phases", "anatomy_bbox"]:
        v = batch.get(k)
        if torch.is_tensor(v):
            print(f"  {k:20s} {tuple(v.shape)}  dtype={v.dtype}  min={v.float().min():.3f} max={v.float().max():.3f}")
    print("  t_target =", np.asarray(data["t_target"]).flatten()[0])
    print("  z_indices vals:", batch["z_indices"].flatten().long().tolist())
    print("  t_indices vals:", batch["t_indices"].flatten().long().tolist())

    out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]},
                                        batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
    V_gt = out["V_gt"]                  # (1, D, H, W)
    print("\n  V_gt shape", tuple(V_gt.shape), "min", float(V_gt.min()), "max", float(V_gt.max()))
    print("  identity-Δ splat (corrupted inputs) bbox PSNR:", float(out["metric_psnr_3d_bbox"]))

    # ---- Oracle: replace each slot's image with the TARGET-PHASE plane at its z, Δ=0 ----
    # phases batch key is splat-order (T, D, H, W). gt_target_volume == phases[t_target].
    D, H, W = GRID_SHAPE
    sc0 = batch["scanner_coords"]                         # (1,S,Hc,Wc,3)
    S = sc0.shape[1]
    # Recover integer canonical plane from the z channel: z_norm = z/(D-1)*2-1.
    z_norm_per_slot = sc0[0, :, 0, 0, 2]                  # (S,)
    z_idx = ((z_norm_per_slot + 1) / 2 * (D - 1)).round().long().clamp(0, D - 1)
    print("  recovered z planes:", z_idx.tolist())
    Vg = V_gt[0]                                          # (D,H,W) = (12,256,256)
    # Build oracle input images at 518 res (mirror pipeline): plane z_i resized 256->518, 3ch.
    img_h, img_w = batch["images"].shape[-2:]
    oracle_imgs = torch.zeros_like(batch["images"])      # (1,S,3,518,518)
    for i in range(S):
        plane = Vg[z_idx[i]].unsqueeze(0).unsqueeze(0)    # (1,1,256,256)
        plane518 = F.interpolate(plane, size=(img_h, img_w), mode="bilinear", align_corners=False)
        oracle_imgs[0, i] = plane518[0].repeat(3, 1, 1)
    oracle_batch = dict(batch)
    oracle_batch["images"] = oracle_imgs
    out_o = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]},
                                          oracle_batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
    bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
    Vg_np = V_gt[0].float().cpu().numpy()
    sgt = inplane_grad_energy(Vg_np, bbox, Vg_np)
    so = inplane_grad_energy(out_o["V_canon"][0].float().cpu().numpy(), bbox, Vg_np)
    print("\n  ORACLE-sparse (target-phase slices @ model's S z's, Δ=0):")
    print("    bbox PSNR:", float(out_o["metric_psnr_3d_bbox"]),
          " full PSNR:", float(out_o["metric_psnr_3d_full"]),
          " coverage_frac:", float(out_o["metric_coverage_frac"]),
          " sharpness/GT:", so / sgt)

    # ---- Oracle FULL: all 12 planes as 12 slots ----
    full_imgs = torch.zeros((1, D, 3, img_h, img_w), device=device)
    sc = batch["scanner_coords"]   # (1,S,Hc,Wc,3) ; rebuild for D slots
    Hc, Wc = sc.shape[2], sc.shape[3]
    full_sc = torch.zeros((1, D, Hc, Wc, 3), device=device)
    # scanner_coords: x,y identical per slot; z = z_i/(D-1)*2-1. Build per plane.
    base_xy = sc[0, 0, :, :, :2]   # (Hc,Wc,2) same for all
    for z in range(D):
        plane = Vg[z].unsqueeze(0).unsqueeze(0)
        plane518 = F.interpolate(plane, size=(img_h, img_w), mode="bilinear", align_corners=False)
        full_imgs[0, z] = plane518[0].repeat(3, 1, 1)
        full_sc[0, z, :, :, :2] = base_xy
        full_sc[0, z, :, :, 2] = z / (D - 1) * 2 - 1
    full_batch = dict(batch)
    full_batch["images"] = full_imgs
    full_batch["scanner_coords"] = full_sc
    out_f = compute_volume_intensity_loss({"world_points": full_sc}, full_batch,
                                          grid_shape=GRID_SHAPE, tv_weight=0.0)
    print("\n  ORACLE-full12 (all planes, Δ=0) -> pure splat round-trip:")
    print("    bbox PSNR:", float(out_f["metric_psnr_3d_bbox"]),
          " full PSNR:", float(out_f["metric_psnr_3d_full"]),
          " coverage_frac:", float(out_f["metric_coverage_frac"]))


if __name__ == "__main__":
    main()
