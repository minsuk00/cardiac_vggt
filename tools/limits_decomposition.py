"""Limitation decomposition for VGGT+refiner cardiac slice-to-volume.

Training-free. Over N val samples (breathing protocol), measures a ladder of
reconstructors and diagnostics, headlined by MOTION PSNR (dynamic heart voxels),
with bbox/full PSNR + sharpness/GT as secondary.

Reconstructors (each → a (12,256,256) volume scored vs the target-phase V_gt):
  identity        Δ=0 splat of the (breathing-corrupted) input slices         [floor]
  model_canon     trained joint refiner model, V_canon (raw splat)            [operating pt]
  model_refined   trained joint refiner model, V_refined (after 3D UNet)      [operating pt]
  oracle_perfect  target-phase planes at the model's z's, Δ=0 (518->256 tri)  [UPPER BOUND:
                  perfect content placement — unachievable by a pure-warp model, see report]
  oracle_native256  target planes splatted at NATIVE 256 (no 518 resize)      [splat attribution]
  oracle_nearest    target planes, nearest-voxel scatter (no trilinear tent)  [splat attribution]
  oracle_super2x    target planes splatted into 512^2 grid, avgpool->256      [splat attribution]

Diagnostics:
  data-consistency: PSNR( sample_volume(V, world_points), input_intensity ) — is the recon
                    self-consistent with the pixels it was built from?
  coverage_frac:    fraction of cube voxels the splat actually fills.
  OOD breathing:    re-score model under a DIFFERENT respiratory waveform/amplitude than it
                    trained on — does the gain survive, or did it memorize the simulator?

Run: micromamba run -n svr python tools/limits_decomposition.py --n 30
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
from vggt.utils.splat import splat_to_volume, sample_volume

JOINT_CKPT = os.path.join(REPO, "scratch", "logs", "218349151_mri_refiner_joint", "ckpts", "checkpoint_last.pt")
OUT = os.path.join(REPO, "result", "limits_eval")

# In-distribution breathing = what the resp runs trained with (matches mri_volume.yaml).
RESP_IND = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35,
                ap_axis="H", per_slot=True, direction_jitter_deg=30.0)
# Out-of-distribution breathing: different waveform shape (cos2n), larger amplitude,
# different AP coupling + more direction jitter. Same family, parameters the model never saw.
RESP_OOD = dict(amplitude_mm=22.0, amplitude_jitter=6.0, cos2n=1, ap_ratio=0.55,
                ap_axis="H", per_slot=True, direction_jitter_deg=45.0)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def vol_metrics(V, V_gt_np, bbox, mmask):
    """V, V_gt_np: (D,H,W) numpy. bbox: list[6]. mmask: (D,H,W) bool numpy."""
    z0, z1, y0, y1, x0, x1 = bbox
    out = {}
    out["psnr_full"] = psnr(V, V_gt_np)
    out["psnr_bbox"] = psnr(V[z0:z1, y0:y1, x0:x1], V_gt_np[z0:z1, y0:y1, x0:x1])
    out["psnr_motion"] = psnr(V[mmask], V_gt_np[mmask]) if mmask.any() else float("nan")
    sgt = inplane_grad_energy(V_gt_np, bbox, V_gt_np)
    out["sharp_rel"] = inplane_grad_energy(V, bbox, V_gt_np) / sgt
    return out


def build_oracle_splat(V_gt, z_idx, base_xy, src_res, grid_hw, nearest=False):
    """Splat target-phase planes at z_idx into a (12, grid_hw, grid_hw) cube.

    V_gt: (D,256,256) target volume. z_idx: (S,) int planes. base_xy: (src_res,src_res,2)
    normalized x,y meshgrid. Returns (12,256,256) on V_gt.device.
    """
    D = V_gt.shape[0]
    S = len(z_idx)
    dev = V_gt.device
    pos = torch.zeros((1, S, src_res, src_res, 3), device=dev)
    inten = torch.zeros((1, S, src_res, src_res), device=dev)
    for i, z in enumerate(z_idx):
        plane = V_gt[int(z)].unsqueeze(0).unsqueeze(0)
        plane_r = F.interpolate(plane, size=(src_res, src_res), mode="bilinear", align_corners=False)[0, 0]
        inten[0, i] = plane_r
        pos[0, i, :, :, :2] = base_xy
        pos[0, i, :, :, 2] = int(z) / (D - 1) * 2 - 1
    pos_flat = pos.reshape(1, S * src_res * src_res, 3)
    int_flat = inten.reshape(1, S * src_res * src_res)
    w = (int_flat > 1e-3).float()
    if nearest:
        # snap positions to nearest voxel center so trilinear weights collapse to {0,1}
        gh = grid_hw
        gx = ((pos_flat[..., 0] + 1) * 0.5 * (gh - 1)).round() / (gh - 1) * 2 - 1
        gy = ((pos_flat[..., 1] + 1) * 0.5 * (gh - 1)).round() / (gh - 1) * 2 - 1
        gz = ((pos_flat[..., 2] + 1) * 0.5 * (D - 1)).round() / (D - 1) * 2 - 1
        pos_flat = torch.stack([gx, gy, gz], dim=-1)
    V, cov = splat_to_volume(pos_flat, int_flat, (D, grid_hw, grid_hw), weight=w)
    if grid_hw != 256:
        V = F.interpolate(V.unsqueeze(1), size=(D, 256, 256), mode="trilinear",
                          align_corners=False)[:, 0]
    return V[0], float((cov > 1e-3).float().mean())


def dc_psnr(V, world_points, batch):
    """Data-consistency: sample V at the input world coords, compare to input intensity."""
    inten = batch["images"].float().mean(dim=2)            # (1,S,H,W)
    B, S, H, W = inten.shape
    pos = world_points.reshape(1, S * H * W, 3)
    samp = sample_volume(V.unsqueeze(0), pos).reshape(S, H, W)
    tgt = inten[0]
    m = tgt > 1e-3
    return psnr(samp[m].cpu().numpy(), tgt[m].cpu().numpy())


def run(n, device):
    os.makedirs(OUT, exist_ok=True)
    ds = build_dataset()
    model, info = make_refiner_model(JOINT_CKPT, device)
    print("model load:", info)

    # meshgrids for oracle src resolutions
    def mesh(res):
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, res, device=device),
                                torch.linspace(-1, 1, res, device=device), indexing="ij")
        return torch.stack([xs, ys], dim=-1)
    mesh518, mesh256 = mesh(518), mesh(256)

    recons = ["identity", "model_canon", "model_refined", "oracle_perfect",
              "oracle_native256", "oracle_nearest", "oracle_super2x"]
    acc = {r: [] for r in recons}
    dc = {"identity": [], "model_canon": [], "model_refined": [], "oracle_perfect": []}
    cov = {"identity": [], "oracle_perfect": []}
    ood = {"model_canon": [], "model_refined": [], "identity": []}  # OOD-breathing motion psnr

    for seq in range(n):
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device,
                                  respiratory_cfg=RespiratoryConfig(enable=True, **RESP_IND), train=False)
        bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
        sc = batch["scanner_coords"]
        z_norm = sc[0, :, 0, 0, 2]
        z_idx = ((z_norm + 1) / 2 * (GRID_SHAPE[0] - 1)).round().long().clamp(0, GRID_SHAPE[0] - 1).tolist()

        out_id = compute_volume_intensity_loss({"world_points": sc}, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out_id["V_gt"][0]
        V_gt_np = V_gt.float().cpu().numpy()
        mmask = compute_motion_mask(batch["phases"])[0].cpu().numpy()

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        V_canon = preds["V_canon"][0].float()
        V_ref = preds["V_refined"][0].float()
        wp = preds["world_points"]

        vols = {
            "identity": out_id["V_canon"][0].float(),
            "model_canon": V_canon,
            "model_refined": V_ref,
        }
        Vo, cov_o = build_oracle_splat(V_gt, z_idx, mesh518, 518, 256)
        vols["oracle_perfect"] = Vo
        vols["oracle_native256"] = build_oracle_splat(V_gt, z_idx, mesh256, 256, 256)[0]
        vols["oracle_nearest"] = build_oracle_splat(V_gt, z_idx, mesh256, 256, 256, nearest=True)[0]
        vols["oracle_super2x"] = build_oracle_splat(V_gt, z_idx, mesh518, 518, 512)[0]

        for r in recons:
            acc[r].append(vol_metrics(vols[r].cpu().numpy(), V_gt_np, bbox, mmask))
        cov["identity"].append(float(out_id["metric_coverage_frac"]))
        cov["oracle_perfect"].append(cov_o)

        # data consistency (sample recon at the geometry it was built from)
        dc["identity"].append(dc_psnr(vols["identity"], sc, batch))
        dc["model_canon"].append(dc_psnr(V_canon, wp, batch))
        dc["model_refined"].append(dc_psnr(V_ref, wp, batch))
        dc["oracle_perfect"].append(dc_psnr(Vo, sc, batch))

        # OOD breathing: rebuild the SAME sample with a different respiratory model
        batch2 = build_batch(data, device, seq_index=seq)
        batch2 = gpu_augment_batch(batch2, None, device,
                                   respiratory_cfg=RespiratoryConfig(enable=True, **RESP_OOD), train=False)
        out_id2 = compute_volume_intensity_loss({"world_points": batch2["scanner_coords"]},
                                                batch2, grid_shape=GRID_SHAPE, tv_weight=0.0)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds2 = model(batch2["images"], batch=batch2)
        ood["identity"].append(psnr(out_id2["V_canon"][0].float().cpu().numpy()[mmask], V_gt_np[mmask]))
        ood["model_canon"].append(psnr(preds2["V_canon"][0].float().cpu().numpy()[mmask], V_gt_np[mmask]))
        ood["model_refined"].append(psnr(preds2["V_refined"][0].float().cpu().numpy()[mmask], V_gt_np[mmask]))

        if seq % 5 == 0:
            print(f"  seq {seq:2d}: model_ref motion={acc['model_refined'][-1]['psnr_motion']:.2f} "
                  f"oracle motion={acc['oracle_perfect'][-1]['psnr_motion']:.2f}")

    def mean(lst, k=None):
        v = [(x[k] if k else x) for x in lst]
        v = [x for x in v if np.isfinite(x)]
        return float(np.mean(v)) if v else float("nan")

    summary = {}
    for r in recons:
        summary[r] = {k: mean(acc[r], k) for k in ["psnr_motion", "psnr_bbox", "psnr_full", "sharp_rel"]}
    summary["coverage"] = {k: mean(v) for k, v in cov.items()}
    summary["data_consistency_psnr"] = {k: mean(v) for k, v in dc.items()}
    summary["ood_breathing_motion_psnr"] = {k: mean(v) for k, v in ood.items()}
    summary["ind_breathing_motion_psnr"] = {k: summary[k]["psnr_motion"] for k in
                                            ["identity", "model_canon", "model_refined"]}
    summary["n"] = n
    json.dump(summary, open(os.path.join(OUT, "decomposition.json"), "w"), indent=2)

    print("\n" + "=" * 78)
    print(f"DECOMPOSITION (breathing val, n={n}) — MOTION PSNR is the headline metric")
    print("=" * 78)
    print(f"{'reconstructor':18s} {'MOTION':>8s} {'bbox':>8s} {'full':>8s} {'sharp/GT':>9s}")
    for r in recons:
        s = summary[r]
        print(f"{r:18s} {s['psnr_motion']:8.2f} {s['psnr_bbox']:8.2f} {s['psnr_full']:8.2f} {s['sharp_rel']:9.3f}")
    print(f"\ncoverage_frac: identity={summary['coverage']['identity']:.3f} "
          f"oracle={summary['coverage']['oracle_perfect']:.3f}")
    print("data-consistency PSNR (recon resampled at its input geometry):")
    for k, v in summary["data_consistency_psnr"].items():
        print(f"   {k:16s} {v:6.2f} dB")
    print("OOD-breathing vs IND-breathing MOTION PSNR (sim-overfit test):")
    for k in ["identity", "model_canon", "model_refined"]:
        print(f"   {k:14s} IND={summary['ind_breathing_motion_psnr'][k]:6.2f}  "
              f"OOD={summary['ood_breathing_motion_psnr'][k]:6.2f}  "
              f"gain_over_id IND={summary['ind_breathing_motion_psnr'][k]-summary['ind_breathing_motion_psnr']['identity']:+.2f}"
              f"  OOD={summary['ood_breathing_motion_psnr'][k]-summary['ood_breathing_motion_psnr']['identity']:+.2f}")
    print("\nWrote", os.path.join(OUT, "decomposition.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    args = ap.parse_args()
    run(args.n, "cuda" if torch.cuda.is_available() else "cpu")
