"""E3 (clean) — Does UNDOING the breath even improve the reconstruction? A noise-free oracle test.

Instead of optimizing Δ (confounded by optimizer instability + coverage bookkeeping), directly
EVALUATE L1/motion-PSNR at two known Δ fields, no optimization:
  identity : Δ = 0 (leave every slice at its labeled plane)
  oracle   : Δ = exactly undo the applied per-slot respiratory displacement (place the breathed
             anatomy back at its true rest location)

If oracle ≫ identity → breath correction DOES help the objective → it is recoverable in principle
(the model/optimizer is the limit → optimist has headroom). If oracle ≈ identity → undoing the
breath does NOT improve the reconstruction against V_gt → the breath is DEGENERATE/unrecoverable
under this renderer+sampling → a genuine wall, no model can beat it.

Matrix (each on N val subjects, pure_resp = all input slots at target phase, so ONLY the rigid SI
breath differs from V_gt):
  one-frame (S=12, one slice/plane)  vs  multi-frame (S=20, planes covered multiply)
  group_by_burst=True (all frames of a plane share ONE breath — realistic)
                 =False (per-slot iid breaths — gives redundant DISTINCT-depth measurements/plane)
The multi-frame × iid cell tests whether redundant distinct-depth measurements make the breath
recoverable (triangulation); the multi-frame × burst cell is the realistic constraint.

Run: micromamba run -n svr python tools/exp_resp_oracle.py --seqs 0-14
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume
from vggt.models.vggt import VGGT
D, H, W = GRID_SHAPE
SP = (12.0, 1.4, 1.4)                                   # (D,H,W) mm canonical spacing
OUT = os.path.join(REPO, "result", "resp_oracle")
REF_CKPT = "scratch/logs/217721337_mri_volume_reference_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"


def build_ref_model(device):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_camera=False, enable_depth=False,
             enable_point=True, enable_track=False, use_z_pose_embedding=True,
             use_t_pose_embedding=False, use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type="dpt").to(device).eval()
    ck = torch.load(os.path.join(REPO, REF_CKPT), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp
    return m


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def splat(world, inten):
    intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
    V, _ = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
    return V[0]


def norm_delta(d_mm, spacing, size):
    return (d_mm / spacing) * (2.0 / (size - 1))


def eval_cell(ds, cfg, device, seqs, img_per_seq, model=None):
    """Return means over subjects: identity / oracle+ / oracle- / model motion PSNR and full L1."""
    ds.t_target_fixed = 0
    rec = {"identity": [], "oracle_undo": [], "oracle_neg": [], "model": [], "l1_id": [], "l1_oracle": [],
           "applied_p95": [], "applied_deep_frac": []}
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=img_per_seq)
        b = build_batch(data, device, seq_index=seq)
        b["timesteps"] = torch.zeros_like(b["timesteps"])           # pure_resp: all slots = target phase
        b = gpu_augment_batch(b, None, device, respiratory_cfg=cfg, train=False)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b,
                                            grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float()
        mmask = compute_motion_mask(b["phases"])[0].cpu().numpy()
        if not mmask.any():
            continue
        disp = b["resp_disp_mm"][0]                                  # (S,3) = (d_D,d_H,d_W) mm
        sc = b["scanner_coords"][0].permute(0, 3, 1, 2)
        sc256 = F.interpolate(sc, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)  # (S,256,256,3) xyz
        it = b["images"][0].float().mean(dim=1, keepdim=True)
        it256 = F.interpolate(it, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        # per-slot oracle Δ in normalized xyz order: x<-d_W, y<-d_H, z<-d_D
        dz = norm_delta(disp[:, 0], SP[0], D)
        dy = norm_delta(disp[:, 1], SP[1], H)
        dx = norm_delta(disp[:, 2], SP[2], W)
        odelta = torch.stack([dx, dy, dz], dim=-1).view(-1, 1, 1, 3)  # (S,1,1,3)
        Vid = splat(sc256, it256)
        Vp = splat(sc256 + odelta, it256)
        Vn = splat(sc256 - odelta, it256)
        rec["identity"].append(psnr(Vid.float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        rec["oracle_undo"].append(psnr(Vp.float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        rec["oracle_neg"].append(psnr(Vn.float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        if model is not None:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                preds = model(b["images"], batch=b)
            Vm = preds.get("V_canon")
            if Vm is None:
                from vggt.utils.splat import splat_predictions
                Vm, _ = splat_predictions({"world_points": preds["world_points"].float()}, b, GRID_SHAPE)
            rec["model"].append(psnr(Vm[0].float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        rec["l1_id"].append(float((Vid - V_gt).abs().mean()))
        rec["l1_oracle"].append(min(float((Vp - V_gt).abs().mean()), float((Vn - V_gt).abs().mean())))
        si = disp[:, 0].abs().cpu().numpy()
        rec["applied_p95"].append(float(np.percentile(si, 95)))
        rec["applied_deep_frac"].append(float((si >= 8).mean()))
    return {k: round(float(np.mean(v)), 3) for k, v in rec.items() if v}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default="0-14")
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    lo, hi = (args.seqs.split("-") if "-" in args.seqs else (args.seqs, args.seqs))
    seqs = list(range(int(lo), int(hi) + 1)) if "-" in args.seqs else [int(s) for s in args.seqs.split(",")]
    device = "cuda"; ds = build_dataset()
    model = build_ref_model(device)
    base = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
                per_slot=True, direction_jitter_deg=30.0)
    cells = [("oneframe_burst", 12, True), ("multiframe_burst", 20, True),
             ("multiframe_iid", 20, False)]
    summary = {}
    for name, ips, burst in cells:
        cfg = RespiratoryConfig(enable=True, group_by_burst=burst, **base)
        r = eval_cell(ds, cfg, device, seqs, ips, model=model)
        r["oracle_gain_dB"] = round(max(r["oracle_undo"], r["oracle_neg"]) - r["identity"], 3)
        r["model_gain_dB"] = round(r["model"] - r["identity"], 3)
        r["model_captures_frac"] = (round(r["model_gain_dB"] / r["oracle_gain_dB"], 2)
                                    if r["oracle_gain_dB"] > 0.1 else None)
        summary[name] = r
        print(f"{name:20s} identity={r['identity']:.2f}  model={r['model']:.2f}(+{r['model_gain_dB']:.2f})  "
              f"oracle={max(r['oracle_undo'],r['oracle_neg']):.2f}(+{r['oracle_gain_dB']:.2f})  "
              f"model/oracle={r['model_captures_frac']}  (p95={r['applied_p95']:.1f}mm)", flush=True)
    summary["_meta"] = {"n": len(seqs), "note": "pure_resp (all slots at target phase). oracle_gain>0 "
                        "=> undoing the breath improves recon => recoverable in principle; ~0 => degenerate/wall."}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\nWrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
