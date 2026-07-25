"""Conclusive DVF + reconstruction analysis of model 4wokxzov (217720691: reference-slot,
1-frame-per-slice, DPT head). Reads the predicted displacement field DIRECTLY (mm), not just PSNR,
and isolates each motion axis. n=30 val, S=12.

Axes:
  A. Breathing through-plane (resp ON, ED target): applied SI vs predicted Δz -> slope, corr, deep
     bins, and the FRACTION of deep breaths (|SI|>=12mm) the model essentially ignores (|Δz|<2mm).
  B. Breathing cost (clean vs breathed, ED target): identity/model on clean vs breathed inputs ->
     raw breathing cost, model residual, recovery %.  (No shift-oracle -> no coverage-hole confound.)
  C. Cardiac motion (resp OFF, so breathing removed): per target phase, predicted in-plane |Δ| and
     through-plane |Δz| over anatomy, and motion PSNR. ED (t0) vs ES (t6) contrast shows whether the
     model applies cardiac in-plane and cardiac through-plane (longitudinal) motion.
  D. Per-phase reconstruction (resp OFF): model motion PSNR vs identity_clean (perfect-placement
     proxy) for t=0..11 -> which phases are hard (ES dip) and the gap to the ceiling.

Run: micromamba run -n svr python tools/exp_4wok_analysis.py --seqs 0-29
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_batch, GRID_SHAPE, DATA_ROOT, SPLIT_FILE
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from data.datasets.mri_dataset import MRIDataset
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume, splat_predictions
from vggt.models.vggt import VGGT
from omegaconf import OmegaConf


def build_dataset_ref():
    """CRITICAL: reference_slot=True so slot 0 = the target-phase reference the model reads. Without
    it the model gets a random slice in slot 0 and cannot see the target phase (invalid eval)."""
    conf = OmegaConf.create({"img_size": 518, "patch_size": 14, "rescale": True, "rescale_aug": False,
                             "landscape_check": False, "augs": {"scales": [1.0, 1.0]}})
    return MRIDataset(conf, DATA_ROOT, split="val", split_file=SPLIT_FILE, mode="dynamic",
                      mri_mode="axial", num_slices=12, target_size=518, reference_slot=True)

D, H, W = GRID_SHAPE
THROUGH_MM = (D - 1) / 2.0 * 12.0            # 66.0  mm per norm z-unit
INPLANE_MM = (256 - 1) / 2.0 * 1.4           # 178.5 mm per norm in-plane unit
ANAT = 0.05
CKPT = "scratch/logs/217720691_mri_volume_diffusion_dynamic_axial_Cine_combined/ckpts/checkpoint_last.pt"
OUT_BASE = os.path.join(REPO, "result", "analysis_4wok")
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
            per_slot=True, group_by_burst=True, direction_jitter_deg=30.0)


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def build_model(device, ckpt, head="dpt", grid=32):
    m = VGGT(img_size=518, patch_size=14, embed_dim=1024, enable_camera=False, enable_depth=False,
             enable_point=True, enable_track=False, use_z_pose_embedding=True,
             use_t_pose_embedding=False, use_target_t_pose_embedding=False, use_reference_token=True,
             train_on_residual_dvf=True, warp_head_type=head, bspline_grid_size=grid).to(device).eval()
    ck = torch.load(os.path.join(REPO, ckpt), map_location="cpu", weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    assert not miss and not unexp, f"missing={miss[:4]} unexpected={unexp[:4]}"
    return m


def fwd(model, b):
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        preds = model(b["images"], batch=b)
    wp = preds["world_points"].float()
    dvf = (wp[0] - b["scanner_coords"][0]).float()                    # (S,H,W,3) norm
    Vm, _ = splat_predictions({"world_points": wp}, b, GRID_SHAPE)
    return Vm[0], dvf


def splat_id(b):
    sc = b["scanner_coords"][0].reshape(1, -1, 3)
    it = b["images"][0].float().mean(1).reshape(1, -1)
    if it.max() > 2: it = it / 255.0
    w = (it > 1e-3).float()
    V, _ = splat_to_volume(sc, it, (D, H, W), weight=w)
    return V[0]


def dvf_mm(dvf, imgs):
    """Per-slot mean |in-plane| and signed/abs Δz over anatomy (mm). imgs (S,H,W) in [0,1]."""
    S = dvf.shape[0]; ip, dzs, dza = [], [], []
    for s in range(S):
        m = imgs[s] > ANAT
        if not m.any(): continue
        dx = dvf[s, :, :, 0][m] * INPLANE_MM; dy = dvf[s, :, :, 1][m] * INPLANE_MM
        dz = dvf[s, :, :, 2][m] * THROUGH_MM
        ip.append(float(torch.sqrt(dx**2 + dy**2).mean())); dzs.append(float(dz.mean())); dza.append(float(dz.abs().mean()))
    return ip, dzs, dza


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default="0-29")
    ap.add_argument("--ckpt", default=CKPT); ap.add_argument("--tag", default="4wok")
    ap.add_argument("--head", default="dpt", choices=["dpt", "bspline"]); ap.add_argument("--grid", type=int, default=32)
    args = ap.parse_args()
    OUT = OUT_BASE if args.tag == "4wok" else OUT_BASE + "_" + args.tag
    os.makedirs(OUT, exist_ok=True)
    lo, hi = args.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    dev = "cuda"; ds = build_dataset_ref(); model = build_model(dev, args.ckpt, args.head, args.grid)
    cfg_on = RespiratoryConfig(enable=True, **RESP); cfg_off = RespiratoryConfig(enable=False, **RESP)

    # ---- A + B: breathing, ED target ----
    A = {"applied_si": [], "pred_dz": [], "is_ref": []}
    B = {"id_clean": [], "id_breath": [], "model_clean": [], "model_breath": []}
    ds.t_target_fixed = 0
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=12)
        b_cl = build_batch(data, dev, seq); b_br = build_batch(data, dev, seq)
        out = compute_volume_intensity_loss({"world_points": b_cl["scanner_coords"]}, b_cl, grid_shape=GRID_SHAPE, tv_weight=0.0)
        Vgt = out["V_gt"][0].float(); mm = compute_motion_mask(b_cl["phases"])[0].cpu().numpy()
        if not mm.any(): continue
        b_cl = gpu_augment_batch(b_cl, None, dev, respiratory_cfg=cfg_off, train=False)
        b_br = gpu_augment_batch(b_br, None, dev, respiratory_cfg=cfg_on, train=False)
        gt = Vgt.cpu().numpy()
        Vm_cl, _ = fwd(model, b_cl); Vm_br, dvf_br = fwd(model, b_br)
        B["id_clean"].append(psnr(splat_id(b_cl).cpu().numpy(), gt, mm))
        B["id_breath"].append(psnr(splat_id(b_br).cpu().numpy(), gt, mm))
        B["model_clean"].append(psnr(Vm_cl.cpu().numpy(), gt, mm))
        B["model_breath"].append(psnr(Vm_br.cpu().numpy(), gt, mm))
        disp = b_br["resp_disp_mm"][0].cpu().numpy(); imgs = b_br["images"][0].float().mean(1)
        for s in range(dvf_br.shape[0]):
            msk = imgs[s] > ANAT
            if not msk.any(): continue
            A["applied_si"].append(float(disp[s, 0]))
            A["pred_dz"].append(float(dvf_br[s, :, :, 2][msk].mean() * THROUGH_MM))
            A["is_ref"].append(s == 0)

    x = np.array(A["applied_si"]); y = np.array(A["pred_dz"]); ref = np.array(A["is_ref"])
    nr = ~ref
    def slope_corr(xx, yy):
        if xx.size < 3: return None, None
        return float(np.polyfit(xx, yy, 1)[0]), float(np.corrcoef(np.abs(xx), yy)[0, 1])
    sl, co = slope_corr(x[nr], y[nr])
    deep = np.abs(x) >= 12
    ignore_frac = float((np.abs(y[nr & deep]) < 2).mean()) if (nr & deep).any() else None
    breath = {
        "slope_scattered": round(sl, 3) if sl else None, "corr_scattered": round(co, 3) if co else None,
        "n_slots": int(nr.sum()),
        "pred_dz_at_SIge12_mean": round(float(y[nr & deep].mean()), 2) if (nr & deep).any() else None,
        "applied_SIge12_mean": round(float(np.abs(x[nr & deep]).mean()), 2) if (nr & deep).any() else None,
        "n_deep_SIge12": int((nr & deep).sum()),
        "frac_deep_ignored_dz_lt2mm": round(ignore_frac, 2) if ignore_frac is not None else None,
        "bins": {f"[{a},{c})": {"n": int(((np.abs(x[nr]) >= a) & (np.abs(x[nr]) < c)).sum()),
                                "pred_dz": round(float(y[nr][(np.abs(x[nr]) >= a) & (np.abs(x[nr]) < c)].mean()), 2)
                                if ((np.abs(x[nr]) >= a) & (np.abs(x[nr]) < c)).any() else None}
                 for a, c in [(0, 2), (2, 8), (8, 12), (12, 40)]},
    }
    Bm = {k: round(float(np.mean(v)), 2) for k, v in B.items()}
    Bm["raw_breath_cost"] = round(Bm["id_clean"] - Bm["id_breath"], 2)
    Bm["model_residual_breath"] = round(Bm["model_clean"] - Bm["model_breath"], 2)
    Bm["model_recovery"] = round(Bm["model_breath"] - Bm["id_breath"], 2)
    Bm["recovery_pct"] = round(100 * (Bm["model_breath"] - Bm["id_breath"]) / max(Bm["id_clean"] - Bm["id_breath"], 1e-6))

    # ---- C + D: cardiac (resp OFF), per target phase ----
    phases = {}
    for t in range(12):
        ds.t_target_fixed = t
        ip_all, dza_all, mp_all, idc_all = [], [], [], []
        for seq in seqs:
            data = ds.get_data(seq_index=seq, img_per_seq=12)
            b = build_batch(data, dev, seq)
            out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
            Vgt = out["V_gt"][0].float(); mm = compute_motion_mask(b["phases"])[0].cpu().numpy()
            if not mm.any(): continue
            b = gpu_augment_batch(b, None, dev, respiratory_cfg=cfg_off, train=False)
            gt = Vgt.cpu().numpy()
            Vm, dvf = fwd(model, b); imgs = b["images"][0].float().mean(1)
            ip, dzs, dza = dvf_mm(dvf, imgs)
            ip_all += ip[1:]; dza_all += dza[1:]                      # exclude reference slot 0
            mp_all.append(psnr(Vm.cpu().numpy(), gt, mm)); idc_all.append(psnr(splat_id(b).cpu().numpy(), gt, mm))
        phases[t] = {"inplane_mm": round(float(np.mean(ip_all)), 2), "through_dz_mm": round(float(np.mean(dza_all)), 2),
                     "model_psnr": round(float(np.mean(mp_all)), 2), "id_clean_psnr": round(float(np.mean(idc_all)), 2),
                     "gap_to_ceiling": round(float(np.mean(idc_all)) - float(np.mean(mp_all)), 2)}
        print(f"  t{t:2d}: inplane={phases[t]['inplane_mm']}mm through_dz={phases[t]['through_dz_mm']}mm "
              f"model={phases[t]['model_psnr']} ceiling={phases[t]['id_clean_psnr']} gap={phases[t]['gap_to_ceiling']}", flush=True)

    summary = {"model": "4wokxzov (217720691, reference-slot, 1-frame, DPT)", "n": len(seqs),
               "A_breathing_dvf": breath, "B_breathing_cost": Bm, "CD_per_phase_resp_off": phases}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n=== A breathing DVF ==="); print(json.dumps(breath, indent=2))
    print("=== B breathing cost ==="); print(json.dumps(Bm, indent=2))
    print("Wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
