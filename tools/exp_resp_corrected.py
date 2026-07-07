"""CORRECTED breathing experiment (supersedes exp_resp_oracle.py's invalid shift-oracle).

The flaw in exp_resp_oracle: the "oracle" re-shifted breathed slices by the true displacement, which
under the coverage-divided splat (V=acc/(cov+1e-6)) EVACUATES single-covered planes -> holes -> the
oracle self-sabotages and is NOT a valid upper bound. Model>oracle was the model under-moving to
avoid holes, i.e. the objective rewarding under-correction (doc 19 E0), NOT "breathing at ceiling."

Correct design (pure_resp = all input slots at the target phase, so the ONLY corruption is the rigid
SI breath; motion mask from the UNSHIFTED phases; same slots for every condition):
  V_clean : respiratory OFF, Δ=0  -> correct content, FULL coverage = the VALID breath-corrected
            ceiling (no reslice, no holes). In pure_resp this is doc-13's perfect-placement oracle.
  V_id    : respiratory ON,  Δ=0  -> do-nothing floor (breathed slices on their planes).
  V_model : respiratory ON,  model Δ -> what the trained model actually achieves.
  V_oracle: respiratory ON,  +undo Δ (the OLD invalid oracle) -> kept only as a diagnostic, scored
            both cov-divided and coverage-restricted, to SHOW the hole penalty.

Honest accounting:
  breathing_cost   = V_clean - V_id           (true cost of breathing; doc 08 says ~ -2.31 dB ON->OFF)
  model_recovery   = V_model - V_id           (what the model recovers)
  model_residual   = V_clean - V_model        (REAL remaining headroom)
  recovery_frac    = model_recovery / breathing_cost

Run: micromamba run -n svr python tools/exp_resp_corrected.py --seqs 0-29 --S 20
"""
import argparse, json, os, sys, copy
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE
from exp_resp_oracle import build_ref_model, norm_delta
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume, splat_predictions
D, H, W = GRID_SHAPE; SP = (12.0, 1.4, 1.4)
OUT = os.path.join(REPO, "result", "resp_corrected")


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def splat(world, inten):
    intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
    V, cov = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
    return V[0], cov[0]


def sc_it(b, device):
    sc = b["scanner_coords"][0].permute(0, 3, 1, 2)
    sc256 = F.interpolate(sc, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    it = b["images"][0].float().mean(dim=1, keepdim=True)
    it256 = F.interpolate(it, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
    return sc256, it256


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default="0-29"); ap.add_argument("--S", type=int, default=20)
    args = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    lo, hi = args.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    device = "cuda"; ds = build_dataset(); ds.t_target_fixed = 0
    model = build_ref_model(device)
    base = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
                per_slot=True, direction_jitter_deg=30.0, group_by_burst=True)
    cfg_on = RespiratoryConfig(enable=True, **base)
    cfg_off = RespiratoryConfig(enable=False, **base)
    def run_model(b):
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(b["images"], batch=b)
        Vm, _ = splat_predictions({"world_points": preds["world_points"].float()}, b, GRID_SHAPE)
        return Vm[0]
    R = {k: [] for k in ["id_clean", "id_breath", "model_clean", "model_breath",
                          "id_breath_cost", "model_breath_cost", "model_recovery"]}
    for seq in seqs:
        # NORMAL multiphase regime (slots at their sampled cardiac phases; ED target). Breathing ON vs
        # OFF is the ONLY toggle, so model_clean - model_breath ISOLATES the breathing effect on the model.
        data = ds.get_data(seq_index=seq, img_per_seq=args.S)
        b_cl = build_batch(data, device, seq_index=seq)
        b_br = build_batch(data, device, seq_index=seq)
        out = compute_volume_intensity_loss({"world_points": b_cl["scanner_coords"]}, b_cl, grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float(); mmask = compute_motion_mask(b_cl["phases"])[0].cpu().numpy()
        if not mmask.any():
            continue
        b_cl = gpu_augment_batch(b_cl, None, device, respiratory_cfg=cfg_off, train=False)
        b_br = gpu_augment_batch(b_br, None, device, respiratory_cfg=cfg_on, train=False)
        sc, it_cl = sc_it(b_cl, device); _, it_br = sc_it(b_br, device)
        gt = V_gt.cpu().numpy()
        pic = psnr(splat(sc, it_cl)[0].float().cpu().numpy(), gt, mmask)   # identity, clean
        pib = psnr(splat(sc, it_br)[0].float().cpu().numpy(), gt, mmask)   # identity, breathed
        pmc = psnr(run_model(b_cl).float().cpu().numpy(), gt, mmask)       # MODEL, clean
        pmb = psnr(run_model(b_br).float().cpu().numpy(), gt, mmask)       # MODEL, breathed
        R["id_clean"].append(pic); R["id_breath"].append(pib)
        R["model_clean"].append(pmc); R["model_breath"].append(pmb)
        R["id_breath_cost"].append(pic - pib)          # raw cost of breathing (do-nothing)
        R["model_breath_cost"].append(pmc - pmb)       # cost of breathing AFTER the model = real headroom
        R["model_recovery"].append(pmb - pib)          # how much the model beats do-nothing on breathed
    S = {k: round(float(np.mean(v)), 2) for k, v in R.items() if v}
    S["_meta"] = {"n": len(R["id_clean"]), "S": args.S, "regime": "multiphase, ED target, group_by_burst, motion PSNR",
                  "reading": "model_breath_cost = model_clean - model_breath = the breathing degradation the "
                             "model FAILS to remove (real headroom; doc08 got ~2.31 for ON->OFF). If small "
                             "=> model already compensates breathing; if large => headroom."}
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print(json.dumps(S, indent=2)); print("Wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
