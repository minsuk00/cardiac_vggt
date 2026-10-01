"""E3 — Respiratory-z direct-optimization UPPER BOUND (is breathing-z a wall or a training gap?).

The trained blind models under-correct deep breaths (measure_dvf_z_correction.py: slope 0.34-0.42,
docs/07: 0.42). Two competing explanations:
  (pessimist) the per-plane SI shift is NON-IDENTIFIABLE / the splat+warp cannot recover it → a wall.
  (optimist)  the shift IS recoverable; the blind model just isn't extracting it → a training gap.

This probe gives the decisive UPPER BOUND. It removes the 941M network and DIRECTLY OPTIMIZES a
per-slot rigid displacement Δ to minimize L1(splat(warped inputs), V_gt) — i.e. it is HANDED the
answer V_gt and given a warp that can EXACTLY represent a rigid SI shift. Direct optimization has
strictly more freedom than any blind network, so:

  * If direct-opt (knowing V_gt) STILL cannot recover the SI shift (recovered-Δz-vs-applied-SI
    slope plateaus ~0.4, like the model) → NO blind network can either → WALL CONFIRMED, and it is a
    representation/renderer limit (matches docs/19 E0: splat through-plane gradient ~2x weak), not
    just missing information.
  * If direct-opt recovers it (slope -> ~1) → the splat+warp CAN represent the correction, so the
    renderer is NOT the culprit; the model's shortfall is inference/training (or blind
    identifiability, tested separately) → optimist has real headroom.

Two variants:
  realistic : input slots at random cardiac t (as trained) + group_by_burst breathing. Mixed
              cardiac+respiratory through-plane — matches deployment.
  pure_resp : ALL input slots forced to the target phase (t=0) so the ONLY thing to correct is the
              rigid SI breath. The cleanest identifiability test (no cardiac appearance confound).

Readout per slot: applied SI = resp_disp_mm[:,0] (mm) vs recovered Δz = mean optimized Δz over
anatomy (mm). slope, corr, and deep-breath-stratified bins (the wall lives in the |SI|>=8 tail;
docs/07: 157/297 slots <2mm).

Run: micromamba run -n svr python tools/exp_resp_directopt.py --seqs 0-11 --steps 1500
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "training"))
sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume

D, H, W = GRID_SHAPE
THROUGH_MM = (D - 1) / 2.0 * 12.0                # 66.0 mm per normalized z-unit (Z pitch 12mm)
ANAT_THR = 0.05
OUT = os.path.join(REPO, "result", "resp_directopt")
# group_by_burst=True to match the ACTIVE mri_volume.yaml regime (one breath per z-plane).
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
            per_slot=True, group_by_burst=True, direction_jitter_deg=30.0)


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def optimize(scanner, inten, V_gt, mmask, G, lr, steps, tv, clamp=True):
    """Direct-opt Δ (rigid G=1 or free) to min L1(splat, V_gt). Returns (best_mpsnr, delta_final)."""
    S, res = scanner.shape[0], scanner.shape[1]
    if G == "free":
        ctrl = torch.zeros_like(scanner, requires_grad=True)
    else:
        ctrl = torch.zeros((S, 3, G, G), device=scanner.device, requires_grad=True)
    opt = torch.optim.Adam([ctrl], lr=lr)
    Vgt_b = V_gt.unsqueeze(0)
    best, delta_best = -1.0, None
    for it in range(steps):
        delta = ctrl if G == "free" else F.interpolate(ctrl, size=(res, res), mode="bilinear",
                                                        align_corners=False).permute(0, 2, 3, 1)
        world = scanner + delta
        if clamp:
            world = world.clamp(-1.05, 1.05)
        intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
        V, _ = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
        l1 = (V - Vgt_b).abs().mean()
        if tv > 0:
            l1 = l1 + tv * ((delta[:, 1:] - delta[:, :-1]).abs().mean()
                            + (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean())
        opt.zero_grad(); l1.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            with torch.no_grad():
                mp = psnr(V[0].detach().float().cpu().numpy(), V_gt.cpu().numpy(), mmask)
                if mp > best:
                    best = mp
                    delta_best = (ctrl.detach() if G == "free" else
                                  F.interpolate(ctrl.detach(), size=(res, res), mode="bilinear",
                                                align_corners=False).permute(0, 2, 3, 1)).clone()
    return best, delta_best


def slot_readout(delta, inten, applied_si):
    """Per-slot recovered mean Δz over anatomy (mm) + applied SI (mm). delta (S,res,res,3) norm."""
    S = delta.shape[0]
    rec_dz, app, isref = [], [], []
    for s in range(S):
        m = inten[s] > ANAT_THR
        if not m.any():
            continue
        rec_dz.append(float(delta[s, :, :, 2][m].mean() * THROUGH_MM))
        app.append(float(applied_si[s]))
        isref.append(s == 0)
    return rec_dz, app, isref


def stats(app, rec):
    app, rec = np.asarray(app), np.asarray(rec)
    if app.size < 3:
        return {}
    slope, intercept = (float(v) for v in np.polyfit(app, rec, 1))
    corr = float(np.corrcoef(np.abs(app), rec)[0, 1])
    bins = [(0, 2), (2, 8), (8, 16), (16, 40)]
    binstat = {}
    for lo, hi in bins:
        sel = (np.abs(app) >= lo) & (np.abs(app) < hi)
        binstat[f"[{lo},{hi})"] = {"n": int(sel.sum()),
                                   "applied_mean": round(float(np.abs(app)[sel].mean()), 2) if sel.any() else None,
                                   "recovered_dz_mean": round(float(rec[sel].mean()), 2) if sel.any() else None}
    deep = np.abs(app) >= 8
    slope_deep = (float(np.polyfit(app[deep], rec[deep], 1)[0]) if deep.sum() >= 3 else None)
    return {"n": int(app.size), "slope": round(slope, 3), "intercept_mm": round(intercept, 2),
            "corr_absSI_recZ": round(corr, 3),
            "slope_deep_SIge8": round(slope_deep, 3) if slope_deep is not None else None,
            "n_deep_SIge8": int(deep.sum()), "bins": binstat}


def run_variant(ds, cfg, device, seqs, variant, configs, steps):
    ds.t_target_fixed = 0
    per = {c[0]: {"app": [], "rec": [], "ref": [], "mpsnr": []} for c in configs}
    idfloor = []
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        b = build_batch(data, device, seq_index=seq)
        if variant == "pure_resp":
            b["timesteps"] = torch.zeros_like(b["timesteps"])     # all slots image the target phase
        b = gpu_augment_batch(b, None, device, respiratory_cfg=cfg, train=False)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b,
                                            grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float()
        mmask = compute_motion_mask(b["phases"])[0].cpu().numpy()
        if not mmask.any():
            continue
        applied_si = b["resp_disp_mm"][0][:, 0].cpu().numpy()      # (S,) mm SI
        sc = b["scanner_coords"][0].permute(0, 3, 1, 2)
        sc256 = F.interpolate(sc, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        it = b["images"][0].float().mean(dim=1, keepdim=True)
        it256 = F.interpolate(it, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        # identity floor (Δ=0)
        intf = it256.reshape(1, -1); w = (intf > 1e-3).float()
        Vid, _ = splat_to_volume(sc256.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
        idfloor.append(psnr(Vid[0].float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        for name, G, lr, tv in configs:
            best, delta = optimize(sc256, it256, V_gt, mmask, G, lr, steps, tv)
            rec_dz, app, isref = slot_readout(delta, it256, applied_si)
            per[name]["app"] += app; per[name]["rec"] += rec_dz; per[name]["ref"] += isref
            per[name]["mpsnr"].append(best)
        print(f"  {variant} seq{seq}: idfloor={idfloor[-1]:.2f} "
              + " ".join(f"{n}={np.mean(per[n]['mpsnr']):.2f}" for n, *_ in configs), flush=True)
    result = {"identity_floor": round(float(np.mean(idfloor)), 2) if idfloor else None,
              "model_ref_slope": "reference 0.34 / bspline 0.31 / diffusion 0.42 (measure_dvf_z_correction)"}
    for name, *_ in configs:
        app, rec, ref = per[name]["app"], per[name]["rec"], per[name]["ref"]
        ref = np.asarray(ref)
        allst = stats(app, rec)
        nonref = stats(list(np.asarray(app)[~ref]), list(np.asarray(rec)[~ref])) if ref.size else {}
        result[name] = {"motion_psnr": round(float(np.mean(per[name]["mpsnr"])), 2),
                        "all_slots": allst, "scattered_only": nonref}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0-11")
    ap.add_argument("--steps", type=int, default=1500)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if "-" in args.seqs:
        lo, hi = args.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    else:
        seqs = [int(s) for s in args.seqs.split(",")]
    device = "cuda"
    ds = build_dataset()
    cfg = RespiratoryConfig(enable=True, **RESP)
    # rigid (G=1) = EXACT per-slot rigid translation = the respiratory model's own form (primary);
    # free = max-freedom upper bound.
    configs = [("rigid", 1, 0.02, 0.0), ("free", "free", 0.005, 0.0)]
    summary = {}
    for variant in ["pure_resp", "realistic"]:
        print(f"=== variant: {variant} (n={len(seqs)}, steps={args.steps}) ===", flush=True)
        summary[variant] = run_variant(ds, cfg, device, seqs, variant, configs, args.steps)
    summary["_meta"] = {"n_subjects": len(seqs), "steps": args.steps, "num_slices": NUM_SLICES,
                        "resp_cfg": RESP, "through_mm_per_unit": THROUGH_MM,
                        "note": "recovered Δz vs applied SI slope: ~1 = warp recovers breath given V_gt; "
                                "~0.4 (=model) = wall even with the answer in hand."}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n=== E3 RESP DIRECT-OPT BOUND ===")
    print(json.dumps(summary, indent=2))
    print("Wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
