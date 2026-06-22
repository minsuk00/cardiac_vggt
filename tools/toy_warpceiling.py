"""CORRECTED direct-opt: the true WARP-ONLY CEILING (proper LR + convergence logging).

prove-it found the earlier direct-opt used lr=0.03 → Adam DIVERGED (L1 went UP, Δ exploded
out of bounds). This version uses sane per-parameterization LRs, logs the loss to PROVE it
decreases (convergence), and reports the best converged motion PSNR = the realistic warp-only
ceiling on the real breathing task.

Question: optimizing Δ (any smooth parameterization) to minimize L1(splat, V_gt) — how high can
motion PSNR go?  ≈ model (20.6) → the warp+splat architecture is the ceiling, motion estimation is
NOT the lever (need appearance synthesis / drop warp).  ≈ oracle (35) → motion estimation IS the
lever and the model is far below what the objective permits.

Run: micromamba run -n svr python tools/toy_warpceiling.py --seqs 0,1,2,3 --steps 2500
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume
D, H, W = GRID_SHAPE
OUT = os.path.join(REPO, "result", "limits_eval")
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H", per_slot=True, direction_jitter_deg=30.0)


def psnr(a, b, m): a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean()); return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def splat256(world, inten):
    intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
    V, _ = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w); return V


def optimize(scanner, inten, V_gt, mmask, G, lr, steps, tv, clamp=True, log=False):
    S, res = scanner.shape[0], scanner.shape[1]
    if G == "free":
        delta = torch.zeros_like(scanner, requires_grad=True); ctrl = delta
    else:
        ctrl = torch.zeros((S, 3, G, G), device=scanner.device, requires_grad=True)
    opt = torch.optim.Adam([ctrl], lr=lr)
    Vgt_b = V_gt.unsqueeze(0); mnp = mmask
    best, curve = -1, []
    for it in range(steps):
        delta = ctrl if G == "free" else F.interpolate(ctrl, size=(res, res), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        world = scanner + delta
        if clamp: world = world.clamp(-1.05, 1.05)   # keep points in-bounds (prevents divergence off-grid)
        intf = inten.reshape(1, -1); w = (intf > 1e-3).float()
        V, _ = splat_to_volume(world.reshape(1, -1, 3), intf, (D, 256, 256), weight=w)
        l1 = (V - Vgt_b).abs().mean()
        loss = l1 + tv * ((delta[:, 1:] - delta[:, :-1]).abs().mean() + (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()) if tv > 0 else l1
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == steps - 1:
            with torch.no_grad():
                mp = psnr(V[0].detach().float().cpu().numpy(), V_gt.cpu().numpy(), mnp)
                best = max(best, mp)
                if log and it % 500 == 0: curve.append((it, round(float(l1), 4), round(mp, 2)))
    return best, float(l1), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0,1,2,3"); ap.add_argument("--steps", type=int, default=2500)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"; ds = build_dataset(); cfg = RespiratoryConfig(enable=True, **RESP)
    seqs = [int(s) for s in args.seqs.split(",")]
    # (name, G, lr, tv) — sane per-parameterization LRs
    configs = [("rigid", 1, 0.02, 0.0), ("lowrank_G16", 16, 0.01, 0.0),
               ("lowrank_G32", 32, 0.008, 0.0), ("free_lr5e-3", "free", 0.005, 0.0),
               ("free_lr5e-3_TV", "free", 0.005, 0.05)]
    res = {c[0]: [] for c in configs}; idf = []; first_curves = {}
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        b = build_batch(data, device, seq_index=seq); b = gpu_augment_batch(b, None, device, respiratory_cfg=cfg, train=False)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float(); mmask = compute_motion_mask(b["phases"])[0].cpu().numpy()
        if not mmask.any(): continue
        sc = b["scanner_coords"][0].permute(0, 3, 1, 2); sc256 = F.interpolate(sc, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        it = b["images"][0].float().mean(dim=1, keepdim=True); it256 = F.interpolate(it, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        idf.append(psnr(splat256(sc256, it256)[0].float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        for name, G, lr, tv in configs:
            best, finall1, curve = optimize(sc256, it256, V_gt, mmask, G, lr, args.steps, tv, log=(seq == seqs[0]))
            res[name].append(best)
            if curve: first_curves[name] = curve
    summary = {k: float(np.mean(v)) for k, v in res.items() if v}
    summary["identity_floor"] = float(np.mean(idf)); summary["model_ref"] = 20.6; summary["oracle_ref"] = 34.98
    json.dump({"summary": summary, "curves": first_curves, "n": len(seqs), "steps": args.steps},
              open(os.path.join(OUT, "toy_warpceiling.json"), "w"), indent=2)
    print(f"=== WARP-ONLY CEILING (proper LR + convergence, real breathing task, n={len(seqs)}, motion PSNR) ===")
    print(f"  identity floor {summary['identity_floor']:.2f} | trained model 20.6 | oracle (perfect place) 35.0")
    for name, *_ in configs:
        if res[name]: print(f"  direct-opt {name:18s} best motion PSNR = {np.mean(res[name]):.2f}")
    print("  convergence (loss should DECREASE — proving proper optimization this time):")
    for name, cv in first_curves.items():
        print(f"    {name:18s}: " + " ".join(f"it{it}(L1={l1},{mp}dB)" for it, l1, mp in cv))
    print("Wrote", os.path.join(OUT, "toy_warpceiling.json"))


if __name__ == "__main__":
    main()
