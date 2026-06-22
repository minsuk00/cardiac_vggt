"""E3: warp-only CEILING via direct optimization of Δ on the REAL task (target-aware).

Remove the network. Take a real breathing-val sample (corrupted input slices at other phases,
+ V_gt target). Directly optimize a free per-pixel Δ (init 0) to minimize the loss through the
splat. This is the BEST a warp+splat can do given the real inputs and a perfect (target-aware)
optimizer — the realistic warp-only ceiling. Compare renderers and losses.

Interpretation:
 - If direct-opt ≈ oracle (35 dB) ≫ model (20.6): the objective is optimizable to near-perfect;
   the model's gap is learning/info (NOT the renderer).
 - If direct-opt ≈ model ≪ oracle: the splat+loss objective itself caps the warp → renderer/loss.
 - Renderer/loss variants that RAISE the ceiling = candidate fixes (upper-bound evidence).

NOTE (debate caveat): direct-opt is target-aware ⇒ NOT r-blind ⇒ it measures the OBJECTIVE's
optimizability, not whether a blind network can reach it. Logs the loss curve to prove convergence.

Run: micromamba run -n svr python tools/toy_directopt.py --seqs 0,1,2 --steps 1500
"""
import argparse, json, os, sys
import numpy as np, torch
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from data.gpu_aug import gpu_augment_batch
from data.respiratory import RespiratoryConfig
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume

D, H, W = GRID_SHAPE
OUT = os.path.join(REPO, "result", "limits_eval")
RESP = dict(amplitude_mm=16.0, amplitude_jitter=8.0, cos2n=3, ap_ratio=0.35, ap_axis="H",
            per_slot=True, direction_jitter_deg=30.0)


def psnr(a, b, m=None):
    if m is not None: a, b = a[m], b[m]
    mse = float(((a - b) ** 2).mean()); return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def splat_render(world, inten, nocovdiv=False):
    S = world.shape[1]
    int_flat = inten.reshape(1, -1); w = (int_flat > 1e-3).float()
    Vc, cov = splat_to_volume(world.reshape(1, -1, 3), int_flat, (D, H, W), weight=w)
    return (Vc * (cov + 1e-6)) if nocovdiv else Vc   # (1,D,H,W)


def directopt(scanner, inten, V_gt, mmask, steps, lr, tv, nocovdiv, motion_w, mmask_t, log=False):
    delta = torch.zeros_like(scanner, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    Vgt_b = V_gt.unsqueeze(0)
    wmap = torch.ones_like(V_gt)
    if motion_w > 1: wmap = wmap + (motion_w - 1) * mmask_t.float()
    curve = []
    for it in range(steps):
        world = scanner + delta
        V = splat_render(world, inten, nocovdiv)
        l1 = ((V - Vgt_b).abs() * wmap).mean()
        if tv > 0:
            tvl = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean() + (delta[:, :, :, 1:] - delta[:, :, :, :-1]).abs().mean()
            loss = l1 + tv * tvl
        else:
            loss = l1
        opt.zero_grad(); loss.backward(); opt.step()
        if log and it % 200 == 0:
            with torch.no_grad():
                mp = psnr(splat_render(scanner + delta, inten)[0].float().cpu().numpy(),
                          V_gt.cpu().numpy(), mmask)
            curve.append((it, float(loss), mp))
    with torch.no_grad():
        V = splat_render(scanner + delta, inten)[0].float().cpu().numpy()
        mp = psnr(V, V_gt.cpu().numpy(), mmask)
    return mp, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0,1,2")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.03)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"; ds = build_dataset()
    seqs = [int(s) for s in args.seqs.split(",")]
    cfg = RespiratoryConfig(enable=True, **RESP)

    variants = [
        ("splat_L1+TV (current obj)", dict(tv=0.1, nocovdiv=False, motion_w=1)),
        ("splat_L1 (no TV)",          dict(tv=0.0, nocovdiv=False, motion_w=1)),
        ("splat_L1_motionW10",        dict(tv=0.0, nocovdiv=False, motion_w=10)),
        ("nocovdiv_L1",               dict(tv=0.0, nocovdiv=True,  motion_w=1)),
    ]
    res = {v[0]: [] for v in variants}
    id_mp, model_ref = [], 20.6
    first_curve = None
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
        out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]}, batch,
                                            grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float()
        mmask_t = compute_motion_mask(batch["phases"])[0]
        mmask = mmask_t.cpu().numpy()
        if not mmask.any(): continue
        scanner = batch["scanner_coords"]                  # (1,S,518,518,3)
        inten = batch["images"].float().mean(dim=2)         # (1,S,518,518)
        id_mp.append(psnr(out["V_canon"][0].float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        for name, kw in variants:
            mp, curve = directopt(scanner, inten, V_gt, mmask, args.steps, args.lr,
                                  kw["tv"], kw["nocovdiv"], kw["motion_w"], mmask_t,
                                  log=(seq == seqs[0] and name == variants[0][0]))
            res[name].append(mp)
            if curve: first_curve = curve

    summary = {k: float(np.mean(v)) for k, v in res.items() if v}
    summary["identity_floor"] = float(np.mean(id_mp))
    summary["model_ref"] = model_ref
    summary["oracle_ref"] = 34.98
    json.dump({"summary": summary, "convergence": first_curve, "n": len(seqs), "steps": args.steps},
              open(os.path.join(OUT, "toy_directopt.json"), "w"), indent=2)

    print(f"=== E3 DIRECT-OPT warp-only ceiling (real breathing task, n={len(seqs)}, motion PSNR) ===")
    print(f"  identity floor       {summary['identity_floor']:6.2f}")
    print(f"  trained model (ref)  {model_ref:6.2f}")
    for name, _ in variants:
        if res[name]:
            print(f"  direct-opt {name:28s} {np.mean(res[name]):6.2f}")
    print(f"  oracle perfect-place 34.98")
    if first_curve:
        print("  [convergence of splat_L1+TV, seq0]: " +
              " ".join(f"it{it}:{mp:.1f}dB" for it, _, mp in first_curve))
    print("Wrote", os.path.join(OUT, "toy_directopt.json"))


if __name__ == "__main__":
    main()
