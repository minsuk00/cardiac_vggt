"""E4: does a CONSTRAINED (low-rank/smooth) Δ avoid the free-Δ degeneracy and reach a high ceiling?

E3 showed free per-pixel Δ + volume-L1 is degenerate (motion PSNR collapses to ~10 dB, below the
identity floor) — the optimizer cheats via the splat's many-to-one scatter. The network avoids this
via implicit feature-smoothness. Here we make smoothness EXPLICIT: parameterize Δ as a coarse
control grid (G×G per slot) bilinearly upsampled — the B-spline/FFD / low-rank motion idea — and
sweep the DOF G. Target-aware direct-opt on the real breathing task (warp-only ceiling).

Prediction (low-rank hypothesis): G=1 (rigid/slot) underfits local cardiac motion; mid G (~8-16)
gives the best motion PSNR (smooth, no degeneracy) and beats the model; G→full collapses (free-Δ
degeneracy). A clean inverted-U vs DOF proves: (a) free Δ is the wrong parameterization, (b) a
smooth/low-rank Δ is a real lever, (c) the objective+splat CAN reach a high ceiling with the right
constraint.

Works at canonical 256 (downsample inputs) — motion-PSNR partition is resolution-independent (256 vs
518 is the separate sharpness axis). Run: micromamba run -n svr python tools/toy_lowrank.py
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
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


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def splat256(world, inten):
    S = world.shape[0]
    int_flat = inten.reshape(1, -1); w = (int_flat > 1e-3).float()
    V, _ = splat_to_volume(world.reshape(1, -1, 3), int_flat, (D, 256, 256), weight=w)
    return V[0]


def opt_grid(scanner, inten, V_gt, mmask, G, steps=800, lr=0.03, tv=0.0, nocovdiv=False, seed=0):
    """Δ parameterized by a (S,3,G,G) control grid upsampled to 256. G=256 => free per-pixel."""
    torch.manual_seed(seed)
    S, res = scanner.shape[0], scanner.shape[1]
    ctrl = torch.zeros((S, 3, G, G), device=scanner.device, requires_grad=True)
    opt = torch.optim.Adam([ctrl], lr=lr)
    Vgt_b = V_gt.unsqueeze(0); m = mmask
    for it in range(steps):
        if G == res:
            delta = ctrl.permute(0, 2, 3, 1)
        else:
            delta = F.interpolate(ctrl, size=(res, res), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        world = scanner + delta
        int_flat = inten.reshape(1, -1); wgt = (int_flat > 1e-3).float()
        Vc, cov = splat_to_volume(world.reshape(1, -1, 3), int_flat, (D, 256, 256), weight=wgt)
        V = (Vc * (cov + 1e-6)) if nocovdiv else Vc
        loss = (V - Vgt_b).abs().mean()
        if tv > 0:
            loss = loss + tv * ((delta[:, 1:] - delta[:, :-1]).abs().mean() + (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        if G == res: delta = ctrl.permute(0, 2, 3, 1)
        else: delta = F.interpolate(ctrl, size=(res, res), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        V = splat256(scanner + delta, inten)[0].float().cpu().numpy() if False else splat256(scanner + delta, inten).float().cpu().numpy()
        return psnr(V, V_gt.cpu().numpy(), m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0,1,2,3"); ap.add_argument("--steps", type=int, default=800)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"; ds = build_dataset(); cfg = RespiratoryConfig(enable=True, **RESP)
    seqs = [int(s) for s in args.seqs.split(",")]
    Gs = [1, 2, 4, 8, 16, 32, 64, 256]
    res = {G: [] for G in Gs}; res_nc = {8: [], 16: []}; idf = []

    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
        out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]}, batch,
                                            grid_shape=GRID_SHAPE, tv_weight=0.0)
        V_gt = out["V_gt"][0].float()
        mmask = compute_motion_mask(batch["phases"])[0].cpu().numpy()
        if not mmask.any(): continue
        # downsample inputs 518 -> 256
        sc = batch["scanner_coords"][0].permute(0, 3, 1, 2)                      # (S,3,518,518)
        sc256 = F.interpolate(sc, size=(256, 256), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
        inten = batch["images"][0].float().mean(dim=1, keepdim=True)              # (S,1,518,518)
        inten256 = F.interpolate(inten, size=(256, 256), mode="bilinear", align_corners=False)[:, 0]
        idf.append(psnr(splat256(sc256, inten256).float().cpu().numpy(), V_gt.cpu().numpy(), mmask))
        for G in Gs:
            res[G].append(opt_grid(sc256, inten256, V_gt, mmask, G, steps=args.steps))
        for G in [8, 16]:
            res_nc[G].append(opt_grid(sc256, inten256, V_gt, mmask, G, steps=args.steps, nocovdiv=True))

    summary = {f"G{G}": float(np.mean(v)) for G, v in res.items() if v}
    summary["identity_floor"] = float(np.mean(idf))
    summary["model_ref"] = 20.6; summary["oracle_ref"] = 34.98
    summary.update({f"G{G}_nocovdiv": float(np.mean(v)) for G, v in res_nc.items() if v})
    json.dump({"summary": summary, "Gs": Gs, "n": len(seqs)}, open(os.path.join(OUT, "toy_lowrank.json"), "w"), indent=2)

    # plot inverted-U
    gs = [G for G in Gs if res[G]]; ys = [np.mean(res[G]) for G in gs]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(gs, ys, "o-", color="#1a73e8", lw=2, label="low-rank Δ (control grid G×G)")
    ax.axhline(summary["identity_floor"], color="#9aa0a6", ls=":", label=f"identity floor {summary['identity_floor']:.1f}")
    ax.axhline(20.6, color="#0b8043", ls="--", label="trained model 20.6")
    ax.axhline(34.98, color="#e8710a", ls="--", label="oracle (perfect place) 35.0")
    ax.set_xscale("log", base=2); ax.set_xlabel("control-grid resolution G (DOF per slot; 256 = free per-pixel)")
    ax.set_ylabel("warp-only ceiling — motion PSNR (dB)")
    ax.set_title("E4: constrained (low-rank) Δ avoids the free-Δ degeneracy\n(target-aware direct-opt, real breathing task)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig_e4_lowrank.png"), bbox_inches="tight", dpi=115, facecolor="white"); plt.close(fig)

    print(f"=== E4 LOW-RANK Δ ceiling (real breathing task, n={len(seqs)}, motion PSNR) ===")
    print(f"  identity floor {summary['identity_floor']:.2f} | model 20.6 | oracle 35.0")
    for G in Gs:
        if res[G]: print(f"  G={G:4d} ({G*G:6d} DOF/slot): {np.mean(res[G]):6.2f} dB")
    for G in [8, 16]:
        if res_nc[G]: print(f"  G={G} no-covdiv: {np.mean(res_nc[G]):6.2f} dB (vs splat {np.mean(res[G]):.2f})")
    best = max(gs, key=lambda G: np.mean(res[G]))
    print(f"  BEST G={best} at {np.mean(res[best]):.2f} dB  (free G=256 = {np.mean(res[256]):.2f}; "
          f"degeneracy if free << best)")
    print("Wrote", os.path.join(OUT, "toy_lowrank.json"), "+ fig_e4_lowrank.png")


if __name__ == "__main__":
    main()
