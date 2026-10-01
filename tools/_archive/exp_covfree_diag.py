"""STEP 1 go/no-go: does a COVERAGE-FREE renderer let direct-optimization recover DEEPER through-plane
breaths than the current coverage-divided splat? (No network, no retrain — pure objective test.)

Setup (per subject, per-plane breathing so the renderer difference is exposed):
  V0 = clean phase-0 volume (D,H,W). Each in-bbox plane z gets an independent breath δ_z (mm, Lujan
  distribution). breathed_slice[z] = V0 resampled at plane z shifted through-plane by δ_z (grid_sample).
  So the slice labelled z now images anatomy from a different depth — exactly what a per-slice breath does.
  Stack -> vol_breathed (D,H,W), one slice per plane (fully covered).

Then directly OPTIMIZE a per-plane correction Δz (D scalars) to minimize L1(render, V0), under:
  covdiv : the training splat  V = acc/(cov+1e-6)   (world_z = z + Δz per plane; scatter)
  invwarp: coverage-free GATHER  V[p] = vol_breathed( p - Δz[p] )  (grid_sample along z; intensity-preserving)

Readout: recovered Δz[z] vs applied δ_z -> slope, per renderer, pooled over subjects + deep-breath bins.
  covdiv slope ≈ invwarp slope  -> renderer is NOT the limiter (info/coarse-z bound); retrain won't help.
  invwarp slope >> covdiv slope  -> coverage-division suppresses recoverable z-motion; a coverage-free
                                    RE-TRAIN is justified (bounded by how much deeper invwarp recovers).

Run: micromamba run -n svr python tools/exp_covfree_diag.py --seqs 0-11 --steps 1200
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume
D, H, W = GRID_SHAPE
THROUGH_MM = (D - 1) / 2.0 * 12.0                 # 66.0 mm per normalized z-unit
OUT = os.path.join(REPO, "result", "covfree_diag")
AMP, JIT, N = 16.0, 8.0, 3                         # Lujan breath: amplitude 16±8mm, sin^{2n}, n=3


def psnr(a, b, m):
    a, b = a[m], b[m]; mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def shift_plane(V0, z, dz_norm):
    """Sample V0 (D,H,W) at plane z shifted by dz_norm (normalized) along z -> (H,W)."""
    ys = torch.linspace(-1, 1, H, device=V0.device); xs = torch.linspace(-1, 1, W, device=V0.device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    z_norm = z / (D - 1) * 2 - 1 + dz_norm
    grid = torch.stack([gx, gy, torch.full_like(gx, z_norm)], dim=-1).view(1, 1, H, W, 3)
    return F.grid_sample(V0.view(1, 1, D, H, W), grid, mode="bilinear", align_corners=True, padding_mode="border")[0, 0, 0]


def render_covdiv(scanner, inten, dz):
    """scatter splat with /cov. scanner (S,H,W,3), inten (S,H,W), dz (S,) per-plane correction (norm)."""
    world = scanner.clone(); world[..., 2] = world[..., 2] + dz.view(-1, 1, 1)
    itf = inten.reshape(1, -1); w = (itf > 1e-3).float()
    V, _ = splat_to_volume(world.reshape(1, -1, 3), itf, (D, H, W), weight=w)
    return V[0]


def render_invwarp(vol, dz):
    """coverage-free gather: output plane p samples vol at (p - dz[p]). vol (D,H,W), dz (D,) norm."""
    ys = torch.linspace(-1, 1, H, device=vol.device); xs = torch.linspace(-1, 1, W, device=vol.device)
    zs = torch.linspace(-1, 1, D, device=vol.device)
    # per-plane z sample coord = plane_norm - dz[p]  (dz in normalized z-units)
    zc = (zs - dz).view(D, 1, 1).expand(D, H, W)
    gy = ys.view(1, H, 1).expand(D, H, W); gx = xs.view(1, 1, W).expand(D, H, W)
    grid = torch.stack([gx, gy, zc], dim=-1).view(1, D, H, W, 3)
    return F.grid_sample(vol.view(1, 1, D, H, W), grid, mode="bilinear", align_corners=True, padding_mode="border")[0, 0]


def optimize(kind, scanner, inten, vol, V0, mmask, applied_norm, steps, lr=0.01):
    dz = torch.zeros(D, device=V0.device, requires_grad=True)
    opt = torch.optim.Adam([dz], lr=lr)
    for it in range(steps):
        V = render_covdiv(scanner, inten, dz) if kind == "covdiv" else render_invwarp(vol, dz)
        loss = (V - V0).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        V = render_covdiv(scanner, inten, dz) if kind == "covdiv" else render_invwarp(vol, dz)
        mp = psnr(V.float().cpu().numpy(), V0.cpu().numpy(), mmask)
    return dz.detach().cpu().numpy() * THROUGH_MM, mp   # recovered per-plane mm, motion PSNR


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seqs", default="0-11"); ap.add_argument("--steps", type=int, default=1200)
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    lo, hi = a.seqs.split("-"); seqs = list(range(int(lo), int(hi) + 1))
    dev = "cuda"; ds = build_dataset()
    rng = np.random.RandomState(0)
    rec = {"covdiv": {"app": [], "rec": [], "mp": []}, "invwarp": {"app": [], "rec": [], "mp": []}, "idfloor": []}
    for seq in seqs:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        b = build_batch(data, dev, seq)
        out = compute_volume_intensity_loss({"world_points": b["scanner_coords"]}, b, grid_shape=GRID_SHAPE, tv_weight=0.0)
        V0 = out["V_gt"][0].float()
        mm = compute_motion_mask(b["phases"])[0].cpu().numpy()
        bbox = [int(v) for v in b["anatomy_bbox"][0].tolist()]
        if not mm.any(): continue
        z0, z1 = max(bbox[0], 0), min(bbox[1], D)
        planes = list(range(z0, z1))
        # per-plane Lujan breath (mm) -> applied δ; build breathed slices + covered vol
        r = rng.uniform(0, 1, D); A = AMP + (rng.uniform(-1, 1, D)) * JIT
        applied_mm = np.clip(A, 0, None) * np.sin(np.pi * r) ** (2 * N)            # (D,) mm, 0 outside used planes
        applied_mm[[z for z in range(D) if z not in planes]] = 0.0
        vol = torch.zeros(D, H, W, device=dev)
        inten = torch.zeros(D, H, W, device=dev)
        for z in planes:
            dz_norm = float(applied_mm[z]) / THROUGH_MM
            sl = shift_plane(V0, z, dz_norm)          # slice images anatomy shifted by breath
            vol[z] = sl; inten[z] = sl
        # scanner coords for covdiv (S=D planes)
        ys = torch.linspace(-1, 1, H, device=dev); xs = torch.linspace(-1, 1, W, device=dev)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        scanner = torch.zeros(D, H, W, 3, device=dev)
        scanner[..., 0] = gx; scanner[..., 1] = gy
        for z in range(D): scanner[z, :, :, 2] = z / (D - 1) * 2 - 1
        applied_norm = torch.tensor(applied_mm / THROUGH_MM, device=dev, dtype=torch.float32)
        # identity floor
        Vid, _ = splat_to_volume(scanner.reshape(1, -1, 3), inten.reshape(1, -1),
                                 (D, H, W), weight=(inten.reshape(1, -1) > 1e-3).float())
        rec["idfloor"].append(psnr(Vid[0].float().cpu().numpy(), V0.cpu().numpy(), mm))
        for kind in ["covdiv", "invwarp"]:
            recovered_mm, mp = optimize(kind, scanner, inten, vol, V0, mm, applied_norm, a.steps)
            for z in planes:
                rec[kind]["app"].append(float(applied_mm[z])); rec[kind]["rec"].append(float(recovered_mm[z]))
            rec[kind]["mp"].append(mp)
        print(f"seq{seq}: idfloor={rec['idfloor'][-1]:.2f} covdiv_mp={rec['covdiv']['mp'][-1]:.2f} invwarp_mp={rec['invwarp']['mp'][-1]:.2f}", flush=True)

    summ = {"idfloor": round(float(np.mean(rec["idfloor"])), 2), "n": len(rec["idfloor"]), "steps": a.steps}
    for kind in ["covdiv", "invwarp"]:
        app = np.array(rec[kind]["app"]); recd = np.array(rec[kind]["rec"])
        # sign-align: recovered should track applied; report |slope| via least squares through matched sign
        slope = float(np.polyfit(app, recd, 1)[0])
        if slope < 0: recd = -recd; slope = -slope        # renderer may recover with either sign; compare magnitude
        deep = app >= 12
        summ[kind] = {"slope": round(slope, 3), "corr": round(float(np.corrcoef(app, recd)[0, 1]), 3),
                      "motion_psnr": round(float(np.mean(rec[kind]["mp"])), 2),
                      "deep_applied_mean": round(float(app[deep].mean()), 1) if deep.any() else None,
                      "deep_recovered_mean": round(float(recd[deep].mean()), 1) if deep.any() else None,
                      "n_planes": int(app.size)}
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("\n=== STEP 1: coverage-free renderer diagnostic ===")
    print(json.dumps(summ, indent=2))
    print("Wrote", os.path.join(OUT, "summary.json"))


if __name__ == "__main__":
    main()
