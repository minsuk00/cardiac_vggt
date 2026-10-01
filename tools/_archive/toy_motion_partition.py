"""TOY: does the splat+L1+TV OBJECTIVE let gradient descent recover a KNOWN displacement?

Partition experiment. Remove the network entirely. Take one subject's target volume V_gt,
synthesize input slices by applying a KNOWN rigid shift (along through-plane z OR in-plane x)
to the scanner positions, then DIRECTLY optimize a displacement field Δ to minimize
L1(splat(scanner+Δ, intensity), V_gt) + tv·TV(Δ). The correct Δ is exactly -shift.

Question: does direct-opt recover the shift?
 - If in-plane x recovers but through-plane z UNDER-recovers → the splat objective itself is
   ill-conditioned for z (the 12-plane/8mm coarse axis) → under-correction is OBJECTIVE-limited.
 - If both recover → objective is fine → the model's under-correction is network/info-limited.

This sees the target (not r-blind) and has no capacity limit, so it is an UPPER BOUND on what the
objective permits. Known-answer (we set the shift), so correct vs incorrect is unambiguous.

Run: micromamba run -n svr python tools/toy_motion_partition.py
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
MM_PER_NORM_Z = (D - 1) / 2.0 * 8.0     # 44.0 mm per norm unit (through-plane)
MM_PER_NORM_X = (W - 1) / 2.0 * 1.4     # 178.5 mm per norm unit (in-plane)
OUT = os.path.join(REPO, "result", "limits_eval")


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def get_sample(ds, seq, device):
    data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
    batch = build_batch(data, device, seq_index=seq)
    out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]}, batch,
                                        grid_shape=GRID_SHAPE, tv_weight=0.0)
    V_gt = out["V_gt"][0].float()                       # (D,H,W)
    mmask = compute_motion_mask(batch["phases"])[0]      # (D,H,W) bool
    bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
    return V_gt, mmask, bbox


def make_shifted(V_gt, axis, shift_mm, res=256, device="cuda"):
    """Inputs = V_gt's own planes, placed at scanner positions SHIFTED by shift_mm along axis.
    Correct Δ to reconstruct V_gt = -shift (uniform). axis: 'z' or 'x'."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, res, device=device),
                            torch.linspace(-1, 1, res, device=device), indexing="ij")
    base = torch.stack([xs, ys], dim=-1)                 # (res,res,2)
    inten = V_gt.clone()                                 # (D,res,res) intensity = true plane content
    scanner = torch.zeros((D, res, res, 3), device=device)
    for z in range(D):
        scanner[z, :, :, :2] = base
        scanner[z, :, :, 2] = z / (D - 1) * 2 - 1
    if axis == "z":
        shift_norm = shift_mm / MM_PER_NORM_Z
        scanner[..., 2] = scanner[..., 2] + shift_norm
        ax = 2
    else:
        shift_norm = shift_mm / MM_PER_NORM_X
        scanner[..., 0] = scanner[..., 0] + shift_norm
        ax = 0
    return scanner, inten, shift_norm, ax


def optimize(scanner, inten, V_gt, mmask, mode, steps=400, tv=0.1, lr=0.02, seed=0):
    torch.manual_seed(seed)
    S = scanner.shape[0]
    w = (inten.reshape(S, -1) > 1e-3).float()
    int_flat = inten.reshape(S, -1)
    if mode == "rigid":
        delta = torch.zeros((S, 1, 1, 3), device=scanner.device, requires_grad=True)
    else:  # free per-pixel
        delta = torch.zeros_like(scanner, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    Vgt_b = V_gt.unsqueeze(0)
    for it in range(steps):
        world = scanner + delta                          # broadcast for rigid
        pos = world.reshape(1, S * scanner.shape[1] * scanner.shape[2], 3)
        Vp, _ = splat_to_volume(pos, int_flat.reshape(1, -1), (D, H, W), weight=w.reshape(1, -1))
        l1 = (Vp - Vgt_b).abs().mean()
        d = world
        tvl = ((d[:, 1:] - d[:, :-1]).abs().mean() + (d[:, :, 1:] - d[:, :, :-1]).abs().mean()) if tv > 0 else 0.0
        loss = l1 + tv * tvl
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        world = scanner + delta
        pos = world.reshape(1, S * scanner.shape[1] * scanner.shape[2], 3)
        Vp, _ = splat_to_volume(pos, int_flat.reshape(1, -1), (D, H, W), weight=w.reshape(1, -1))
        Vp = Vp[0]
        # recovered shift along axis = mean Δ over anatomy (nonzero-intensity) pixels
        d_full = (world - scanner)                       # (S,res,res,3)
        m = inten > 1e-3
        rec_z = float(d_full[..., 2][m].mean()) if m.any() else 0.0
        rec_x = float(d_full[..., 0][m].mean()) if m.any() else 0.0
        mp = psnr(Vp[mmask].cpu().numpy(), V_gt[mmask].cpu().numpy()) if mmask.any() else float("nan")
    return dict(rec_norm_z=rec_z, rec_norm_x=rec_x, Vp=Vp, motion_psnr=mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0,1,2,3")
    ap.add_argument("--shifts", default="4,8,16,24")
    ap.add_argument("--mode", default="rigid", choices=["rigid", "free"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--tv", type=float, default=0.1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    ds = build_dataset()
    seqs = [int(s) for s in args.seqs.split(",")]
    shifts = [float(s) for s in args.shifts.split(",")]

    rows = []
    for axis in ["z", "x"]:
        mm_per_norm = MM_PER_NORM_Z if axis == "z" else MM_PER_NORM_X
        for shift_mm in shifts:
            recs, mps = [], []
            for seq in seqs:
                V_gt, mmask, bbox = get_sample(ds, seq, device)
                scanner, inten, shift_norm, ax = make_shifted(V_gt, axis, shift_mm, device=device)
                r = optimize(scanner, inten, V_gt, mmask, args.mode, steps=args.steps, tv=args.tv)
                rec_norm = r["rec_norm_z"] if axis == "z" else r["rec_norm_x"]
                rec_mm = rec_norm * mm_per_norm
                recs.append(rec_mm); mps.append(r["motion_psnr"])
            applied_mm = -shift_mm   # correct Δ = -shift
            rec_mean = float(np.mean(recs))
            ratio = rec_mean / applied_mm if applied_mm != 0 else 0.0
            row = dict(axis=axis, shift_mm=shift_mm, applied_mm=applied_mm,
                       recovered_mm=round(rec_mean, 2), recovered_frac=round(ratio, 3),
                       motion_psnr=round(float(np.mean(mps)), 2))
            rows.append(row)
            print(f"  axis={axis} shift={shift_mm:5.1f}mm  applied_Δ={applied_mm:+6.1f}  "
                  f"recovered_Δ={rec_mean:+7.2f}mm  frac={ratio:5.2f}  motionPSNR={np.mean(mps):5.2f}")

    json.dump({"mode": args.mode, "steps": args.steps, "tv": args.tv, "rows": rows},
              open(os.path.join(OUT, f"toy_partition_{args.mode}.json"), "w"), indent=2)
    print("\n=== SUMMARY (recovered_frac: 1.0 = fully recovered the known shift) ===")
    for ax in ["z", "x"]:
        fr = [r["recovered_frac"] for r in rows if r["axis"] == ax]
        print(f"  axis={ax}: recovered_frac per shift {[r['recovered_frac'] for r in rows if r['axis']==ax]} "
              f"(mean {np.mean(fr):.2f})")
    print("Wrote", os.path.join(OUT, f"toy_partition_{args.mode}.json"))


if __name__ == "__main__":
    main()
