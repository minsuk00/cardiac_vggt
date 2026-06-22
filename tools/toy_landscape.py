"""KEYSTONE (E0): 1-D loss-landscape sweep — NO optimizer, NO network.

Take one subject's single-phase volume V0. Inputs = V0's own planes at ALIGNED scanner
positions (so a zero offset is a perfect reconstruction). Then sweep a KNOWN rigid offset δ
along through-plane z vs in-plane x and record L1(splat, V0) + motion-PSNR as a function of δ.

This measures the OBJECTIVE's gradient landscape directly, immune to optimizer/LR/steps confounds:
 - If the z-loss basin is FLAT/WIDE (loss barely rises per mm of misalignment) while the x-basin
   is SHARP → the splat+L1 objective is ill-conditioned through-plane → ∂L/∂Δz is tiny → any
   optimizer/network will under-correct z. OBJECTIVE/RENDERER-limited.
 - If z and x basins are comparably sharp → the objective is fine → under-correction is
   network/info-limited.

Confound guards: interior-z only (avoid zero-pad info loss); fp32 (splat forces it); metric on
motion-mask voxels; also runs a coverage-division ABLATION (raw accumulate, no /coverage) and an
INVERSE-WARP renderer (sample_volume) to test whether the splat coverage-division is the culprit.

Run: micromamba run -n svr python tools/toy_landscape.py
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools")); sys.path.insert(0, os.path.join(REPO, "training")); sys.path.insert(0, REPO)
from eval_variants_matrix import build_dataset, build_batch, GRID_SHAPE, NUM_SLICES
from loss import compute_volume_intensity_loss, compute_motion_mask
from vggt.utils.splat import splat_to_volume, sample_volume

D, H, W = GRID_SHAPE
MM_Z = (D - 1) / 2.0 * 8.0      # 44.0 mm per norm unit (through-plane)
MM_X = (W - 1) / 2.0 * 1.4      # 178.5 mm per norm unit (in-plane)
OUT = os.path.join(REPO, "result", "limits_eval")


def psnr_mask(a, b, m):
    a, b = a[m], b[m]
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse < 1e-12 else 10.0 * np.log10(1.0 / mse)


def get_sample(ds, seq, device):
    data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
    batch = build_batch(data, device, seq_index=seq)
    out = compute_volume_intensity_loss({"world_points": batch["scanner_coords"]}, batch,
                                        grid_shape=GRID_SHAPE, tv_weight=0.0)
    V0 = out["V_gt"][0].float()                          # single-phase target (D,H,W)
    mmask = compute_motion_mask(batch["phases"])[0]      # (D,H,W) bool
    bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
    return V0, mmask, bbox


def build_aligned(V0, z_planes, res=256, device="cuda"):
    """Inputs = V0's own planes at z_planes, scanner aligned (offset 0 → perfect)."""
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, res, device=device),
                            torch.linspace(-1, 1, res, device=device), indexing="ij")
    base = torch.stack([xs, ys], dim=-1)
    S = len(z_planes)
    inten = torch.stack([V0[z] for z in z_planes])       # (S,res,res)
    scanner = torch.zeros((S, res, res, 3), device=device)
    for i, z in enumerate(z_planes):
        scanner[i, :, :, :2] = base
        scanner[i, :, :, 2] = z / (D - 1) * 2 - 1
    return scanner, inten


def render(scanner, inten, offset_norm, axis, mode="splat"):
    """Return reconstructed volume (D,H,W) with a rigid offset added to scanner along axis."""
    S = scanner.shape[0]
    world = scanner.clone()
    world[..., axis] = world[..., axis] + offset_norm
    int_flat = inten.reshape(S, -1)
    w = (int_flat > 1e-3).float()
    pos = world.reshape(1, -1, 3)
    if mode == "splat":
        V, _ = splat_to_volume(pos, int_flat.reshape(1, -1), (D, H, W), weight=w.reshape(1, -1))
        return V[0]
    elif mode == "splat_nocovdiv":
        # raw accumulate (no /coverage): replicate splat numerator only
        Vc, cov = splat_to_volume(pos, int_flat.reshape(1, -1), (D, H, W), weight=w.reshape(1, -1))
        return (Vc * (cov + 1e-6))[0]    # undo the division → raw accumulated intensity
    elif mode == "invwarp":
        # inverse warp: for each output voxel, sample the *input stack* treated as a volume.
        # Build a (D,H,W) volume by nearest-z placement of the input planes, then sample it
        # shifted by -offset (so a +offset of content is undone). This is a coverage-free
        # backward renderer for the same rigid-shift recovery.
        vol = torch.zeros((1, D, H, W), device=scanner.device)
        for i in range(S):
            z = int(round((scanner[i, 0, 0, 2].item() + 1) / 2 * (D - 1)))
            vol[0, z] = inten[i]
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, H, device=scanner.device),
                                torch.linspace(-1, 1, W, device=scanner.device), indexing="ij")
        zs = torch.linspace(-1, 1, D, device=scanner.device)
        gz, gy, gx = torch.meshgrid(zs, ys[:, 0], xs[0], indexing="ij")
        samp_pos = torch.stack([gx, gy, gz], dim=-1).reshape(1, -1, 3).clone()
        samp_pos[..., axis] = samp_pos[..., axis] + offset_norm   # sample shifted
        return sample_volume(vol, samp_pos).reshape(D, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--npts", type=int, default=33)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"
    ds = build_dataset()
    seqs = [int(s) for s in args.seqs.split(",")]
    offs_mm = np.linspace(-24, 24, args.npts)

    curves = {}   # (axis, mode) -> list of L1 curves (per subject)
    for axis_name, axis, mm_per in [("z", 2, MM_Z), ("x", 0, MM_X)]:
        for mode in ["splat", "splat_nocovdiv", "invwarp"]:
            curves[(axis_name, mode)] = []

    for seq in seqs:
        V0, mmask, bbox = get_sample(ds, seq, device)
        if not mmask.any():
            continue
        z0, z1 = bbox[0], bbox[1]
        z_planes = list(range(max(z0 + 1, 0), min(z1 - 1, D)))   # INTERIOR z only (dodge zero-pad)
        if len(z_planes) < 3:
            z_planes = list(range(z0, z1))
        scanner, inten = build_aligned(V0, z_planes, device=device)
        V0np = V0.cpu().numpy(); mnp = mmask.cpu().numpy()
        for axis_name, axis, mm_per in [("z", 2, MM_Z), ("x", 0, MM_X)]:
            for mode in ["splat", "splat_nocovdiv", "invwarp"]:
                l1c = []
                for mm in offs_mm:
                    with torch.no_grad():
                        V = render(scanner, inten, float(mm) / mm_per, axis, mode=mode).cpu().numpy()
                    l1c.append(float(np.abs(V[mnp] - V0np[mnp]).mean()))
                curves[(axis_name, mode)].append(l1c)

    # average curves
    def mean_curve(k):
        a = np.array(curves[k]); return a.mean(0) if len(a) else np.full(len(offs_mm), np.nan)

    # quantify: loss rise per mm near 0 (slope of |L1| at small offset) + half-width
    def stats(curve):
        c = np.array(curve); i0 = len(c) // 2
        # average |ΔL1| per mm over ±8 mm
        m8 = np.abs(offs_mm) <= 8
        rise = (c[m8] - c[i0])
        slope = float(np.mean(rise[offs_mm[m8] != 0] / np.abs(offs_mm[m8][offs_mm[m8] != 0])))
        return dict(min=float(c[i0]), at8mm=float(np.interp(8, offs_mm, c)),
                    at16mm=float(np.interp(16, offs_mm, c)), rise_per_mm=slope)

    summary = {}
    for axis_name in ["z", "x"]:
        for mode in ["splat", "splat_nocovdiv", "invwarp"]:
            summary[f"{axis_name}_{mode}"] = stats(mean_curve((axis_name, mode)))

    # plot the main splat landscape z vs x
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cz = mean_curve(("z", "splat")); cx = mean_curve(("x", "splat"))
    ax[0].plot(offs_mm, cz, "o-", color="#c5221f", label="through-plane z (12 planes, 8mm)")
    ax[0].plot(offs_mm, cx, "s-", color="#1a73e8", label="in-plane x (256, 1.4mm)")
    ax[0].set_xlabel("rigid misalignment offset δ (mm)"); ax[0].set_ylabel("L1 loss on motion voxels")
    ax[0].set_title("E0 landscape: loss vs known misalignment\n(flatter = weaker gradient = under-correction)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    # normalized to compare shape
    ax[1].plot(offs_mm, cz / cz.max(), "o-", color="#c5221f", label="z splat")
    ax[1].plot(offs_mm, cx / cx.max(), "s-", color="#1a73e8", label="x splat")
    ax[1].plot(offs_mm, mean_curve(("z", "invwarp")) / max(mean_curve(("z", "invwarp")).max(), 1e-9),
               "^--", color="#e8710a", label="z inverse-warp")
    ax[1].set_xlabel("offset δ (mm)"); ax[1].set_ylabel("L1 (normalized)")
    ax[1].set_title("Normalized basin shape (z vs x vs z-invwarp)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig_e0_landscape.png"), bbox_inches="tight", dpi=115, facecolor="white")
    plt.close(fig)

    json.dump({"offs_mm": offs_mm.tolist(),
               "curves": {f"{a}_{m}": mean_curve((a, m)).tolist() for a in ["z", "x"]
                          for m in ["splat", "splat_nocovdiv", "invwarp"]},
               "summary": summary, "n": len(seqs)},
              open(os.path.join(OUT, "toy_landscape.json"), "w"), indent=2)

    print(f"=== E0 LANDSCAPE (n={len(seqs)} subjects, motion-voxel L1 vs rigid misalignment) ===")
    print(f"{'axis_mode':22s} {'L1@0':>8s} {'L1@8mm':>9s} {'L1@16mm':>9s} {'rise/mm':>9s}")
    for k, s in summary.items():
        print(f"{k:22s} {s['min']:8.4f} {s['at8mm']:9.4f} {s['at16mm']:9.4f} {s['rise_per_mm']:9.5f}")
    zr = summary["z_splat"]["rise_per_mm"]; xr = summary["x_splat"]["rise_per_mm"]
    print(f"\nKEY: in-plane x rises {xr/max(zr,1e-9):.1f}x faster per mm than through-plane z (splat).")
    print(f"  → z gradient is ~{xr/max(zr,1e-9):.0f}x weaker ⇒ objective is ill-conditioned through-plane.")
    iz = summary["z_invwarp"]["rise_per_mm"]
    print(f"  inverse-warp z rise/mm = {iz:.5f} (vs splat z {zr:.5f}) — does a backward renderer sharpen z?")
    print("Wrote", os.path.join(OUT, "toy_landscape.json"), "+ fig_e0_landscape.png")


if __name__ == "__main__":
    main()
