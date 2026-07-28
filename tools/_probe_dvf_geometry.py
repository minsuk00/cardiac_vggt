"""Brute-force the correct DVF application convention (empirical, no reasoning).

Enumerate {warp direction} x {global sign} x {per-axis component sign / transpose} and report
which one makes a warped frame best match its partner. The correct convention should give
residual MSE ratio << 1 (a validated elastix reg aligns frames well). "measure, don't reason."
"""
import os, glob, itertools
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

SUBJ = "scratch/data/CMRxRecon2024/Cine_combined/CMRx24_Val_P002/sax"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_subject(sax):
    ph = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
    T = len(ph)
    aff = nib.load(ph[0]).affine
    spacing = np.array([aff[0, 0], aff[1, 1], aff[2, 2]], np.float32)
    phases = np.stack([np.asarray(nib.load(p).dataobj).astype(np.float32) for p in ph], 0)
    v0 = phases[0]; nz = v0[v0 > 0]
    lo, hi = np.percentile(nz, 0.5), np.percentile(nz, 99.5)
    phases = np.clip((phases - lo) / (hi - lo + 1e-8), 0, 1)
    X, Y, Z = phases.shape[1:]
    dvf = np.zeros((T, X, Y, Z, 3), np.float32)
    for t in range(1, T):
        dvf[t] = np.asarray(nib.load(os.path.join(sax, "dvf_elastix", f"dvf_frame_{t:02d}.nii.gz")).dataobj)[..., 0, :]
    return phases, dvf, spacing


def warp(vol_xyz, disp_xyz):
    """warp[p] = vol(p + disp(p)); disp in index units, component order (dX,dY,dZ)."""
    X, Y, Z = vol_xyz.shape
    ii, jj, kk = torch.meshgrid(torch.arange(X, device=DEV, dtype=torch.float32),
                                torch.arange(Y, device=DEV, dtype=torch.float32),
                                torch.arange(Z, device=DEV, dtype=torch.float32), indexing="ij")
    src_i = ii + disp_xyz[..., 0]; src_j = jj + disp_xyz[..., 1]; src_k = kk + disp_xyz[..., 2]
    gx = src_k / (Z - 1) * 2 - 1; gy = src_j / (Y - 1) * 2 - 1; gz = src_i / (X - 1) * 2 - 1
    grid = torch.stack([gx, gy, gz], -1).unsqueeze(0)
    return F.grid_sample(vol_xyz.view(1, 1, X, Y, Z), grid, mode="bilinear",
                         padding_mode="border", align_corners=True).view(X, Y, Z)


def synthetic_selftest():
    """Validate grid_sample mechanics: a constant disp of +3 along X must reproduce a
    -3 index roll of the volume (gather at p+3 pulls content from +3)."""
    X, Y, Z = 40, 36, 12
    vol = torch.rand(X, Y, Z, device=DEV)
    disp = torch.zeros(X, Y, Z, 3, device=DEV); disp[..., 0] = 3.0
    w = warp(vol, disp)
    ref = torch.roll(vol, shifts=-3, dims=0)
    err = (w[5:-5, 5:-5, 2:-2] - ref[5:-5, 5:-5, 2:-2]).abs().mean().item()
    print(f"[selftest] warp(+3 X) vs roll(-3 X) interior MAE = {err:.2e}  (should be ~0)\n")


def main():
    synthetic_selftest()
    phases, dvf, spacing = load_subject(SUBJ)
    T = phases.shape[0]
    ph = torch.from_numpy(phases).to(DEV)
    d_mm = torch.from_numpy(dvf).to(DEV)
    spc = torch.from_numpy(spacing).to(DEV)
    # MOTION mask: voxels with real temporal change (this is where the DVF must help;
    # static tissue dilutes the ratio toward 1). matches compute_motion_mask (amax-amin>tau).
    mot = (ph.amax(0) - ph.amin(0)) > 0.05
    print(f"motion-mask voxels: {int(mot.sum())} ({100*mot.float().mean():.1f}% of volume)")
    f0 = ph[0]

    # candidate displacement builders (index units), component order variants
    def variants(dt):  # dt: (X,Y,Z,3)
        cands = {}
        for name, base in [("signed", dt / spc), ("abs", dt / spc.abs())]:
            cands[f"{name}"] = base
            cands[f"{name}_negXY"] = base * torch.tensor([-1, -1, 1.], device=DEV)
            cands[f"{name}_negZ"] = base * torch.tensor([1, 1, -1.], device=DEV)
            cands[f"{name}_neg"] = -base
        return cands

    # For each (direction, sign-variant) average residual ratio over t=1..T-1
    results = {}
    for direction in ["f0_to_ft", "ft_to_f0"]:
        for vname in ["signed", "signed_negXY", "signed_negZ", "signed_neg",
                      "abs", "abs_negXY", "abs_negZ", "abs_neg"]:
            ratios = []
            for t in range(1, T):
                dt = d_mm[t]
                cand = variants(dt)[vname]
                ft = ph[t]
                if direction == "f0_to_ft":
                    recon = warp(f0, cand); tgt = ft; base = f0
                else:
                    recon = warp(ft, cand); tgt = f0; base = ft
                m = mot  # score ONLY on the moving region
                nowarp = ((base - tgt)[m] ** 2).mean().item()
                w = ((recon - tgt)[m] ** 2).mean().item()
                ratios.append(w / (nowarp + 1e-9))
            results[(direction, vname)] = float(np.mean(ratios))

    print("mean residual ratio (warp_mse / nowarp_mse), lower=better; <<1 means correct:")
    for k in sorted(results, key=lambda x: results[x]):
        print(f"  {k[0]:>10}  {k[1]:>14}  ratio={results[k]:.3f}")
    best = min(results, key=lambda x: results[x])
    print(f"\nBEST: direction={best[0]}  variant={best[1]}  ratio={results[best]:.3f}")


if __name__ == "__main__":
    main()
