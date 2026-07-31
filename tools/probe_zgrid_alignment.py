#!/usr/bin/env python3
"""Provenance for docs/58 — why the canonical z grid must follow native slice spacing.

Runs four probes and prints the numbers quoted in the doc:

  E1  identity-splat PSNR when input slices land ON the output grid's planes vs OFF them.
      This is the measurement that kills "native slices into a fixed 12 mm target".
  E2  identity-splat PSNR under `continuous_z` jitter (the same failure, self-inflicted).
  E3  identity-splat PSNR with the proposed design: in-plane resampled to 1.4 mm, z left
      at native spacing, D = native slice count.
  E4  ZIndexEmbedder feature aliasing for |z_norm| > 1 (sets Z_HALF_MM).

"Identity splat" = splat the input slices back with Delta = 0 on a single static frame.
A perfect model can do no better, so it is a hard ceiling, not a baseline.

Usage:
    PYTHONPATH=training:. micromamba run -n svr python tools/probe_zgrid_alignment.py
"""
from __future__ import annotations

import math

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ResizeWithPadOrCropd,
    Spacingd,
)

from vggt.utils.splat import splat_to_volume

CMRX = ("scratch/data/CMRxRecon2024/Cine_combined/CMRx24_Test_P001/sax"
        "/3d_recon/sax_frame_00.nii.gz")
ACDC = "scratch/data/ACDC/training/patient001/patient001_frame01.nii.gz"


def _load(path, dz_out, d_out):
    """(256, 256, Z) volume at (1.4, 1.4, dz_out). dz_out <= 0 keeps the native z spacing;
    d_out None keeps the resulting Z."""
    tf = [
        LoadImaged(keys=["im"], image_only=True),
        EnsureChannelFirstd(keys=["im"]),
        Orientationd(keys=["im"], axcodes="LPS"),
        Spacingd(keys=["im"], pixdim=(1.4, 1.4, dz_out), mode="bilinear"),
        ResizeWithPadOrCropd(keys=["im"], spatial_size=(256, 256, d_out if d_out else -1),
                             mode="constant", value=0),
    ]
    v = Compose(tf)({"im": path})["im"]
    return v.as_tensor().squeeze(0).float()          # (X, Y, Z)


def _norm_splat_order(v):
    """Percentile-normalize to [0, 1] and permute monai (X, Y, Z) -> splat (D, H, W)."""
    nz = v[v != 0]
    lo, hi = torch.quantile(nz, 0.005).item(), torch.quantile(nz, 0.999).item()
    return ((v - lo) / max(hi - lo, 1e-8)).clamp(0, 1).permute(2, 1, 0).contiguous()


def _identity_psnr(gt, planes, znorms):
    """Splat `planes` at normalized depths `znorms` into gt's grid with Delta = 0."""
    D, H, W = gt.shape
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xn = (xx.float() / (W - 1)) * 2 - 1
    yn = (yy.float() / (H - 1)) * 2 - 1
    pos, inten = [], []
    for sl, zn in zip(planes, znorms):
        pos.append(torch.stack([xn, yn, torch.full_like(xn, zn)], -1).reshape(-1, 3))
        inten.append(sl.reshape(-1))
    pos = torch.cat(pos)[None]
    inten = torch.cat(inten)[None]
    V, cov = splat_to_volume(pos, inten, (D, H, W), weight=(inten > 1e-3).float())
    m = (cov[0] > 0.5) & (gt > 0)
    mse = ((V[0] - gt) ** 2)[m].mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def e1_fixed_grid_alignment():
    print("\nE1  fixed 12 mm target grid: inputs ON grid vs native/OFF grid")
    for name, path, dz_native in [("CMRx24 P001", CMRX, 12.0), ("ACDC p001", ACDC, 10.0)]:
        gt = _norm_splat_order(_load(path, 12.0, 12))              # canonical 12-plane cube
        nat = _norm_splat_order(_load(path, 0.0, None))            # native-z volume
        D = gt.shape[0]
        half_mm = (D - 1) * 12.0 / 2

        on = _identity_psnr(gt, [gt[k] for k in range(D)],
                            [k / (D - 1) * 2 - 1 for k in range(D)])
        Zn = nat.shape[0]
        off = _identity_psnr(gt, [nat[j] for j in range(Zn)],
                             [((j - (Zn - 1) / 2) * dz_native) / half_mm for j in range(Zn)])
        print(f"  {name:14s} native {dz_native:>4.1f} mm x {Zn:2d}   "
              f"on-grid {on:6.2f} dB   native/off-grid {off:6.2f} dB")


def e2_continuous_z():
    print("\nE2  continuous_z jitter on the fixed 12 mm grid (CMRx24 P001)")
    gt = _norm_splat_order(_load(CMRX, 12.0, 12))
    D = gt.shape[0]
    base = list(range(D))

    def run(zs):
        z0 = [int(np.floor(min(max(z, 0), D - 1 - 1e-3))) for z in zs]
        fr = [min(max(z, 0), D - 1 - 1e-3) - a for z, a in zip(zs, z0)]
        planes = [(1 - f) * gt[a] + f * gt[min(a + 1, D - 1)] for a, f in zip(z0, fr)]
        return _identity_psnr(gt, planes, [z / (D - 1) * 2 - 1 for z in zs])

    print(f"  jitter 0.0 (continuous_z OFF)        {run(base):6.2f} dB")
    for j in (0.25, 0.5):
        rng = np.random.default_rng(0)
        zs = [min(max(k + rng.uniform(-j, j), 0), D - 1 - 1e-3) for k in base]
        print(f"  jitter +-{j} plane (= +-{j*12:.0f} mm)      {run(zs):6.2f} dB")


def e3_native_grid():
    print("\nE3  proposed design: z NEVER resampled, D = native slice count")
    for name, path in [("CMRx24 P001", CMRX), ("ACDC p001", ACDC),
                       ("ACDC p002", ACDC.replace("patient001", "patient002"))]:
        pre = Compose([
            LoadImaged(keys=["im"], image_only=True),
            EnsureChannelFirstd(keys=["im"]),
            Orientationd(keys=["im"], axcodes="LPS"),
            Spacingd(keys=["im"], pixdim=(1.4, 1.4, 0.0), mode="bilinear"),
        ])({"im": path})
        dz = float(pre["im"].pixdim[2])
        v = ResizeWithPadOrCropd(keys=["im"], spatial_size=(256, 256, -1),
                                 mode="constant", value=0)(pre)["im"]
        gt = _norm_splat_order(v.as_tensor().squeeze(0).float())
        D = gt.shape[0]
        psnr = _identity_psnr(gt, [gt[k] for k in range(D)],
                              [k / (D - 1) * 2 - 1 for k in range(D)])
        print(f"  {name:14s} dz={dz:6.3f} mm (native)  D={D:2d}  "
              f"span={(D-1)*dz:6.1f} mm   identity {psnr:6.2f} dB")


def e4_embedder_aliasing():
    print("\nE4  ZIndexEmbedder input features (aggregator.py:33-42), num_freqs=3")

    def feats(z):
        z = torch.as_tensor([z], dtype=torch.float64)
        f = [z]
        for i in range(3):
            f += [torch.sin((2 ** i) * math.pi * z), torch.cos((2 ** i) * math.pi * z)]
        return torch.cat(f)

    for a, b in [(1.167, -0.833), (1.05, -0.95), (0.933, -0.933)]:
        fa, fb = feats(a), feats(b)
        print(f"  z={a:+.3f} vs z={b:+.3f}   max|d sinusoidal| = "
              f"{(fa[1:] - fb[1:]).abs().max().item():.2e}   "
              f"|d linear| = {abs(fa[0] - fb[0]).item():.3f}")
    print("  sinusoids have period 2 -> any pair 2.0 apart collides on 6 of 7 channels.")
    print("  within-subject z_norm range = span / Z_HALF_MM; tallest stack span = 168 mm")
    print("    Z_HALF_MM=72 -> 2.33 (aliases)   Z_HALF_MM=90 -> 1.87 (cannot alias)")


if __name__ == "__main__":
    e1_fixed_grid_alignment()
    e2_continuous_z()
    e3_native_grid()
    e4_embedder_aliasing()
    print()
