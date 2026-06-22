#!/usr/bin/env python
"""Reconstruct one OCMR real-time slice and quantify respiratory motion.

For us_*.h5 (real-time, undersampled). Zero-filled IFFT + SoS coil combine -> aliased
but adequate for tracking gross motion. Outputs:
  - cine.gif  (real-time slice playback)
  - trajectory.png  (centroid drift over time + frequency spectrum, with
    respiratory band 0.2-0.33 Hz and cardiac band 0.8-1.5 Hz shaded)

Centroid trajectory is the empirical breathing signal: a slow ~0.2-0.3 Hz oscillation
in the y (SI) direction = visible respiratory motion in the data.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_ocmr import read_ocmr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--out", default="result/ocmr_recon/motion_check")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"reading {args.path}")
    data, param = read_ocmr(args.path)
    print(f"  shape (kx,ky,kz,coil,phase,set,slice,rep,avg): {data.shape}")
    nx, ny, nz, nc, npha, nset, nsl, nrep, navg = data.shape
    sl = min(args.slice, nsl - 1)

    # collapse rep/avg/set; pick one slice; assume 2D (kz=1)
    k = data[:, :, 0, :, :, 0, sl, 0, 0]  # (kx,ky,coil,phase) complex
    print(f"  slice {sl} k: {k.shape}")

    # centered 2D IFFT over (kx,ky), per coil per phase
    img = np.fft.fftshift(
        np.fft.ifftn(np.fft.ifftshift(k, axes=(0, 1)), axes=(0, 1)),
        axes=(0, 1),
    )
    mag = np.sqrt((np.abs(img) ** 2).sum(axis=2))  # (kx, ky, phase) SoS
    # crop 2x readout oversampling
    mag = mag[nx // 4 : 3 * nx // 4, :, :]
    cine = np.transpose(mag, (2, 1, 0))  # (phase, y, x)
    T, H, W = cine.shape
    print(f"  cine: {cine.shape}")

    # temporal resolution from param.TRes (ms)
    try:
        dt = float(np.array(param.TRes).ravel()[0]) * 1e-3  # s
    except Exception:
        dt = 0.045
    print(f"  TRes = {dt*1e3:.1f} ms ; total time = {T*dt:.2f} s")

    # render cine GIF
    lo, hi = np.percentile(cine, [1, 99.5])
    norm = np.clip((cine - lo) / (hi - lo + 1e-9), 0, 1)
    frames = (norm * 255).astype(np.uint8)
    gif_p = os.path.join(args.out, "cine.gif")
    imageio.mimsave(gif_p, frames, duration=dt, loop=0)
    print(f"  wrote {gif_p}")

    # intensity-weighted centroid per frame (y = SI in most clinical views)
    y_grid = np.arange(H).reshape(H, 1)
    x_grid = np.arange(W).reshape(1, W)
    w = cine + 1e-9
    cy = (w * y_grid).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    cx = (w * x_grid).sum(axis=(1, 2)) / w.sum(axis=(1, 2))
    cy -= cy.mean()
    cx -= cx.mean()
    t = np.arange(T) * dt

    # FFT of detrended centroid
    freqs = np.fft.rfftfreq(T, dt)
    Py = np.abs(np.fft.rfft(cy))
    Px = np.abs(np.fft.rfft(cx))

    # also a static-region temporal-std (for visual confirmation)
    tstd = cine.std(axis=0)

    fig = plt.figure(figsize=(11, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cine.mean(axis=0), cmap="gray")
    ax1.set_title("temporal mean")
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(tstd, cmap="magma")
    ax2.set_title("temporal std (where motion is)")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(t, cy, label="y (mostly SI)", color="C0")
    ax3.plot(t, cx, label="x (mostly AP)", color="C1")
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("centroid drift (pixels)")
    ax3.set_title(f"Centroid trajectory (T={T} frames @ {dt*1e3:.0f} ms = {T*dt:.1f} s)")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(freqs, Py, label="y", color="C0")
    ax4.plot(freqs, Px, label="x", color="C1")
    ax4.axvspan(0.2, 0.33, alpha=0.15, color="blue", label="respiratory 0.2-0.33 Hz")
    ax4.axvspan(0.8, 1.5, alpha=0.15, color="red", label="cardiac 0.8-1.5 Hz")
    ax4.set_xlim(0, 3)
    ax4.set_xlabel("frequency (Hz)")
    ax4.set_ylabel("|FFT|")
    ax4.set_title("Spectrum of centroid drift")
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)

    fig.suptitle(os.path.basename(args.path), fontsize=10)
    fig.tight_layout()
    traj_p = os.path.join(args.out, "trajectory.png")
    fig.savefig(traj_p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {traj_p}")

    # numeric report
    resp_pwr = ((freqs >= 0.15) & (freqs <= 0.40)).astype(float) @ Py
    card_pwr = ((freqs >= 0.7) & (freqs <= 1.6)).astype(float) @ Py
    print(f"  y centroid: respiratory-band power = {resp_pwr:.3g}")
    print(f"  y centroid: cardiac-band     power = {card_pwr:.3g}")
    print(f"  y centroid: drift range = {cy.ptp():.2f} pixels")


if __name__ == "__main__":
    main()
