"""Visualize the respiratory-motion simulation (training/data/respiratory.py).

Loads ONE subject's canonical phases and renders, using the SAME functions the
training augmentation uses (so the panels show exactly what training applies):

  (a) lujan_curve.png   — the d(r) waveform (SI + AP), end-expiration → inspiration.
  (b) reslice_sweep.png — single cardiac phase, coronal + sagittal reslices across
                          a respiratory-displacement sweep (heart slides along D).
  (c) axial_sweep.png   — single cardiac phase, a FIXED scanner plane across the
                          sweep (same plane images different anatomy = the point).
  (d) combined.png      — multi-phase (beating + breathing), axial at a fixed plane.

No model/checkpoint — this is a pure data-transform visualization.

Run:
    PYTHONPATH=training:. micromamba run -n svr python tools/render_respiratory_examples.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from data.preprocess import build_data_dicts, get_canonical_transforms  # noqa: E402
from data.respiratory import (  # noqa: E402
    RespiratoryConfig,
    lujan_displacement,
    reslice_volume,
)

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
OUT_DIR = Path("/home/minsukc/vggt/result/respiratory_examples")
SUBJECT = "Train_P053"          # known decent FOV (bbox z≈[1,10])
T_FIXED = 0                     # cardiac phase for the single-phase core demo (ED)
T_PAIR = (0, 6)                # phases for the combined beating+breathing panel
CFG = RespiratoryConfig(enable=True, amplitude_mm=12.0, ap_ratio=0.35, ap_axis="H", cos2n=2)
SWEEP_MM = [0.0, 3.0, 6.0, 9.0, 12.0]   # SI displacement sweep (rest → deep inspiration)


def load_subject(name: str):
    """Canonical preprocess → (T, D, H, W) splat-order phases + (D, H, W) mask."""
    sax_dir = os.path.join(DATA_ROOT, name, "sax")
    out = get_canonical_transforms()(build_data_dicts([sax_dir])[0])
    phases = out["phases"].squeeze(1).permute(0, 3, 2, 1).contiguous().float()  # (T,D,H,W)
    mask = out["content_mask"].squeeze(0).permute(2, 1, 0).contiguous().float()  # (D,H,W)
    return phases, mask


def heart_center(mask):
    """Centroid (z, y, x) of the content mask → indices through the anatomy."""
    idx = mask.nonzero(as_tuple=False).float().mean(0)
    return tuple(int(round(float(v))) for v in idx)


def title_for(d_mm):
    return f"d={d_mm:+.0f} mm ({d_mm / 8.0:+.2f} vox)"


# ── (a) Lujan waveform ────────────────────────────────────────────────────────
def render_lujan_curve(path):
    r = torch.linspace(0, 1, 400)
    d_si = lujan_displacement(r, CFG.amplitude_mm, n=CFG.cos2n)
    d_ap = CFG.ap_ratio * d_si
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    ax.plot(r.numpy(), d_si.numpy(), color="#1f77b4", lw=2.2, label=f"SI  (A={CFG.amplitude_mm:.0f} mm)")
    ax.plot(r.numpy(), d_ap.numpy(), color="#0a7d28", lw=2.0, ls="--",
            label=f"AP  (= {CFG.ap_ratio:.2f}·SI)")
    # mark the sweep displacements on the curve (solve r from d: sin(pi r)=(d/A)^(1/2n))
    for d in SWEEP_MM:
        frac = (d / CFG.amplitude_mm) ** (1.0 / (2 * CFG.cos2n))
        rm = float(np.arcsin(np.clip(frac, 0, 1)) / np.pi)
        ax.scatter([rm], [d], color="#1f77b4", zorder=5, s=28)
    ax.axvline(0.0, color="#888", ls=":", lw=1)
    ax.axvline(0.5, color="#888", ls=":", lw=1)
    ax.annotate("end-expiration\n(rest, r=0)", xy=(0.0, 0), xytext=(0.04, CFG.amplitude_mm * 0.55),
                fontsize=9, color="#555")
    ax.annotate("peak inspiration\n(r=0.5)", xy=(0.5, CFG.amplitude_mm), xytext=(0.54, CFG.amplitude_mm * 0.6),
                fontsize=9, color="#555")
    ax.set_xlabel("respiratory phase  r"); ax.set_ylabel("displacement from rest (mm)")
    ax.set_title(f"Lujan waveform  d(r) = A·sin²ⁿ(πr),  n={CFG.cos2n}  (dwells at end-expiration)", fontsize=11)
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def show(ax, img, vmax, title=None, aspect="equal"):
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=aspect, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


# ── (b) coronal + sagittal sweep, single cardiac phase ────────────────────────
def render_reslice_sweep(path, V, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5.2), dpi=110)
    for j, d in enumerate(SWEEP_MM):
        Vs = reslice_volume(V, d, CFG.ap_ratio * d, ap_axis=CFG.ap_axis).numpy()
        show(axes[0, j], Vs[:, cy, :], vmax, title_for(d), aspect="auto")   # coronal (D×W)
        show(axes[1, j], Vs[:, :, cx], vmax, aspect="auto")                 # sagittal (D×H)
    axes[0, 0].set_ylabel("coronal  (D×W)", fontsize=10)
    axes[1, 0].set_ylabel("sagittal (D×H)", fontsize=10)
    fig.suptitle(f"Respiratory sweep — single cardiac phase t={T_FIXED} (depth axis D is vertical)",
                 fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (c) axial at a FIXED plane across the sweep ───────────────────────────────
def render_axial_sweep(path, V, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.0), dpi=110)
    for j, d in enumerate(SWEEP_MM):
        Vs = reslice_volume(V, d, CFG.ap_ratio * d, ap_axis=CFG.ap_axis).numpy()
        show(axes[j], Vs[cz], vmax, title_for(d))
    fig.suptitle(f"Same scanner plane z={cz} across the sweep — content changes as the heart slides "
                 f"through-plane (t={T_FIXED})", fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (d) combined beating + breathing (multi-phase) ────────────────────────────
def render_combined(path, phases, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    fig, axes = plt.subplots(len(T_PAIR), n, figsize=(2.5 * n, 2.7 * len(T_PAIR)), dpi=110)
    for i, t in enumerate(T_PAIR):
        Vt = phases[t]
        for j, d in enumerate(SWEEP_MM):
            Vs = reslice_volume(Vt, d, CFG.ap_ratio * d, ap_axis=CFG.ap_axis).numpy()
            show(axes[i, j], Vs[cz], vmax, title_for(d) if i == 0 else None)
        axes[i, 0].set_ylabel(f"cardiac t={t}", fontsize=10)
    fig.suptitle(f"Combined: cardiac phase (rows) × respiratory displacement (cols), axial z={cz}",
                 fontsize=10, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading subject {SUBJECT} …")
    phases, mask = load_subject(SUBJECT)
    center = heart_center(mask)
    V = phases[T_FIXED]
    vmax = float(V.max())
    print(f"  phases {tuple(phases.shape)}  heart center (z,y,x)={center}  vmax={vmax:.4f}")

    render_lujan_curve(OUT_DIR / "lujan_curve.png")
    render_reslice_sweep(OUT_DIR / "reslice_sweep.png", V, center, vmax)
    render_axial_sweep(OUT_DIR / "axial_sweep.png", V, center, vmax)
    render_combined(OUT_DIR / "combined.png", phases, center, vmax)
    for p in ["lujan_curve", "reslice_sweep", "axial_sweep", "combined"]:
        print(f"  saved {OUT_DIR / (p + '.png')}")


if __name__ == "__main__":
    main()
