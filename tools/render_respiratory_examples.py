"""Visualize the respiratory-motion simulation (training/data/respiratory.py).

Loads ONE subject's canonical phases and renders, using the SAME functions the
training augmentation uses (so the panels show exactly what training applies):

  lujan_curve.png   — d(r) waveform: n-comparison (single cycle) + multi-cycle trace.
  reslice_sweep.png — single cardiac phase, coronal + sagittal reslices across a
                      displacement sweep, each with a difference-vs-rest row.
  axial_sweep.png   — single cardiac phase, a FIXED scanner plane across the sweep
                      + difference-vs-rest (same plane → different anatomy).
  zmontage.png      — through-plane montage: depth planes (cols) × displacement (rows).
  input_view.png    — the actual 518² model input for one (t,z) slot across the sweep.
  combined.png      — multi-phase (beating + breathing), axial at a fixed plane.

All grayscale panels use nearest-neighbour display (no smoothing) so the 12-plane
(8 mm) through-plane sampling is shown honestly, not smeared.

No model/checkpoint — a pure data-transform visualization.

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

from data.gpu_aug import extract_slices_from_phases  # noqa: E402
from data.preprocess import build_data_dicts, get_canonical_transforms  # noqa: E402
from data.respiratory import (  # noqa: E402
    RespiratoryConfig,
    extract_slices_with_respiratory,
    extract_slices_with_respiratory_vec,
    lujan_displacement,
    reslice_volume,
    sample_resp_disp,
)

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
OUT_DIR = Path("/home/minsukc/vggt/result/respiratory_examples")
SUBJECT = "Train_P053"          # known decent FOV (bbox z≈[1,10])
T_FIXED = 0                     # cardiac phase for the single-phase core demo (ED)
T_PAIR = (0, 6)                # phases for the combined beating+breathing panel
CFG = RespiratoryConfig(enable=True, amplitude_mm=24.0, ap_ratio=0.35, ap_axis="H", cos2n=3)
SWEEP_MM = [0.0, 6.0, 12.0, 18.0, 24.0]  # SI displacement sweep (rest → deep inspiration ~24mm)
RESP_PERIOD_S = 5.0             # breathing cycle (for the multi-cycle time trace)


def load_subject(name: str):
    """Canonical preprocess → (T, D, H, W) splat-order phases + (D, H, W) mask."""
    sax_dir = os.path.join(DATA_ROOT, name, "sax")
    out = get_canonical_transforms()(build_data_dicts([sax_dir])[0])
    phases = out["phases"].squeeze(1).permute(0, 3, 2, 1).contiguous().float()  # (T,D,H,W)
    mask = out["content_mask"].squeeze(0).permute(2, 1, 0).contiguous().float()  # (D,H,W)
    return phases, mask


def heart_center(mask):
    idx = mask.nonzero(as_tuple=False).float().mean(0)
    return tuple(int(round(float(v))) for v in idx)


def title_for(d_mm):
    return f"d={d_mm:+.0f}mm ({d_mm / 8.0:+.2f} vox)"


def shifted(V, d):
    return reslice_volume(V, d, CFG.ap_ratio * d, ap_axis=CFG.ap_axis).numpy()


def show(ax, img, vmax, title=None, aspect="equal"):
    ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=aspect,
              origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def show_diff(ax, img, vlim, title=None, aspect="equal"):
    im = ax.imshow(img, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect=aspect,
                   origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


# ── (1) Lujan waveform: n-comparison + multi-cycle trace ──────────────────────
def render_lujan_curve(path):
    n = CFG.cos2n
    A = CFG.amplitude_mm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.4), dpi=110)

    # (left) single cycle: SI + AP
    r = torch.linspace(0, 1, 400)
    d_si = lujan_displacement(r, A, n=n)
    d_ap = CFG.ap_ratio * d_si
    ax1.plot(r.numpy(), d_si.numpy(), color="#1f77b4", lw=2.4, label=f"SI  (A={A:.0f} mm)")
    ax1.plot(r.numpy(), d_ap.numpy(), color="#0a7d28", lw=2.0, ls="--",
             label=f"AP  (= {CFG.ap_ratio:.2f}·SI)")
    for d in SWEEP_MM:
        frac = (d / A) ** (1.0 / (2 * n))
        rm = float(np.arcsin(np.clip(frac, 0, 1)) / np.pi)
        ax1.scatter([rm], [d], color="#1f77b4", zorder=5, s=26)
    ax1.axvline(0.0, color="#bbb", ls=":", lw=1); ax1.axvline(0.5, color="#bbb", ls=":", lw=1)
    ax1.annotate("end-expiration\n(rest, r=0)", xy=(0.02, A * 0.5), fontsize=8.5, color="#555")
    ax1.annotate("peak\ninspiration", xy=(0.52, A * 0.2), fontsize=8.5, color="#555")
    ax1.set_xlabel("respiratory phase  r"); ax1.set_ylabel("displacement from rest (mm)")
    ax1.set_title(f"d(r) = A·sin²ⁿ(πr),  n={n},  A={A:.0f} mm  (dwells at end-expiration)")
    ax1.legend(loc="upper right"); ax1.grid(alpha=0.25)

    # (right) over 3 breaths: SI + AP
    t = torch.linspace(0, 3 * RESP_PERIOD_S, 1200)
    r_t = (t / RESP_PERIOD_S) % 1.0
    d_si_t = lujan_displacement(r_t, A, n=n)
    d_ap_t = CFG.ap_ratio * d_si_t
    ax2.plot(t.numpy(), d_si_t.numpy(), color="#1f77b4", lw=2.0, label="SI")
    ax2.plot(t.numpy(), d_ap_t.numpy(), color="#0a7d28", lw=2.0, ls="--", label="AP")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("displacement from rest (mm)")
    ax2.set_title(f"Over 3 breaths (period {RESP_PERIOD_S:.0f} s),  n={n}")
    ax2.legend(loc="upper right"); ax2.grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (2) coronal + sagittal sweep, with difference-vs-rest rows ────────────────
def render_reslice_sweep(path, V, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    ref = shifted(V, 0.0)
    vlim = 0.6 * vmax
    fig, axes = plt.subplots(4, n, figsize=(2.5 * n, 9.2), dpi=110)
    for j, d in enumerate(SWEEP_MM):
        Vs = shifted(V, d)
        show(axes[0, j], Vs[:, cy, :], vmax, title_for(d), aspect="auto")            # coronal
        show_diff(axes[1, j], Vs[:, cy, :] - ref[:, cy, :], vlim, aspect="auto")     # coronal Δ
        show(axes[2, j], Vs[:, :, cx], vmax, aspect="auto")                          # sagittal
        show_diff(axes[3, j], Vs[:, :, cx] - ref[:, :, cx], vlim, aspect="auto")     # sagittal Δ
    for r, lab in zip(range(4), ["coronal (D×W)", "Δ vs rest", "sagittal (D×H)", "Δ vs rest"]):
        axes[r, 0].set_ylabel(lab, fontsize=10)
    fig.suptitle(f"Respiratory sweep — single cardiac phase t={T_FIXED}. Depth axis D vertical "
                 f"(12 planes @ 8mm, nearest-interp). Δ = shifted − rest.", fontsize=11, y=1.0)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (3) axial at a FIXED plane across the sweep + difference ───────────────────
def render_axial_sweep(path, V, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    ref = shifted(V, 0.0)
    vlim = 0.6 * vmax
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n + 0.7, 5.2), dpi=110, constrained_layout=True)
    diff_im = None
    for j, d in enumerate(SWEEP_MM):
        Vs = shifted(V, d)
        show(axes[0, j], Vs[cz], vmax, title_for(d))
        diff_im = show_diff(axes[1, j], Vs[cz] - ref[cz], vlim)
    axes[0, 0].set_ylabel(f"axial z={cz}", fontsize=10)
    axes[1, 0].set_ylabel("Δ vs rest", fontsize=10)
    cbar = fig.colorbar(diff_im, ax=list(axes[1, :]), fraction=0.046, pad=0.02)
    cbar.set_label("residual (shifted − rest), norm. intensity", fontsize=9)
    fig.suptitle(f"Same scanner plane z={cz} across the sweep (t={T_FIXED}) — the fixed plane images "
                 f"DIFFERENT anatomy as the heart slides through-plane.", fontsize=10, y=1.02)
    fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (4) through-plane montage: depth planes × displacement ────────────────────
def render_zmontage(path, V, center, vmax):
    cz, cy, cx = center
    zs = [z for z in range(cz - 3, cz + 4) if 0 <= z < V.shape[0]]
    rows = [0.0, 6.0, 12.0]
    fig, axes = plt.subplots(len(rows), len(zs), figsize=(1.7 * len(zs), 1.9 * len(rows)), dpi=110)
    for i, d in enumerate(rows):
        Vs = shifted(V, d)
        for j, z in enumerate(zs):
            show(axes[i, j], Vs[z], vmax, f"z={z}" if i == 0 else None)
        axes[i, 0].set_ylabel(title_for(d), fontsize=9)
    fig.suptitle(f"Through-plane montage (t={T_FIXED}): depth plane (cols) × displacement (rows). "
                 f"At a larger shift each fixed plane shows deeper anatomy.", fontsize=10, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (5) the actual 518² model input for one (t,z) slot across the sweep ───────
def render_input_view(path, phases, center):
    cz, cy, cx = center
    S = len(SWEEP_MM)
    d_si = torch.tensor([[float(d) for d in SWEEP_MM]])
    d_ap = CFG.ap_ratio * d_si
    t_seq = torch.full((1, S), T_FIXED, dtype=torch.int64)
    z_seq = torch.full((1, S), cz, dtype=torch.int64)
    imgs = extract_slices_with_respiratory(
        phases.unsqueeze(0), t_seq, z_seq, d_si, d_ap, ap_axis=CFG.ap_axis,
    )[0, :, :, :, 0].numpy()                       # (S, 518, 518) in [0,255]
    ref = imgs[0]
    vlim = 0.6 * 255.0
    fig, axes = plt.subplots(2, S, figsize=(2.5 * S, 5.2), dpi=110)
    for j in range(S):
        show(axes[0, j], imgs[j], 255.0, title_for(SWEEP_MM[j]))
        show_diff(axes[1, j], imgs[j] - ref, vlim)
    axes[0, 0].set_ylabel("model input (518²)", fontsize=10)
    axes[1, 0].set_ylabel("Δ vs rest", fontsize=10)
    fig.suptitle(f"What the model sees: the upsampled input slice for one slot (t={T_FIXED}, z={cz}) "
                 f"across the sweep — the SAME geometric plane, shifted anatomy.", fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (5b) ACTUAL training input: scattered slots, per-slot iid breathing draw ──
def render_training_input(path, phases, mask, seed=7):
    """The real thing the model trains on: S scattered (t,z) slots, each with an
    INDEPENDENT respiratory draw (per-slot iid) AND randomized SI direction — sampled
    by the same `sample_resp_disp` the trainer uses (val/deterministic branch here so
    the figure is reproducible). Rows: reference (no breathing) | breathing input
    (what the model sees) | Δ."""
    T, D, H, W = phases.shape
    znz = mask.nonzero(as_tuple=False)[:, 0]
    z0, z1 = int(znz.min()), int(znz.max()) + 1
    S = 6
    zs = np.linspace(z0, z1 - 1, S).round().astype(int)
    ts = [(2 * i) % T for i in range(S)]                 # varied cardiac phases
    t_seq = torch.tensor([ts], dtype=torch.int64)
    z_seq = torch.tensor([zs.tolist()], dtype=torch.int64)

    ref = extract_slices_from_phases(phases.unsqueeze(0), t_seq, z_seq)[0, ..., 0].numpy()
    disp = sample_resp_disp(1, S, CFG, "cpu", train=False, seq_index=torch.tensor([[seed]]))
    brt = extract_slices_with_respiratory_vec(
        phases.unsqueeze(0), t_seq, z_seq, disp)[0, ..., 0].numpy()
    dd = disp[0].numpy()                                  # (S, 3) canonical mm (D,H,W)

    vlim = 0.6 * 255.0
    fig, axes = plt.subplots(3, S, figsize=(2.5 * S, 7.7), dpi=110)
    for j in range(S):
        show(axes[0, j], ref[j], 255.0, f"t={ts[j]}, z={int(zs[j])}")
        mag = float(np.linalg.norm(dd[j]))
        show(axes[1, j], brt[j], 255.0,
             f"|d|={mag:.0f}mm  D/H/W={dd[j, 0]:+.0f}/{dd[j, 1]:+.0f}/{dd[j, 2]:+.0f}")
        show_diff(axes[2, j], brt[j] - ref[j], vlim)
    axes[0, 0].set_ylabel("reference input\n(no breathing)", fontsize=10)
    axes[1, 0].set_ylabel("breathing input\n(what the model sees)", fontsize=10)
    axes[2, 0].set_ylabel("Δ (breathing − ref)", fontsize=10)
    fig.suptitle("Actual training input — 6 scattered slots, each an INDEPENDENT per-slot "
                 "breathing draw with randomized SI direction (D/H/W mm per slot). "
                 "Some slots land near exhale (small |d|), some near inspiration.",
                 fontsize=10, y=1.01)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ── (6) combined beating + breathing (multi-phase) ────────────────────────────
def render_combined(path, phases, center, vmax):
    cz, cy, cx = center
    n = len(SWEEP_MM)
    fig, axes = plt.subplots(len(T_PAIR), n, figsize=(2.5 * n, 2.7 * len(T_PAIR)), dpi=110)
    for i, t in enumerate(T_PAIR):
        Vt = phases[t]
        for j, d in enumerate(SWEEP_MM):
            show(axes[i, j], shifted(Vt, d)[cz], vmax, title_for(d) if i == 0 else None)
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
    render_zmontage(OUT_DIR / "zmontage.png", V, center, vmax)
    render_input_view(OUT_DIR / "input_view.png", phases, center)
    render_training_input(OUT_DIR / "training_input.png", phases, mask)
    render_combined(OUT_DIR / "combined.png", phases, center, vmax)
    for p in ["lujan_curve", "reslice_sweep", "axial_sweep", "zmontage",
              "input_view", "training_input", "combined"]:
        print(f"  saved {OUT_DIR / (p + '.png')}")


if __name__ == "__main__":
    main()
