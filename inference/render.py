"""Dataset-agnostic render helpers for RTFB inference (moved verbatim from the original
tools/eval_ocmr_inference.py). Beating-heart GIF, per-z volume sheet, input contact sheet,
and predicted-DVF panel. No ground truth needed.
"""
import os

import numpy as np
from PIL import Image

from inference.adapters.base import MM_PER_NORM


def save_dvf_png(world_points, coords, picks, path, t=0):
    """Per-slot predicted displacement Δ = world_points - scanner_coords, in mm (Δx/Δy/Δz rows),
    full field with a faint input overlay for context. Mirrors training's _DVF figure. No GT."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    delta = (world_points - coords)                    # (S,518,518,3) normalized
    S = delta.shape[0]
    labels = ["Δx (mm)", "Δy (mm)", "Δz (mm)"]
    fig, axes = plt.subplots(4, S, figsize=(1.7 * S, 7.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for s in range(S):
        z_canon, slice_idx, f, up = picks[s]
        axes[0][s].imshow(up, cmap="gray")
        axes[0][s].set_title(f"z{z_canon} s{slice_idx} f{f}", fontsize=7)
        for c in range(3):
            dm = delta[s, ..., c] * MM_PER_NORM[c]
            # Full Δ field, NO intensity mask (the old up>0.03 mask erased low-signal anatomy);
            # vlim from the 99th pct over the whole slice + a faint input overlay for context.
            vlim = max(float(np.percentile(np.abs(dm), 99)), 1e-3)
            im = axes[c + 1][s].imshow(dm, cmap="bwr", vmin=-vlim, vmax=vlim)
            axes[c + 1][s].imshow(up, cmap="gray", alpha=0.15)   # faint anatomy overlay
            if s == 0:
                axes[c + 1][s].set_ylabel(labels[c], fontsize=9)
            if s == S - 1:
                fig.colorbar(im, ax=axes[c + 1][s], fraction=0.046, pad=0.02)
    axes[0][0].set_ylabel("input slice", fontsize=9)
    fig.suptitle(f"{os.path.basename(path)}  predicted DVF (target t={t}, full field)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_cycle_gif(vols, path, planes=None, n_slices=5):
    """1×n pred strip — n z-planes spanning the reconstructed content — animated over the swept
    frames (beating heart). Pred only (OOD: no GT). `planes` overrides the auto-picked z indices;
    otherwise pick n planes evenly spanning the planes that carry signal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    stack = np.stack(vols)                                   # (F, D, H, W)
    D = stack.shape[1]
    if planes is None:
        e = stack.max(0).reshape(D, -1).max(1)              # per-plane peak intensity over frames
        nz = np.where(e > 0.05 * (float(e.max()) or 1e-3))[0]
        z0, z1 = (int(nz.min()), int(nz.max())) if len(nz) else (0, D - 1)
        planes = np.unique(np.clip(np.linspace(z0, z1, n_slices).round().astype(int), 0, D - 1))
    vmax = max(float(stack[:, planes].max()), 1e-3)
    frames = []
    for v in vols:
        fig, axes = plt.subplots(1, len(planes), figsize=(1.6 * len(planes), 2.0), squeeze=False)
        for c, z in enumerate(planes):
            ax = axes[0][c]
            ax.imshow(v[z], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"z={z}", fontsize=8)
        fig.tight_layout()
        buf = BytesIO(); fig.savefig(buf, format="png", dpi=90); buf.seek(0)
        frames.append(Image.open(buf).convert("RGB")); plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=200, loop=0)


def save_inputs_png(picks, path):
    """Contact sheet of the S input slices actually fed (orientation/quality sanity)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(picks)
    cols = min(n, 6); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (z_canon, slice_idx, f, up) in enumerate(picks):
        ax = axes[i // cols][i % cols]
        ax.imshow(up, cmap="gray"); ax.axis("off")
        ax.set_title(f"z{z_canon} s{slice_idx} f{f}", fontsize=7)
    fig.suptitle(os.path.basename(path), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def save_volume_png(vols, path, t=0):
    """All 12 canonical z-planes of V_canon at one target phase (volume coverage check)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = vols[t]; D = V.shape[0]; vmax = float(V.max()) or 1e-3
    fig, axes = plt.subplots(2, (D + 1) // 2, figsize=(1.6 * ((D + 1) // 2), 3.4), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for z in range(D):
        ax = axes[z // ((D + 1) // 2)][z % ((D + 1) // 2)]
        ax.imshow(V[z], cmap="gray", vmin=0, vmax=vmax); ax.axis("off")
        ax.set_title(f"z={z}", fontsize=7)
    fig.suptitle(f"{os.path.basename(path)}  (target t={t})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)
