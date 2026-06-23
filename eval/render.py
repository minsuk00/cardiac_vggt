"""Dataset-agnostic render helpers for RTFB inference (moved verbatim from the original
tools/eval_ocmr_inference.py). Beating-heart GIF, per-z volume sheet, input contact sheet,
and predicted-DVF panel. No ground truth needed.
"""
import os

import numpy as np
from PIL import Image

from eval.adapters.base import MM_PER_NORM


def save_dvf_png(world_points, coords, picks, path, t=0):
    """Per-slot predicted displacement Δ = world_points - scanner_coords, in mm
    (Δx/Δy/Δz rows), masked to anatomy. Mirrors training's _DVF figure. No GT needed."""
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
        mask = up > 0.03
        axes[0][s].imshow(up, cmap="gray")
        axes[0][s].set_title(f"z{z_canon} s{slice_idx} f{f}", fontsize=7)
        for c in range(3):
            dm = delta[s, ..., c] * MM_PER_NORM[c]
            vlim = float(np.percentile(np.abs(dm[mask]), 99)) if mask.any() else 1.0
            vlim = max(vlim, 1e-3)
            disp = np.where(mask, dm, np.nan)
            im = axes[c + 1][s].imshow(disp, cmap="bwr", vmin=-vlim, vmax=vlim)
            if s == 0:
                axes[c + 1][s].set_ylabel(labels[c], fontsize=9)
            if s == S - 1:
                fig.colorbar(im, ax=axes[c + 1][s], fraction=0.046, pad=0.02)
    axes[0][0].set_ylabel("input slice", fontsize=9)
    fig.suptitle(f"{os.path.basename(path)}  predicted DVF (target t={t}, anatomy-masked)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90); plt.close(fig)


def _u8(a, vmax):
    return np.clip(a / vmax * 255.0, 0, 255).astype(np.uint8)


def save_cycle_gif(vols, path, mid_d=None):
    """Mid-z V_canon across the 12 target phases -> beating-heart GIF."""
    if mid_d is None:
        mid_d = vols[0].shape[0] // 2
    vmax = max(float(v[mid_d].max()) for v in vols) or 1e-3
    frames = [Image.fromarray(_u8(v[mid_d], vmax)) for v in vols]
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
