"""Visualize the aggressive-tier acquisition-artifact post-ops (docs/63 §5).

Figures on a real CMRx SAX subject (mid-ventricular slice, phase 0):
  figs/aggressive_aug_perop.png    — each new op in isolation at 3 magnitudes
  figs/aggressive_aug_20draws.png  — 20 random draws of the FULL aggressive
                                     pipeline (flip+affine+photometric Compose
                                     + zoom/low-res/Gibbs/ghosting post-ops)
  figs/aggressive_aug_minmax.png   — EVERY aug in the pipeline at its weakest
                                     and strongest reachable setting
  figs/aggressive_aug_10draws.png  — 10 random full-pipeline draws

All panels are displayed with per-image MINMAX scaling (vmin=min, vmax=max),
not percentile windowing, so the artifact amplitude is visible as-is.

Run from repo root: PYTHONPATH=training:. python tools/render_aggressive_aug_examples.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "training")
sys.path.insert(0, ".")

import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import batchaug as _B
_B.set_backend("pytorch")

from omegaconf import OmegaConf
from data.gpu_aug import build_gpu_transforms, _apply_ghosting

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NIFTI = "/home/minsukc/vggt/data/CMRxRecon2024/Cine_combined/CMRx24_Test_P001/sax/4d_recon.nii.gz"
OUT_PEROP = "/home/minsukc/vggt/figs/aggressive_aug_perop.png"
OUT_DRAWS = "/home/minsukc/vggt/figs/aggressive_aug_20draws.png"
OUT_MINMAX = "/home/minsukc/vggt/figs/aggressive_aug_minmax.png"
OUT_10DRAWS = "/home/minsukc/vggt/figs/aggressive_aug_10draws.png"
SEED = 0

KEYS = ["phases", "content_mask"]
MODE = {"phases": "bilinear", "content_mask": "nearest"}


def load_phases():
    """(X, Y, Z, T) NIfTI → (1, T, D, H, W) float in [0, 1] on DEVICE."""
    img = nib.load(NIFTI).get_fdata().astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    phases = torch.from_numpy(img).permute(3, 2, 1, 0)  # (T, D=Z, H=Y, W=X)
    return phases.unsqueeze(0).to(DEVICE)


def show(ax, img2d, title=""):
    """Per-image MINMAX display — no percentile windowing."""
    a = img2d.detach().cpu().float().numpy()
    ax.imshow(a, cmap="gray", vmin=a.min(), vmax=a.max())
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _affine2d(x, angle_deg=0.0, tx=0.0, ty=0.0, sh=1.0, sw=1.0):
    """Deterministic in-plane affine on (1, T, D, H, W) — for the minmax panel
    (batchaug samples its params randomly, so extremes are applied directly)."""
    import math
    _, T, D, H, W = x.shape
    a = math.radians(angle_deg)
    # inverse-mapping (grid_sample) convention, translations in normalized units
    theta = torch.tensor([[math.cos(a) / sw, -math.sin(a) / sw, -2.0 * tx / W],
                          [math.sin(a) / sh,  math.cos(a) / sh, -2.0 * ty / H]],
                         device=x.device, dtype=torch.float32).unsqueeze(0)
    flat = x.reshape(1, T * D, H, W)
    grid = torch.nn.functional.affine_grid(theta, (1, T * D, H, W), align_corners=False)
    out = torch.nn.functional.grid_sample(flat, grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=False)
    return out.reshape(1, T, D, H, W)


def render_minmax(phases, mid, resh, back):
    """Every aug in the aggressive pipeline at its weakest / strongest setting."""
    rows = [
        ("flip W  (p=.5)",
         phases, torch.flip(phases, dims=[-1])),
        ("rotate ±180°  (p=.9)",
         _affine2d(phases, angle_deg=5), _affine2d(phases, angle_deg=180)),
        ("translate ±32 px  (p=.9)",
         _affine2d(phases, tx=3, ty=3), _affine2d(phases, tx=32, ty=32)),
        ("gamma 0.6–1.5  (p=.75)",
         phases.clamp(0, 1) ** 0.6, phases.clamp(0, 1) ** 1.5),
        ("bias field ±0.4  (p=.7)", None, None),  # filled below (random field, pinned coeff)
        ("zoom 0.8–1.2  (p=.5)", None, None),
        ("low-res 0.5–0.85  (p=.4)", None, None),
        ("gibbs α 0.5–0.75  (p=.3)", None, None),
        ("ghost n2-5 i.15-.4  (p=.3)",
         _apply_ghosting(phases, 5, 0.15, "W"), _apply_ghosting(phases, 2, 0.4, "W")),
    ]
    bias_lo = _B.RandBiasFieldd(keys=["phases"], prob=1.0, degree=3, coeff_range=(0.1, 0.1))
    bias_hi = _B.RandBiasFieldd(keys=["phases"], prob=1.0, degree=3, coeff_range=(0.4, 0.4))
    rows[4] = (rows[4][0], bias_lo({"phases": phases.clone()})["phases"],
               bias_hi({"phases": phases.clone()})["phases"])
    zo = lambda z: back(_B.RandZoomd(keys=["phases"], prob=1.0, min_zoom=z, max_zoom=z,
                                     mode={"phases": "bilinear"},
                                     padding_mode={"phases": "zeros"})(
        {"phases": resh(phases)})["phases"], phases.shape)
    rows[5] = (rows[5][0], zo(1.2), zo(0.8))
    lr = lambda f: back(_B.RandSimulateLowResolutiond(
        keys=["phases"], prob=1.0, zoom_range=(f, f), downsample_mode="nearest",
        upsample_mode="trilinear", align_corners=True)({"phases": resh(phases)})["phases"],
        phases.shape)
    rows[6] = (rows[6][0], lr(0.85), lr(0.5))
    gb = lambda a: back(_B.RandGibbsNoised(keys=["phases"], prob=1.0, alpha=(a, a))(
        {"phases": resh(phases)})["phases"], phases.shape)
    rows[7] = (rows[7][0], gb(0.5), gb(0.75))

    fig, axes = plt.subplots(len(rows), 3, figsize=(9.5, 3.1 * len(rows)))
    for r, (name, lo, hi) in enumerate(rows):
        show(axes[r, 0], lo[0, 0, mid], f"{name} — MIN")
        show(axes[r, 1], phases[0, 0, mid], "original")
        show(axes[r, 2], hi[0, 0, mid], f"{name} — MAX")
    fig.suptitle("Aggressive pipeline: weakest vs strongest reachable setting per aug "
                 "(minmax display; fire-prob in label)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_MINMAX, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_MINMAX}")


def render_draws(phases, mask, mid, agg, n_draws, out_path):
    ncol = 5
    nrow = (n_draws + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.1 * nrow))
    for i in range(n_draws):
        d = {"phases": phases.clone().float(), "content_mask": mask.clone()}
        d = agg(d)
        d = agg.vggt_post_ops(d)
        show(axes[i // ncol, i % ncol], d["phases"][0, 0, mid].clamp(0, 1), f"draw {i}")
    fig.suptitle(f"{n_draws} random draws — full aggressive pipeline "
                 "(flip+affine+gamma+bias + zoom/low-res/Gibbs/ghosting), minmax display",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    torch.manual_seed(SEED)
    phases = load_phases()
    _, T, D, H, W = phases.shape
    mid = D // 2
    mask = (phases[:, 0:1] > 0).float()  # (1, 1, D, H, W)
    resh = lambda v: v.reshape(v.shape[0], -1, *v.shape[3:]).unsqueeze(-1)
    back = lambda v, s: v.squeeze(-1).reshape(s)

    # ── Figure 1: each op in isolation, 3 magnitudes ─────────────────────────
    rows = []
    for z in (0.85, 1.0, 1.15):
        tr = _B.RandZoomd(keys=["phases"], prob=1.0, min_zoom=z, max_zoom=z,
                          mode={"phases": "bilinear"},
                          padding_mode={"phases": "zeros"})
        out = back(tr({"phases": resh(phases)})["phases"], phases.shape)
        rows.append((f"zoom {z:.2f}", out))
    for f in (0.85, 0.65, 0.5):
        tr = _B.RandSimulateLowResolutiond(keys=["phases"], prob=1.0, zoom_range=(f, f),
                                           downsample_mode="nearest",
                                           upsample_mode="trilinear", align_corners=True)
        out = back(tr({"phases": resh(phases)})["phases"], phases.shape)
        rows.append((f"low-res {f:.2f}", out))
    for a in (0.5, 0.65, 0.75):
        tr = _B.RandGibbsNoised(keys=["phases"], prob=1.0, alpha=(a, a))
        out = back(tr({"phases": resh(phases)})["phases"], phases.shape)
        rows.append((f"gibbs α={a:.2f}", out))
    for n, i in ((2, 0.4), (3, 0.3), (5, 0.15)):
        out = _apply_ghosting(phases, n, i, "W")
        rows.append((f"ghost n={n} i={i:.2f}", out))

    ncol = 3
    fig, axes = plt.subplots(5, ncol, figsize=(3 * ncol, 15))
    show(axes[0, 1], phases[0, 0, mid], "ORIGINAL (phase 0, mid slice)")
    axes[0, 0].axis("off"); axes[0, 2].axis("off")
    for i, (name, vol) in enumerate(rows):
        r, c = 1 + i // ncol, i % ncol
        show(axes[r, c], vol[0, 0, mid], name)
    fig.suptitle("Aggressive-tier post-ops in isolation (minmax display)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PEROP, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PEROP}")

    # ── Figure 2/4: random draws of the FULL aggressive pipeline ─────────────
    agg = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": "aggressive"}))
    render_draws(phases, mask, mid, agg, 20, OUT_DRAWS)
    render_draws(phases, mask, mid, agg, 10, OUT_10DRAWS)

    # ── Figure 3: min/max extremes of every aug in the pipeline ──────────────
    render_minmax(phases, mid, resh, back)


if __name__ == "__main__":
    main()
