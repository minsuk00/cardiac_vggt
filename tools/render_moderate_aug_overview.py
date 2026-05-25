"""Single-PNG overview of a GPU-augmentation tier (default: aggressive).

Usage: PYTHONPATH=training:. python tools/render_moderate_aug_overview.py [tier]
       tier ∈ {conservative, moderate, aggressive}  (default: aggressive)

Top row    : each op in isolation (prob=1, tier magnitude; for each op we sample
             a few draws and show the most-visible one). Columns: original,
             H-flip, rotate, translate, scale, gaussian noise, gamma, bias field.
Bottom rows: 16 random draws of the FULL probabilistic pipeline (real
             fire-probabilities for that tier) — what training actually sees.
All panels : ED phase (t=0), mid-z slice of one real subject.
"""
import os, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import batchaug as B
B.set_backend("pytorch")
from data.preprocess import build_data_dicts, get_canonical_transforms
from data.gpu_aug import build_gpu_transforms

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SUBJECT = "Train_P053"
KEYS = ["phases", "content_mask"]
MODE = {"phases": "bilinear", "content_mask": "nearest"}
T_ED = 0

# (rot_deg, translate_vox, scale_frac, noise_std, gamma_lo, gamma_hi, bias_coeff)
TIERS = {
    "conservative": (5,  4.0,  0.05, 0.02, 0.8, 1.25, 0.2),
    "moderate":     (12, 8.0,  0.10, 0.03, 0.7, 1.4,  0.3),
    "aggressive":   (25, 16.0, 0.20, 0.05, 0.6, 1.7,  0.5),
}


def load_subject(name):
    d = build_data_dicts([os.path.join(DATA_ROOT, name, "sax")])[0]
    out = get_canonical_transforms()(d)
    phases = out["phases"].squeeze(1).permute(0, 3, 2, 1).contiguous().float()
    mask = out["content_mask"].squeeze(0).permute(2, 1, 0).contiguous().float()
    return phases, mask


def apply(op, phases, mask):
    d = {"phases": phases.clone().unsqueeze(0), "content_mask": mask.clone().unsqueeze(0).unsqueeze(0)}
    return op(d)["phases"][0]


def affine(rot=0.0, tr=(0., 0., 0.), sc=(0., 0., 0.)):
    return B.Compose(transforms=[B.RandAffined(
        keys=KEYS, prob=1.0, rotate_range=(float(np.deg2rad(rot)), 0., 0.),
        translate_range=tr, scale_range=sc, padding_mode="zeros")], lazy=True, mode=MODE)


def most_visible(op, phases, mask, mid, base, n=8):
    best, best_d = None, -1.0
    for _ in range(n):
        sl = apply(op, phases, mask)[T_ED, mid].numpy()
        d = np.abs(sl - base).mean()
        if d > best_d:
            best, best_d = sl, d
    return best


def main():
    tier = sys.argv[1] if len(sys.argv) > 1 else "aggressive"
    rot, tr, sc, nstd, glo, ghi, bias = TIERS[tier]
    out = Path(f"/home/minsukc/vggt/result/augmentation_examples/{tier}_overview.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    phases, mask = load_subject(SUBJECT)
    T, D, H, W = phases.shape
    mid = D // 2
    base = phases[T_ED, mid].numpy()

    indiv = [
        (f"H-flip",            B.RandFlipd(keys=KEYS, prob=1.0, spatial_axis=[2])),
        (f"rotate ±{rot}°",    affine(rot=rot)),
        (f"translate ±{tr:.0f}vox", affine(tr=(0., tr, tr))),
        (f"scale ±{int(sc*100)}%", affine(sc=(0., sc, sc))),
        (f"noise std {nstd}",  B.RandGaussianNoised(keys=["phases"], prob=1.0, std=(0.0, nstd))),
        (f"gamma {glo}–{ghi}", B.RandAdjustContrastd(keys=["phases"], prob=1.0, gamma=(glo, ghi))),
        (f"bias ±{bias}",      B.RandBiasFieldd(keys=["phases"], prob=1.0, degree=3, coeff_range=(-bias, bias))),
    ]
    pipe = build_gpu_transforms(OmegaConf.create({"enable": True, "tier": tier}))

    ncol = 9  # original + 7 isolated ops + combined(train-time)
    fig = plt.figure(figsize=(2.0 * ncol, 8.6), constrained_layout=True)
    fig.suptitle(f"{tier.upper()} augmentation tier — Train_P053, ED (t=0), mid-z", fontsize=14)
    sub = fig.subfigures(2, 1, height_ratios=[1.0, 2.0])
    sub[0].suptitle("INDIVIDUAL OPS (prob=1)    ·    rightmost = ALL augs combined, probabilistic (train-time)",
                    fontsize=10.5, color="#1c5b8c", weight="bold")
    sub[1].suptitle("FULL PROBABILISTIC PIPELINE — random draws (real fire-probs = exactly what training sees)",
                    fontsize=10.5, color="#1c5b8c", weight="bold")
    axA = sub[0].subplots(1, ncol)
    axB = sub[1].subplots(2, ncol)

    def show(ax, img, title):
        ax.imshow(img, cmap="gray", origin="lower", vmin=0, vmax=1)
        ax.set_title(title, fontsize=8); ax.axis("off")

    top = [("original", base)]
    top += [(name, most_visible(op, phases, mask, mid, base)) for name, op in indiv]
    top.append(("ALL (train-time)", apply(pipe, phases, mask)[T_ED, mid].numpy()))
    for i, (name, img) in enumerate(top):
        show(axA[i], img, name)

    for i in range(ncol * 2):
        sl = apply(pipe, phases, mask)[T_ED, mid].numpy()
        show(axB[i // ncol, i % ncol], sl, f"draw {i+1}  Δ={np.abs(sl - base).mean():.3f}")

    fig.savefig(out, dpi=115, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
