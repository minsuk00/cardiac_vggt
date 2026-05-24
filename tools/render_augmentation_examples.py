"""Visual sanity check for the batchaug GPU augmentation pipeline.

Loads ONE subject through the canonical preprocess pipeline (so the input is
a real (T=12, D=12, H=256, W=256) tensor) and renders:

  1. Per-op variant PNGs to `result/augmentation_examples/{op}.png`
     A 4×3 grid: rows = (original, variant 1, variant 2, variant 3); columns =
     mid-z slice of phase 0 (ED), mid-z slice of phase 6 (ES), XY MIP across Z.
     One PNG per op so each augmentation type can be eyeballed in isolation.

  2. Combined-pipeline cardiac-cycle GIF
     `result/augmentation_examples/combined_cycle.gif` — 12 frames showing
     ONE augmented variant playing through all 12 cardiac phases at mid-z.
     If the heart beats smoothly with no per-frame rotation jitter, the
     spatial aug is being applied consistently across T (which is what we want).

  3. Combined-pipeline still PNG
     `result/augmentation_examples/combined.png` — 4 variants × 3 columns,
     same layout as (1) but with the full conservative-tier pipeline applied.

Run:
    PYTHONPATH=training:. micromamba run -n svr python tools/render_augmentation_examples.py
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

import batchaug as B  # noqa: E402

B.set_backend("pytorch")

from data.preprocess import (  # noqa: E402
    build_data_dicts,
    get_canonical_transforms,
)

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
OUT_DIR = Path("/home/minsukc/vggt/result/augmentation_examples")
SUBJECT_NAME = "Train_P053"   # the val-subj-0 from our earlier smoke; has bbox z=[1, 10]
NUM_VARIANTS = 3              # number of augmented variants per op
T_ED = 0                      # end-diastole phase
T_ES = 6                      # population-median ES (per CLAUDE.md)


def load_subject(subj_name: str):
    """Run the canonical preprocess once and return a (T, D, H, W) splat-order tensor + mask."""
    sax_dir = os.path.join(DATA_ROOT, subj_name, "sax")
    data_dict = build_data_dicts([sax_dir])[0]
    out = get_canonical_transforms()(data_dict)
    # monai (X, Y, Z) → splat (D=Z, H=Y, W=X). Match mri_dataset.get_data.
    phases = out["phases"].squeeze(1).permute(0, 3, 2, 1).contiguous().float()  # (T, D, H, W)
    mask = out["content_mask"].squeeze(0).permute(2, 1, 0).contiguous().float()  # (D, H, W)
    return phases, mask


def to_batch5d(phases: torch.Tensor, mask: torch.Tensor):
    """(T, D, H, W) + (D, H, W) → batched dict {phases: (1, T, D, H, W), content_mask: (1, 1, D, H, W)}."""
    return {
        "phases": phases.unsqueeze(0),
        "content_mask": mask.unsqueeze(0).unsqueeze(0),
    }


def apply_op(phases: torch.Tensor, mask: torch.Tensor, op):
    """Run `op` (a batchaug transform or Compose) on a 5D dict; return augmented (T, D, H, W)."""
    aug_dict = to_batch5d(phases.clone(), mask.clone())
    aug_dict = op(aug_dict)
    p = aug_dict["phases"][0]                       # (T, D, H, W)
    m = aug_dict["content_mask"][0, 0]              # (D, H, W)
    return p, m


def render_op_panel(phases, mask, op, op_name: str, out_path: Path, num_variants: int = NUM_VARIANTS):
    """4-row × 3-col PNG: row 0 = original; rows 1..N = augmented variants.
       Columns: mid-d slice phase ED | mid-d slice phase ES | XY MIP across D."""
    T, D, H, W = phases.shape
    mid_d = D // 2

    fig, axes = plt.subplots(num_variants + 1, 3, figsize=(9, 3 * (num_variants + 1)))
    fig.suptitle(f"augmentation: {op_name}", fontsize=12)

    for row in range(num_variants + 1):
        if row == 0:
            p, m = phases, mask
            label = "original"
        else:
            p, m = apply_op(phases, mask, op)
            label = f"variant {row}"

        ed = p[T_ED, mid_d].cpu().numpy()
        es = p[T_ES, mid_d].cpu().numpy()
        mip = p[T_ED].cpu().numpy().max(axis=0)  # XY MIP across all 12 z planes for phase 0

        axes[row, 0].imshow(ed, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label} — ED (phase {T_ED}), z={mid_d}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(es, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"{label} — ES (phase {T_ES}), z={mid_d}", fontsize=9)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(mip, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 2].set_title(f"{label} — XY MIP across z", fontsize=9)
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def make_cardiac_cycle_gif(phases, mask, combined_op, out_path: Path, fps: int = 4):
    """One augmented variant of the full pipeline → 12-frame cardiac-cycle GIF at mid-z.
       Sanity check: the rotation/flip applied to phase 0 is the SAME as phase 11.
       If the heart appears to jitter rotation between phases, the aug is broken."""
    T, D, H, W = phases.shape
    mid_d = D // 2

    # One sampled variant — apply once to all 12 phases jointly.
    p_aug, _m_aug = apply_op(phases, mask, combined_op)

    frames = []
    for t in range(T):
        sl = p_aug[t, mid_d].cpu().numpy()
        # Composite side-by-side: original phase t | augmented phase t.
        orig_sl = phases[t, mid_d].cpu().numpy()
        composite = np.concatenate([orig_sl, np.full((H, 4), 0.5), sl], axis=1)  # 4-pixel gray gap
        composite = (np.clip(composite, 0, 1) * 255).astype(np.uint8)
        frames.append(Image.fromarray(composite, mode="L"))

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path,
        save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"  wrote {out_path}  ({T} frames @ {fps} fps)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"loading {SUBJECT_NAME} through canonical preprocess pipeline...")
    phases, mask = load_subject(SUBJECT_NAME)
    print(f"  phases: {tuple(phases.shape)}  mask: {tuple(mask.shape)}  "
          f"mask sum: {int(mask.sum().item())} / {mask.numel()}")

    # Per-op variants — each one with prob=1.0 so the variant is guaranteed to be aug'd
    # (otherwise prob=0.5 means half the "variants" are no-op and the panel is useless).
    keys = ["phases", "content_mask"]
    mode_dict = {"phases": "bilinear", "content_mask": "nearest"}
    single_ops = {
        "flip_W": B.RandFlipd(keys=keys, prob=1.0, spatial_axis=[2]),
        "rotate_5deg": B.Compose(
            transforms=[B.RandAffined(
                keys=keys, prob=1.0,
                rotate_range=(float(np.deg2rad(5)), 0.0, 0.0),  # in-plane (H-W) — see gpu_aug.py
                padding_mode="zeros",
            )],
            lazy=True, mode=mode_dict,
        ),
        "translate": B.Compose(
            transforms=[B.RandAffined(
                keys=keys, prob=1.0,
                translate_range=(0.0, 4.0, 4.0),
                padding_mode="zeros",
            )],
            lazy=True, mode=mode_dict,
        ),
        "scale": B.Compose(
            transforms=[B.RandAffined(
                keys=keys, prob=1.0,
                scale_range=(0.0, 0.05, 0.05),
                padding_mode="zeros",
            )],
            lazy=True, mode=mode_dict,
        ),
        "gaussian_noise": B.RandGaussianNoised(keys=["phases"], prob=1.0, std=(0.0, 0.02)),
        "gamma": B.RandAdjustContrastd(keys=["phases"], prob=1.0, gamma=(0.8, 1.25)),
        "bias_field": B.RandBiasFieldd(keys=["phases"], prob=1.0, degree=3, coeff_range=(-0.2, 0.2)),
    }

    print("\nrendering per-op variant panels...")
    for op_name, op in single_ops.items():
        out_path = OUT_DIR / f"{op_name}.png"
        render_op_panel(phases, mask, op, op_name, out_path)

    # Combined pipeline (conservative tier) — same as build_gpu_transforms with prob=1.0
    # on each op so every variant is fully aug'd.
    print("\nrendering combined-pipeline panel...")
    combined = B.Compose(
        transforms=[
            B.RandFlipd(keys=keys, prob=1.0, spatial_axis=[2]),
            B.RandAffined(
                keys=keys, prob=1.0,
                rotate_range=(float(np.deg2rad(5)), 0.0, 0.0),  # in-plane (H-W) — see gpu_aug.py
                translate_range=(0.0, 4.0, 4.0),
                scale_range=(0.0, 0.05, 0.05),
                padding_mode="zeros",
            ),
            B.RandGaussianNoised(keys=["phases"], prob=1.0, std=(0.0, 0.02)),
            B.RandAdjustContrastd(keys=["phases"], prob=1.0, gamma=(0.8, 1.25)),
            B.RandBiasFieldd(keys=["phases"], prob=1.0, degree=3, coeff_range=(-0.2, 0.2)),
        ],
        lazy=True, mode=mode_dict,
    )
    render_op_panel(phases, mask, combined, "combined", OUT_DIR / "combined.png")

    print("\nrendering combined-pipeline cardiac-cycle GIF...")
    make_cardiac_cycle_gif(phases, mask, combined, OUT_DIR / "combined_cycle.gif", fps=4)

    print("\ndone.")


if __name__ == "__main__":
    main()
