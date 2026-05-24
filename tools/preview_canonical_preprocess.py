"""Eyeball-check the canonical preprocess pipeline on a handful of subjects.

Picks subjects spanning the shape extremes (smallest Z, largest Z, smallest H,
largest H) plus one typical case. For each, renders a PNG to
`result/canonical_preview/{subj_id}.png` showing:

    row 1 (native):    mid-z slice from the loaded NIfTI (W × H pixels)
    row 2 (canonical): mid-z slice from the resampled (256, 256, 12) tensor
                        with the anatomy_bbox drawn as a red rectangle

Run:
    PYTHONPATH=training:. micromamba run -n svr python tools/preview_canonical_preprocess.py
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from data.preprocess import (  # noqa: E402
    TARGET_SHAPE,
    TARGET_SPACING,
    build_data_dicts,
    compute_geometric_bbox,
    get_canonical_transforms,
)

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
OUT_DIR = Path("/home/minsukc/vggt/result/canonical_preview")


def find_extreme_subjects(n_typical: int = 1):
    """Scan all subjects, pick representatives of shape extremes."""
    sax_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "sax")))
    if not sax_dirs:
        raise RuntimeError(f"No subjects found under {DATA_ROOT}")

    records = []  # (sax_dir, shape (W,H,Z), spacing)
    for d in sax_dirs:
        p = os.path.join(d, "3d_recon", "sax_frame_00.nii.gz")
        if not os.path.exists(p):
            continue
        h = nib.load(p).header
        shape = tuple(int(x) for x in h.get_data_shape()[:3])
        spacing = tuple(float(x) for x in h.get_zooms()[:3])
        records.append((d, shape, spacing))

    by_z = sorted(records, key=lambda r: r[1][2])
    by_h = sorted(records, key=lambda r: r[1][1])

    picks = {}
    picks["min_z"] = by_z[0]
    picks["max_z"] = by_z[-1]
    picks["min_h"] = by_h[0]
    picks["max_h"] = by_h[-1]
    # Typical: a record with the most common (H, Z) = (246, 11)
    typical = next((r for r in records if r[1][1] == 246 and r[1][2] == 11), records[len(records) // 2])
    picks["typical"] = typical

    selected = []
    seen = set()
    for tag, rec in picks.items():
        if rec[0] in seen:
            continue
        seen.add(rec[0])
        selected.append((tag, *rec))
    return selected


def render_one(tag: str, sax_dir: str, native_shape, native_spacing, transforms, out_path: Path):
    subj_id = os.path.basename(os.path.dirname(sax_dir))

    # Native phase 0
    p0_path = os.path.join(sax_dir, "3d_recon", "sax_frame_00.nii.gz")
    native_vol = nib.load(p0_path).get_fdata()  # (W, H, Z) in nibabel order
    native_mid_z = native_vol.shape[2] // 2
    native_slice = native_vol[:, :, native_mid_z]  # (W, H)

    # Run cached pipeline (no PersistentDataset — direct transform call, just to
    # render the preprocessed output).
    data_dict = build_data_dicts([sax_dir])[0]
    out = transforms(data_dict)
    phases = out["phases"]  # (T=12, 1, X=256, Y=256, Z=12) in monai order, float16
    content_mask = out["content_mask"]  # (1, X=256, Y=256, Z=12) uint8
    phases_splat = phases.squeeze(1).float().permute(0, 3, 2, 1).contiguous()  # → (T, D=12, H=256, W=256)
    mask_splat = content_mask.squeeze(0).permute(2, 1, 0).contiguous()  # → (D=12, H=256, W=256)
    canon_mid_d = phases_splat.shape[1] // 2
    canon_slice = phases_splat[0, canon_mid_d].cpu().numpy()  # (H=256, W=256) phase 0 mid-z
    mask_slice = mask_splat[canon_mid_d].cpu().numpy()  # (H=256, W=256) content mask mid-z

    bbox = compute_geometric_bbox(mask_splat).tolist()
    z0, z1, y0, y1, x0, x1 = bbox

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{subj_id}  ({tag})  native shape {native_shape}  spacing {tuple(round(s, 3) for s in native_spacing)}",
        fontsize=11,
    )

    axes[0].imshow(native_slice.T, cmap="gray", origin="lower")
    axes[0].set_title(f"native phase 0, mid-z slice (W×H = {native_slice.shape[0]}×{native_slice.shape[1]})")
    axes[0].axis("off")

    axes[1].imshow(canon_slice, cmap="gray", origin="lower")
    axes[1].set_title(
        f"canonical phase 0, mid-d=6 slice (256×256)\n"
        f"geometric bbox z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}]"
    )
    rect = mpatches.Rectangle(
        (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="red", facecolor="none", label="geometric bbox",
    )
    axes[1].add_patch(rect)
    axes[1].axis("off")

    axes[2].imshow(mask_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[2].set_title(
        f"content_mask, mid-d=6 slice (256×256)\n"
        f"white = native FOV reaches here, black = zero-pad"
    )
    axes[2].add_patch(mpatches.Rectangle(
        (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
        linewidth=1.5, edgecolor="red", facecolor="none",
    ))
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}  bbox={bbox}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Target: spacing={TARGET_SPACING}  shape={TARGET_SHAPE}")
    print(f"Output: {OUT_DIR}")
    selections = find_extreme_subjects()
    print(f"Picked {len(selections)} subjects:")
    for tag, sax_dir, shape, sp in selections:
        print(f"  {tag:>10s}  {os.path.basename(os.path.dirname(sax_dir)):>10s}  shape={shape}  spacing={tuple(round(x, 3) for x in sp)}")

    transforms = get_canonical_transforms()
    for tag, sax_dir, shape, sp in selections:
        subj_id = os.path.basename(os.path.dirname(sax_dir))
        out_path = OUT_DIR / f"{tag}_{subj_id}.png"
        render_one(tag, sax_dir, shape, sp, transforms, out_path)

    print("done.")


if __name__ == "__main__":
    main()
