"""Visual gate for `tools/convert_to_sax_layout.py` (docs/58 A3).

Renders the converted ED frame (`sax_frame_00.nii.gz`) for a few subjects per source,
rows = subjects, cols = ALL z-planes base->apex, with **CMRx rows first as the orientation
reference**. What to look for:

  * an LV donut (bright ring of myocardium around the dark blood pool) in the mid slices
  * the RV crescent on the SAME side of the LV in every row -> chirality is consistent
  * plane order base -> apex consistent across sources

Every z-plane is shown, not just the mid one: the mid plane is the easiest and least
informative, and orientation/roll problems show up at the base and apex.

Usage:
    python tools/render_converted_sax.py                 # 4 subjects per source
    python tools/render_converted_sax.py --per-source 8
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
OUTDIR = os.path.join(ROOT, "result/converted_sax_check")

SOURCES = [
    ("CMRx24 (reference)", os.path.join(DATA, "CMRxRecon2024/Cine_combined/*/sax")),
    ("ACDC",               os.path.join(DATA, "ACDC_sax/*/sax")),
    ("M&Ms",               os.path.join(DATA, "MNMs_sax/*/sax")),
]


def load_ed(sax_dir):
    """ED frame as (Z, Y, X) — permuted to match the splat/display order used everywhere else."""
    p = os.path.join(sax_dir, "3d_recon", "sax_frame_00.nii.gz")
    im = nib.load(p)
    v = np.asarray(im.dataobj).astype(np.float32)          # (X, Y, Z)
    zooms = [float(z) for z in im.header.get_zooms()[:3]]
    lo, hi = np.percentile(v[v > 0], [0.5, 99.5]) if (v > 0).any() else (0.0, 1.0)
    v = np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1)
    return v.transpose(2, 1, 0), zooms                     # (Z, Y, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    rows = []
    for label, pattern in SOURCES:
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"  (skip {label}: nothing at {pattern})")
            continue
        # Spread the picks across the cohort rather than taking the first N.
        step = max(1, len(dirs) // args.per_source)
        for d in dirs[::step][: args.per_source]:
            try:
                vol, zooms = load_ed(d)
            except Exception as e:
                print(f"  FAILED {d}: {e!r}")
                continue
            sid = os.path.basename(os.path.dirname(d))
            rows.append((f"{label}\n{sid}\nD={vol.shape[0]} dz={zooms[2]:.1f}mm", vol))

    if not rows:
        print("nothing to render")
        return

    ncol = max(v.shape[0] for _, v in rows)
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(1.5 * ncol, 1.7 * len(rows)), squeeze=False)
    for r, (lab, vol) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < vol.shape[0]:
                ax.imshow(vol[c], cmap="gray", vmin=0, vmax=1, aspect="equal")
                if r == 0:
                    ax.set_title(f"z{c}", fontsize=7)
            else:
                ax.axis("off")
        axes[r][0].set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
    fig.suptitle("Converted SAX, ED frame, all z-planes — CMRx rows are the orientation reference",
                 fontsize=10)
    fig.tight_layout(rect=[0.06, 0, 1, 0.97])
    out = os.path.join(OUTDIR, "converted_sax_panel.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({len(rows)} rows x {ncol} planes)")


if __name__ == "__main__":
    main()
