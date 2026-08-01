#!/usr/bin/env python3
"""Render the 6 byte-identical duplicate subject pairs found by docs/59 F3.

For each pair: mid-slice of frame_00 for both members, plus their absolute difference,
annotated with split membership and the (self-contradicting) manifest metadata.

Usage:
    micromamba run -n svr python tools/render_duplicate_pairs.py
Output: result/duplicate_pairs/duplicate_pairs.png
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "result", "duplicate_pairs")

PAIRS = [
    ("ACDC_patient055", "ACDC_patient118"),
    ("MNMs_A7G0P5", "MNMs_K3R0Y7"),
    ("MNMs_C8J7L5", "MNMs_C8O0P2"),
    ("ACDC_patient074", "ACDC_patient076"),
    ("MNMs_A8C9H8", "MNMs_Q0Q1Y4"),
    ("MNMs_C5Q2Y5", "MNMs_E9L4N2"),
]

SOURCE_DIR = {"ACDC": "ACDC_sax", "MNMs": "MNMs_sax"}


def load_manifest():
    path = os.path.join(REPO, "training", "splits", "manifest.csv")
    with open(path) as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def load_frame0(subj_id, meta):
    root = SOURCE_DIR[subj_id.split("_")[0]]
    path = os.path.join(REPO, "scratch", "data", root, subj_id, "sax", "3d_recon", "sax_frame_00.nii.gz")
    return np.asarray(nib.load(path).dataobj, dtype=np.float32)


def norm(a):
    lo, hi = np.percentile(a, [0.5, 99.5])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    man = load_manifest()

    fig, axes = plt.subplots(len(PAIRS), 3, figsize=(9.5, 3.1 * len(PAIRS)))
    for row, (a_id, b_id) in enumerate(PAIRS):
        va = load_frame0(a_id, man)
        vb = load_frame0(b_id, man)
        z = va.shape[2] // 2
        sa, sb = va[:, :, z], vb[:, :, z]
        diff = np.abs(sa - sb)
        maxdiff = float(np.abs(va - vb).max()) if va.shape == vb.shape else float("nan")

        for col, (img, title) in enumerate((
            (norm(sa), a_id), (norm(sb), b_id), (diff, f"|diff|  max(vol)={maxdiff:g}"),
        )):
            ax = axes[row, col]
            ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1 if col < 2 else max(diff.max(), 1e-6))
            ax.set_xticks([]); ax.set_yticks([])
            if col < 2:
                m = man.get(title, {})
                sub = (f"split={m.get('split','?')}  {m.get('vendor','?')}/centre {m.get('centre','?')}  "
                       f"sex={m.get('sex','?')} age={m.get('age','?')}")
                ax.set_title(f"{title}\n{sub}", fontsize=7.5)
            else:
                ax.set_title(title, fontsize=7.5)
        axes[row, 0].set_ylabel(f"pair {row + 1}", fontsize=9)

    fig.suptitle("docs/59 F3 — byte-identical duplicate subjects (frame_00, mid slice)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(OUT_DIR, "duplicate_pairs.png")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
