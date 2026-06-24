"""Per-subject PNG: input slices (left) | red line | newseed V_refined (right).

For each subject, places the model INPUT slices into a (12,256,256) canonical
volume by their z-index (sparse — empty planes are black), and pairs it with the
saved newseed V_refined volume. Renders a 3x4 input grid and a 3x4 output grid
side by side, separated by a vertical red line (so "3x8 with a line in middle").
Inputs are rebuilt deterministically (same rng/seq) so they match the saved output.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from tools.five_row_compare import (
    DEV, VAL_SEQS, OCMR_SUBJECTS, GOTT_SUBJECTS, MIITT_SUBJECTS, MIITT_RECON,
    val_batch, ocmr_batch, goettingen_batch, miitt_batch,
    build_val_dataset, build_gott_dataset,
)

NIFTI_DIR = os.path.join(_ROOT, "result", "4way_refiner", "nifti_newseed_refined")
OUT = os.path.join(_ROOT, "result", "4way_refiner", "refined_io_slices")
D = 12


def input_volume(batch):
    """Place each input slice into a (12,256,256) canonical volume by z-index."""
    imgs = batch["images"][0]                       # (S,3,518,518) in [0,1]
    z = batch["z_indices"][0, :, 0].float().cpu().numpy()
    V = np.zeros((D, 256, 256), np.float32)
    for s in range(imgs.shape[0]):
        zi = int(round((z[s] + 1) / 2 * (D - 1)))
        if not (0 <= zi <= D - 1):
            continue
        sl = imgs[s, 0].float().cpu()               # (518,518)
        sl256 = F.interpolate(sl[None, None], size=(256, 256),
                              mode="bilinear", align_corners=True)[0, 0].numpy()
        V[zi] = sl256
    return V


def window_pct(V):
    nz = V[V > 0]
    ref = nz if nz.size else V
    hi = float(np.percentile(ref, 99.5))
    lo = float(np.percentile(ref, 1.0))
    return np.clip((V - lo) / (hi - lo + 1e-9), 0, 1)


def render_io(Vin, Vout, title, path):
    Vin_w, Vout_w = window_pct(Vin), window_pct(Vout)
    fig = plt.figure(figsize=(8 * 2.6 + 0.5, 3 * 2.6))
    gs = gridspec.GridSpec(3, 9, figure=fig,
                           width_ratios=[1, 1, 1, 1, 0.12, 1, 1, 1, 1],
                           wspace=0.05, hspace=0.12)
    panels = [(Vin_w, 0, "in"), (Vout_w, 5, "out")]
    for Vw, c0, tag in panels:
        for k in range(D):
            r, c = k // 4, c0 + (k % 4)
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(Vw[k], cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{tag} z={k}", fontsize=8)
            ax.axis("off")
    # red separator column
    sep = fig.add_subplot(gs[:, 4])
    sep.set_xlim(0, 1); sep.set_ylim(0, 1)
    sep.axvline(0.5, color="red", lw=3)
    sep.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}", flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    val_ds, rcfg = build_val_dataset()
    gott_ds = build_gott_dataset()

    jobs = []
    for i in VAL_SEQS:
        jobs.append(("val_ON", f"seq{i}", lambda i=i: val_batch(val_ds, rcfg, i, True)))
    for i in VAL_SEQS:
        jobs.append(("val_OFF", f"seq{i}", lambda i=i: val_batch(val_ds, rcfg, i, False)))
    for sub in OCMR_SUBJECTS:
        sd = os.path.join(_ROOT, "scratch/data/ocmr/recon", sub)
        jobs.append(("OCMR", sub, lambda sd=sd: ocmr_batch(sd)))
    for sub in GOTT_SUBJECTS:
        jobs.append(("Goett", sub, lambda sub=sub: goettingen_batch(gott_ds, sub)))
    for sub in MIITT_SUBJECTS:
        nii = os.path.join(MIITT_RECON, sub, "realtime", "sax", "4d_recon.nii.gz")
        if os.path.exists(nii):
            jobs.append(("MIITT", sub, lambda sub=sub: miitt_batch(sub)))

    for mode, lbl, build in jobs:
        out_nii = os.path.join(NIFTI_DIR, f"{mode}_{lbl}_Vrefined.nii.gz")
        if not os.path.exists(out_nii):
            print(f"  skip {mode}_{lbl} (no saved V_refined)"); continue
        Vin = input_volume(build())
        Vout = nib.load(out_nii).get_fdata().astype(np.float32)
        render_io(Vin, Vout,
                  f"{mode}_{lbl}  —  INPUT slices (left)  |  V_refined newseed (right)",
                  os.path.join(OUT, f"{mode}_{lbl}_io.png"))


if __name__ == "__main__":
    main()
