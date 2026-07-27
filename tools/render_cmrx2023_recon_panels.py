"""Visual QC panels for the CMRxRecon2023 reconstruction.

Spans EVERY z-plane, never just the mid slice — the mid plane is the easiest and least informative
one, and a recon can look perfect there while the apex/base are broken.

Per subject, two rows:
  * all z at t=0  -> look for the LV donut through the stack, and for wrap/fold artifacts at the edges
  * all 12 frames at mid z -> the LV cavity should visibly contract and re-dilate

Always includes the three 6 mm-thickness protocol variants, whose slice pitch is an assumption
(10 mm), so they get looked at rather than trusted.

Usage: python tools/render_cmrx2023_recon_panels.py [--n 6]
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D23 = os.path.join(REPO, "scratch", "data", "CMRxRecon2023")
CINE = os.path.join(D23, "Cine_combined")
SIX_MM = ["CMRx23_Train_P040", "CMRx23_Train_P046", "CMRx23_Test_P116"]


def panel(cid, out_dir):
    f4 = os.path.join(CINE, cid, "sax", "4d_recon.nii.gz")
    if not os.path.exists(f4):
        return None
    im = nib.load(f4)
    a = np.asanyarray(im.dataobj)          # (x, y, z, t)
    zooms = tuple(round(float(z), 3) for z in im.header.get_zooms())
    Z, T = a.shape[2], a.shape[3]
    zmid = Z // 2
    ncol = max(Z, T)
    fig, axes = plt.subplots(2, ncol, figsize=(1.55 * ncol, 4.2))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    for c in range(ncol):
        for r in range(2):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].axis("off")
    for z in range(Z):
        img = a[:, :, z, 0].T             # -> (y, x) for display
        axes[0, z].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5) or 1)
        axes[0, z].set_title(f"z={z}", fontsize=7)
        axes[0, z].axis("on"); axes[0, z].set_xticks([]); axes[0, z].set_yticks([])
    for t in range(T):
        img = a[:, :, zmid, t].T
        axes[1, t].imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.5) or 1)
        axes[1, t].set_title(f"t={t}", fontsize=7)
        axes[1, t].axis("on"); axes[1, t].set_xticks([]); axes[1, t].set_yticks([])
    axes[0, 0].set_ylabel(f"all {Z} slices\n(t=0)", fontsize=8)
    axes[1, 0].set_ylabel(f"all {T} frames\n(z={zmid})", fontsize=8)
    tag = "  ⚠️ 6 mm variant — pitch 10 mm is an ASSUMPTION" if cid in SIX_MM else ""
    fig.suptitle(f"{cid}   shape={a.shape}  spacing={zooms}{tag}", fontsize=10)
    out = os.path.join(out_dir, f"{cid}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="how many regular subjects (6 mm ones are always added)")
    args = ap.parse_args()

    with open(os.path.join(D23, "SUBJECT_MANIFEST.csv")) as f:
        ids = [r["combined_id"] for r in csv.DictReader(f) if r["reconstruct"] == "1"]
    done = [c for c in ids if os.path.exists(os.path.join(CINE, c, "sax", "4d_recon.nii.gz"))]
    # spread the sample across the list rather than taking the first N (which are all Train_P00x)
    step = max(1, len(done) // max(1, args.n))
    sample = done[::step][: args.n]
    for s in SIX_MM:
        if s in done and s not in sample:
            sample.append(s)

    out_dir = os.path.join(REPO, "result", "cmrx2023_recon_qc")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{len(done)}/{len(ids)} reconstructed; rendering {len(sample)}")
    for cid in sample:
        p = panel(cid, out_dir)
        print(f"  {'wrote ' + p if p else 'skip ' + cid}")


if __name__ == "__main__":
    main()
