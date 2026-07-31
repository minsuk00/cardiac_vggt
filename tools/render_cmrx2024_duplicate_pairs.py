"""Visual side-by-side of the 7 CMRxRecon2024 duplicate pairs (see CMRxRecon2024/DUPLICATES.txt).

Each pair is one row: N z-planes of the KEPT subject, the same z-planes of the REDUNDANT copy,
then |difference|. If the pair really is one person released twice, the difference panels are
black and the annotated max|diff| is ~0.

Both members are rendered with the SAME intensity window (taken from the kept volume) -- a
per-panel window would let a brightness difference hide a real mismatch, or manufacture one.

Both sides must come from the SAME recon version. The archived Train_P19x carry v1 (image-domain
ESPIRiT) recons, so they are re-reconstructed with the fixed pipeline into `_dupcheck_v2/` first
(scratchpad `recon_dups.py`); this script reads that.

Panels are drawn at TRUE physical aspect from the header in-plane spacing -- a square-pixel
render distorts anisotropic grids.

    micromamba run -n svr python tools/render_cmrx2024_duplicate_pairs.py
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVE = os.path.join(REPO, "scratch", "data", "CMRxRecon2024", "Cine_combined")
DUPV2 = os.path.join(REPO, "scratch", "data", "CMRxRecon2024", "_dupcheck_v2")
OUT = os.path.join(REPO, "result", "cmrx2024_duplicate_pairs.png")

# (kept subject in the live tree, redundant copy that was archived)
PAIRS = [("Test_P009", "Train_P196"), ("Test_P010", "Train_P194"), ("Test_P011", "Train_P192"),
         ("Test_P012", "Train_P199"), ("Test_P013", "Train_P198"), ("Val_P052", "Train_P193"),
         ("Val_P024", "Train_P200")]

NZ = 5      # z-planes shown per subject
T = 0       # cardiac phase


def load(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj), img.header.get_zooms()


def main():
    rows = []
    for kept, redun in PAIRS:
        pk = os.path.join(LIVE, f"CMRx24_{kept}", "sax", "4d_recon.nii.gz")
        pr = os.path.join(DUPV2, redun, "sax", "4d_recon.nii.gz")
        if not (os.path.exists(pk) and os.path.exists(pr)):
            print(f"  SKIP {kept} <-> {redun}: missing "
                  f"{'kept' if not os.path.exists(pk) else ''}{'redundant' if not os.path.exists(pr) else ''}")
            continue
        a, za = load(pk)
        b, zb = load(pr)
        rows.append((kept, redun, a, b, za, zb))

    if not rows:
        print("nothing to render")
        return

    ncol = NZ * 3
    fig, axes = plt.subplots(len(rows), ncol,
                             figsize=(ncol * 1.25, len(rows) * 1.75), dpi=130)
    axes = np.atleast_2d(axes)

    for r, (kept, redun, a, b, za, zb) in enumerate(rows):
        same_shape = a.shape == b.shape
        nz = a.shape[2]
        zs = np.unique(np.linspace(0, nz - 1, NZ).round().astype(int))
        aspect = za[1] / za[0]                     # rows=Y, cols=X -> physical aspect
        vmax = float(np.percentile(a[..., T], 99.5)) or 1.0
        kw = dict(cmap="gray", vmin=0, vmax=vmax, aspect=aspect, interpolation="nearest")

        if same_shape:
            d = np.abs(a[..., T].astype(np.float64) - b[..., T].astype(np.float64))
            mx = float(d.max())
            fa, fb = a[..., T].ravel(), b[..., T].ravel()
            corr = float(np.corrcoef(fa, fb)[0, 1])
            verdict = f"max|diff|={mx:.3g}   corr={corr:.6f}"
        else:
            d, verdict = None, f"SHAPE MISMATCH {a.shape} vs {b.shape}"

        for i, z in enumerate(zs):
            axes[r, i].imshow(a[:, :, z, T].T, **kw)
            axes[r, NZ + i].imshow(b[:, :, z, T].T, **kw)
            if d is not None:
                axes[r, 2 * NZ + i].imshow(d[:, :, z].T, **kw)
            if r == 0:
                axes[r, i].set_title(f"z{z}", fontsize=6)
                axes[r, NZ + i].set_title(f"z{z}", fontsize=6)
                axes[r, 2 * NZ + i].set_title(f"z{z}", fontsize=6)
        for i in range(len(zs), NZ):                     # blank any unused columns
            for grp in range(3):
                axes[r, grp * NZ + i].set_visible(False)

        for c in range(ncol):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"{kept}\nvs {redun}", fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=26)
        ok = (d is not None and mx == 0.0)
        axes[r, 2 * NZ].text(0.02, -0.13, verdict + ("   [IDENTICAL]" if ok else ""),
                             transform=axes[r, 2 * NZ].transAxes, fontsize=6.5,
                             color=("green" if ok else "darkorange"))

    for c, lab in [(NZ // 2, "KEPT (live v2)"),
                   (NZ + NZ // 2, "REDUNDANT copy (re-reconstructed v2)"),
                   (2 * NZ + NZ // 2, "|difference|")]:
        axes[0, c].annotate(lab, xy=(0.5, 1.42), xycoords="axes fraction",
                            ha="center", fontsize=8.5, fontweight="bold")

    fig.suptitle("CMRxRecon2024 — the 7 duplicate pairs, both sides reconstructed with the fixed "
                 f"ESPIRiT (t={T}, true physical aspect)", fontsize=10, y=0.998)
    fig.tight_layout(rect=[0.02, 0, 1, 0.965])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
