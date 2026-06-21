"""Assess M&Ms nnU-Net segmentations of VGGT SAX volumes.

For each subject we segmented both the GT volume and the VGGT-predicted volume.
Reports per-structure voxel counts (coverage / did it segment anything sensible)
and Dice(seg(pred_vol), seg(gt_vol)) per structure -- i.e. does the reconstructed
volume yield the SAME anatomy the GT volume does. Also dumps mid-slice overlay PNGs.

M&Ms / Task114 labels: 1=LV blood pool, 2=LV myocardium, 3=RV blood pool.
"""
import argparse, glob, os, re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {1: "LV", 2: "MYO", 3: "RV"}


def load(seg_dir, case):
    f = os.path.join(seg_dir, case + ".nii.gz")
    return np.asarray(nib.load(f).dataobj).astype(np.int16) if os.path.exists(f) else None


def dice(a, b, lbl):
    A, B = a == lbl, b == lbl
    s = A.sum() + B.sum()
    return float("nan") if s == 0 else 2.0 * (A & B).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg_dir", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_png_dir", default=None)
    args = ap.parse_args()

    segs = sorted(glob.glob(os.path.join(args.seg_dir, "*.nii.gz")))
    bases = sorted({re.sub(r"_(gt|pred)$", "", os.path.basename(s)[:-7]) for s in segs})

    print(f"{'subject':28s} {'kind':5s} | " + " ".join(f"{LABELS[l]:>7s}" for l in LABELS) +
          "  zslices_with_seg")
    rows = {}
    for base in bases:
        for kind in ("gt", "pred"):
            case = f"{base}_{kind}"
            seg = load(args.seg_dir, case)
            if seg is None:
                continue
            rows[(base, kind)] = seg
            counts = {l: int((seg == l).sum()) for l in LABELS}
            zslices = int(((seg > 0).reshape(-1, seg.shape[-1]).sum(0) > 0).sum()) \
                if seg.ndim == 3 else -1
            print(f"{base:28s} {kind:5s} | " +
                  " ".join(f"{counts[l]:7d}" for l in LABELS) + f"   {zslices}")

    print("\n=== Dice( seg(pred_vol) , seg(gt_vol) ) per structure ===")
    print(f"{'subject':28s} | " + " ".join(f"{LABELS[l]:>6s}" for l in LABELS))
    agg = {l: [] for l in LABELS}
    for base in bases:
        g, p = rows.get((base, "gt")), rows.get((base, "pred"))
        if g is None or p is None:
            continue
        ds = {l: dice(p, g, l) for l in LABELS}
        for l in LABELS:
            if not np.isnan(ds[l]):
                agg[l].append(ds[l])
        print(f"{base:28s} | " + " ".join(f"{ds[l]:6.3f}" for l in LABELS))
    print(f"{'MEAN':28s} | " +
          " ".join(f"{np.mean(agg[l]) if agg[l] else float('nan'):6.3f}" for l in LABELS))

    if args.out_png_dir:
        os.makedirs(args.out_png_dir, exist_ok=True)
        cmap = matplotlib.colors.ListedColormap(["none", "red", "yellow", "cyan"])
        for base in bases:
            fig, axes = plt.subplots(2, 1, figsize=(4, 8))
            for ax, kind in zip(axes, ("gt", "pred")):
                seg = rows.get((base, kind))
                img = np.asarray(nib.load(os.path.join(
                    args.input_dir, f"{base}_{kind}_0000.nii.gz")).dataobj)
                if seg is None:
                    continue
                z = seg.shape[-1] // 2
                ax.imshow(img[:, :, z].T, cmap="gray")
                ax.imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]).T,
                          cmap=cmap, vmin=0, vmax=3, alpha=0.5)
                ax.set_title(f"{kind} z={z}"); ax.axis("off")
            fig.suptitle(base, fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out_png_dir, f"{base}.png"), dpi=90)
            plt.close(fig)
        print(f"\nwrote overlays to {args.out_png_dir}")


if __name__ == "__main__":
    main()
