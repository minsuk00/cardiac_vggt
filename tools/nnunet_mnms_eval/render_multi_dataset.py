"""Render Task114 segs across datasets for a trust check. Per dataset-regime group:
a figure with rows=cases, cols=evenly-spaced z-planes, filled masks (LV=red, MYO=green, RV=blue).
Also prints a per-case numeric summary (labeled planes, per-structure voxels, contiguity)."""
import os, glob, sys
import numpy as np
import nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

IN = "/home/minsukc/vggt/scratch/data/nnunet_mnms/multi_ds_inputs"
SEG = "/home/minsukc/vggt/scratch/data/nnunet_mnms/multi_ds_seg"
OUTDIR = "/home/minsukc/vggt/result/whs_check"
os.makedirs(OUTDIR, exist_ok=True)
PAL = {1: (1, 0, 0, 0.55), 2: (0, 1, 0, 0.55), 3: (0, 0.5, 1, 0.55)}
NCOL = 8

GROUPS = {
    "cmrx":        lambda c: c.startswith("cmrx_"),
    "miitt_gated": lambda c: c.startswith("miitt_") and c.endswith("_gated"),
    "miitt_rt":    lambda c: c.startswith("miitt_") and c.endswith("_rt"),
    "ocmr_gated":  lambda c: c.startswith("ocmr_") and c.endswith("_gated"),
    "ocmr_rtfb":   lambda c: c.startswith("ocmr_") and c.endswith("_rtfb"),
}


def summarize(seg):
    Z = seg.shape[-1]
    per = [(seg[..., z] > 0).sum() for z in range(Z)]
    labeled = [z for z in range(Z) if per[z] > 0]
    lv = int((seg == 1).sum()); myo = int((seg == 2).sum()); rv = int((seg == 3).sum())
    contig = (labeled == list(range(labeled[0], labeled[-1] + 1))) if labeled else True
    return dict(Z=Z, n_labeled=len(labeled), zr=(labeled[0], labeled[-1]) if labeled else None,
                lv=lv, myo=myo, rv=rv, contig=contig)


def main():
    print(f"{'case':46} {'Z':>3} {'lab':>3} {'zrange':>8} {'LV':>6} {'MYO':>6} {'RV':>6} contig")
    for gname, pred in GROUPS.items():
        cases = sorted(c for c in
                       (os.path.basename(p)[:-7] for p in glob.glob(os.path.join(SEG, "*.nii.gz")))
                       if pred(c))
        if not cases:
            print(f"[{gname}] no cases"); continue
        fig, axes = plt.subplots(len(cases), NCOL, figsize=(2.0 * NCOL, 2.0 * len(cases)))
        if len(cases) == 1: axes = axes[None, :]
        for r, c in enumerate(cases):
            img = np.asarray(nib.load(os.path.join(IN, c + "_0000.nii.gz")).dataobj).astype(np.float32)
            seg = np.asarray(nib.load(os.path.join(SEG, c + ".nii.gz")).dataobj).astype(np.int16)
            s = summarize(seg)
            print(f"{c:46} {s['Z']:>3} {s['n_labeled']:>3} {str(s['zr']):>8} "
                  f"{s['lv']:>6} {s['myo']:>6} {s['rv']:>6} {'Y' if s['contig'] else 'N'}")
            Z = img.shape[-1]
            zcols = np.linspace(0, Z - 1, NCOL).round().astype(int)
            vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1
            for k, z in enumerate(zcols):
                ax = axes[r, k]
                ax.imshow(img[..., z].T, cmap="gray", vmin=0, vmax=vmax, origin="lower")
                m = seg[..., z].T
                rgba = np.zeros((*m.shape, 4))
                for lbl, col in PAL.items(): rgba[m == lbl] = col
                ax.imshow(rgba, origin="lower")
                ax.set_title(f"z{z}", fontsize=6); ax.axis("off")
            axes[r, 0].text(-0.15, 0.5, f"{c.replace('miitt_','').replace('ocmr_exam_','')[:24]}\n{s['n_labeled']}/{s['Z']}pl",
                            transform=axes[r, 0].transAxes, rotation=90, va="center", fontsize=6)
        fig.suptitle(f"{gname}   Red=LV Green=MYO Blue=RV   ({len(cases)} cases)", fontsize=12, y=1.005)
        plt.tight_layout()
        out = os.path.join(OUTDIR, f"multiDS_{gname}.png")
        plt.savefig(out, dpi=90, bbox_inches="tight"); plt.close()
        print(f"  -> {out}\n")


if __name__ == "__main__":
    main()
