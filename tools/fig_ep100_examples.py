"""Visual examples for _html/46: recon vs GT panels per cohort + the appearance-wall visual.
Offline; reads the frozen bundles' gt/ + the hub recon_breath/ NIfTIs (canonical, X,Y,Z)."""
import json, glob, os
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = "/home/minsukc/vggt/scratch/eval"
OUT = "/home/minsukc/vggt/result/1frame_ep100"; os.makedirs(OUT, exist_ok=True)
HUB = "vggt_20260719_1f_gather05_ep99"
COH = [("cmrxrecon", "CMRx (in-dist)"), ("miitt", "MIITT (OOD)"),
       ("ocmr", "OCMR (OOD)"), ("acdc", "ACDC (OOD, pathology)")]


def load_xyz(p):
    return np.asarray(nib.load(p).dataobj, dtype=np.float32)   # (X,Y,Z)


def rep_subject(cohort):
    """subject closest to the cohort's median hub breath PSNR."""
    f = glob.glob(f"{E}/{cohort}/out/{HUB}_summary.json")[0]
    ps = json.load(open(f))["per_subject"]
    med = np.median([r["breath_psnr"] for r in ps])
    return min(ps, key=lambda r: abs(r["breath_psnr"] - med))


def midz(gt):
    filled = [z for z in range(gt.shape[2]) if gt[..., z].max() > 0]
    return filled[len(filled) // 2]


def fig_examples():
    fig, axes = plt.subplots(len(COH), 3, figsize=(8.2, 2.5 * len(COH)))
    for i, (c, lbl) in enumerate(COH):
        rs = rep_subject(c); subj = rs["subject"]
        sd = f"{E}/{c}/out/{subj}/{HUB}"
        gt = load_xyz(f"{E}/{c}/out/{subj}/gt/gt_t00.nii.gz")
        rec = load_xyz(f"{sd}/recon_breath/vol_t00.nii.gz")
        z = midz(gt)
        g, r = gt[..., z].T, rec[..., z].T
        vmax = max(g.max(), 1e-3)
        for j, (im, ttl) in enumerate([(g, "clean GT"), (r, "recon (breath in)"), (np.abs(g - r), "|diff|")]):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(ttl, fontsize=10)
        axes[i, 0].set_ylabel(f"{lbl}\n{subj}\nbreath PSNR {rs['breath_psnr']:.1f} dB", fontsize=8)
    fig.suptitle("ED mid-ventricular slice: model recon (breathing-corrupted input) vs clean GT — hub, one representative subject/cohort", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{OUT}/fig_examples.png", dpi=135); plt.close(fig)
    print("wrote fig_examples.png")


def fig_wall_visual():
    """Same in-dist subject, GT vs 4 models — they look near-identical = the appearance wall."""
    c = "cmrxrecon"; subj = rep_subject(c)["subject"]
    models = [("gather05", "gather05 (hub)"), ("aug_moderate", "aug"),
              ("dino_ft", "dino"), ("no_gather", "no_gather")]
    gt = load_xyz(f"{E}/{c}/out/{subj}/gt/gt_t00.nii.gz"); z = midz(gt)
    fig, ax = plt.subplots(1, 5, figsize=(12, 2.6))
    ax[0].imshow(gt[..., z].T, cmap="gray", origin="lower"); ax[0].set_title("clean GT", fontsize=10)
    for k, (m, ml) in enumerate(models):
        rec = load_xyz(f"{E}/{c}/out/{subj}/vggt_20260719_1f_{m}_ep99/recon_breath/vol_t00.nii.gz")
        ax[k + 1].imshow(rec[..., z].T, cmap="gray", origin="lower"); ax[k + 1].set_title(ml, fontsize=10)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"The appearance wall, visually — {subj}: 4 models with wildly different C1–C5 breathing behavior produce near-identical recons (they share the same appearance-synthesis error)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/fig_wall_visual.png", dpi=135); plt.close(fig)
    print("wrote fig_wall_visual.png")


if __name__ == "__main__":
    fig_examples(); fig_wall_visual()
