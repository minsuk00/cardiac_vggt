"""Visualize the proposed motion mask on real val subjects.

motion_mag[z,y,x] = V_gt[:, z,y,x].max(t) - min(t)   over the 12 cardiac phases
motion_mask       = motion_mag > tau

Renders, per subject:
  fig A: rows = in-bbox z-planes, cols = [ED frame, ES frame, motion magnitude, mask overlay]
  fig B: tau sweep on one mid-bbox plane (how the mask grows/shrinks with the threshold)

Outputs PNGs to result/motion_mask_preview/.
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.datasets.mri_dataset import MRIDataset
from data.datasets.mri_dataset import compute_geometric_bbox

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
SPLIT_FILE = "/home/minsukc/vggt/training/splits/random_8_1_1.txt"
OUT_DIR = "/home/minsukc/vggt/result/motion_mask_preview"
SUBJECTS = [0, 7]               # which val subjects to render
TAU_OVERLAY = 0.05             # threshold used for the overlay column (matches MOTION_MASK_TAU)
TAU_SWEEP = [0.05, 0.10, 0.15, 0.20]


def load_phases(ds, subj_idx):
    """Return (phases_splat (T,D,H,W) float32, mask_splat (D,H,W) bool)."""
    cached = ds.cache[subj_idx]
    phases = cached["phases"]
    content_mask = cached["content_mask"]
    if phases.ndim == 5:
        phases = phases.squeeze(1)
    phases_splat = phases.permute(0, 3, 2, 1).contiguous().float().cpu().numpy()  # (T,D,H,W)
    mask_splat = content_mask.squeeze(0).permute(2, 1, 0).contiguous().cpu().numpy() > 0.5
    return phases_splat, mask_splat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    common_conf = OmegaConf.create({
        "img_size": 518, "patch_size": 14, "rescale": True,
        "rescale_aug": False, "landscape_check": False,
        "augs": {"scales": [1.0, 1.0]},
    })
    ds = MRIDataset(common_conf, DATA_ROOT, split="val", split_file=SPLIT_FILE,
                    mode="dynamic", mri_mode="axial", num_slices=12, target_size=518)

    for subj_idx in SUBJECTS:
        name = os.path.basename(os.path.dirname(ds.subjects[subj_idx]))
        phases, cmask = load_phases(ds, subj_idx)          # (T,D,H,W), (D,H,W)
        T, D, H, W = phases.shape

        ed = phases[0]                                      # (D,H,W)
        # ES = phase that differs most from ED globally
        es_t = int(np.argmax(((phases - ed[None]) ** 2).reshape(T, -1).mean(1)))
        es = phases[es_t]

        motion_mag = phases.max(0) - phases.min(0)         # (D,H,W)
        vmax_anat = max(float(ed.max()), 1e-3)
        mmax = max(float(motion_mag.max()), 1e-3)

        # in-bbox z-planes
        bbox = compute_geometric_bbox(
            torch.from_numpy(cmask.astype(np.float32))
        ).cpu().numpy().astype(int)
        z0, z1 = int(bbox[0]), int(bbox[1])
        zs = np.linspace(z0, z1 - 1, min(5, z1 - z0)).round().astype(int)

        # ---- fig A: per-plane panel ----
        nrows = len(zs)
        fig, axes = plt.subplots(nrows, 4, figsize=(4 * 2.4, nrows * 2.4))
        if nrows == 1:
            axes = axes[None]
        col_titles = ["ED (t=0)", f"ES (t={es_t})", "motion = max-min", f"mask (tau={TAU_OVERLAY}) on ED"]
        for r, z in enumerate(zs):
            axes[r, 0].imshow(ed[z], cmap="gray", vmin=0, vmax=vmax_anat)
            axes[r, 1].imshow(es[z], cmap="gray", vmin=0, vmax=vmax_anat)
            im = axes[r, 2].imshow(motion_mag[z], cmap="magma", vmin=0, vmax=mmax)
            axes[r, 3].imshow(ed[z], cmap="gray", vmin=0, vmax=vmax_anat)
            mask_z = motion_mag[z] > TAU_OVERLAY
            overlay = np.zeros((*mask_z.shape, 4))
            overlay[mask_z] = [1, 0, 0, 0.45]
            axes[r, 3].imshow(overlay)
            axes[r, 0].set_ylabel(f"z={z}", fontsize=10)
            for c in range(4):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                if r == 0:
                    axes[r, c].set_title(col_titles[c], fontsize=10)
        frac = float((motion_mag > TAU_OVERLAY)[z0:z1].mean())
        fig.suptitle(f"{name}  (val subj {subj_idx})   motion mask covers {frac*100:.1f}% of in-bbox voxels at tau={TAU_OVERLAY}",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        pa = os.path.join(OUT_DIR, f"subj{subj_idx:02d}_{name}_planes.png")
        fig.savefig(pa, dpi=90); plt.close(fig)

        # ---- fig B: tau sweep on the mid-bbox plane ----
        zmid = (z0 + z1) // 2
        fig, axes = plt.subplots(1, len(TAU_SWEEP) + 1, figsize=((len(TAU_SWEEP) + 1) * 2.4, 2.6))
        axes[0].imshow(motion_mag[zmid], cmap="magma", vmin=0, vmax=mmax)
        axes[0].set_title(f"motion mag\nz={zmid}", fontsize=10)
        axes[0].set_xticks([]); axes[0].set_yticks([])
        for k, tau in enumerate(TAU_SWEEP):
            ax = axes[k + 1]
            ax.imshow(ed[zmid], cmap="gray", vmin=0, vmax=vmax_anat)
            mask_z = motion_mag[zmid] > tau
            overlay = np.zeros((*mask_z.shape, 4)); overlay[mask_z] = [1, 0, 0, 0.45]
            ax.imshow(overlay)
            pct = float((motion_mag[z0:z1] > tau).mean()) * 100
            ax.set_title(f"tau={tau}\n{pct:.1f}% vox", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{name}: motion mask vs threshold (overlay on ED, z={zmid})", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pb = os.path.join(OUT_DIR, f"subj{subj_idx:02d}_{name}_tau_sweep.png")
        fig.savefig(pb, dpi=90); plt.close(fig)

        # ---- stats ----
        in_bbox_mag = motion_mag[z0:z1]
        print(f"[subj {subj_idx}] {name}  ES=t{es_t}  bbox_z=[{z0},{z1})")
        print(f"    motion_mag in-bbox: mean={in_bbox_mag.mean():.3f} "
              f"p50={np.percentile(in_bbox_mag,50):.3f} p95={np.percentile(in_bbox_mag,95):.3f} "
              f"max={in_bbox_mag.max():.3f}")
        for tau in TAU_SWEEP:
            print(f"    tau={tau}: {float((in_bbox_mag>tau).mean())*100:5.1f}% of in-bbox voxels")
        print(f"    wrote {pa}")
        print(f"    wrote {pb}")


if __name__ == "__main__":
    main()
