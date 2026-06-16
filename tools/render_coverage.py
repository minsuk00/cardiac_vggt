"""Render coverage maps to SEE whether the dark spots are low-coverage regions.

For the resp model (v2) on breathing val, per in-bbox z-plane:
  row 1 V_gt | row 2 V_canon | row 3 coverage (how much input landed) | row 4 low-coverage mask
  (coverage<thr where GT has tissue = where dark/blurry spots come from).

Run: micromamba run -n svr python tools/render_coverage.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
from eval_variants_matrix import (build_dataset, build_batch, make_model, RUNS, LOGS,  # noqa: E402
                                  PROTOCOLS, GRID_SHAPE, NUM_SLICES)
sys.path.insert(0, os.path.join(REPO, "training"))
from data.gpu_aug import gpu_augment_batch          # noqa: E402
from loss import compute_volume_intensity_loss       # noqa: E402

OUT = os.path.join(REPO, "result", "variants_eval", "panels")
SEQS = [0, 7]
VAR = 2
COV_THR = 0.25   # voxels with less accumulated weight than this are "under-covered"


def main():
    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_dataset()
    run = next(r for r in RUNS if r["var"] == VAR)
    model, _ = make_model(run["use_t"], os.path.join(LOGS, run["exp_dir"], "ckpts", "checkpoint_last.pt"), device)
    cfg = PROTOCOLS["breathing"]

    for seq in SEQS:
        data = ds.get_data(seq_index=seq, img_per_seq=NUM_SLICES)
        batch = build_batch(data, device, seq_index=seq)
        batch = gpu_augment_batch(batch, None, device, respiratory_cfg=cfg, train=False)
        bbox = [int(v) for v in batch["anatomy_bbox"][0].tolist()]
        t_target = int(np.asarray(data["t_target"]).flatten()[0])
        subj = os.path.basename(ds.subjects[seq % len(ds.subjects)])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            preds = model(batch["images"], batch=batch)
        out = compute_volume_intensity_loss(preds, batch, grid_shape=GRID_SHAPE, tv_weight=0.0)
        Vgt = out["V_gt"][0].float().cpu().numpy()
        Vc = out["V_canon"][0].float().cpu().numpy()
        cov = out["coverage"][0].float().cpu().numpy()
        cov_frac = float((cov > 1e-3).mean())

        z0, z1 = bbox[0], bbox[1]
        zs = list(range(z0, z1))
        vmax = float(max(Vgt.max(), Vc.max(), 1e-3))
        gt_tissue = Vgt > 1e-2
        under = (cov < COV_THR) & gt_tissue   # under-covered tissue → dark/blur risk
        rows = [("V_gt", Vgt, "gray", 0, vmax),
                ("V_canon (resp)", Vc, "gray", 0, vmax),
                ("coverage", cov, "viridis", 0, float(np.percentile(cov, 99))),
                (f"under-covered\ntissue (cov<{COV_THR})", under.astype(float), "Reds", 0, 1)]
        nrow, ncol = len(rows), len(zs)
        fig = plt.figure(figsize=(1.5 * ncol + 1.2, 1.5 * nrow + 0.8), dpi=130)
        gs = gridspec.GridSpec(nrow, ncol, wspace=0.04, hspace=0.08)
        for r, (label, vol, cmap, vmn, vmx) in enumerate(rows):
            for c, z in enumerate(zs):
                ax = fig.add_subplot(gs[r, c]); ax.imshow(vol[z], cmap=cmap, vmin=vmn, vmax=vmx)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0: ax.set_title(f"z={z}", fontsize=8)
                if c == 0: ax.set_ylabel(label, fontsize=8)
        frac_under = float(under.sum() / max(1, gt_tissue.sum()))
        fig.suptitle(f"Coverage map — {subj} t={t_target} (breathing val, resp model). "
                     f"coverage_frac={cov_frac:.2f}; under-covered tissue={frac_under*100:.1f}%",
                     fontsize=11, y=1.01)
        p = os.path.join(OUT, f"coverage_seq{seq}_{subj}_t{t_target}.png")
        plt.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
        print(f"saved {p}  cov_frac={cov_frac:.3f} under_tissue={frac_under*100:.1f}%")


if __name__ == "__main__":
    main()
