#!/usr/bin/env python
"""FAST multi-frame check on a FEW CMRxRecon val subjects (not the whole set): reference-slot
model 4wokxzov, both breathing on/off, frames_per_slice ∈ {1,5} (no fps=12 -> no OOM, no 6-min
passes). Dumps EVERY output slice as PNG (all 12 phases × all 12 z-planes) plus a cycle GIF, so
the reconstruction can be eyeballed. Loads the 8.8 GB model ONCE.

  micromamba run -n svr python tools/exp_multiframe_quick.py --subjects 0 3 7 --frames 1 5
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from eval.inference import load_rtfb_model_reference
from eval.run_cmrxrecon import (reconstruct_cycle, build_mri_dataset, save_multislice_gif)

SNAPSHOT = "/tmp/vggt_4wokxzov_snapshot.pt"


def montage_all_slices(vol_tdhw, path, title, vmax=None):
    """Full grid: rows = 12 cardiac phases, cols = 12 z-planes. Every reconstructed slice."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    T, D = vol_tdhw.shape[0], vol_tdhw.shape[1]
    vmax = vmax if vmax is not None else float(max(vol_tdhw.max(), 1e-3))
    fig, axes = plt.subplots(T, D, figsize=(0.95 * D, 0.95 * T), squeeze=False)
    for t in range(T):
        for z in range(D):
            ax = axes[t][z]
            ax.imshow(vol_tdhw[t, z], cmap="gray", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_title(f"z{z}", fontsize=6)
            if z == 0:
                ax.set_ylabel(f"t{t}", fontsize=6, rotation=0, labelpad=8, va="center")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=95); plt.close(fig)
    return vmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=SNAPSHOT)
    ap.add_argument("--subjects", nargs="*", type=int, default=[0, 3, 7])
    ap.add_argument("--frames", nargs="*", type=int, default=[1, 5])
    ap.add_argument("--modes", nargs="*", default=["clean", "breathing"])
    ap.add_argument("--out", default="result/multiframe_quick")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    t0 = time.time()
    model = load_rtfb_model_reference(args.ckpt, refiner=False, device=device)
    mri_ds, rcfg = build_mri_dataset()
    print(f"[setup] ready in {time.time()-t0:.0f}s", flush=True)

    summary = []
    gt_done = set()
    for seq_index in args.subjects:
        for mode in args.modes:
            breathing = mode == "breathing"
            for fps in args.frames:
                ts = time.time()
                res = reconstruct_cycle(model, mri_ds, rcfg, seq_index, breathing, device,
                                        frames_per_slice=fps)
                m = res["metrics"]
                def nm(xs):
                    xs = [x for x in xs if x == x]
                    return float(np.mean(xs)) if xs else float("nan")
                row = dict(subject=int(seq_index), mode=mode, frames_per_slice=int(fps),
                           motion=nm(m["motion"]), bbox=nm(m["bbox"]), full=nm(m["full"]),
                           ssim=nm(m["ssim"]),
                           motion_per_phase=[float(x) for x in m["motion"]])
                summary.append(row)
                base = os.path.join(args.out, f"subj{seq_index}_{mode}_fps{fps:02d}")
                gmax = float(max(res["gt_vols"].max(), res["pred_vols"].max(), 1e-3))
                montage_all_slices(res["pred_vols"], base + "_pred_allslices.png",
                                   f"subj{seq_index} {mode} fps={fps}  PRED (rows=phase, cols=z)  "
                                   f"motion={row['motion']:.2f} bbox={row['bbox']:.2f}dB", vmax=gmax)
                save_multislice_gif(res["pred_vols"], res["gt_vols"], res["bbox"], base + "_cycle.gif")
                if seq_index not in gt_done:   # GT is mode/fps-independent -> once per subject
                    montage_all_slices(res["gt_vols"], os.path.join(
                        args.out, f"subj{seq_index}_GT_allslices.png"),
                        f"subj{seq_index}  GROUND TRUTH (rows=phase, cols=z)", vmax=gmax)
                    gt_done.add(seq_index)
                print(f"[subj{seq_index} {mode} fps={fps}] motion={row['motion']:.2f} "
                      f"bbox={row['bbox']:.2f} full={row['full']:.2f} ssim={row['ssim']:.3f} "
                      f"[{time.time()-ts:.0f}s] -> {base}_pred_allslices.png", flush=True)
                torch.cuda.empty_cache()
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE ({time.time()-t0:.0f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
