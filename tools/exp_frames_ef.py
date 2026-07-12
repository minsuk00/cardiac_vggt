#!/usr/bin/env python
"""EF stage of the frames-per-slice experiment: does feeding more frames per slice recover
contraction AMPLITUDE (ejection fraction), the depth-motion a single frame can't resolve
(docs/24-25)? Dumps per-phase pred/GT volumes at each frames_per_slice into SEPARATE dirs so the
committed seg pipeline (inference/seg_cmrxrecon.sh + inference/seg_metrics_cmrxrecon.py) can be run per
frame-count and the resulting EF scatters compared.

Loads the 8.8 GB model ONCE and reuses inference.run_cmrxrecon.reconstruct_cycle + save_nnunet_nii.
GT is mode-independent -> dumped once per subject per frame-count.

  micromamba run -n svr python tools/exp_frames_ef.py --subjects $(seq 0 14) --frames 1 5 --modes clean
Then per frame-count:
  bash inference/seg_cmrxrecon.sh result/frames_ef/fps01/vols result/frames_ef/fps01/seg
  micromamba run -n svr python inference/seg_metrics_cmrxrecon.py \
      --seg_dir result/frames_ef/fps01/seg --vol_dir result/frames_ef/fps01/vols \
      --out_json result/frames_ef/fps01/ef.json --out_png result/frames_ef/fps01/scatter.png
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "training"))

from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import (reconstruct_cycle, build_mri_dataset, save_nnunet_nii,
                                save_multislice_gif, save_ed_montage)

SNAPSHOT = "/tmp/vggt_4wokxzov_snapshot.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=SNAPSHOT)
    ap.add_argument("--subjects", nargs="*", type=int, default=list(range(15)))
    ap.add_argument("--frames", nargs="*", type=int, default=[1, 5])
    ap.add_argument("--modes", nargs="*", default=["clean"])
    ap.add_argument("--out", default="result/frames_ef")
    ap.add_argument("--n_viz", type=int, default=2,
                    help="first N subjects also get a qualitative cycle GIF + ED montage per fps")
    args = ap.parse_args()

    device = torch.device("cuda")
    t0 = time.time()
    model = load_rtfb_model_reference(args.ckpt, refiner=False, device=device)
    mri_ds, rcfg = build_mri_dataset()
    print(f"[setup] ready in {time.time()-t0:.0f}s", flush=True)

    for fps in args.frames:
        base = os.path.join(args.out, f"fps{fps:02d}")
        vols = os.path.join(base, "vols")
        viz = os.path.join(base, "viz")
        os.makedirs(vols, exist_ok=True)
        os.makedirs(viz, exist_ok=True)
        gt_done = set()
        for seq_index in args.subjects:
            for mode in args.modes:
                breathing = mode == "breathing"
                ts = time.time()
                res = reconstruct_cycle(model, mri_ds, rcfg, seq_index, breathing, device,
                                        frames_per_slice=fps)
                for t in range(res["pred_vols"].shape[0]):
                    save_nnunet_nii(res["pred_vols"][t], os.path.join(
                        vols, f"subj{seq_index}_{mode}_pred_t{t:02d}_0000.nii.gz"))
                if seq_index not in gt_done:   # GT is mode-independent -> once per subject
                    for t in range(res["gt_vols"].shape[0]):
                        save_nnunet_nii(res["gt_vols"][t], os.path.join(
                            vols, f"subj{seq_index}_gt_t{t:02d}_0000.nii.gz"))
                    gt_done.add(seq_index)
                # Qualitative side-by-side fuel for the report (first n_viz subjects, clean mode):
                # the cycle GIF makes the fps=1 temporal blur vs fps=5 sharpness directly visible.
                if seq_index in args.subjects[:args.n_viz] and mode == "clean":
                    tag = os.path.join(viz, f"subj{seq_index}_fps{fps:02d}")
                    save_multislice_gif(res["pred_vols"], res["gt_vols"], res["bbox"],
                                        tag + "_cycle.gif")
                    save_ed_montage(res["pred_vols"][0], res["gt_vols"][0], tag + "_ED.png")
                print(f"[fps={fps:2d} subj{seq_index} {mode}] dumped 12 phases "
                      f"[{time.time()-ts:.0f}s]", flush=True)
        print(f"[fps={fps}] volumes -> {vols}", flush=True)
    print(f"DONE ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
