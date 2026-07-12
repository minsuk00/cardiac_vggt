#!/usr/bin/env python
"""Experiment: does feeding MORE frames per slice help the reference-slot model (4wokxzov)?

The model was trained multi-frame but S-slot-budget-capped (~random coverage). At inference we
are NOT budget-capped, so we can feed a genuinely short real-time acquisition: the mid reference
plane contributes all T phases; every other in-bbox plane contributes `frames_per_slice`
consecutive phases. This driver sweeps `frames_per_slice` on CMRxRecon val (which HAS ground
truth) and measures, per config:

  motion / bbox / full PSNR + SSIM  (compute_volume_intensity_loss, identical to training)
  coverage  = fraction of in-bbox voxels the splat filled (pred>eps), averaged over phases
  motion@ED / motion@ES  (peak-motion phase is the hardest; ED is near-static)

Two opposing effects are under test:
  (+) more phases per plane -> better coverage + more chance a frame sits near the target phase
  (-) the model is NOT told each frame's phase (t inert) -> the splat AVERAGES unresolved phases
      -> temporal blur of the moving myocardium.
So the read is: coverage/bbox should rise with frames, but motion PSNR may plateau/fall.

Loads the 8.8 GB model ONCE and reuses inference.run_cmrxrecon.reconstruct_cycle across all configs
(running the CLI per frame-count would reload the model each time).

  micromamba run -n svr python tools/exp_frames_sweep.py --subjects 0 3 7 --frames 1 2 3 5 8 12
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

from inference.inference import load_rtfb_model_reference
from inference.run_cmrxrecon import reconstruct_cycle, build_mri_dataset, ED_PHASE

SNAPSHOT = "/tmp/vggt_4wokxzov_snapshot.pt"


def coverage_frac(pred_vols, bbox, eps=1e-4):
    """Fraction of in-bbox voxels the splat filled (pred > eps), averaged over phases.
    The splat zeroes uncovered voxels (acc=0 -> 0/(0+eps)), so pred>eps ~= 'a frame reached here'."""
    z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox[:6]]
    sub = pred_vols[:, z0:z1, y0:y1, x0:x1]
    if sub.size == 0:
        return float("nan")
    return float((sub > eps).mean())


def es_phase(gt_vols, bbox):
    """End-systole = phase with smallest in-bbox blood-pool-ish volume proxy. We don't have a
    segmentation here, so use the phase whose in-bbox intensity mass deviates most from ED as a
    cheap 'peak motion' pick; falls back to mid-cycle. Only used to report motion@peak."""
    z0, z1, y0, y1, x0, x1 = [int(v) for v in bbox[:6]]
    sub = gt_vols[:, z0:z1, y0:y1, x0:x1]
    ref = sub[ED_PHASE]
    diff = np.abs(sub - ref[None]).reshape(sub.shape[0], -1).mean(1)
    return int(np.argmax(diff))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=SNAPSHOT)
    ap.add_argument("--subjects", nargs="*", type=int, default=[0, 3, 7])
    ap.add_argument("--frames", nargs="*", type=int, default=[1, 2, 3, 5, 8, 12])
    ap.add_argument("--modes", nargs="*", default=["clean", "breathing"])
    ap.add_argument("--out", default="result/frames_sweep")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda")
    t0 = time.time()
    model = load_rtfb_model_reference(args.ckpt, refiner=False, device=device)
    mri_ds, rcfg = build_mri_dataset()
    print(f"[setup] model+dataset ready in {time.time()-t0:.0f}s", flush=True)

    rows = []
    outpath = os.path.join(args.out, "results.json")
    for seq_index in args.subjects:
        for mode in args.modes:
            breathing = mode == "breathing"
            for fps in args.frames:
                ts = time.time()
                try:
                    res = reconstruct_cycle(model, mri_ds, rcfg, seq_index, breathing, device,
                                            frames_per_slice=fps)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"[subj{seq_index} {mode} fps={fps:2d}] SKIPPED (CUDA OOM at this S) "
                          f"[{time.time()-ts:.0f}s]", flush=True)
                    continue
                m = res["metrics"]
                bbox = res["bbox"]
                es = es_phase(res["gt_vols"], bbox)

                def nm(xs):
                    xs = [x for x in xs if x == x]
                    return float(np.mean(xs)) if xs else float("nan")

                row = dict(
                    subject=int(seq_index), mode=mode, frames_per_slice=int(fps),
                    S=None,  # filled below
                    motion_mean=nm(m["motion"]), bbox_mean=nm(m["bbox"]),
                    full_mean=nm(m["full"]), ssim_mean=nm(m["ssim"]),
                    motion_ED=float(m["motion"][ED_PHASE]), motion_ES=float(m["motion"][es]),
                    es_phase=es, coverage=coverage_frac(res["pred_vols"], bbox),
                    motion_per_phase=[float(x) for x in m["motion"]],
                    bbox_per_phase=[float(x) for x in m["bbox"]],
                )
                rows.append(row)
                torch.cuda.empty_cache()
                with open(outpath, "w") as f:
                    json.dump(rows, f, indent=2)
                print(f"[subj{seq_index} {mode} fps={fps:2d}] motion={row['motion_mean']:.2f} "
                      f"(ED={row['motion_ED']:.1f}/ES@{es}={row['motion_ES']:.1f}) "
                      f"bbox={row['bbox_mean']:.2f} full={row['full_mean']:.2f} "
                      f"ssim={row['ssim_mean']:.3f} cover={row['coverage']:.3f} "
                      f"[{time.time()-ts:.0f}s]", flush=True)
    print(f"wrote {outpath}  ({time.time()-t0:.0f}s total)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
