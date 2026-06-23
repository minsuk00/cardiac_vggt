#!/usr/bin/env python
"""Single-subject RTFB inference runner — one entry point for all OOD real-time datasets.

Replaces the per-dataset scripts (tools/eval_ocmr_inference.py CLI + goettingen_infer.py):
loads the trained z-only + target_t model, adapts a real real-time cine into the canonical
input contract via the dataset's RTFB adapter, sweeps target_t over the cardiac cycle, and
renders a beating-heart GIF (+ per-z volume sheet, input contact sheet, predicted-DVF panel).

There is NO ground-truth volume for these prospectively-acquired datasets — this is a
qualitative beating-heart transfer check, not a metric.

Usage:
  micromamba run -n svr python eval/run_rtfb.py --dataset ocmr      [--subjects us_0084_1_5T ...]
  micromamba run -n svr python eval/run_rtfb.py --dataset goettingen --refiner [--ckpt PATH]
  micromamba run -n svr python eval/run_rtfb.py --dataset miitt     [--draws 3]
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from eval.adapters import OCMRAdapter, GoettingenAdapter, MIITTAdapter
from eval.inference import load_rtfb_model, phase_sweep
from eval.render import save_cycle_gif, save_dvf_png, save_inputs_png, save_volume_png
from eval.adapters.base import DEFAULT_CKPT

# Per-dataset: default recon root, subject discovery, adapter factory (subject -> adapter).
DATASETS = {
    "ocmr": dict(
        root="scratch/data/ocmr/recon",
        discover=lambda root: sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, "sax_cine.nii.gz"))),
        adapter=lambda root, s: OCMRAdapter(os.path.join(root, s)),
    ),
    "goettingen": dict(
        root="scratch/data/goettingen/recon",
        discover=lambda root: sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, os.path.basename(d) + ".nii.gz"))),
        adapter=lambda root, s: GoettingenAdapter(os.path.join(root, s, s + ".nii.gz")),
    ),
    "miitt": dict(
        root="scratch/data/MIITT/nifti",
        discover=lambda root: sorted(
            os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
            if os.path.exists(os.path.join(d, "realtime", "sax", "4d_recon.nii.gz"))),
        adapter=lambda root, s: MIITTAdapter(os.path.join(root, s, "realtime", "sax", "4d_recon.nii.gz")),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--refiner", action="store_true", help="model has a coverage refiner head")
    ap.add_argument("--root", default=None, help="override the dataset recon root")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all discovered")
    ap.add_argument("--draws", type=int, default=3, help="random input draws per subject")
    ap.add_argument("--out", default=None, help="default: result/<dataset>_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = args.root or spec["root"]
    out = args.out or f"result/{args.dataset}_eval"
    os.makedirs(out, exist_ok=True)

    if args.dataset == "miitt" and getattr(MIITTAdapter, "SPACING_IS_PLACEHOLDER", False):
        print("  !! MIITT spacing is a PLACEHOLDER (2.6/8.0 mm) — qualitative only, no EF/distances.",
              flush=True)

    device = torch.device("cuda")
    model = load_rtfb_model(args.ckpt, refiner=args.refiner, device=device)

    subjects = args.subjects or spec["discover"](root)
    if not subjects:
        print(f"  no {args.dataset} subjects found under {root}", flush=True)
        return

    for name in subjects:
        adapter = spec["adapter"](root, name)
        odir = os.path.join(out, name); os.makedirs(odir, exist_ok=True)
        for d in range(args.draws):
            rng = np.random.default_rng(args.seed + d)
            batch, S, picks = adapter.build_batch(rng, device)
            coords0 = batch["scanner_coords"][0].cpu().numpy()        # (S,518,518,3)
            vols, wp_by_t = phase_sweep(model, batch, return_world_points=True, device=device)
            save_cycle_gif(vols, os.path.join(odir, f"draw{d}_cycle.gif"))
            save_inputs_png(picks, os.path.join(odir, f"draw{d}_inputs.png"))
            if d == 0:
                save_volume_png(vols, os.path.join(odir, "draw0_volume_t0.png"))
                save_dvf_png(wp_by_t[0], coords0, picks, os.path.join(odir, "draw0_dvf_t0.png"), t=0)
            print(f"[{name}] draw {d}: S={S} -> {odir}/draw{d}_cycle.gif", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
