#!/usr/bin/env python
"""Single-subject RTFB inference runner — one entry point for all OOD real-time datasets.

Replaces the per-dataset scripts (tools/eval_ocmr_inference.py CLI + goettingen_infer.py):
loads the reference-slot z-only model (docs/25), adapts a real real-time cine into a
multi-frame + reference-slot canonical batch via the dataset's RTFB adapter (docs/28), sweeps
the reference slot over real acquired frames at the mid-z plane, and renders a beating-heart
GIF (+ per-z volume sheet, input contact sheet, predicted-DVF panel).

There is NO ground-truth volume for these prospectively-acquired datasets — this is a
qualitative beating-heart transfer check, not a metric.

Deterministic — no random draws (that's a training-time augmentation only): every discovered
z-plane is sampled at its first `--frames-per-slice` CONSECUTIVE real frames (simulating a
short acquisition burst per slice, the whole point of the fast-acquisition project — not an
even subsample of a recording that ran far longer), except the mid-ventricular reference plane
which gets a longer consecutive burst (`--frames-for-reference`) to drive the beating-heart
animation.

Usage:
  micromamba run -n svr python eval/run_rtfb.py --dataset ocmr      [--subjects us_0084_1_5T ...]
  micromamba run -n svr python eval/run_rtfb.py --dataset goettingen --refiner [--ckpt PATH]
  micromamba run -n svr python eval/run_rtfb.py --dataset miitt
"""
import argparse
import glob
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
from eval.adapters import OCMRAdapter, GoettingenAdapter, MIITTAdapter
from eval.inference import load_rtfb_model_reference, reference_sweep
from eval.render import save_cycle_gif, save_dvf_png, save_inputs_png, save_volume_png
from eval.adapters.base import DEFAULT_CKPT_REFERENCE

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
    ap.add_argument("--ckpt", default=DEFAULT_CKPT_REFERENCE)
    ap.add_argument("--frames-per-slice", type=int, default=5, help="first N consecutive real frames per non-reference z-plane")
    ap.add_argument("--frames-for-reference", type=int, default=30, help="first N consecutive real frames at the mid-z reference plane, swept as the query")
    ap.add_argument("--refiner", action="store_true", help="model has a coverage refiner head")
    ap.add_argument("--root", default=None, help="override the dataset recon root")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all discovered")
    ap.add_argument("--out", default=None, help="default: result/<dataset>_eval")
    args = ap.parse_args()

    spec = DATASETS[args.dataset]
    root = args.root or spec["root"]
    out = args.out or f"result/{args.dataset}_eval"
    os.makedirs(out, exist_ok=True)

    if args.dataset == "miitt" and getattr(MIITTAdapter, "SPACING_IS_PLACEHOLDER", False):
        print("  !! MIITT spacing is a PLACEHOLDER (2.6/8.0 mm) — qualitative only, no EF/distances.",
              flush=True)

    device = torch.device("cuda")
    model = load_rtfb_model_reference(args.ckpt, refiner=args.refiner, device=device)

    subjects = args.subjects or spec["discover"](root)
    if not subjects:
        print(f"  no {args.dataset} subjects found under {root}", flush=True)
        return

    for name in subjects:
        adapter = spec["adapter"](root, name)
        odir = os.path.join(out, name); os.makedirs(odir, exist_ok=True)
        batch, S, picks, ref_ctx = adapter.build_batch_multiframe(
            device, frames_per_slice=args.frames_per_slice, frames_for_reference=args.frames_for_reference)
        coords0 = batch["scanner_coords"][0].cpu().numpy()        # (S,518,518,3)
        vols, wp_by_t, _ = reference_sweep(model, batch, ref_ctx, return_world_points=True, device=device)
        save_cycle_gif(vols, os.path.join(odir, "cycle.gif"))
        save_inputs_png(picks, os.path.join(odir, "inputs.png"))
        save_volume_png(vols, os.path.join(odir, "volume_t0.png"))
        save_dvf_png(wp_by_t[0], coords0, picks, os.path.join(odir, "dvf_t0.png"), t=0)
        print(f"[{name}]: S={S} -> {odir}/cycle.gif", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
