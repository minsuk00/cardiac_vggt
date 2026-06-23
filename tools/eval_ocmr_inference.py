#!/usr/bin/env python
"""BACK-COMPAT SHIM — the OCMR RTFB inference logic now lives in the `eval/` package.

This module was the original direct OCMR adapter + inference + render + CLI. Its logic moved
to `eval/adapters/` (data→batch), `eval/inference.py` (batch→volume), and `eval/render.py`
(figures); the unified runner is `eval/run_rtfb.py`. This shim re-exports the names that
existing `tools/` diagnostic scripts import, so they keep working unchanged. The OCMR batch
is byte-identical to the original (guarded by tests/test_eval_ocmr_equivalence.py).

Prefer `python eval/run_rtfb.py --dataset ocmr ...` for new work.
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# ── re-exported pipeline pieces (the names consumers import) ───────────────
from eval.adapters.base import (  # noqa: F401
    percentile_scale, assign_canonical_z, to_canonical_inplane, _build_batch_core,
    INPUT_IMG_SIZE, TARGET_INPLANE_MM, GRID_SHAPE, D_CANON, CANON_Z_SPACING_MM,
    PCT_LO, PCT_HI, MM_PER_NORM, DEFAULT_CKPT,
)
from eval.adapters.ocmr import OCMRAdapter
from eval.inference import load_rtfb_model, forward, phase_sweep  # noqa: F401
from eval.render import (  # noqa: F401
    save_dvf_png, save_cycle_gif, save_inputs_png, save_volume_png,
)


# ── back-compat free functions (original signatures) ──────────────────────
def load_model(ckpt_path, device):
    """Original signature: z-only (no refiner) model + weights."""
    return load_rtfb_model(ckpt_path, refiner=False, device=device)


def load_cine(subj_dir):
    """-> cine[frame, slice, H, W] float32, meta dict (original return contract)."""
    a = OCMRAdapter(subj_dir)
    return a.load(), a._meta


def build_batch(cine, meta, scale, z_map, rng, device):
    """Original signature: reads inplane from meta['inplane_mm']; delegates to the shared core."""
    return _build_batch_core(cine, meta["inplane_mm"], scale, z_map, rng, device)


def reconstruct_cycle(model, batch, S, device):
    """Original signature (S unused now). Sweep target_t -> (vols, wp_by_t)."""
    return phase_sweep(model, batch, return_world_points=True, device=device)


# ── CLI (delegates to the shared adapter + inference + render) ─────────────
def main():
    ap = argparse.ArgumentParser(description="OCMR RTFB qualitative inference (shim for eval/run_rtfb.py).")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--recon_dir", default="scratch/data/ocmr/recon")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all in recon_dir")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--out", default="result/ocmr_eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda")
    model = load_rtfb_model(args.ckpt, refiner=False, device=device)
    subj_dirs = ([os.path.join(args.recon_dir, s) for s in args.subjects] if args.subjects
                 else sorted(d for d in glob.glob(os.path.join(args.recon_dir, "*"))
                             if os.path.isdir(d)))
    os.makedirs(args.out, exist_ok=True)

    for sd in subj_dirs:
        name = os.path.basename(sd)
        if not os.path.exists(os.path.join(sd, "sax_cine.nii.gz")):
            continue
        odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
        for d in range(args.draws):
            rng = np.random.default_rng(args.seed + d)
            batch, S, picks = OCMRAdapter(sd).build_batch(rng, device)
            coords0 = batch["scanner_coords"][0].cpu().numpy()
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
