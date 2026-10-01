#!/usr/bin/env python
"""Verify tools/framesel_data.py against the frozen rhythm-arm bundles (docs/128).

For a few test subjects of a rhythm cohort: framesel_data.render_frames, given the manifest's
true frame positions and the bundle's breathing model, must reproduce the model images
run_vggt.py builds from the stored breath/stack_t* frames -- i.e. the selector trains on exactly
the image formation it is evaluated on.

    PYTHONPATH=training:. python tools/framesel_check_data.py --cohort cmrx2024_af12 --n 3
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
import build_af_bundle as afb                                            # noqa: E402
import framesel_data as fd                                               # noqa: E402
from data.gpu_aug import _resize_to_model_res, extract_slices_from_phases  # noqa: E402
from data.respiratory import lujan_displacement                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cmrx2024_af12")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    dev = torch.device("cuda")
    subs = sorted(glob.glob(f"scratch/eval/{args.cohort}/out/*/manifest.json"))[: args.n]
    for mp in subs:
        d = os.path.dirname(mp)
        man = json.load(open(mp))
        src = os.path.join("scratch/eval", man["source"], "out", man["subject"])
        _, gt, _ = afb.load_subject(src, need_vol=False)                # (T,D,H,W) clean cine
        rh = man["rhythm"]
        F_ = int(rh["params"]["n_frames"])
        D = int(man["D"])
        pos = np.asarray(rh["pos_per_plane"])                           # (D, F) true positions
        # frame-wise breathing of build_af_bundle.simulate
        u, amp, r0, n, _ = afb.breathing_model(man)
        planes = np.repeat(np.arange(D), F_)
        frames = np.tile(np.arange(F_), D)
        r = (r0[planes] + frames * afb.DT / afb.T_BREATH) % 1.0
        disp = u[None] * amp * np.array([float(lujan_displacement(float(x), 1.0, n=n)) for x in r])[:, None]
        mine = fd.render_frames(gt.float().to(dev), planes, pos.ravel(), disp, man["dz_mm"], 518)
        stack = np.stack([np.asarray(__import__("nibabel").load(
            os.path.join(d, "breath", f"stack_t{f:02d}.nii.gz")).dataobj, np.float32).transpose(2, 1, 0)
            for f in range(F_)])                                         # (F, D, H, W)
        # The exact eval path (run_vggt._extract -> gpu_augment_batch, aug off): native 256 slice,
        # clamped to [0,1], then _resize_to_model_res.
        native = extract_slices_from_phases(torch.from_numpy(stack)[None].to(dev),
                                            torch.as_tensor(frames, device=dev)[None],
                                            torch.as_tensor(planes, device=dev)[None],
                                            out_size=stack.shape[-1])[..., 0] / 255.0
        ref = _resize_to_model_res(native, 518)[0]
        ierr = (mine - ref).abs()
        print(f"{man['subject']}: pixel max|err| {ierr.max():.2e} "
              f"mean {ierr.mean():.2e}  (images in [0,1], {len(planes)} frames)")


if __name__ == "__main__":
    main()
