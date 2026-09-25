#!/usr/bin/env python
"""LV volume curve over time for a real-time (RT) recon vs the naive RT stack, by 2D nnU-Net.

Both 4D inputs are on the same canonical grid (run_vggt_rt.py outputs): the naive stack
`rt_input.nii.gz` (frame k of every slice, no correction) and ours `recon_rt.nii.gz`. Each frame is
segmented with nnU-Net Task114 **2D** (slice by slice), and LV volume = LV-label (1) pixel count x
voxel volume (disc summation, the clinical SAX method). 2D on purpose: per-slice segmentation and
slice summation are unaffected by inter-slice misalignment, so the naive stack is not penalised
by a 3D segmenter seeing an out-of-distribution misaligned stack.

Stages (nnU-Net lives in the isolated `nnunet` env, so it is a separate call):
    dump  : write every frame as a 3D nnU-Net input into <work>/in_s<k> (k = shard, for multi-GPU)
    seg   : (shell, per shard)  micromamba run -n nnunet bash -c "source <env.sh> && nnUNet_predict \
              -i <work>/in_s<k> -o <work>/seg -t 114 -m 2d -tr nnUNetTrainerV2_MMS"
    curve : LV volume per frame -> <work>/lv_curves.json + <work>/lv_curves.png

    PYTHONPATH=training:. python tools/rt_lv_curve_2d.py dump --work <dir> --shards 3 \
        --case afib_naive=<subj>/rt_input.nii.gz --case afib_ours=<subj>/<arm>/recon_rt.nii.gz
    PYTHONPATH=training:. python tools/rt_lv_curve_2d.py curve --work <dir> --pairs afib volunteer1
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

FRAME_MS = 25.0


def dump(args):
    n = 0
    for spec in args.case:
        tag, path = spec.split("=", 1)
        img = nib.load(path)
        a = np.asarray(img.dataobj, np.float32)                  # (X, Y, Z, F)
        for k in range(a.shape[3]):
            sd = os.path.join(args.work, f"in_s{n % args.shards}")
            os.makedirs(sd, exist_ok=True)
            nib.save(nib.Nifti1Image(a[..., k], img.affine), os.path.join(sd, f"{tag}_f{k:03d}_0000.nii.gz"))
            n += 1
        print(f"{tag}: {a.shape[3]} frames from {path}")
    print(f"dumped {n} volumes into {args.shards} shards under {args.work}")


def curve(args):
    seg_dir = os.path.join(args.work, "seg")
    out = {}
    for tag in sorted({os.path.basename(p).rsplit("_f", 1)[0] for p in glob.glob(os.path.join(seg_dir, "*_f*.nii.gz"))}):
        files = sorted(glob.glob(os.path.join(seg_dir, f"{tag}_f*.nii.gz")))
        vox_ml = float(np.prod(nib.load(files[0]).header.get_zooms()[:3])) / 1000.0
        out[tag] = [float((np.asarray(nib.load(f).dataobj) == 1).sum() * vox_ml) for f in files]
        print(f"{tag}: {len(files)} frames, LV {min(out[tag]):.0f}-{max(out[tag]):.0f} mL")
    json.dump(out, open(os.path.join(args.work, "lv_curves.json"), "w"), indent=1)

    fig, ax = plt.subplots(len(args.pairs), 1, figsize=(10, 2.8 * len(args.pairs)), squeeze=False)
    for a, name in zip(ax[:, 0], args.pairs):
        for kind, style in (("naive", dict(color="0.5", lw=1.2, label="naive stack")),
                            ("ours", dict(color="C3", lw=1.8, label="ours"))):
            v = out.get(f"{name}_{kind}")
            if v is not None:
                a.plot(np.arange(len(v)) * FRAME_MS / 1000, v, **style)
        a.set_title(name, fontsize=10); a.set_ylabel("LV volume (mL)"); a.legend(fontsize=8, loc="upper right")
    ax[-1, 0].set_xlabel("time (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.work, "lv_curves.png"), dpi=120)
    print(f"-> {os.path.join(args.work, 'lv_curves.png')}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--work", required=True)
    d.add_argument("--case", nargs="+", required=True, help="tag=path/to/4d.nii.gz (tag = <name>_naive|<name>_ours)")
    d.add_argument("--shards", type=int, default=1)
    c = sub.add_parser("curve")
    c.add_argument("--work", required=True)
    c.add_argument("--pairs", nargs="+", required=True, help="names; plots <name>_naive vs <name>_ours")
    args = ap.parse_args()
    {"dump": dump, "curve": curve}[args.stage](args)


if __name__ == "__main__":
    main()
