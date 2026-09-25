#!/usr/bin/env python
"""Per-subject stacked-input vs ours PNGs from the volumes saved by tools/run_vggt_rt_stacked.py.

One PNG per subject (<out>/<subject>.png), one row per saved frame: stacked input | ours, each as
SAX (LV-centre plane) + x-z + y-z cuts through the LV centre, base at top. No model run.

    PYTHONPATH=training:. python tools/render_rt_stacked_vs_ours.py --dir <run_vggt_rt_stacked out>
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation", "src", "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import run_vggt_rt as rt                                  # noqa: E402
from render_rt_orthoviews import lv_centre                # noqa: E402
from run_vggt_rt_stacked import cuts                      # noqa: E402


def load_dhw(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, np.float32).transpose(3, 2, 1, 0)[0], float(img.header.get_zooms()[2])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    heads = ["stacked input\nSAX", "stacked input\nx-z cut", "stacked input\ny-z cut",
             "ours\nSAX", "ours\nx-z cut", "ours\ny-z cut"]
    for sdir in sorted(d for d in glob.glob(os.path.join(args.dir, "MIITT_*")) if os.path.isdir(d)):
        subj = os.path.basename(sdir)
        fdirs = sorted(glob.glob(os.path.join(sdir, "frame_*")))
        seg = os.path.join(rt.RT_ROOT, subj.replace("MIITT_", ""), "realtime/sax/heart_seg.nii.gz")
        x, y, zc = lv_centre(seg)
        vols = [(os.path.basename(fd), *load_dhw(os.path.join(fd, "input.nii.gz")),
                 load_dhw(os.path.join(fd, "ours.nii.gz"))[0]) for fd in fdirs]
        vmax = float(np.percentile(np.concatenate([v[1].ravel() for v in vols]), 99.5))
        fig, ax = plt.subplots(len(vols), 6, figsize=(6 * 2.4, len(vols) * 2.5), squeeze=False)
        for i, (name, inp, dz, ours) in enumerate(vols):
            zs = dz / rt.rv.INPLANE_MM
            for j, img in enumerate(cuts(inp, x, y, zc) + cuts(ours, x, y, zc)):
                ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=1 if j % 3 == 0 else zs,
                                interpolation="nearest")
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(heads[j], fontsize=10)
            ax[i, 0].set_ylabel(name.replace("_", " "), fontsize=10)
        fig.suptitle(f"{subj}: naive stack of frame f of every slice vs ours (SAX = z{zc})", fontsize=11)
        plt.tight_layout()
        out = os.path.join(args.dir, f"{subj}.png")
        plt.savefig(out, dpi=110); plt.close(fig)
        print(f"-> {out}")


if __name__ == "__main__":
    main()
