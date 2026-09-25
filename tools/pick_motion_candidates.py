#!/usr/bin/env python
"""Candidate (subject, plane) list for tools/sweep_input_frame.py (ED/ES motion figure).

Ranks test subjects by the smaller of their AF and HRV ED-ES target LV volume change (GT LV curve of the
scored arm; more contraction -> more visible motion) and, per subject, takes the --planes non-reference planes
with the largest LV cross-section (GT seg at target frame 0). Prints one 'source:subject:z' per line.

Usage: micromamba run -n svr python tools/pick_motion_candidates.py --n 40 --planes 2 > candidates.txt
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

EV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
VOL, RES = f"{EV}/volumes", f"{EV}/metric_results/test"
ARM = "vggt_final518_diff1000_ep300"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of subjects")
    ap.add_argument("--planes", type=int, default=2, help="planes per subject")
    a = ap.parse_args()
    rng = {}
    for rh in ("af12", "hrv12"):
        for f in glob.glob(f"{RES}/*_{rh}/ef/{ARM}.json"):
            src = f.split("/")[-3][:-len(rh) - 1]
            for r in json.load(open(f))["per_subject"]:
                c = np.array(r["lv_curve_gt"])
                rng.setdefault((src, r["subject"]), {})[rh] = float(c.max() - c.min())
    ranked = sorted((k for k, v in rng.items() if len(v) == 2), key=lambda k: -min(rng[k].values()))
    for src, sid in ranked[:a.n]:
        sd = f"{VOL}/{src}_af12/out/{sid}"
        ref = json.load(open(f"{sd}/manifest.json"))["scatter"]["ref_plane"]
        seg = nib.load(f"{sd}/seg_gt_3d_fullres/seg_t00.nii.gz").get_fdata()
        area = [(seg[:, :, z] == 1).sum() if z != ref else -1 for z in range(seg.shape[2])]
        for z in np.argsort(area)[::-1][:a.planes]:
            print(f"{src}:{sid}:{z}")


if __name__ == "__main__":
    main()
