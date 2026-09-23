#!/usr/bin/env python
"""Breathing-motion EPE for NeSVoR, from its own per-slice estimated pose, against the TRUE applied
shift, on the SAME D slices VGGT was fed (common.py: scatter/stack_t00).

run_nesvor.sh (`--output-slices`) writes recon_breath/slices_t{t:02d}/{i}.nii.gz: one (W,H,1) NIfTI per
input slice i of stack_t{t} (mask_{i}.nii.gz alongside). NeSVoR drops planes whose mask (the
--stack-masks heart mask AND intensity > --background-threshold 0) is empty and numbers the rest
0.. in stack order, so slice i = the i-th kept plane (rule verified: kept-plane count == slice count
on all 180 plain-cohort subjects). Each slice's affine
IS NeSVoR's estimated 6-DOF pose (nesvor transformation2affine; its 3rd column is scaled by the slice
THICKNESS, not the pitch, so only the in-plane columns + translation are comparable to the stack's).
There is no separate transform file, so the predicted shift is a reference-affine DIFF:

    pred[z] = A_slice_z @ c  -  A_stack @ (c + z*e_z)       c = plane z's heart-mask centroid voxel (i, j, 0)

i.e. where NeSVoR put that point minus where the input stack says it is, in world mm, expressed
along the stack's own (unit) voxel axes (sx, sy, sz), so (sz, sy, sx) = true (dz, dy, dx). Same
measurement point as niftymic.py / svrtk3d.py (heart-mask centroid).
Only the t00 run is scored (see niftymic.py: its input IS scatter/stack_t00). A subject whose slice
count != the keep rule's is skipped (index i could not be trusted). Requires input_stack == scatter.

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/nesvor.py af12
            [--arm-name nesvor_scatter] [--json temp/rhythm24_ef/af12_resp_epe.json]
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                              # noqa: E402


def pred_t00(recon_dir, stack_path, mask_path, D):
    """(D, 3) NeSVoR slice-centre displacement (mm) along the stack's unit voxel axes, NaN for planes
    NeSVoR dropped, or None if its slice count does not match the keep rule."""
    fs = [f for f in glob.glob(os.path.join(recon_dir, "slices_t00", "*.nii.gz"))
          if not os.path.basename(f).startswith("mask_")]
    st = nib.load(stack_path)
    s, m = np.asarray(st.dataobj), np.asarray(nib.load(mask_path).dataobj) > 0
    keep = [z for z in range(D) if (m[..., z] & (s[..., z] > 0)).any()]
    if len(fs) != len(keep):
        return None
    A0 = st.affine
    axes = A0[:3, :3] / np.linalg.norm(A0[:3, :3], axis=0)          # unit stack axes (columns)
    out = np.full((D, 3), np.nan)
    for i, z in enumerate(keep):
        im = nib.load(os.path.join(recon_dir, "slices_t00", f"{i}.nii.gz"))
        # heart-mask centroid of plane z (slice and stack share the in-plane grid) -- the same point
        # niftymic.py / svrtk3d.py measure at, so a tilt about a different point cannot differ
        c = np.r_[np.argwhere(m[..., z]).mean(0), 0.0, 1.0]
        disp = (im.affine @ c)[:3] - (A0 @ (c + [0, 0, z, 0]))[:3]
        out[z] = (axes.T @ disp)[::-1]                              # (sz, sy, sx) = true (dz, dy, dx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="nesvor_scatter")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()

    subs = common.subjects(a.arm, os.path.join(a.arm_name, "recon_breath", "slices_t00", "0.nii.gz"))
    if not subs:
        sys.exit(f"no subject has {a.arm_name}/recon_breath/slices_t00 for arm {a.arm}")
    per_subj, skipped = {}, []
    for sid, sd, man in subs:
        rd = os.path.join(sd, a.arm_name, "recon_breath")
        common.check_scatter_input(rd)
        P = pred_t00(rd, os.path.join(sd, "scatter", "stack_t00.nii.gz"), os.path.join(sd, "mask_heart.nii.gz"), man["D"])
        if P is None:
            skipped.append(sid)
            continue
        ok = ~np.isnan(P).any(1)
        per_subj[sid] = (common.applied(man, common.slots(man))[ok], P[ok], np.arange(man["D"])[ok])
    if skipped:
        print(f"skipped {len(skipped)} subjects (slice count != D): {skipped}")
    out, _, slices = common.score(per_subj, "nesvor", ["sz", "sy", "sx"])
    if a.slices:
        common.write_slices(a.slices, "nesvor", common.AXES, slices)
    if a.json:
        common.write_json(a.json, out)


if __name__ == "__main__":
    main()
