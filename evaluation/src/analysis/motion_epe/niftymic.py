#!/usr/bin/env python
"""Breathing-motion EPE for NiftyMIC, from its own per-slice ITK registration transforms, against the
TRUE applied shift, on the SAME D slices VGGT was fed (common.py: scatter/stack_t00).

run_niftymic (evaluation/src/engine) writes, per target phase t: recon_breath/transforms_t{t:02d}/
stack_t{t:02d}_slice{z}.tfm -- an ITK "Insight Transform File" (plain text), one per slice that
SURVIVED outlier rejection (missing -> NaN, dropped). Format (Euler3DTransform_double_3_3):
    Parameters: rx ry rz(rad)  tx ty tz(mm)          FixedParameters: rotation centre (0,0,0)
The pose lives in ITK's LPS world and rotates about the ORIGIN (angles up to 0.024 rad on cmrx2024,
i.e. several mm at the heart), so the raw translation is not the slice's shift: it is converted to
the shift of the slice's heart-mask centroid (centroid_disp), back in RAS. The stack is
axis-aligned, so world (Z, Y, X) = true (dz, dy, dx).
Only the t00 run is scored: its input IS scatter/stack_t00, so slice z is VGGT's slot z. The t01..t11
runs see the same non-reference slices again (only the reference plane changes), so scoring them
would give NiftyMIC 12 estimates per slice where VGGT gets one. Requires input_stack == scatter.

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/niftymic.py af12
            [--arm-name niftymic_scatter] [--json temp/rhythm24_ef/af12_resp_epe.json]
"""
import argparse
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                              # noqa: E402


LPS = np.array([-1.0, -1.0, 1.0])     # NIfTI RAS world <-> ITK LPS world


def read_tfm(path):
    """(params (6,) = rx ry rz tx ty tz, fixed (4,) = centre xyz + ComputeZYX) from an ITK .tfm."""
    kv = dict(line.split(":", 1) for line in open(path) if ":" in line)
    return (np.array(kv["Parameters"].split(), float), np.array(kv["FixedParameters"].split(), float))


def centroid_disp(path, p_ras):
    """RAS world mm shift of point p under the Euler3D pose: R (p - c) + c + t - p, in ITK's LPS world.
    Default ZXY order (R = Rz Rx Ry), checked against SimpleITK's TransformPoint."""
    prm, fx = read_tfm(path)
    if fx[3] != 0:
        raise ValueError(f"{path}: ComputeZYX=1 not handled")
    p, c = LPS * p_ras, fx[:3]
    R = Rotation.from_euler("yxz", [prm[1], prm[0], prm[2]])
    return LPS * (R.apply(p - c) + c + prm[3:6] - p)


def pred_t00(recon_dir, mask_path, D):
    """(D, 3) world (Z, Y, X) shift of each t00 slice's heart-mask centroid, NaN where rejected/missing."""
    mk = nib.load(mask_path)
    M = np.asarray(mk.dataobj) > 0
    out = np.full((D, 3), np.nan)
    for z in range(D):
        p = os.path.join(recon_dir, "transforms_t00", f"stack_t00_slice{z}.tfm")
        if os.path.isfile(p):
            out[z] = centroid_disp(p, common.mask_centroid_world(M, mk.affine, z))[::-1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="niftymic_scatter")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()

    subs = common.subjects(a.arm, os.path.join(a.arm_name, "recon_breath", "transforms_t00"))
    if not subs:
        sys.exit(f"no subject has {a.arm_name}/recon_breath/transforms_t00 for arm {a.arm}")
    per_subj = {}
    for sid, sd, man in subs:
        rd = os.path.join(sd, a.arm_name, "recon_breath")
        common.check_scatter_input(rd)
        A, P = common.applied(man, common.slots(man)), pred_t00(rd, os.path.join(sd, "mask_heart.nii.gz"), man["D"])
        ok = ~np.isnan(P).any(1)
        per_subj[sid] = (A[ok], P[ok], np.arange(man["D"])[ok])
    out, _, slices = common.score(per_subj, "niftymic", ["centroid_disp_Z", "centroid_disp_Y", "centroid_disp_X"])
    if a.slices:
        common.write_slices(a.slices, "niftymic", common.AXES, slices)
    if a.json:
        common.write_json(a.json, out)


if __name__ == "__main__":
    main()
