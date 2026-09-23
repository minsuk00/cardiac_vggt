#!/usr/bin/env python
"""Breathing-motion EPE for SVRTK 3D, from its own per-slice rigid registration transforms, against
the TRUE applied shift, on the SAME D slices VGGT was fed (common.py: scatter/stack_t00).

run_svrtk3d.sh (`-debug`, `DEBUG=1` re-run only -- default runs never write these) keeps MIRTK's
per-phase per-slice `.dof` files: recon_breath/transforms_t{t:02d}/transformation{z}.dof, one per
INPUT slice z (verified: file count == D, in the stack's own native slice order, same order every
other file in this codebase uses). `global-transformation0.dof` in the same dir is SVRTK's one
whole-stack correction and is NOT read here. Only the t00 run is scored (see niftymic.py: its input
IS scatter/stack_t00; t01..t11 re-estimate the same non-reference slices). Requires input_stack ==
scatter.

`.dof` is MIRTK's small binary rigid-transform format -- reverse-engineered from real files (no
`mirtk`/singularity dependency needed to read it): big-endian header of 3 uint32 (magic, transform
type, parameter count = 6 for rigid), then 6 big-endian float64 = (tx, ty, tz [mm], rx, ry, rz
[deg]). Confirmed stable across files/subjects (same magic/type/count, plausible mm/deg-scale
values, e.g. tz ~= 7.9 mm on a real mid-slice). The raw translation is about the world ORIGIN, so it
is converted to the slice's actual shift at its heart-mask centroid, R p + t - p
(common.mirtk_centroid_disp, convention verified against reconstructCardiac's MeanDisplacement);
world Z/Y/X = true dz/dy/dx (axis-aligned stack, common.py).

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/svrtk3d.py af12
            [--arm-name svrtk3d_debug_scatter] [--json temp/rhythm24_ef/af12_resp_epe.json]
"""
import argparse
import os
import struct
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                              # noqa: E402


def read_dof_rigid(path):
    """(tx,ty,tz,rx,ry,rz) from a MIRTK rigid .dof -- big-endian [magic,type,nparam]:3xu32 then
    nparam x f64. Refuses anything that isn't a 6-parameter rigid transform (never silently
    misreads an affine/global .dof with a different parameter count)."""
    data = open(path, "rb").read()
    magic, ttype, nparam = struct.unpack(">III", data[:12])
    if nparam != 6:
        raise ValueError(f"{path}: expected a 6-param rigid .dof, got nparam={nparam} (type={ttype})")
    return np.array(struct.unpack(">6d", data[12:12 + 48]), float)


def pred_t00(recon_dir, mask_path, D):
    """(D, 3) world (Z, Y, X) mm shift of each t00 slice's heart-mask centroid under its .dof pose
    (common.mirtk_centroid_disp), NaN where a .dof is missing/unreadable."""
    mk = nib.load(mask_path)
    M = np.asarray(mk.dataobj) > 0
    out = np.full((D, 3), np.nan)
    for z in range(D):
        p = os.path.join(recon_dir, "transforms_t00", f"transformation{z}.dof")
        if os.path.isfile(p):
            try:
                out[z] = common.mirtk_centroid_disp(read_dof_rigid(p), common.mask_centroid_world(M, mk.affine, z))[::-1]
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="svrtk3d_debug_scatter")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()

    subs = common.subjects(a.arm, os.path.join(a.arm_name, "recon_breath", "transforms_t00", "transformation0.dof"))
    if not subs:
        sys.exit(f"no subject has {a.arm_name}/recon_breath/transforms_t00/transformation0.dof for arm {a.arm}")
    per_subj = {}
    for sid, sd, man in subs:
        rd = os.path.join(sd, a.arm_name, "recon_breath")
        common.check_scatter_input(rd)
        A, P = common.applied(man, common.slots(man)), pred_t00(rd, os.path.join(sd, "mask_heart.nii.gz"), man["D"])
        ok = ~np.isnan(P).any(1)
        per_subj[sid] = (A[ok], P[ok], np.arange(man["D"])[ok])
    out, _, slices = common.score(per_subj, "svrtk3d", ["centroid_disp_Z", "centroid_disp_Y", "centroid_disp_X"])
    if a.slices:
        common.write_slices(a.slices, "svrtk3d", common.AXES, slices)
    if a.json:
        common.write_json(a.json, out)


if __name__ == "__main__":
    main()
