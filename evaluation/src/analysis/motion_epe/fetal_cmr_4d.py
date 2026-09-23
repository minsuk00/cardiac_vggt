#!/usr/bin/env python
"""Breathing-motion EPE for Fetal CMR 4D, from MIRTK's own per-frame rigid pose (info.tsv), against
the TRUE applied shift, on the SAME D slices VGGT was fed (common.py: scatter/stack_t00).

MIRTK's `reconstructCardiac` (`run_fetal4d.sh`) writes info.tsv every run: one row per input frame
(InputIndex = z*T + f over the rolled/ stack). We read MeanDisplacementX/Y/Z: SVRTK's own mean over
the slice's pixels of T(p) - p in world mm (CalculateDisplacement), i.e. the slice's actual shift.
NOT TranslationX/Y/Z: MIRTK rotates about the world origin, so a ~1 deg tilt puts several mm into
the raw translation (af12: mean |TranslationZ| 3.77 vs |MeanDisplacementZ| 2.18 mm). The stack is
axis-aligned, so world Z/Y/X = true dz/dy/dx (common.py). Fetal sees all T frames of every plane; per plane z we
read its pose at the one frame VGGT saw (breath/ frame f). On a rhythm cohort rolled/ -> breath/, so
the rolled frame is f; on a plain cohort rolled/stack_t{k}[z] = breath/stack_t{(k - r_z) mod T}[z]
(pooled.add_rolled), so the rolled frame is (f + r_z) mod T.

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/fetal_cmr_4d.py af12
            [--arm-name fetal_cmr_4d] [--json temp/rhythm24_ef/af12_resp_epe.json] [--per-subject out.json]
`--arm <source>` (e.g. cmrx2024) = plain-cohort mode: pools all 5 plain sources (common.subjects).
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                              # noqa: E402

MIRTK_AX = ["MeanDisplacementZ", "MeanDisplacementY", "MeanDisplacementX"]   # = true (dz, dy, dx)


def pred_at(info_tsv, man, sl, slice0_roll):
    """(len(sl), 3) MIRTK MeanDisplacement{Z,Y,X} at each breath/ slot (z, f), NaN where missing or
    excluded (MeanDisplacement = -1). `slice0_roll` = the gate's gate.json["slice0_frame_roll"]:
    fetal4d_gate re-rolls plane 0 of stack4d so its detected ED is frame 0, so stack4d plane-0 frame
    k is rolled frame (k - slice0_roll) mod T, i.e. rolled frame g sits at k = (g + slice0_roll) mod T."""
    T = man["rhythm"]["params"]["n_frames"] if "rhythm" in man else man["T"]
    roll = None if "rhythm" in man else man["rolled"]["roll_per_plane"]
    pose = {int(r["InputIndex"]): [float(r[a]) for a in MIRTK_AX]
            for r in csv.DictReader(open(info_tsv), delimiter="\t") if float(r["MeanDisplacement"]) >= 0}
    out = []
    for z, f in sl:
        g = f if roll is None else (f + roll[z]) % T                    # rolled/ frame
        k = (g + slice0_roll) % T if z == 0 else g                       # stack4d frame
        out.append(pose.get(z * T + k, [np.nan] * 3))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="fetal_cmr_4d")
    ap.add_argument("--json", default=None)
    ap.add_argument("--per-subject", default=None)
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()

    subs = common.subjects(a.arm, os.path.join(a.arm_name, "recon_breath", "info.tsv"))
    if not subs:
        sys.exit(f"no subject has {a.arm_name}/recon_breath/info.tsv for arm {a.arm}")
    per_subj = {}
    for sid, sd, man in subs:
        sl = common.slots(man)
        rd = os.path.join(sd, a.arm_name, "recon_breath")
        gate_arm = json.load(open(os.path.join(rd, "stamp.json"))).get("gate_arm", "fetal_cmr_4d")  # run_fetal4d.sh default
        roll0 = json.load(open(os.path.join(sd, gate_arm, "gate", "gate.json")))["slice0_frame_roll"]
        A, P = common.applied(man, sl), pred_at(os.path.join(rd, "info.tsv"), man, sl, roll0)
        ok = ~np.isnan(P).any(1)
        per_subj[sid] = (A[ok], P[ok], np.array([z for z, _ in sl])[ok])
    out, dump, slices = common.score(per_subj, "fetal_cmr_4d", MIRTK_AX)
    if a.slices:
        common.write_slices(a.slices, "fetal_cmr_4d", common.AXES, slices)
    if a.per_subject:
        json.dump(dump, open(a.per_subject, "w"), indent=1)
        print(f"wrote {a.per_subject}")
    if a.json:
        common.write_json(a.json, out)


if __name__ == "__main__":
    main()
