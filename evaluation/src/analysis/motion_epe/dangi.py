#!/usr/bin/env python
"""In-plane motion EPE for the Dangi centring baseline, from its own per-plane shift, against the
TRUE applied shift, on the SAME D slices VGGT was fed (common.py: scatter/stack_t00).

Dangi (evaluation/src/engine/run_dangi.py) is a rigid IN-PLANE-ONLY centring method by design (its
own docstring: "rigid in-plane translation cannot represent... through-plane motion" -- it never
attempts dz, so this script has NO dz row). It writes recon_breath/centres.json = {"t{k:02d}":
{"centres_px": [...], "shifts_mm": [[sx,sy], ...]}}, one (sx, sy) per plane z per target phase k,
already in mm, in the native stack's in-plane (X,Y) axes, so (sy, sx) = true (dy, dx). Only "t00" is scored (see niftymic.py: its input IS scatter/stack_t00). Requires
input_stack == scatter.

Output keys "dangi_dy"/"dangi_dx" only (no "dangi" alias -- there is no through-plane number).

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/dangi.py af12
            [--arm-name dangi_scatter] [--json temp/rhythm24_ef/af12_resp_epe.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common                                                              # noqa: E402


def pred_t00(centres_json, D):
    """(D, 2) Dangi's own (sx, sy) shifts_mm per plane at t00, NaN where absent."""
    c = json.load(open(centres_json))
    out = np.full((D, 2), np.nan)
    if "t00" in c:
        s = np.asarray(c["t00"]["shifts_mm"], float)[:, ::-1]   # (sy, sx) = true (dy, dx)
        out[:min(D, len(s))] = s[:D]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arm-name", default="dangi_scatter")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slices", default=None, help="per-slice dump for common_set.py")
    a = ap.parse_args()

    subs = common.subjects(a.arm, os.path.join(a.arm_name, "recon_breath", "centres.json"))
    if not subs:
        sys.exit(f"no subject has {a.arm_name}/recon_breath/centres.json for arm {a.arm}")
    per_subj = {}
    for sid, sd, man in subs:
        rd = os.path.join(sd, a.arm_name, "recon_breath")
        common.check_scatter_input(rd)
        A, P = common.applied(man, common.slots(man))[:, 1:], pred_t00(os.path.join(rd, "centres.json"), man["D"])
        ok = ~np.isnan(P).any(1)
        per_subj[sid] = (A[ok], P[ok], np.arange(man["D"])[ok])
    out, _, slices = common.score(per_subj, "dangi", ["sy", "sx"], axes=("dy", "dx"))
    if a.slices:
        common.write_slices(a.slices, "dangi", ("dy", "dx"), slices)
    if a.json:
        common.write_json(a.json, out)


if __name__ == "__main__":
    main()
