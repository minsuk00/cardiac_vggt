#!/usr/bin/env python
"""Cross-method breathing-motion EPE table on the COMMON slice set: only (subject, plane) slices that
EVERY listed method scored, so all rows share one set of slices, one truth, and one predict-nothing
baseline. Each method still drops its own slices (VGGT: blank inputs; CiNeVol / NeSVoR: planes
without heart-mask content; NiftyMIC: rejected slices; SVRTK: missing .dof), so per-method tables
are scored on slightly different sets.

Inputs are the `--slices` dumps of the per-method scripts (vggt.py --slices-dir, cinevol.py,
fetal_cmr_4d.py, niftymic.py, svrtk3d.py, nesvor.py, dangi.py): {"method", "axes", "slices":
{subject_id: {z: {"applied": [...], "pred": [...]}}}}, pred already carrying the method's sign.
The truth is recomputed independently by each script; it is checked to agree across methods on
every common slice (a disagreement means two scripts scored different slices and aborts).

Per axis, the common set is intersected over the methods that HAVE that axis (dz rows need
through-plane methods; Dangi only enters dy/dx). Error is demeaned per subject, as in common.score.

Usage:  python evaluation/src/analysis/motion_epe/common_set.py <slices.json> [<slices.json> ...]
            [--json temp/af12_common_set.json]
"""
import argparse
import json

import numpy as np

AXES = ("dz", "dy", "dx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slices", nargs="+")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dumps = [json.load(open(p)) for p in a.slices]
    out = {}
    for ax in AXES:
        ms = [d for d in dumps if ax in d["axes"]]
        if not ms:
            continue
        keys = set.intersection(*[{(s, z) for s, zs in d["slices"].items() for z in zs} for d in ms])
        subj = sorted({s for s, _ in keys})
        truth = {}
        for d in ms:
            i = d["axes"].index(ax)
            for s, z in keys:
                t = d["slices"][s][z]["applied"][i]
                if (s, z) in truth and abs(truth[(s, z)] - t) > 1e-3:
                    raise SystemExit(f"{ax} truth disagrees on {s} z={z}: {truth[(s, z)]} vs {t} ({d['method']})")
                truth[(s, z)] = t
        print(f"\n=== {ax}: {len(keys)} common slices, {len(subj)} subjects, {len(ms)} methods ===")
        print(f"{'method':38}{'EPE demeaned':>14}{'raw EPE':>10}{'corr':>7}")
        rows = {}
        preds = {d["method"]: {k: d["slices"][k[0]][k[1]]["pred"][d["axes"].index(ax)] for k in keys} for d in ms}
        preds["predict_nothing"] = {k: 0.0 for k in keys}
        planes = {s: sorted((z for s2, z in keys if s2 == s), key=int) for s in subj}
        for name, pr in preds.items():
            dm, raw, P, A = [], [], [], []
            for s in subj:
                if len(planes[s]) < 2:
                    continue
                p = np.array([pr[(s, z)] for z in planes[s]]); t = np.array([truth[(s, z)] for z in planes[s]])
                e = p - t
                dm.append(np.abs(e - e.mean()).mean()); raw.append(np.abs(e).mean()); P += list(p); A += list(t)
            c = float(np.corrcoef(A, P)[0, 1]) if np.std(P) > 1e-9 else 0.0
            rows[name] = {"epe_demeaned": float(np.mean(dm)), "epe": float(np.mean(raw)), "corr": c,
                          "n_subjects": len(dm), "n_slices": len(keys)}
            print(f"{name:38}{rows[name]['epe_demeaned']:14.2f}{rows[name]['epe']:10.2f}{c:7.2f}")
        out[ax] = rows
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
