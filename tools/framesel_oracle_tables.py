#!/usr/bin/env python
"""Oracle per-slice frame selection tables for `run_vggt.py --input sel` (docs/128).

For every rhythm-arm bundle, plane z and queried frame f, pick the frame of plane z whose TRUE
cardiac position (manifest rhythm.pos_per_plane) is circularly closest to the reference slice's
position at frame f (rhythm.ref_pos). This is the ceiling any learned selector can reach.

    PYTHONPATH=training:. python tools/framesel_oracle_tables.py --cohorts cmrx2024_af12 ... \
        --out scratch/framesel/sel_tables/oracle
Writes <out>/<cohort>/<subject>.json = {"sel": (frames, D), "err": (frames, D) phase error}.
"""
import argparse
import glob
import json
import os

import numpy as np


def oracle_sel(man):
    T = int(man["n_cardphase"])
    pos = np.asarray(man["rhythm"]["pos_per_plane"], dtype=np.float64)    # (D, frames)
    ref = np.asarray(man["rhythm"]["ref_pos"], dtype=np.float64)          # (frames,)
    d = np.abs(pos[None, :, :] - ref[:, None, None]) % T                  # (frames, D, frames)
    d = np.minimum(d, T - d)
    return d.argmin(-1), d.min(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", required=True)
    ap.add_argument("--eval-root", default="scratch/eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for c in args.cohorts:
        errs, n = [], 0
        os.makedirs(os.path.join(args.out, c), exist_ok=True)
        for mp in sorted(glob.glob(os.path.join(args.eval_root, c, "out", "*", "manifest.json"))):
            man = json.load(open(mp))
            sel, err = oracle_sel(man)
            ref = int(man["rhythm"]["ref_plane"])
            errs.append(np.delete(err, ref, axis=1).ravel())   # companions only
            json.dump({"sel": sel.tolist(), "err": err.round(4).tolist(), "selector": "oracle"},
                      open(os.path.join(args.out, c, f"{man['subject']}.json"), "w"))
            n += 1
        e = np.concatenate(errs)
        print(f"{c}: {n} subjects  companion phase err mean {e.mean():.3f}  <=1: {(e <= 1).mean():.1%}")


if __name__ == "__main__":
    main()
