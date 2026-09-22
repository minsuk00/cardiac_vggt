"""Verify tools/cinevol_prepare.py's rebuilt beat timeline + state labels against every rhythm bundle,
BEFORE any CiNeVol fit. A timeline off by one frame gives silently wrong labels, and the damage
would read as a CiNeVol failure.

Hard checks (exit 1 on any failure), per subject, every plane:
  A  beat index per frame, rebuilt from rr/roll/BURN/DT      == manifest beat_per_plane
  B  regular/hrv: uniform-stretch position from OUR timeline == manifest pos_per_plane (exact times)
  C  regular:     Feinstein phase                            == true position / T
  D  breath level at frame 0 is proportional to the stored |disp_dhw_mm| with ONE constant per
     subject (the label really is the factor the simulator multiplied the displacement by)
Report only (not a pass/fail): label-vs-true-position mismatch per arm, in gated phases of 12, for
the Feinstein label and for the plain linear label elapsed/RR.

    PYTHONPATH=baselines/cinevol:training:. python tools/cinevol_verify_labels.py
    ... --fault    # inject a one-frame timeline shift: checks A/B/C MUST fail (proves they can)
"""
import argparse
import json
import sys

import numpy as np

import cinevol_prepare as cp
import paths

ARMS = ("regular24", "hrv24", "af24")
SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")


def circ(a, b, T):
    d = np.abs(a - b) % T
    return np.minimum(d, T - d)


def check_subject(man):
    r, P = man["rhythm"], man["rhythm"]["params"]
    T, D = P["T"], man["D"]
    fails = []
    pos_true = np.asarray(r["pos_per_plane"], float) % T
    phi = cp.cardiac_labels(man)
    lin = np.empty_like(phi)
    for z in range(D):
        times, starts, beat = cp.timeline(man, z)
        rr = np.asarray(r["rr_per_plane"][z], float)[beat]
        lin[z] = (times - starts[beat]) / rr
        if not np.array_equal(beat, np.asarray(r["beat_per_plane"][z])):
            fails.append("A")
        if r["rhythm"] in ("regular", "hrv") and circ(lin[z] * T, pos_true[z], T).max() > 1e-6:
            fails.append("B")
    if r["rhythm"] == "regular" and circ(phi * T, pos_true, T).max() > 1e-6:
        fails.append("C")
    psi0 = cp.resp_labels(man)[:, 0]
    mag = np.linalg.norm(np.asarray(man["breath"]["disp_dhw_mm"], float), axis=1)
    ok = psi0 > 1e-3
    if ok.sum() >= 2:
        k = mag[ok] / psi0[ok]
        if (k.max() - k.min()) > 1e-3 * k.mean():
            fails.append("D")
    return sorted(set(fails)), circ(phi * T, pos_true, T), circ(lin * T, pos_true, T)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fault", action="store_true", help="shift the rebuilt timeline by one frame")
    a = ap.parse_args()
    if a.fault:
        real = cp.timeline
        cp.timeline = lambda man, z: (lambda t, s, b: (
            t + man["rhythm"]["params"]["DT"], s,
            np.searchsorted(s, t + man["rhythm"]["params"]["DT"], side="right") - 1))(*real(man, z))
    bad = 0
    for arm in ARMS:
        n, failed, mf, ml = 0, {}, [], []
        for src in SOURCES:
            ds = f"{src}_{arm}"
            keep, _ = paths.filter_by_split(ds, paths.subjects(ds), "test")
            for s in keep:
                man = json.load(open(paths.manifest(ds, s)))
                f, e_f, e_l = check_subject(man)
                n += 1
                mf.append(e_f.ravel()); ml.append(e_l.ravel())
                if f:
                    failed[f"{ds}/{s}"] = f
        mf, ml = np.concatenate(mf), np.concatenate(ml)
        print(f"{arm:10s} subjects {n:3d} | FAILED {len(failed):3d} "
              f"| label-vs-true mismatch in phases/12:  Feinstein mean {mf.mean():.3f} p95 {np.percentile(mf, 95):.3f} "
              f"max {mf.max():.2f}   linear mean {ml.mean():.3f} p95 {np.percentile(ml, 95):.3f} max {ml.max():.2f}")
        for k, v in list(failed.items())[:5]:
            print(f"    {k}: checks {v}")
        bad += len(failed)
    print("FAULT INJECTED: failures above are EXPECTED" if a.fault else ("ALL CHECKS PASS" if not bad else "CHECKS FAILED"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
