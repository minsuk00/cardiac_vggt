"""Summarise E2 JSONs (breathing pathway, dose-response, z-token lie, common-mode shift)."""
import glob
import json
import sys

import numpy as np

DELTAS = [-24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24]


def epe_slope(appl, pred):
    """Companions only (slot>=1). EPE after per-subject demeaning (paper convention) + slope/corr."""
    a, p = appl[:, 1:].ravel(), pred[:, 1:].ravel()
    ok = ~np.isnan(p)
    a, p = a[ok], p[ok]
    e = p - a
    e = e - e.mean()
    sl = np.polyfit(a, p, 1)[0] if a.std() > 1e-3 else np.nan
    r = np.corrcoef(a, p)[0, 1] if a.std() > 1e-3 and p.std() > 1e-6 else np.nan
    return np.mean(np.abs(e)), sl, r


def main(d):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s*.json"))]
    print(f"n subjects = {len(recs)}")
    conds = list(recs[0]["A"][0]["pred"])
    print("A. pathway knockouts — through-plane (dz) and in-plane AP (dy) recovery, companions")
    print(f"{'cond':<14}{'EPEz':>8}{'slopez':>8}{'rz':>8}{'EPEy':>8}{'slopey':>8}")
    for c in conds:
        res = []
        for r in recs:
            appl = np.array([row["disp"] for row in r["A"]])            # (K,S,3)
            pred = np.array([row["pred"][c] for row in r["A"]])        # (K,S,3)
            ez = epe_slope(appl[..., 0], pred[..., 0])
            ey = epe_slope(appl[..., 1], pred[..., 1])
            res.append(ez + ey[:2])
        res = np.array(res)
        print(f"{c:<14}" + "".join(f"{v:8.3f}" for v in np.nanmean(res, 0)))
    print("\nB. single-slot content shift δz (others unshifted): mean predicted dz of probe slot")
    for c in recs[0]["B"]:
        ys = np.array([[row[r["probe_slot"]][0] for row in r["B"][c]] for r in recs])
        others = np.array([[np.nanmean(np.abs(np.delete(np.array(row)[:, 0], r["probe_slot"]))) for row in r["B"][c]] for r in recs])
        print(f"  {c:<10} probe: " + " ".join(f"{v:6.1f}" for v in ys.mean(0)) + f"   | slope {np.polyfit(DELTAS, ys.mean(0), 1)[0]:.3f}")
        print(f"  {'':<10} |others|: " + " ".join(f"{v:5.1f}" for v in others.mean(0)))
    print("\nC. z-token lie δ on probe slot (image unchanged): predicted dz")
    ys = np.array([[row[r["probe_slot"]][0] for row in r["C"]["full"]] for r in recs])
    others = np.array([[np.nanmean(np.abs(np.delete(np.array(row)[:, 0], r["probe_slot"]))) for row in r["C"]["full"]] for r in recs])
    print("  probe:   " + " ".join(f"{v:6.1f}" for v in ys.mean(0)) + f"   | slope {np.polyfit(DELTAS, ys.mean(0), 1)[0]:.3f}")
    print("  |others|: " + " ".join(f"{v:5.1f}" for v in others.mean(0)))
    print("\nD. common-mode shift δ: mean predicted dz over companions")
    for v in recs[0]["D_"]:
        ys = np.array([[np.nanmean(np.array(row)[1:, 0]) for row in r["D_"][v]] for r in recs])
        ref = np.array([[np.array(row)[0, 0] for row in r["D_"][v]] for r in recs])
        print(f"  {v:<12} comp: " + " ".join(f"{x:6.1f}" for x in ys.mean(0)) + f"   | slope {np.polyfit(DELTAS, ys.mean(0), 1)[0]:.3f}")
        print(f"  {'':<12} ref:  " + " ".join(f"{x:6.1f}" for x in ref.mean(0)))
    print("δ:", DELTAS)


if __name__ == "__main__":
    main(sys.argv[1])
