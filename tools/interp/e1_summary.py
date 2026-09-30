"""Summarise E1 JSONs: phase tracking on companion planes + breathing recovery, per condition."""
import glob
import json
import sys

import numpy as np


def circ(a, b, T=12):
    d = abs(a - b) % T
    return min(d, T - d)


def per_subject(rec, cond):
    rows = rec["cond"][cond]
    T = rec["T"]
    M = np.array([r["ncc_comp"] for r in rows])          # (t_target, t_gt)
    Mr = np.array([r["ncc_ref"] for r in rows])
    diag = np.mean(np.diag(M))
    off = (M.sum() - np.trace(M)) / (T * T - T)
    # decode: argmax over GT phase for each target
    dec = np.mean([circ(int(np.argmax(M[t])), t, T) <= 1 for t in range(T)])
    # "contrast": does V_t match GT_t better than it matches GT at other targets (column-normalised)
    Mc = M - M.mean(0, keepdims=True)
    dec_c = np.mean([circ(int(np.argmax(Mc[:, t])), t, T) <= 1 for t in range(T)])
    appl = np.array(rec["disp"])[:, 0]
    dz = np.array([r["disp_fov"] for r in rows])[:, :, 0]   # (T,S)
    err = dz - appl[None]
    err = err - np.nanmean(err)                             # demean per subject (paper EPE)
    epe = np.nanmean(np.abs(err[:, 1:]))                    # companions
    slope = np.polyfit(appl[1:], np.nanmean(dz, 0)[1:], 1)[0] if appl[1:].std() > 1e-3 else np.nan
    obs = np.nanmean([np.nanmean(r["obs"][1:]) for r in rows])
    return dict(diag=diag, off=off, gap=diag - off, dec=dec, dec_c=dec_c, ref_gap=np.mean(np.diag(Mr)) - (Mr.sum() - np.trace(Mr)) / (T * T - T),
                epe=epe, slope=slope, obs=obs)


def main(d):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{d}/s*.json"))]
    recs = [r for r in recs if r["cond"]]
    conds = list(recs[0]["cond"])
    print(f"n subjects = {len(recs)}")
    keys = ["diag", "off", "gap", "dec", "dec_c", "ref_gap", "epe", "slope", "obs"]
    print(f"{'cond':<16}" + "".join(f"{k:>9}" for k in keys))
    for c in conds:
        vals = [per_subject(r, c) for r in recs if c in r["cond"]]
        print(f"{c:<16}" + "".join(f"{np.nanmean([v[k] for v in vals]):9.4f}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1])
