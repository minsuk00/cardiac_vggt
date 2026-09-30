"""E4 scoring: LV (label 1) volume-time curves from nnU-Net Task114 segs, on (a) all planes and (b)
companion planes only (the reference plane excluded, as in the paper's real-data Fig.).
Per arm: VTC err (mean |V_arm(t) - V_gt(t)| / EDV_gt, %), EF abs err (pp), and 'swing ratio'
(max-min of arm curve / max-min of GT curve)."""
import glob
import json
import sys

import nibabel as nib
import numpy as np

ARMS = ["full", "ko_ref_L3", "self_L12_17", "identity"]


def curve(seg_dir, tag, arm, T, planes=None, lab=1):
    out = []
    for t in range(T):
        s = np.asarray(nib.load(f"{seg_dir}/{tag}__{arm}__t{t:02d}.nii.gz").dataobj)   # (X,Y,Z)
        if planes is not None:
            s = s[..., planes]
        out.append(float((s == lab).sum()))
    return np.array(out)


def metrics(c, g):
    edv = g.max()
    vtc = np.mean(np.abs(c - g)) / edv * 100
    ef = lambda x: (x.max() - x.min()) / x.max() * 100 if x.max() > 0 else np.nan
    swing = (c.max() - c.min()) / max(g.max() - g.min(), 1e-9)
    r = np.corrcoef(c, g)[0, 1] if c.std() > 0 else np.nan
    return vtc, abs(ef(c) - ef(g)), swing, r


def main(root):
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/meta/*.json"))]
    metas = [m for m in metas if m]
    scopes = ("all", "comp", "far")
    res = {a: {s: [] for s in scopes} for a in ARMS}
    per = []
    for m in metas:
        tag, T = f"s{m['idx']:03d}", m["T"]
        planes = {"all": None,
                  "comp": [z for z in range(m["D"]) if z != m["ref_plane"]],
                  # far: beyond the reference slice's own trilinear-splat leakage (|z - z_ref| >= 2)
                  "far": [z for z in range(m["D"]) if abs(z - m["ref_plane"]) >= 2]}
        try:
            g = {s: curve(f"{root}/seg", tag, "gt", T, planes[s]) for s in scopes}
        except FileNotFoundError:
            continue
        row = {"idx": m["idx"]}
        for a in ARMS:
            c = {s: curve(f"{root}/seg", tag, a, T, planes[s]) for s in scopes}
            for s in scopes:
                res[a][s].append(metrics(c[s], g[s]))
            row[a] = {s: c[s].tolist() for s in scopes}
        row["gt"] = {s: g[s].tolist() for s in scopes}
        per.append(row)
    print(f"n = {len(per)}")
    for scope in scopes:
        print(f"[{scope} planes]  arm            VTC%    EFerr   swing    r")
        for a in ARMS:
            v = np.array(res[a][scope])
            print(f"   {a:<14} {np.nanmean(v[:,0]):7.2f} {np.nanmean(v[:,1]):7.2f} {np.nanmedian(v[:,2]):7.2f} {np.nanmedian(v[:,3]):6.2f}")
    json.dump({"per_subject": per, "summary": {a: {s: np.array(res[a][s]).tolist() for s in res[a]} for a in ARMS}},
              open(f"{root}/e4_curves.json", "w"))


if __name__ == "__main__":
    main(sys.argv[1])
