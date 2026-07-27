"""Census every extracted CMRxRecon2025 SAX subject against the two invariants the 2024 recon assumes.

The 2024 recon (`_archive/batch_reconstruct_cmrxrecon2024.py`) silently depends on:
    ny == ReconMatrix_Y          (no phase-encode zero-fill needed; it only ever CROPS)
    nx == 2 * ReconMatrix_X      (the centre-crop removes exactly the 2x readout oversampling)
Both hold for every 2024 and 2023 subject. This measures how often they hold for 2025.

Where they fail the 2024 script does NOT raise — it pads to the top-left corner (`:126-130`)
and/or stamps a wrong voxel spacing. Silent geometric corruption, hence this census.

Usage:
    python tools/scan_cmrx2025_geometry.py
    python tools/scan_cmrx2025_geometry.py --json-out /tmp/census2025.json

Re-run after extraction completes — files that fail to open are counted separately.
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import h5py
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")


def read_kspace_meta(mat):
    """-> (h5key, dtype, (nf, nz, nc, ny, nx), matver).

    2025 ships TWO container formats. Most subjects are MATLAB v7.3 (= HDF5), read lazily by
    h5py, dims already reversed by HDF5 to (nf, nz, nc, ny, nx). But 33 subjects are plain
    **MATLAB v5** ('MATLAB 5.0 MAT-file' magic), which h5py rejects with
    'file signature not found' -- previously mistaken for corruption/incomplete extraction.
    Those are fine, just a different format: `scipy.io.whosmat` reads their shape without
    loading the array, and MATLAB-native order is the REVERSE, (nx, ny, nc, nz, nf), so it
    must be flipped here or every downstream dim is silently transposed.
    """
    with open(mat, "rb") as f:
        is_v5 = f.read(10) == b"MATLAB 5.0"
    if is_v5:
        name, shape, dtype = next(e for e in sio.whosmat(mat) if e[0] in ("kspace", "kspace_full"))
        return name, str(dtype), tuple(int(x) for x in shape[::-1]), "v5"
    with h5py.File(mat, "r") as f:
        key = "kspace" if "kspace" in f else "kspace_full"
        return key, str(f[key].dtype), tuple(int(x) for x in f[key].shape), "v7.3"


def read_csv_normalized(p):
    """Read Parameter,Value csv -> (normalized_dict, geometry_keys_are_suffixed).

    2025 ships two conventions: plain `FOVx` and suffixed `FOVx(mm)`. Strip a trailing
    '(units)' from every key so downstream lookups work for both.

    The flag tracks only the GEOMETRY keys. Every file -- both conventions -- suffixes the
    timing keys (`TR(ms)`, `TE(ms)`, `FlipAngle(degree)`), so an any-key test reports 100%
    suffixed and tells you nothing about the lookups the recon actually performs.
    """
    GEOM = {"FOVx", "FOVy", "SliceThickness", "ReconMatrix_X", "ReconMatrix_Y", "SliceNum"}
    m, suffixed = {}, False
    with open(p) as f:
        r = csv.reader(f)
        next(r)  # 'Parameter,Value'
        for row in r:
            if len(row) == 2:
                k = row[0].strip()
                base = re.sub(r"\(.*\)$", "", k)
                if k.endswith(")") and base in GEOM:
                    suffixed = True
                m[base] = row[1].strip()
    return m, suffixed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="/tmp/cmrx2025_geometry_census.json")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, "**", "cine_sax_info.csv"), recursive=True)):
        mat = os.path.join(os.path.dirname(p), "cine_sax.mat")
        if not os.path.exists(mat):
            continue
        parts = p.split("/")
        rec = {"csv": p, "scanner": parts[-3], "center": parts[-4], "pid": parts[-2]}
        try:
            m, suffixed = read_csv_normalized(p)
            key, dtype, (nf, nz, nc, ny, nx), matver = read_kspace_meta(mat)
            rec.update(
                h5key=key, dtype=dtype, matver=matver, nf=nf, nz=nz, nc=nc, ny=ny, nx=nx,
                ry=int(m["ReconMatrix_Y"]), rx=int(m["ReconMatrix_X"]),
                thick=m.get("SliceThickness", ""), fx=m.get("FOVx", ""), fy=m.get("FOVy", ""),
                suffixed_header=suffixed,
            )
        except Exception as e:
            rec["error"] = str(e)[:100]
        rows.append(rec)

    ok = [r for r in rows if "error" not in r]
    print(f"total={len(rows)} readable={len(ok)} unreadable={len(rows)-len(ok)}")
    if len(rows) != len(ok):
        print("  (unreadable => extraction incomplete/corrupt; re-run when it finishes)")
    print(f"\nh5 keys: {dict(Counter(r['h5key'] for r in ok))}")
    print(f"mat ver: {dict(Counter(r['matver'] for r in ok))}   (v5 => scipy.io, REVERSED dim order)")
    print(f"dtypes : {dict(Counter(r['dtype'] for r in ok))}")
    print(f"ncoil  : {dict(Counter(r['nc'] for r in ok))}")
    print(f"nframe : {dict(Counter(r['nf'] for r in ok))}")

    print(f"\nINVARIANT 1  ny == ReconMatrix_Y")
    print(f"  ny <  ry : {sum(1 for r in ok if r['ny'] < r['ry'])}")
    print(f"  ny == ry : {sum(1 for r in ok if r['ny'] == r['ry'])}")
    print(f"  ny >  ry : {sum(1 for r in ok if r['ny'] > r['ry'])}")
    severe = [r for r in ok if r["ny"] / r["ry"] <= 0.93]
    print(f"  of which SEVERE (ratio <= 0.93, needs 1.5-2.4x zero-fill): {len(severe)}")
    print(f"  minor (0.94-0.99, short by 1-2 rows)                    : {sum(1 for r in ok if 0.94 <= r['ny']/r['ry'] < 1.0)}")

    print(f"\nINVARIANT 2  nx == 2 * ReconMatrix_X")
    print(f"  holds    : {sum(1 for r in ok if r['nx'] == 2*r['rx'])} / {len(ok)}")
    buckets = Counter("~1.0" if r["nx"]/r["rx"] < 1.15 else "~1.33" if r["nx"]/r["rx"] < 1.6 else "~2.0" for r in ok)
    print(f"  ratio buckets: {dict(buckets)}")

    print(f"\nOTHER")
    print(f"  blank SliceThickness : {sum(1 for r in ok if not r['thick'])}")
    print(f"  nslice <= 5          : {sum(1 for r in ok if r['nz'] <= 5)}")
    print(f"  suffixed csv headers : {sum(1 for r in ok if r['suffixed_header'])} (plain: {sum(1 for r in ok if not r['suffixed_header'])})")

    print(f"\nPER-SCANNER  ny<ry / total   [distinct ny/ry values]")
    tot, bad, vals = Counter(), Counter(), defaultdict(set)
    for r in ok:
        tot[r["scanner"]] += 1
        vals[r["scanner"]].add(round(r["ny"]/r["ry"], 2))
        if r["ny"] < r["ry"]:
            bad[r["scanner"]] += 1
    for s in sorted(tot):
        print(f"  {s:24s} {bad[s]:3d}/{tot[s]:3d}   {sorted(vals[s])}")
    print("  ^ ratios varying WITHIN a scanner model => no per-vendor lookup table can work")

    fully = [r for r in ok if r["ny"] == r["ry"] and r["nx"] == 2*r["rx"] and r["thick"] and r["nz"] > 5]
    print(f"\nFULLY 2024-COMPATIBLE (all invariants + thickness + nz>5): {len(fully)} / {len(ok)}")
    print(f"  by scanner: {dict(Counter(r['scanner'] for r in fully))}")

    json.dump(rows, open(args.json_out, "w"), indent=1)
    print(f"\njson -> {args.json_out}")


if __name__ == "__main__":
    main()
