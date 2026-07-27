"""Duplicate scan for CMRxRecon2025 Cine SAX, across TrainingData / TaskR1 / TaskR2.

2023 turned out to re-release 68 train/val subjects inside its test split (see
`scratch/data/CMRxRecon2023/README.md`), and R1/R2 here are two TASKS that may well have been
built from an overlapping subject pool -- so the same leakage check is due before anyone pools
2025 into a train/val/test partition.

Method (same spirit as the 2023/2024 scans): hash raw k-space straight out of `cine_sax.mat`,
not a recon and not file bytes, so a differing header/compression cannot mask a real duplicate.

⚠️ READ BEFORE CHANGING THE SAMPLING PATTERN. The 2025 HDF5 files are gzip-chunked
`(nt, nz, nc, 2, 1)` -- every chunk spans ALL frames/slices/coils but only 2 ky x 1 kx. So the
2023/2024 habit of reading one `[ny, nx]` plane at fixed (t,z,c) touches ~ny/2*nx chunks and
decompresses essentially the WHOLE file: measured **73 s for a single plane** on a 1.85 GB
subject (~17 h for this scan). Instead we read three CHUNK-ALIGNED blocks, each one chunk in
the ky/kx dims and full in (t,z,c). That is ~46 KB apiece, and is a *better* signature anyway:
it samples every frame, slice and coil rather than one arbitrary plane.

Two independent block sets are hashed per subject:
    h1 = block at ky/kx origin                 -- the grouping key
    h2 = blocks at two interior ky/kx offsets   -- corroboration; a group agreeing on h1 but
                                                  not h2 is reported suspicious, not a dupe.
Each block's L2 norm is recorded: a group built on all-zero blocks would be a degenerate
artifact rather than a duplicate, and shows up here as norm == 0.

Both container formats are handled. Most subjects are MATLAB v7.3 (HDF5, dims already reversed
to (nt,nz,nc,ny,nx)); 33 are plain MATLAB v5, which h5py rejects with 'file signature not
found'. Those are read with scipy and transposed into the same logical order, so hashes are
comparable ACROSS formats (a subject re-released in the other container still matches).

Subjects are keyed by SPLIT/CENTER/SCANNER/P### -- 2025 reuses patient IDs across centres, so a
bare P### key would itself manufacture collisions.

Usage:
    python tools/scan_cmrx2025_duplicates.py
    python tools/scan_cmrx2025_duplicates.py --json-out /tmp/dupes2025.json
"""

import argparse
import glob
import hashlib
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(REPO, "scratch", "data", "CMRxRecon2025")

# ky/kx sampling positions as fractions of (ny, nx); floored onto the chunk grid.
POSITIONS = [(0.0, 0.0), (0.5, 0.5), (0.25, 0.75)]


def _blocks_v73(mat):
    with h5py.File(mat, "r") as f:
        key = "kspace" if "kspace" in f else "kspace_full"
        d = f[key]
        nt, nz, nc, ny, nx = d.shape
        cy, cx = (d.chunks[-2:] if d.chunks else (2, 1))
        out = []
        for fy, fx in POSITIONS:
            y0 = min(int(ny * fy) // cy * cy, ny - cy)
            x0 = min(int(nx * fx) // cx * cx, nx - cx)
            b = d[:, :, :, y0:y0 + cy, x0:x0 + cx]
            out.append(np.ascontiguousarray(b["real"] + 1j * b["imag"]))
        return out, (nt, nz, nc, ny, nx), "v7.3"


def _blocks_v5(mat):
    name = next(e[0] for e in sio.whosmat(mat) if e[0] in ("kspace", "kspace_full"))
    # MATLAB-native order is the reverse of the HDF5 view -> transpose into (nt,nz,nc,ny,nx).
    a = np.transpose(sio.loadmat(mat)[name], (4, 3, 2, 1, 0))
    nt, nz, nc, ny, nx = a.shape
    cy, cx = 2, 1
    out = []
    for fy, fx in POSITIONS:
        y0 = min(int(ny * fy) // cy * cy, ny - cy)
        x0 = min(int(nx * fx) // cx * cx, nx - cx)
        out.append(np.ascontiguousarray(a[:, :, :, y0:y0 + cy, x0:x0 + cx].astype(np.complex128)))
    return out, (nt, nz, nc, ny, nx), "v5"


def scan_one(mat):
    with open(mat, "rb") as f:
        is_v5 = f.read(10) == b"MATLAB 5.0"
    blocks, shape, ver = (_blocks_v5 if is_v5 else _blocks_v73)(mat)
    md5 = lambda arrs: hashlib.md5(b"".join(a.tobytes() for a in arrs)).hexdigest()
    return {
        "shape": tuple(int(x) for x in shape),
        "matver": ver,
        "h1": md5(blocks[:1]),
        "h2": md5(blocks[1:]),
        "n1": float(np.linalg.norm(blocks[0])),
        "n2": float(sum(np.linalg.norm(b) for b in blocks[1:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="/tmp/cmrx2025_duplicates.json")
    args = ap.parse_args()

    mats = sorted(glob.glob(os.path.join(ROOT, "*_extracted", "**", "cine_sax.mat"), recursive=True))
    print(f"{len(mats)} cine_sax.mat found", flush=True)

    recs, errors = {}, {}
    for i, mat in enumerate(mats):
        parts = mat.split("/")
        split = next(p for p in parts if p.endswith("_extracted")).replace("_extracted", "")
        # Include the Set level (`TrainingSet`/`ValidationSet`/`TestSet`): within TaskR1 the SAME
        # center/scanner/P### occurs in both ValidationSet and TestSet (4 cases), so a
        # split/center/scanner/pid key silently overwrites one of each pair and they never get
        # compared. They are different people -- 2025 reuses IDs per split -- but the scan must
        # still see them.
        key = f"{split}/{parts[-5]}/{parts[-4]}/{parts[-3]}/{parts[-2]}"
        try:
            recs[key] = scan_one(mat)
            recs[key]["path"] = mat
        except Exception as e:
            errors[key] = f"{type(e).__name__}: {e}"[:120]
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(mats)}", flush=True)

    print(f"\nreadable={len(recs)}  unreadable={len(errors)}", flush=True)
    for k, v in list(errors.items())[:10]:
        print(f"  ERR {k}: {v}", flush=True)

    zero = [k for k, v in recs.items() if v["n1"] == 0.0]
    print(f"degenerate (zero-norm primary block): {len(zero)} {zero[:5]}", flush=True)

    groups = defaultdict(list)
    for k, v in recs.items():
        groups[v["h1"]].append(k)
    dupes = {h: sorted(v) for h, v in groups.items() if len(v) > 1}

    confirmed, suspicious = {}, {}
    for h, members in dupes.items():
        agree = len({recs[m]["h2"] for m in members}) == 1 and len({recs[m]["shape"] for m in members}) == 1
        (confirmed if agree else suspicious)[h] = members

    splits_of = lambda members: sorted({m.split("/")[0] for m in members})

    print("\n=== RESULT ===", flush=True)
    print(f"subjects scanned : {len(recs)}", flush=True)
    print(f"unique h1 hashes : {len(groups)}", flush=True)
    print(f"duplicate groups : {len(dupes)}  (confirmed on BOTH block sets: {len(confirmed)}, "
          f"one-set-only/suspicious: {len(suspicious)})", flush=True)

    cross = {h: m for h, m in confirmed.items() if len(splits_of(m)) > 1}
    print(f"  CROSS-split groups : {len(cross)}   <- these are the leakage risk", flush=True)
    print(f"  within-split groups: {len(confirmed) - len(cross)}", flush=True)
    pat = defaultdict(int)
    for h, m in confirmed.items():
        pat["+".join(splits_of(m))] += 1
    print(f"  split patterns: {dict(pat)}", flush=True)

    print("\n--- confirmed duplicate groups ---", flush=True)
    for h, m in sorted(confirmed.items(), key=lambda kv: kv[1][0]):
        print(f"  {m}  shape={recs[m[0]]['shape']}", flush=True)
    if suspicious:
        print("\n--- SUSPICIOUS (h1 matches, h2 or shape does not) ---", flush=True)
        for h, m in sorted(suspicious.items(), key=lambda kv: kv[1][0]):
            print(f"  {m}", flush=True)

    redundant = sum(len(v) - 1 for v in confirmed.values())
    print(f"\nredundant copies : {redundant}", flush=True)
    print(f"unique subjects  : {len(recs) - redundant}", flush=True)

    # Negative control: many subjects share an identical SHAPE. If shape-sharing is common yet
    # hash collisions are rare, the signature is discriminating rather than degenerate.
    byshape = defaultdict(list)
    for k, v in recs.items():
        byshape[v["shape"]].append(k)
    multi = {k: len(v) for k, v in byshape.items() if len(v) > 1}
    print(f"distinct shapes  : {len(byshape)}", flush=True)
    print(f"shared-shape grps: {len(multi)}  (largest {max(multi.values()) if multi else 0})", flush=True)

    json.dump(
        {
            "records": {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
                        for k, v in recs.items()},
            "confirmed": confirmed,
            "suspicious": suspicious,
            "errors": errors,
        },
        open(args.json_out, "w"),
        indent=1,
    )
    print(f"\njson -> {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
