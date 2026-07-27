"""Confirm the 2024 duplicate pairs on RAW k-space, not the recon hash.

The recon is deterministic, so identical recon almost certainly implies identical
input -- but "almost certainly" is not proof. This reads independent full
[t,z,c] slabs straight from cine_sax.mat and compares bytes, plus a same-shape
negative control.
"""
import os

import h5py
import numpy as np

ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"

PAIRS = [
    ("Test_P013", "Train_P198"),   # leaks into our TEST split
    ("Test_P012", "Train_P199"),   # leaks into our VAL split
    ("Train_P193", "Val_P052"),    # leaks into our VAL split
]
# same-shape distinct subjects -> must DIFFER
NEG = [("Test_P012", "Test_P013")]  # both (256,246,11) per the recon scan


def slabs(path, picks):
    out = {}
    with h5py.File(path, "r") as f:
        d = f["kspace_full"]
        out["shape"] = d.shape
        nt, nz, nc = d.shape[:3]
        for (t, z, c) in picks:
            if t < nt and z < nz and c < nc:
                p = d[t, z, c]
                out[(t, z, c)] = p["real"] + 1j * p["imag"]
    return out


PICKS = [(0, 0, 0), (5, 3, 4), (11, 1, 9)]


def compare(a, b, expect_same):
    pa = os.path.join(ROOT, a, "sax", "cine_sax.mat")
    pb = os.path.join(ROOT, b, "sax", "cine_sax.mat")
    ia, ib = os.stat(pa), os.stat(pb)
    print(f"\n{a}  <==>  {b}   (expect {'IDENTICAL' if expect_same else 'DIFFERENT'})", flush=True)
    print(f"  size   {ia.st_size} vs {ib.st_size}", flush=True)
    print(f"  inode  {ia.st_ino} vs {ib.st_ino}   nlink {ia.st_nlink}/{ib.st_nlink} "
          f"-> {'SAME FILE (hardlink)' if ia.st_ino == ib.st_ino else 'distinct files'}", flush=True)
    sa, sb = slabs(pa, PICKS), slabs(pb, PICKS)
    print(f"  shape  {sa['shape']} vs {sb['shape']}", flush=True)
    if sa["shape"] != sb["shape"]:
        print("  -> shapes differ, not duplicates", flush=True)
        return
    for k in PICKS:
        if k not in sa or k not in sb:
            continue
        A, B = sa[k], sb[k]
        same = np.array_equal(A, B)
        denom = np.linalg.norm(np.abs(A)) * np.linalg.norm(np.abs(B))
        corr = float((np.abs(A) * np.abs(B)).sum() / denom) if denom else float("nan")
        print(f"  slab t,z,c={k}: bitwise_identical={same}  |k|_corr={corr:.6f}", flush=True)


for a, b in PAIRS:
    compare(a, b, True)
for a, b in NEG:
    compare(a, b, False)
print("\nDONE", flush=True)
