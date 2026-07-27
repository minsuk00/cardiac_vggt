"""Duplicate scan for CMRxRecon2024 Cine_combined.

Hashes the VOXEL ARRAY (not file bytes) of sax_frame_00.nii.gz so a differing
affine/header cannot mask a genuine duplicate. Frame 0 only -> immune to any
per-copy tail-frame trimming (the CMR-300 lesson).
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"

subjects = sorted(os.listdir(ROOT))
print(f"{len(subjects)} subjects", flush=True)

hashes = defaultdict(list)
shapes = {}
missing = []

for i, s in enumerate(subjects):
    p = os.path.join(ROOT, s, "sax", "3d_recon", "sax_frame_00.nii.gz")
    if not os.path.exists(p):
        missing.append(s)
        continue
    arr = np.asarray(nib.load(p).dataobj, dtype=np.float32)
    h = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    hashes[h].append(s)
    shapes[s] = tuple(int(x) for x in arr.shape)
    if (i + 1) % 25 == 0:
        print(f"  {i+1}/{len(subjects)}", flush=True)

dupes = {h: v for h, v in hashes.items() if len(v) > 1}

print("\n=== RESULT ===", flush=True)
print(f"scanned          : {len(shapes)}", flush=True)
print(f"missing frame_00 : {len(missing)} {missing[:10]}", flush=True)
print(f"unique hashes    : {len(hashes)}", flush=True)
print(f"duplicate groups : {len(dupes)}", flush=True)
for h, v in sorted(dupes.items(), key=lambda kv: kv[1][0]):
    print(f"  {v}  shape={shapes[v[0]]}", flush=True)

# How many subjects would be lost if we deduped?
redundant = sum(len(v) - 1 for v in dupes.values())
print(f"\nredundant copies : {redundant}", flush=True)
print(f"unique subjects  : {len(shapes) - redundant}", flush=True)

# Shape-collision sanity: how many subjects SHARE a shape? If many share a
# shape yet no hash collides, that is a strong negative control.
byshape = defaultdict(list)
for s, sh in shapes.items():
    byshape[sh].append(s)
multi = {k: len(v) for k, v in byshape.items() if len(v) > 1}
print(f"distinct shapes  : {len(byshape)}", flush=True)
print(f"shared-shape grps: {len(multi)}  (largest {max(multi.values()) if multi else 0})", flush=True)

out = os.environ.get("DUP_JSON", "/tmp/cmrx2024_dupes.json")
with open(out, "w") as f:
    json.dump(
        {"dupes": dupes, "shapes": {k: list(v) for k, v in shapes.items()}, "missing": missing},
        f,
        indent=1,
    )
print("done", flush=True)
