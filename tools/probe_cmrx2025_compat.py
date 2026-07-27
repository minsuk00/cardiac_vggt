"""Is CMRxRecon2025 Cine SAX compatible with the 2024 reconstruction pipeline?

_archive/batch_reconstruct_cmrxrecon2024.py requires ALL of:
  1. h5 key 'kspace_full'
  2. 5D shape (nframe, nslice, ncoil, ny, nx)
  3. FULLY SAMPLED k-space (it does a direct iFFT + SENSE-combine; R>1 => aliasing)
  4. info.csv with ReconMatrix_X/Y, FOVx/FOVy, SliceThickness

Samples one subject per (center, scanner) so vendor heterogeneity is visible.
"""
import csv
import os
from collections import defaultdict

import h5py
import numpy as np

ROOT = ("/home/minsukc/vggt/scratch/data/CMRxRecon2025/TrainingData_extracted/"
        "TrainingData/MultiCoil/Cine/TrainingSet/FullSample")

NEEDED = ["ReconMatrix_X", "ReconMatrix_Y", "FOVx", "FOVy", "SliceThickness"]

# one subject per scanner
by_scanner = defaultdict(list)
for center in sorted(os.listdir(ROOT)):
    cpath = os.path.join(ROOT, center)
    if not os.path.isdir(cpath):
        continue
    for scanner in sorted(os.listdir(cpath)):
        spath = os.path.join(cpath, scanner)
        if not os.path.isdir(spath):
            continue
        for subj in sorted(os.listdir(spath)):
            m = os.path.join(spath, subj, "cine_sax.mat")
            if os.path.exists(m):
                by_scanner[(center, scanner)].append((subj, m))

print(f"scanners with SAX: {len(by_scanner)}", flush=True)

for (center, scanner), subs in sorted(by_scanner.items()):
    subj, mat = subs[0]
    print(f"\n{'='*72}\n{center}/{scanner}  ({len(subs)} subj w/ SAX)  -> {subj}", flush=True)

    # --- info.csv ---
    csvp = os.path.join(os.path.dirname(mat), "cine_sax_info.csv")
    meta = {}
    if os.path.exists(csvp):
        with open(csvp) as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) == 2:
                    meta[row[0].strip()] = row[1].strip()
        miss = [k for k in NEEDED if k not in meta]
        print(f"  info.csv: {len(meta)} fields, missing-required={miss or 'NONE'}", flush=True)
        print(f"    Recon={meta.get('ReconMatrix_X')}x{meta.get('ReconMatrix_Y')} "
              f"FOV={meta.get('FOVx')}x{meta.get('FOVy')} "
              f"Thk={meta.get('SliceThickness')} Slices={meta.get('SliceNum')} "
              f"Coils={meta.get('CoilNumber')} T={meta.get('TemporalPhase')}", flush=True)
    else:
        print("  info.csv: MISSING", flush=True)

    # --- k-space ---
    try:
        with h5py.File(mat, "r") as f:
            keys = list(f.keys())
            print(f"  h5 keys: {keys}  ('kspace_full' present: {'kspace_full' in keys})", flush=True)
            k = "kspace_full" if "kspace_full" in keys else keys[0]
            d = f[k]
            print(f"  {k}: shape={d.shape} ndim={d.ndim} dtype={d.dtype}", flush=True)
            if d.ndim == 5:
                plane = d[0, 0, 0]
            elif d.ndim == 4:
                plane = d[0, 0]
            else:
                print("  -> unexpected ndim, skipping sampling check", flush=True)
                continue
            if plane.dtype.names and "real" in plane.dtype.names:
                plane = plane["real"] + 1j * plane["imag"]
            rows = np.abs(plane).sum(axis=-1)
            nz, n = int((rows > 0).sum()), len(rows)
            print(f"  sampling: {nz}/{n} ky lines non-zero -> R_eff={n/max(nz,1):.2f} "
                  f"{'FULLY SAMPLED (2024-compatible)' if nz == n else '*** UNDERSAMPLED ***'}", flush=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)

print("\nDONE", flush=True)
