"""Probe the k-space sampling pattern of each release.

The decisive question: which releases ship k-space that the EXISTING 2024 recon
(iFFT + ESPIRiT-from-frame-0 + SENSE combine, no parallel-imaging unfolding)
can actually consume? That pipeline assumes a FULLY SAMPLED grid. If a release
is R=3 zero-filled, running it would produce 3x-aliased garbage.
"""
import h5py
import numpy as np

TARGETS = [
    ("2024-challenge", "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined/Test_P001/sax/cine_sax.mat"),
    ("2023-challenge-MC", "/home/minsukc/vggt/scratch/data/CMRxRecon2023/ChallengeData/MultiCoil/Cine/TrainingSet/FullSample/P001/cine_sax.mat"),
    ("2023-challenge-SC", "/home/minsukc/vggt/scratch/data/CMRxRecon2023/ChallengeData/SingleCoil/Cine/TrainingSet/FullSample/P001/cine_sax.mat"),
    ("CMR300-ks", "/home/minsukc/vggt/scratch/data/CMRxRecon-300/TrainingSet/P003/cine_sax_ks.mat"),
    ("CMR300-calib", "/home/minsukc/vggt/scratch/data/CMRxRecon-300/TrainingSet/P003/cine_sax_calib.mat"),
]


def describe(tag, path):
    print(f"\n{'='*70}\n{tag}\n  {path}", flush=True)
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"  keys: {keys}", flush=True)
            for k in keys:
                d = f[k]
                if not hasattr(d, "shape"):
                    print(f"  {k}: (group)", flush=True)
                    continue
                print(f"  {k}: shape={d.shape} dtype={d.dtype}", flush=True)
                if d.ndim < 4:
                    continue
                # Grab ONE contiguous [t,z,c] plane (GPFS: contiguous >> strided)
                idx = (0,) * (d.ndim - 2)
                plane = d[idx]
                if plane.dtype.names and "real" in plane.dtype.names:
                    plane = plane["real"] + 1j * plane["imag"]
                mag = np.abs(plane)
                # ky is the second-to-last axis in the h5py-reversed order
                rowsum = mag.sum(axis=-1)
                nz = int((rowsum > 0).sum())
                n = len(rowsum)
                print(f"    plane {plane.shape}: {nz}/{n} ky lines non-zero "
                      f"-> R_eff = {n/max(nz,1):.2f}", flush=True)
                if nz < n:
                    on = np.nonzero(rowsum > 0)[0]
                    gaps = np.diff(on)
                    vals, cnts = np.unique(gaps, return_counts=True)
                    print(f"    sampled idx: {on[:12].tolist()} ... {on[-4:].tolist()}", flush=True)
                    print(f"    step histogram: {dict(zip(vals.tolist(), cnts.tolist()))}", flush=True)
                    # contiguous ACS block?
                    runs, cur = [], 1
                    for g in gaps:
                        if g == 1:
                            cur += 1
                        else:
                            runs.append(cur)
                            cur = 1
                    runs.append(cur)
                    print(f"    longest contiguous ACS run: {max(runs)} lines", flush=True)
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}", flush=True)


for tag, path in TARGETS:
    describe(tag, path)
print("\nDONE", flush=True)
