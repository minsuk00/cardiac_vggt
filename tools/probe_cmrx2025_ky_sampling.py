"""Measure WHY 2025 ships `ny < ReconMatrix_Y`, from the data rather than from documentation.

298 of 405 subjects have fewer acquired ky lines than the recon matrix (down to 0.42x). The 2024
recon only ever CROPS, so it silently corner-pads those subjects -- misregistering the image by
up to ~50 rows with no exception. Fixing it correctly requires knowing which of these it is:

  (a) SYMMETRIC truncation / reduced phase FOV  -> centred zero-fill to ReconMatrix_Y is right.
      Signature: the DC (max-energy) ky line sits at the CENTRE of the stored array.
  (b) ASYMMETRIC partial Fourier                -> the fill must be one-sided (or POCS), and a
      centred zero-fill would itself shift the image.
      Signature: the DC ky line sits OFF-centre, by roughly the amount that is missing.

Also reported: the fraction of all-zero ky lines, which would mean the "FullSample" grid is not
actually fully sampled (it is -- see results) and no plain iFFT+SENSE recon would be valid.

⚠️ Reading strategy matters. These files are gzip-chunked `(nt, nz, nc, 2, 1)`, so a chunk spans
every frame/slice/coil but only 2 ky x 1 kx. Reading one ky COLUMN therefore costs ny/2 chunks
and automatically delivers all (t,z,c) for free -- so we average the profile over them, which
costs nothing extra and is far less noisy than a single plane. (Reading a whole [ny,nx] plane
the obvious way costs ~ny/2*nx chunks = ~73 s/subject; don't.)

Usage (run the census first -- this reads its json to pick a spread of subjects):
    python tools/scan_cmrx2025_geometry.py
    python tools/probe_cmrx2025_ky_sampling.py
    python tools/probe_cmrx2025_ky_sampling.py --census /path/to/census.json --n-per-bucket 3
"""

import argparse
import json
import os
from collections import defaultdict

import h5py
import numpy as np
import scipy.io as sio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Default matches `tools/scan_cmrx2025_geometry.py --json-out`; run that first.
CENSUS = "/tmp/cmrx2025_geometry_census.json"


def ky_profile(mat, n_cols=3):
    """-> (profile over ky, shape). Averages |k| over all t, z, c and a few central kx columns."""
    with open(mat, "rb") as f:
        is_v5 = f.read(10) == b"MATLAB 5.0"
    if is_v5:
        name = next(e[0] for e in sio.whosmat(mat) if e[0] in ("kspace", "kspace_full"))
        a = np.transpose(sio.loadmat(mat)[name], (4, 3, 2, 1, 0))
        nt, nz, nc, ny, nx = a.shape
        x0 = nx // 2 - n_cols // 2
        return np.abs(a[:, :, :, :, x0:x0 + n_cols]).sum(axis=(0, 1, 2, 4)), (nt, nz, nc, ny, nx)
    with h5py.File(mat, "r") as f:
        d = f["kspace"] if "kspace" in f else f["kspace_full"]
        nt, nz, nc, ny, nx = d.shape
        x0 = nx // 2 - n_cols // 2
        b = d[:, :, :, :, x0:x0 + n_cols]
        arr = b["real"] + 1j * b["imag"]
        return np.abs(arr).sum(axis=(0, 1, 2, 4)), (nt, nz, nc, ny, nx)


def analyse(prof):
    ny = len(prof)
    dc = int(np.argmax(prof))
    centre = (ny - 1) / 2.0
    # Energy centroid is a smoother locator than the single argmax.
    centroid = float((np.arange(ny) * prof).sum() / max(prof.sum(), 1e-30))
    return {
        "ny": ny,
        "dc_idx": dc,
        "centre_idx": centre,
        "dc_offset": dc - centre,
        "centroid_offset": centroid - centre,
        "zero_lines": int((prof <= 0).sum()),
        "frac_energy_first_half": float(prof[: ny // 2].sum() / max(prof.sum(), 1e-30)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default=CENSUS)
    ap.add_argument("--n-per-bucket", type=int, default=2)
    ap.add_argument("--json-out", default="/tmp/cmrx2025_ky_probe.json")
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.census)) if "error" not in r]
    buckets = defaultdict(list)
    for r in rows:
        ratio = r["ny"] / r["ry"]
        key = ("1.00" if ratio >= 0.995 else "0.94-0.99" if ratio >= 0.94
               else "0.60-0.93" if ratio >= 0.60 else "<0.60")
        buckets[f"{key}  {r['scanner']}"].append(r)

    picks = []
    for k in sorted(buckets):
        picks += buckets[k][: args.n_per_bucket]
    print(f"probing {len(picks)} subjects across {len(buckets)} (ratio x scanner) buckets\n", flush=True)

    out = []
    hdr = f"{'ratio':>6} {'scanner':22} {'ny':>4}/{'ry':<4} {'DCoff':>7} {'centrOff':>9} {'zeroLn':>7} {'E<half':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in picks:
        mat = os.path.join(os.path.dirname(r["csv"]), "cine_sax.mat")
        try:
            prof, shape = ky_profile(mat)
            a = analyse(prof)
        except Exception as e:
            print(f"  ERR {r['scanner']} {r['pid']}: {type(e).__name__}: {e}", flush=True)
            continue
        a.update(scanner=r["scanner"], center=r["center"], pid=r["pid"],
                 ry=r["ry"], ratio=r["ny"] / r["ry"], csv=r["csv"])
        out.append(a)
        print(f"{a['ratio']:6.2f} {r['scanner']:22} {a['ny']:>4}/{r['ry']:<4} "
              f"{a['dc_offset']:+7.1f} {a['centroid_offset']:+9.2f} {a['zero_lines']:>7} "
              f"{a['frac_energy_first_half']:7.3f}", flush=True)

    json.dump(out, open(args.json_out, "w"), indent=1)
    print(f"\njson -> {args.json_out}", flush=True)
    print("\nREAD: dc_offset ~ 0  => DC is centred => symmetric truncation => CENTRED zero-fill.", flush=True)
    print("      dc_offset far from 0 => partial Fourier => one-sided fill / POCS.", flush=True)
    print("      zero_lines > 0 => grid not fully sampled => plain iFFT+SENSE is invalid.", flush=True)


if __name__ == "__main__":
    main()
