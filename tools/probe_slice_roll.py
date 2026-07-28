"""Detect a cyclic roll of the z (slice) axis in reconstructed SAX volumes.

Finding it was written for: every CMRxRecon **challenge** release (2023/2024/2025) ships
odd-Z SAX stacks cyclically rolled by -1 — the most basal slice is stored *last*, after
the apex. Even-Z stacks are fine. That is the exact signature of an `fftshift` used where
`ifftshift` was needed on the slice axis (the two differ by a roll of -1 for odd N and are
identical for even N). See `docs/56`.

Two independent estimators are computed per subject; they agreed 849/849 when this was run.

  1. `k_adjacent` — local. In a correct base->apex stack the ONE big discontinuity is the
     wrap pair (z=Z-1 -> z=0). Find the argmin over all Z cyclic-adjacent correlations;
     `k = Z-1-argmin` is the implied roll (0 = correct order).
  2. `k_global` — global, cannot be fooled by a single odd slice. Similarity must decay
     monotonically with |z_i - z_j|. For each candidate roll k, relabel positions
     t_i = (i+k) mod Z and score by Spearman corr between |t_i - t_j| and C[i,j] over ALL
     pairs; the best k is the most negative.

Usage:
    python tools/probe_slice_roll.py out.json                     # all 3 CMRxRecon years
    python tools/probe_slice_roll.py out.json --glob '<pattern>'  # any *.nii.gz SAX stacks
"""
import argparse
import glob as globmod
import json
import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
from scipy.stats import spearmanr

ROOT = "/home/minsukc/vggt/scratch/data"
CMRX_YEARS = {
    "2023": f"{ROOT}/CMRxRecon2023/Cine_combined",
    "2024": f"{ROOT}/CMRxRecon2024/Cine_combined",
    "2025": f"{ROOT}/CMRxRecon2025/Cine_combined",
}


def analyze(args):
    tag, subj, path = args
    try:
        vol = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        return {"tag": tag, "subj": subj, "error": repr(e)}
    if vol.ndim == 4:
        vol = vol[..., 0]
    if vol.ndim != 3 or vol.shape[2] < 4:
        return {"tag": tag, "subj": subj, "error": f"shape={vol.shape}"}
    Z = vol.shape[2]

    # zero-mean / unit-norm each slice once -> C is the full slice correlation matrix
    F = vol.reshape(-1, Z).astype(np.float64)
    F -= F.mean(0)
    F /= np.linalg.norm(F, axis=0) + 1e-12
    C = F.T @ F

    r = np.array([C[z, (z + 1) % Z] for z in range(Z)])
    argmin = int(np.nanargmin(r))
    k_adj = Z - 1 - argmin

    iu = np.triu_indices(Z, 1)
    c = C[iu]
    scores = np.array([
        spearmanr(np.abs(((np.arange(Z) + k) % Z)[iu[0]] - ((np.arange(Z) + k) % Z)[iu[1]]).astype(float), c).statistic
        for k in range(Z)
    ])
    k_glob = int(np.argmin(scores))

    return {
        "tag": tag, "subj": subj, "Z": Z,
        "k_adjacent": int(k_adj),
        "k_global": k_glob,
        "agree": int(k_adj) == k_glob,
        "r_cyclic": [round(float(v), 4) for v in r],
        "r_min": round(float(r[argmin]), 4),
        "r_second_min": round(float(np.sort(r)[1]), 4),
        "score_best": round(float(scores[k_glob]), 4),
        "score_second": round(float(np.sort(scores)[1]), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--glob", default=None, help="glob of NIfTI stacks instead of the CMRxRecon tree")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    tasks = []
    if a.glob:
        tasks = [("glob", os.path.basename(p), p) for p in sorted(globmod.glob(a.glob))]
    else:
        for year, d in CMRX_YEARS.items():
            for subj in sorted(os.listdir(d)):
                p = os.path.join(d, subj, "sax", "3d_recon", "sax_frame_00.nii.gz")
                if os.path.exists(p):
                    tasks.append((year, subj, p))
    print(f"{len(tasks)} volumes", flush=True)

    out = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(analyze, tasks, chunksize=4)):
            out.append(res)
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)

    ok = [r for r in out if "error" not in r]
    print(f"\nestimators agree: {sum(r['agree'] for r in ok)}/{len(ok)}")
    for tag in sorted({r["tag"] for r in ok}):
        rows = [r for r in ok if r["tag"] == tag]
        for par, lab in ((0, "even"), (1, "odd ")):
            sub = [r for r in rows if r["Z"] % 2 == par]
            if sub:
                n = sum(r["k_adjacent"] == 1 for r in sub)
                print(f"  {tag} {lab} Z: rolled(k=1) {n}/{len(sub)}")


if __name__ == "__main__":
    main()
