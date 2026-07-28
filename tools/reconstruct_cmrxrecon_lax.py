#!/usr/bin/env python
"""Reconstruct CMRxRecon2024 fully-sampled LAX + LVOT cine (ESPIRiT + SENSE) into Cine_combined,
mirroring the existing sax/ layout. CPU-only (fully sampled -> no CS -> no GPU needed).

Per subject <Train|Val|Test>_P### in Cine_combined:
  lax/  4d_recon.nii.gz + cine_lax_info.csv (copy) + cine_lax.mat (symlink to FullSample)
  lvot/ 4d_recon.nii.gz + cine_lvot_info.csv (copy) + cine_lvot.mat (symlink)

Algorithm matches _archive/batch_reconstruct_cmrxrecon2024.py (the SAX pipeline), on CPU sigpy.
NOTE: for lax the slice axis = VIEW index (3 long-axis planes), not depth; and CMRxRecon carries NO
per-slice position/orientation, so these cines are NOT spatially registered to the SAX (see docs).

Usage:  micromamba run -n svr python tools/reconstruct_cmrxrecon_lax.py [--limit N] [--subjects CMRx24_Train_P001 ...]
"""
import argparse
import csv
import os
import shutil

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri as mr
import SimpleITK as sitk

ROOT = "scratch/data/CMRxRecon2024"
CC = os.path.join(ROOT, "Cine_combined")
# prefix -> (mat_root, csv_root)  [authoritative mapping from batch_reconstruct_cmrxrecon2024.py]
SRC = {
    "Train": (f"{ROOT}/ChallengeData/Cine/TrainingSet/FullSample",
              f"{ROOT}/ChallengeData/Cine/TrainingSet/ImgSnapshot"),
    "Val":   (f"{ROOT}/ChallengeData_AfterCompetition/Cine/ValidationSet/FullSample",
              f"{ROOT}/ChallengeData_AfterCompetition/Cine/ValidationSet/ImgSnapshot"),
    "Test":  (f"{ROOT}/ChallengeData_AfterCompetition/Cine/TestSet/FullSample",
              f"{ROOT}/ChallengeData_AfterCompetition/Cine/TestSet/ImgSnapshot"),
}


def read_info(csv_path):
    m = {}
    if not os.path.exists(csv_path):
        return None
    for row in csv.reader(open(csv_path)):
        if len(row) == 2:
            k, v = row[0].strip(), row[1].strip()
            try:
                m[k] = float(v) if "." in v else int(v)
            except ValueError:
                m[k] = v
    return m if "ReconMatrix_X" in m else None


def recon_view(mat_file, info):
    """ESPIRiT+SENSE per slice -> (frame, slice, recon_y, recon_x) float32."""
    with h5py.File(mat_file, "r") as h:
        k = h["kspace_full"]
        ksp = k["real"][:] + 1j * k["imag"][:]          # (frame, slice, coil, ny, nx)
    nf, ns, nc, ny, nx = ksp.shape
    rx, ry = int(info["ReconMatrix_X"]), int(info["ReconMatrix_Y"])
    out = np.zeros((nf, ns, ry, rx), np.float32)
    for s in range(ns):
        smap = mr.app.EspiritCalib(sp.ifft(ksp[0, s], axes=[-2, -1]),
                                   crop=0.80, thresh=0.01, calib_width=32, show_pbar=False).run()
        for f in range(nf):
            img = sp.ifft(ksp[f, s], axes=[-2, -1])
            comb = np.abs(np.sum(np.conj(smap) * img, 0) / (np.sum(np.abs(smap) ** 2, 0) + 1e-8))
            y0, x0 = max(0, (ny - ry) // 2), max(0, (nx - rx) // 2)
            crop = comb[y0:y0 + ry, x0:x0 + rx]
            out[f, s, :crop.shape[0], :crop.shape[1]] = crop
    spacing = (info["FOVx"] / rx, info["FOVy"] / ry, float(info["SliceThickness"]))
    return out, spacing


def save_view(cine, spacing, out_dir, view):
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for f in range(cine.shape[0]):
        im = sitk.GetImageFromArray(cine[f])       # (slice, H, W) -> (W, H, slice)
        im.SetSpacing(spacing)
        frames.append(im)
    sitk.WriteImage(sitk.JoinSeries(frames), os.path.join(out_dir, "4d_recon.nii.gz"))


def process(subject, limit_views=("lax", "lvot")):
    prefix, pid = subject.split("_", 1)
    mat_root, csv_root = SRC[prefix]
    for view in limit_views:
        mat = os.path.join(mat_root, pid, f"cine_{view}.mat")
        info_csv = os.path.join(csv_root, pid, f"cine_{view}_info.csv")
        out_dir = os.path.join(CC, subject, view)
        if os.path.exists(os.path.join(out_dir, "4d_recon.nii.gz")):
            continue
        info = read_info(info_csv)
        if not os.path.exists(mat) or info is None:
            print(f"  [{subject}] {view}: missing mat/info -> skip", flush=True)
            continue
        cine, spacing = recon_view(mat, info)
        save_view(cine, spacing, out_dir, view)
        shutil.copy2(info_csv, os.path.join(out_dir, f"cine_{view}_info.csv"))
        link = os.path.join(out_dir, f"cine_{view}.mat")
        if not os.path.islink(link) and not os.path.exists(link):
            os.symlink(os.path.abspath(mat), link)      # symlink, do NOT copy the big .mat
        print(f"  [{subject}] {view}: {cine.shape} -> {out_dir}/4d_recon.nii.gz", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    subjects = args.subjects or sorted(d for d in os.listdir(CC) if os.path.isdir(os.path.join(CC, d)))
    if args.limit:
        subjects = subjects[:args.limit]
    print(f"processing {len(subjects)} subjects", flush=True)
    for i, s in enumerate(subjects):
        process(s)
        if (i + 1) % 25 == 0:
            print(f"... {i+1}/{len(subjects)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
