"""Reproduce the CMRxRecon2024 SAX recon from raw k-space and diff it against the on-disk NIfTIs.

Imports `reconstruct_subject` from `_archive/batch_reconstruct_cmrxrecon2024.py` *unmodified*
(so this is literally the same code path that produced `Cine_combined/`), runs it into a scratch
output dir, and compares voxel-for-voxel against the shipped recon.

Usage:
    python tools/verify_cmrx2024_recon_repro.py --subjects Train/P010 Test/P020

Writes side-by-side PNG panels to result/recon_verify_2024/ and a JSON summary.
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_SCRIPT = os.path.join(REPO, "_archive", "batch_reconstruct_cmrxrecon2024.py")
DATA = os.path.join(REPO, "scratch", "data", "CMRxRecon2024")
CINE = os.path.join(DATA, "Cine_combined")

DS_MAP = {
    "Train": ("ChallengeData/Cine/TrainingSet/FullSample", "ChallengeData/Cine/TrainingSet/ImgSnapshot"),
    "Val": (
        "ChallengeData_AfterCompetition/Cine/ValidationSet/FullSample",
        "ChallengeData_AfterCompetition/Cine/ValidationSet/ImgSnapshot",
    ),
    "Test": (
        "ChallengeData_AfterCompetition/Cine/TestSet/FullSample",
        "ChallengeData_AfterCompetition/Cine/TestSet/ImgSnapshot",
    ),
}


def load_archive_module():
    spec = importlib.util.spec_from_file_location("batch_reconstruct_cmrxrecon2024", ARCHIVE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def psnr(ref, test):
    mse = float(np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2))
    if mse == 0:
        return float("inf")
    peak = float(ref.max())
    return 10.0 * np.log10(peak**2 / mse)


def load_sitk(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img), img  # array is (t, z, y, x) for 4D


def panel(disk, fresh, idx_list, titles, out_png, suptitle):
    """3 rows (disk / fresh / |diff|) x len(idx_list) cols. idx_list = [(t, z), ...]."""
    n = len(idx_list)
    fig, axes = plt.subplots(3, n, figsize=(1.9 * n, 6.4))
    if n == 1:
        axes = axes[:, None]
    vmax = float(np.percentile(disk, 99.5))
    diff_all = np.abs(disk.astype(np.float64) - fresh.astype(np.float64))
    dmax = float(diff_all.max()) if diff_all.max() > 0 else 1.0
    for c, (t, z) in enumerate(idx_list):
        a, b = disk[t, z], fresh[t, z]
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        axes[0, c].imshow(a, cmap="gray", vmin=0, vmax=vmax)
        axes[1, c].imshow(b, cmap="gray", vmin=0, vmax=vmax)
        im = axes[2, c].imshow(d, cmap="magma", vmin=0, vmax=dmax)
        axes[0, c].set_title(titles[c], fontsize=8)
        for r in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("ON DISK\n(shipped)", fontsize=8)
    axes[1, 0].set_ylabel("FRESH RE-RUN\n(archive script)", fontsize=8)
    axes[2, 0].set_ylabel(f"|diff|\nmax={dmax:.2e}", fontsize=8)
    cbar = fig.colorbar(im, ax=axes[2, :].tolist(), fraction=0.02, pad=0.01)
    cbar.ax.tick_params(labelsize=6)
    fig.suptitle(suptitle, fontsize=10)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def verify_one(mod, ds_name, p_id, work_root, fig_dir, stage_dir):
    combined = f"{ds_name}_{p_id}"
    mat_root, csv_root = DS_MAP[ds_name]
    mat_src = os.path.join(DATA, mat_root, p_id, "cine_sax.mat")
    csv_src = os.path.join(DATA, csv_root, p_id, "cine_sax_info.csv")
    disk_4d = os.path.join(CINE, combined, "sax", "4d_recon.nii.gz")

    for p in (mat_src, csv_src, disk_4d):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Stage the .mat to node-local /tmp (GPFS small/strided reads are pathologically slow).
    os.makedirs(stage_dir, exist_ok=True)
    mat_local = os.path.join(stage_dir, f"{combined}_cine_sax.mat")
    if not os.path.exists(mat_local):
        t0 = time.time()
        shutil.copy2(mat_src, mat_local)
        print(f"[{combined}] staged {os.path.getsize(mat_local)/1e9:.2f} GB GPFS->/tmp in {time.time()-t0:.0f}s")

    out_dir = os.path.join(work_root, combined)
    if os.path.exists(os.path.join(out_dir, "sax", "4d_recon.nii.gz")):
        os.remove(os.path.join(out_dir, "sax", "4d_recon.nii.gz"))

    t0 = time.time()
    mod.reconstruct_subject(combined, mat_local, csv_src, out_dir, device_id=0)
    recon_s = time.time() - t0

    fresh_4d = os.path.join(out_dir, "sax", "4d_recon.nii.gz")
    a_disk, img_disk = load_sitk(disk_4d)
    a_fresh, img_fresh = load_sitk(fresh_4d)

    res = {
        "subject": combined,
        "recon_seconds": round(recon_s, 1),
        "shape_disk": list(a_disk.shape),
        "shape_fresh": list(a_fresh.shape),
        "spacing_disk": [round(float(v), 6) for v in img_disk.GetSpacing()],
        "spacing_fresh": [round(float(v), 6) for v in img_fresh.GetSpacing()],
        "dtype_disk": str(a_disk.dtype),
        "dtype_fresh": str(a_fresh.dtype),
    }
    if a_disk.shape != a_fresh.shape:
        res["MATCH"] = False
        res["reason"] = "shape mismatch"
        print(json.dumps(res, indent=2))
        return res

    diff = np.abs(a_disk.astype(np.float64) - a_fresh.astype(np.float64))
    res.update(
        {
            "bitwise_identical": bool(np.array_equal(a_disk, a_fresh)),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "data_max": float(a_disk.max()),
            "rel_max_diff": float(diff.max() / a_disk.max()),
            "psnr_db": round(psnr(a_disk, a_fresh), 2),
            "corr": float(np.corrcoef(a_disk.ravel(), a_fresh.ravel())[0, 1]),
        }
    )
    # Per-frame 3d_recon files too (the ones the training pipeline actually reads).
    per_frame = []
    for f in range(a_disk.shape[0]):
        dp = os.path.join(CINE, combined, "sax", "3d_recon", f"sax_frame_{f:02d}.nii.gz")
        fp = os.path.join(out_dir, "sax", "3d_recon", f"sax_frame_{f:02d}.nii.gz")
        if os.path.exists(dp) and os.path.exists(fp):
            d3 = sitk.GetArrayFromImage(sitk.ReadImage(dp))
            f3 = sitk.GetArrayFromImage(sitk.ReadImage(fp))
            per_frame.append(round(psnr(d3, f3), 2))
    res["per_frame_3d_recon_psnr_db"] = per_frame

    T, Z = a_disk.shape[0], a_disk.shape[1]
    zmid = Z // 2
    os.makedirs(fig_dir, exist_ok=True)
    panel(
        a_disk,
        a_fresh,
        [(0, z) for z in range(Z)],
        [f"t=0 z={z}" for z in range(Z)],
        os.path.join(fig_dir, f"gpu_repro_{combined}_all_z_t00.png"),
        f"{combined} — ALL {Z} slices at t=0   |   shipped vs fresh re-run   |   PSNR {res['psnr_db']} dB",
    )
    panel(
        a_disk,
        a_fresh,
        [(t, zmid) for t in range(T)],
        [f"t={t}" for t in range(T)],
        os.path.join(fig_dir, f"gpu_repro_{combined}_all_t_zmid.png"),
        f"{combined} — ALL {T} frames at z={zmid}   |   shipped vs fresh re-run   |   PSNR {res['psnr_db']} dB",
    )
    print(json.dumps(res, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["Train/P010", "Test/P020"])
    ap.add_argument("--work-root", default=f"/tmp/cmrx2024_recon_verify_{os.environ.get('USER','u')}")
    ap.add_argument("--stage-dir", default=f"/tmp/cmrx2024_recon_verify_{os.environ.get('USER','u')}/_staged")
    ap.add_argument("--fig-dir", default=os.path.join(REPO, "result", "recon_verify_2024"))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mod = load_archive_module()
    print(f"loaded recon module from {ARCHIVE_SCRIPT}")

    results = []
    for sub in args.subjects:
        ds_name, p_id = sub.split("/", 1)
        results.append(verify_one(mod, ds_name, p_id, args.work_root, args.fig_dir, args.stage_dir))

    out_json = args.json_out or os.path.join(args.fig_dir, "gpu_repro_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_json}")

    print("\n=== SUMMARY ===")
    for r in results:
        print(
            f"{r['subject']:>12}  shape {r['shape_disk']}  PSNR {r.get('psnr_db')} dB  "
            f"max|diff| {r.get('max_abs_diff'):.3e} (rel {r.get('rel_max_diff'):.2e})  "
            f"spacing disk={r['spacing_disk']} fresh={r['spacing_fresh']}"
        )


if __name__ == "__main__":
    sys.exit(main())
