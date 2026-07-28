"""Full QC pass over every reconstructed CMRxRecon2025 subject.

Two silent failure modes survive a clean recon run: a misplaced k-space fill (heart off-centre)
and a wrong spacing (heart the wrong physical size / aspect). Neither raises, so the only proof is
to look at every subject AND to measure a few things that separate "fine" from "suspicious".

For each subject this computes, from the reconstructed NIfTI:
  * centroid offset (% of FOV)  -- a good fill leaves the heart near centre; >15% is suspect
  * body extent (mm)            -- adult thorax in SAX is ~250-420 mm; wildly off => spacing wrong
  * in-plane anisotropy         -- reported, not failed (UIH is legitimately ~1.5x from ky ZIP)
  * dead-slice fraction         -- slices whose mean signal < 5% of the max (empty/failed slices)

Outputs (project dir, so they are visible):
  result/cmrx2025_recon_check/qc_montage_<scanner>.png   compact grid, one thumbnail per subject,
                                                          drawn at TRUE physical aspect, RED box = flagged
  result/cmrx2025_recon_check/qc_report.json             per-subject metrics + flags
  a printed summary with the flagged outliers listed

Usage:
    python tools/qc_cmrx2025_recon.py
    python tools/qc_cmrx2025_recon.py --root scratch/data/CMRxRecon2025/Cine_combined
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "result", "cmrx2025_recon_check")

# thresholds for flagging (deliberately loose -- catch gross failures, not borderline softness)
MAX_CENTROID_OFF = 15.0      # % of FOV
BODY_MIN_MM, BODY_MAX_MM = 200.0, 460.0
MAX_DEAD_FRAC = 0.20


def load(cid_dir):
    img = nib.load(os.path.join(cid_dir, "sax", "4d_recon.nii.gz"))
    a = np.transpose(np.asarray(img.dataobj, dtype=np.float32), (3, 2, 1, 0))  # (T,Z,Y,X)
    z = img.header.get_zooms()
    return a, (float(z[1]), float(z[0]), float(z[2]))


def norm(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99.5)
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def metrics(vol, sp):
    py, px, pz = sp
    m0 = vol.mean(axis=(0, 1))                     # (Y,X) time+slice average silhouette
    cy, cx = ndimage.center_of_mass(m0)
    off_y = (cy - (m0.shape[0] - 1) / 2) / m0.shape[0] * 100
    off_x = (cx - (m0.shape[1] - 1) / 2) / m0.shape[1] * 100
    # body extent: largest connected component above a robust absolute threshold
    m = ndimage.binary_opening(m0 > 0.15 * np.percentile(m0, 99.5), np.ones((3, 3)))
    m = ndimage.binary_closing(m, np.ones((9, 9)))
    lab, n = ndimage.label(m)
    if n:
        m = lab == (1 + int(np.argmax(ndimage.sum(m, lab, range(1, n + 1)))))
    ys, xs = np.where(m)
    bx = (xs.max() - xs.min() + 1) * px if len(xs) else 0.0
    by = (ys.max() - ys.min() + 1) * py if len(ys) else 0.0
    # dead slices: mean signal < 5% of the brightest slice
    e = vol[0].mean(axis=(1, 2)); e = e / max(e.max(), 1e-8)
    dead = float((e < 0.05).mean())
    return {
        "offset_y_pct": round(float(off_y), 1), "offset_x_pct": round(float(off_x), 1),
        "body_x_mm": round(float(bx)), "body_y_mm": round(float(by)),
        "aniso": round(max(px, py) / min(px, py), 2),
        "dead_slice_frac": round(dead, 2),
        "shape": [int(v) for v in vol.shape], "spacing": [round(py, 3), round(px, 3), round(pz, 1)],
    }


def flags(m):
    f = []
    if abs(m["offset_x_pct"]) > MAX_CENTROID_OFF or abs(m["offset_y_pct"]) > MAX_CENTROID_OFF:
        f.append("off-centre")
    if not (BODY_MIN_MM <= m["body_x_mm"] <= BODY_MAX_MM):
        f.append(f"body_x={m['body_x_mm']}mm")
    if m["dead_slice_frac"] > MAX_DEAD_FRAC:
        f.append(f"dead={m['dead_slice_frac']}")
    return f


def montage(entries, scanner, path):
    n = len(entries)
    ncol = min(8, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.4 * ncol, 2.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (cid, vol, sp, m, fl) in zip(axes, entries):
        py, px, _ = sp; Z, Y, X = vol.shape[1], vol.shape[2], vol.shape[3]
        ax.imshow(norm(vol[0, Z // 2]), cmap="gray", extent=[0, X * px, Y * py, 0], aspect="equal")
        ax.set_title(f"{cid.replace('CMRx25_','').replace('_'+scanner,'')}\n"
                     f"{m['body_x_mm']}x{m['body_y_mm']}mm" + (f"\n⚠{','.join(fl)}" if fl else ""),
                     fontsize=6, color=("red" if fl else "black"))
        if fl:
            for s in ax.spines.values():
                s.set_visible(True); s.set_color("red"); s.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("on")
    fig.suptitle(f"CMRxRecon2025 QC — {scanner}  (n={n})  — mid-z ED, true physical aspect, RED=flagged",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=95); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(REPO, "scratch", "data", "CMRxRecon2025", "Cine_combined"))
    ap.add_argument("--out", default=OUT,
                    help="output dir (default result/cmrx2025_recon_check). Use a distinct dir to "
                         "render a new recon version without overwriting the previous figures.")
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    by_scanner = defaultdict(list)
    report = []
    dirs = sorted(glob.glob(os.path.join(args.root, "CMRx25_*")))
    print(f"scanning {len(dirs)} reconstructed subjects", flush=True)
    for i, d in enumerate(dirs, 1):
        cid = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "sax", "4d_recon.nii.gz")):
            continue
        vol, sp = load(d)
        m = metrics(vol, sp); fl = flags(m)
        # scanner = the token before the trailing _P###
        scanner = "_".join(cid.split("_")[3:-1])
        by_scanner[scanner].append((cid, vol, sp, m, fl))
        report.append({"cid": cid, "scanner": scanner, "flags": fl, **m})
        if i % 40 == 0:
            print(f"  {i}/{len(dirs)}", flush=True)

    for scanner, entries in sorted(by_scanner.items()):
        montage(entries, scanner, os.path.join(out_dir, f"qc_montage_{scanner}.png"))

    json.dump(report, open(os.path.join(out_dir, "qc_report.json"), "w"), indent=1)
    flagged = [r for r in report if r["flags"]]
    print(f"\n=== QC SUMMARY: {len(report)} subjects, {len(flagged)} flagged ===")
    for r in flagged:
        print(f"  {r['cid']:55} {r['flags']}  (off {r['offset_y_pct']:+},{r['offset_x_pct']:+}%  "
              f"body {r['body_x_mm']}x{r['body_y_mm']}mm)")
    print(f"\nmontages + qc_report.json -> {out_dir}")


if __name__ == "__main__":
    main()
