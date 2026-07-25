#!/usr/bin/env python3
"""Scan a directory of NIfTI volumes and report geometry distributions + a figure.

Dataset-agnostic: works on ANY dataset of 3D/4D NIfTIs (cine or static). Reads
headers ONLY (nibabel lazy-load — no pixel data), so it is reasonably fast even
on large sets. Reports, over all matched files, the distributions of:

    T (cardiac phases / timepoints, if 4D), Z (slices), in-plane X/Y,
    in-plane spacing, Z (slice) spacing, and orientation (nibabel axcodes).

and saves a 4-panel distribution figure (slices / Z-spacing / orientation / T-frames).

Usage
-----
    micromamba run -n svr python tools/scan_dataset_geometry.py <root_dir> \
        [--glob '**/*.nii.gz'] [--name "My Dataset"] \
        [--out result/<name>/geometry_scan.png] [--limit N]

Examples
--------
    # M&Ms-1 short-axis cines
    python tools/scan_dataset_geometry.py scratch/data/MNMs/MNMs1 --glob '**/*_sa.nii.gz' --name "M&Ms-1"
    # any dataset, all nii.gz
    python tools/scan_dataset_geometry.py scratch/data/ACDC --name "ACDC"

Notes
-----
- Axis order assumed nibabel/(X,Y,Z[,T]); orientation is the raw stored axcodes
  (LPS is the training-canonical target — see project CLAUDE.md LPS rule).
- Written 2026-07-24 (generalized from tools/scan_mnms1.py, which additionally
  cross-tabs the M&Ms CSV metadata: vendor / pathology / centre / split).
"""
from __future__ import annotations
import argparse, os, glob, collections
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def scan(root: str, pattern: str, limit: int | None):
    files = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
    if limit:
        files = files[:limit]
    rows = []
    for f in files:
        try:
            im = nib.load(f)                       # lazy: no pixel load
            sh = im.shape
            X, Y, Z = (list(sh) + [1, 1, 1])[:3]
            T = sh[3] if len(sh) > 3 else 1
            zx, zy, zz = [float(v) for v in (list(im.header.get_zooms()) + [0, 0, 0])[:3]]
            ax = "".join(nib.aff2axcodes(im.affine))
            rows.append(dict(X=X, Y=Y, Z=Z, T=T, zx=zx, zy=zy, zz=zz, ax=ax))
        except Exception as e:
            print(f"  [skip] {f}: {e}")
    return files, rows


def counts(vals, round_dp=None):
    if round_dp is not None:
        vals = [round(v, round_dp) for v in vals]
    return dict(sorted(collections.Counter(vals).items()))


def stat_line(name, vals):
    v = np.asarray(vals, float)
    return f"  {name:16} min={v.min():.4g}  max={v.max():.4g}  median={np.median(v):.4g}  mean={v.mean():.4g}"


def report(name, rows):
    n = len(rows)
    print(f"\n===== {name}: {n} volumes =====")
    T = [r["T"] for r in rows]
    print(stat_line("T (frames)", T) + f"   hist={counts(T)}")
    print(stat_line("Z (slices)", [r["Z"] for r in rows]) + f"   hist={counts([r['Z'] for r in rows])}")
    print(stat_line("X in-plane", [r["X"] for r in rows]))
    print(stat_line("Y in-plane", [r["Y"] for r in rows]))
    print(stat_line("spacing x", [r["zx"] for r in rows]))
    print(stat_line("spacing y", [r["zy"] for r in rows]))
    zz = [r["zz"] for r in rows]
    print(stat_line("spacing z", zz) + f"   hist={counts(zz, round_dp=2)}")
    print(f"  {'orientation':16} {counts([r['ax'] for r in rows])}")
    return n


def _bar(ax, d, title, xlabel, color, max_unique_before_hist=24):
    """Bar of value->count; falls back to a binned histogram if too many unique values."""
    if len(d) > max_unique_before_hist:
        vals = np.array([k for k, c in d.items() for _ in range(c)], float)
        ax.hist(vals, bins=24, color=color, edgecolor="black", linewidth=0.4)
    else:
        ks = list(d); vs = [d[k] for k in ks]
        xs = np.arange(len(ks))
        ax.bar(xs, vs, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(xs); ax.set_xticklabels([f"{k:g}" for k in ks], rotation=0)
        for x, v in zip(xs, vs):
            if v > 0:
                ax.text(x, v + max(vs) * 0.01, str(v), ha="center", fontsize=7)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel(xlabel); ax.set_ylabel("# volumes")


def figure(name, rows, out):
    n = len(rows)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{name} — geometry distributions (n={n})", fontsize=14, fontweight="bold")
    _bar(ax[0, 0], counts([r["Z"] for r in rows]), "Number of slices (Z)", "slices per stack", "#4C78A8")
    _bar(ax[0, 1], counts([r["zz"] for r in rows], round_dp=2), "Z (slice) spacing", "mm", "#F58518")
    _bar(ax[1, 1], counts([r["T"] for r in rows]), "Timepoints (T frames)", "frames per volume", "#54A24B")
    # orientation: horizontal bar, LPS highlighted
    oc = sorted(counts([r["ax"] for r in rows]).items(), key=lambda kv: -kv[1])
    labels = [k for k, _ in oc]; vals = [v for _, v in oc]
    cols = ["#E45756" if k == "LPS" else "#B279A2" for k in labels]
    ys = np.arange(len(labels))[::-1]
    a = ax[1, 0]
    a.barh(ys, vals, color=cols, edgecolor="black", linewidth=0.5)
    a.set_yticks(ys); a.set_yticklabels(labels)
    a.set_title(f"Orientation (axcodes) — LPS(red)={counts([r['ax'] for r in rows]).get('LPS',0)}/{n} canonical",
                fontweight="bold")
    a.set_xlabel("# volumes")
    for y, v in zip(ys, vals):
        a.text(v + max(vals) * 0.01, y, str(v), va="center", fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nfigure saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="dataset root directory")
    ap.add_argument("--glob", default="**/*.nii.gz", help="recursive glob under root (default '**/*.nii.gz')")
    ap.add_argument("--name", default=None, help="display name (default: basename of root)")
    ap.add_argument("--out", default=None, help="figure path (default result/<name>/geometry_scan.png)")
    ap.add_argument("--limit", type=int, default=None, help="scan only first N files (quick sample)")
    a = ap.parse_args()
    name = a.name or os.path.basename(os.path.normpath(a.root))
    out = a.out or os.path.join("result", name.replace("/", "_").replace(" ", "_"), "geometry_scan.png")
    files, rows = scan(a.root, a.glob, a.limit)
    if not rows:
        print(f"No NIfTIs matched {a.root}/{a.glob}"); return
    report(name, rows)
    figure(name, rows, out)


if __name__ == "__main__":
    main()
