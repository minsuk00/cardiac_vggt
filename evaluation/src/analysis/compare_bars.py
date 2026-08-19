#!/usr/bin/env python
"""compare_bars.py — bar-figure companion to compare_table.py: PSNR/SSIM/NCC across arms of a
dataset, clean vs breath, from the git-tracked cohort summaries (metric_results/<ds>/<arm>.json).

The visual read of compare_table's numbers. Reveals the breathing-robustness gap: classical SVR is
strong on CLEAN input but collapses under BREATHING, while VGGT holds. Pure metric_results/ read — no
recompute. Writes to comparison_figures/<ds>/compare_bars.png (GPFS, gitignored) unless --out overrides.

Run:
  python evaluation/src/analysis/compare_bars.py cmrx2024
  python evaluation/src/analysis/compare_bars.py cmrx2024 --arms svrtk3d nesvor vggt_augaggr224hw2_ep300
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()
ROOT = next(p for p in HERE.parents if (p / "evaluation").is_dir())
sys.path.insert(0, str(ROOT / "evaluation"))
import paths  # noqa: E402

METRICS = [("PSNR (dB)", "psnr"), ("SSIM", "ssim"), ("NCC", "ncc")]


def short(arm):
    """Compact display name: strip the vggt_<date>_1f_ prefix + _ep## suffix; baselines pass through."""
    return re.sub(r"_ep\d+.*$", "", re.sub(r"^vggt_\d+_1f_", "", arm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(paths.DATASETS))
    ap.add_argument("--arms", nargs="*", default=None, help="default: every arm with a results json")
    ap.add_argument("--out", default=None, help="default: comparison_figures/<ds>/compare_bars.png")
    a = ap.parse_args()

    rdir = paths.RESULTS / a.dataset
    files = ([rdir / f"{arm}.json" for arm in a.arms] if a.arms else sorted(rdir.glob("*.json")))
    arms, data = [], {}                                            # data[metric] = ([clean...],[breath...])
    for f in files:
        if not f.is_file():
            print(f"  !! missing results json: {f.name}", file=sys.stderr)
            continue
        allm = (json.load(open(f)).get("all") or {})
        arms.append(short(f.stem))
        for _, m in METRICS:
            # A breath-only cohort has no clean mean; aggregate.py encodes that as JSON null, so
            # normalize both "key absent" and "[null, null]" to NaN — matplotlib skips NaN bars but
            # raises on None.
            def _mean(key):
                v = allm.get(key) or [np.nan]
                return np.nan if v[0] is None else v[0]
            c, b = _mean(f"clean_{m}"), _mean(f"breath_{m}")
            data.setdefault(m, ([], []))
            data[m][0].append(c); data[m][1].append(b)
    if not arms:
        sys.exit(f"no results found under {rdir}")

    x = np.arange(len(arms)); w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(4.6 * 3, 4.6))
    for ax, (label, m) in zip(axes, METRICS):
        clean, breath = data[m]
        ax.bar(x - w / 2, clean, w, label="clean input", color="#b8c4d0")
        ax.bar(x + w / 2, breath, w, label="breathing input", color="#c0392b")
        for xi, (c, b) in enumerate(zip(clean, breath)):
            fmt = (lambda v: f"{v:.1f}") if m == "psnr" else (lambda v: f"{v:.3f}")
            if np.isfinite(c):
                ax.text(xi - w / 2, c, fmt(c), ha="center", va="bottom", fontsize=6.5)
            if np.isfinite(b):
                ax.text(xi + w / 2, b, fmt(b), ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(arms, fontsize=8, rotation=15, ha="right")
        ax.set_title(f"{a.dataset} — {label}", fontsize=11)
        if m != "psnr":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        if m == "psnr":
            ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("clean vs breathing input, heart ROI — classical SVR is strong on CLEAN but collapses "
                 "under BREATHING; VGGT is breathing-robust", fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = a.out or str(paths.cohort_fig_dir(a.dataset) / "compare_bars.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135); plt.close(fig)
    print(f"-> {out}  ({len(arms)} arms x 3 metrics)")


if __name__ == "__main__":
    main()
