"""Distribution of z-slice count (and cardiac phase count) across the pooled CMRxRecon cohort.

Reads the per-subject shapes recorded by tools/verify_recon_v2.py -- no volume is re-read.
Marks the canonical cube depth (Z=12), because subjects above it are CROPPED in z by the
canonical resample and subjects below it are zero-padded.

    micromamba run -n svr python tools/plot_cohort_shape_distribution.py
"""
import collections
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT = os.path.join(REPO, "scratch", "data", "recon_v2_verification_full.json")
OUT = os.path.join(REPO, "result", "cohort_shape_distribution.png")
CANON_Z = 12
# Archived after the report was written (2025 duplicate) -- exclude so counts match the live tree.
EXCLUDE = {"CMRx25_R2val_Center004_UIH_15T_umr680_P006"}
YEARS = ["2023", "2024", "2025"]
COLORS = {"2023": "#4C72B0", "2024": "#DD8452", "2025": "#55A868"}


def main():
    blob = json.load(open(REPORT))
    data = {}
    for y in YEARS:
        subs = [s for s in blob[y]["subjects"] if s["subject"] not in EXCLUDE]
        data[y] = {
            "n": len(subs),
            "z": collections.Counter(s["shape_4d"][2] for s in subs),
            "t": collections.Counter(s["shape_4d"][3] for s in subs),
        }

    zmin = min(min(d["z"]) for d in data.values())
    zmax = max(max(d["z"]) for d in data.values())
    bins = list(range(zmin, zmax + 1))

    fig, axes = plt.subplots(len(YEARS), 1, figsize=(9, 7.5), dpi=140, sharex=True)
    for ax, y in zip(axes, YEARS):
        d = data[y]
        counts = [d["z"].get(b, 0) for b in bins]
        ax.bar(bins, counts, color=COLORS[y], width=0.75, edgecolor="black", linewidth=0.4)
        for b, c in zip(bins, counts):
            if c:
                ax.text(b, c + max(counts) * 0.02, str(c), ha="center", fontsize=6.5)

        over = sum(v for k, v in d["z"].items() if k > CANON_Z)
        under = sum(v for k, v in d["z"].items() if k < CANON_Z)
        exact = d["z"].get(CANON_Z, 0)
        zs = sorted(k for k, v in d["z"].items() for _ in range(v))
        med = zs[len(zs) // 2]

        ax.axvline(CANON_Z, color="crimson", ls="--", lw=1.3)
        ax.text(CANON_Z + 0.12, max(counts) * 0.92, f"canonical Z={CANON_Z}",
                color="crimson", fontsize=7, va="top")
        tset = ", ".join(f"{k}" for k in sorted(d["t"]))
        ax.set_title(
            f"CMRxRecon{y}   n={d['n']}   z-slices {min(d['z'])}–{max(d['z'])} (median {med})   "
            f"|   phases T = {tset} for all {d['n']}   "
            f"|   z<12 padded: {under} ({under/d['n']:.0%})   "
            f"z=12: {exact}   z>12 CROPPED: {over} ({over/d['n']:.0%})",
            fontsize=8, loc="left")
        ax.set_ylabel("subjects", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, max(counts) * 1.18)

    axes[-1].set_xlabel("number of z slices (native, before the canonical resample)", fontsize=9)
    axes[-1].set_xticks(bins)

    tot = sum(d["n"] for d in data.values())
    tot_over = sum(sum(v for k, v in d["z"].items() if k > CANON_Z) for d in data.values())
    fig.suptitle(
        f"Pooled CMRxRecon cohort — native z-slice distribution   (n={tot}: "
        f"{data['2023']['n']} + {data['2024']['n']} + {data['2025']['n']})\n"
        f"Every subject has exactly T=12 cardiac phases. {tot_over} subjects "
        f"({tot_over/tot:.0%}) exceed the canonical Z={CANON_Z} and are cropped in z.",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")

    for y in YEARS:
        d = data[y]
        print(f"  {y}: n={d['n']}  T={dict(d['t'])}  Z={dict(sorted(d['z'].items()))}")


if __name__ == "__main__":
    main()
