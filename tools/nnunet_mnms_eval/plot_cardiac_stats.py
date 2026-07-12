"""Visualize cardiac-phase statistics (ED/ES timing, EF, EDV/ESV) across datasets.

Reads scratch/data/whs/cardiac_phase.csv. ED/ES normalized to cycle-fraction idx/(T-1) so datasets
with different phase counts (CMRx 12, MIITT 30, OCMR ~20, ACDC ~30) compare. Uses box + jittered
points (box = summary, points = the actual data + small-n honesty + CMRx's ES quantization).
Excludes unimodal_ok==0 (broken segs). EF-by-pathology panel appears when ACDC is present.
"""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "scratch/data/whs/cardiac_phase.csv"
OUT = "result/whs_check/cardiac_stats.png"
DPAL = {"cmrx": "#4C72B0", "miitt": "#DD8452", "ocmr": "#55A868", "acdc": "#8172B3"}
GPAL = {"NOR": "#55A868", "MINF": "#DD8452", "DCM": "#C44E52", "HCM": "#8172B3", "RV": "#4C72B0"}
DS_ORDER = ["cmrx", "miitt", "ocmr", "acdc"]
G_ORDER = ["NOR", "MINF", "DCM", "HCM", "RV"]


def load():
    rows = [r for r in csv.DictReader(open(CSV)) if r["EF_pct"] != "" and int(r["unimodal_ok"]) == 1]
    for r in rows:
        T, ed, es = int(r["T"]), int(r["ED"]), int(r["ES"])
        r["_edf"], r["_esf"] = ed / (T - 1), es / (T - 1)    # raw fraction in [0,1] (viz only)
        for k in ("EDV_mL", "ESV_mL", "EF_pct"):
            r[k] = float(r[k])
    return rows


def boxstrip(ax, groups, title, ylab, colors, ref=None, seed=0):
    labels = list(groups.keys())
    data = [groups[k] for k in labels]
    bp = ax.boxplot(data, showfliers=False, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black", lw=1.8),
                    whiskerprops=dict(color="gray"), capprops=dict(color="gray"),
                    boxprops=dict(edgecolor="gray"))
    for patch, k in zip(bp["boxes"], labels):
        patch.set_facecolor(colors.get(k, "gray")); patch.set_alpha(0.25)
    for i, k in enumerate(labels):
        x = np.random.RandomState(seed).normal(i + 1, 0.07, len(groups[k]))
        ax.scatter(x, groups[k], s=11, alpha=0.55, color=colors.get(k, "gray"), edgecolor="none", zorder=3)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{k}\n(n={len(groups[k])})" for k in labels])
    ax.set_title(title, fontsize=11); ax.set_ylabel(ylab)
    if ref is not None:
        ax.axhline(ref, ls="--", c="gray", lw=1)


def main():
    rows = load()
    dss = [d for d in DS_ORDER if any(r["dataset"] == d for r in rows)]
    by = lambda key: {d: [r[key] for r in rows if r["dataset"] == d] for d in dss}
    has_acdc = "acdc" in dss

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.5))
    boxstrip(ax[0], by("_edf"), "ED timing", "ED frame / (T-1)", DPAL, ref=0)
    ax[0].set_ylim(-0.03, 1.03)
    boxstrip(ax[1], by("_esf"), "ES timing", "ES frame / (T-1)", DPAL)
    ax[1].axhspan(0.30, 0.45, color="gray", alpha=0.08, zorder=0)   # normal end-systolic window
    boxstrip(ax[2], by("EF_pct"), "Ejection fraction", "EF (%)", DPAL, ref=55)

    a = ax[3]        # EF by ACDC pathology (only ACDC has Group labels)
    if has_acdc:
        gg = {g: [r["EF_pct"] for r in rows if r["dataset"] == "acdc" and r["group"] == g] for g in G_ORDER}
        gg = {g: v for g, v in gg.items() if v}
        boxstrip(a, gg, "ACDC EF by pathology", "EF (%)", GPAL, ref=55)
    else:
        a.set_title("ACDC EF by pathology (pending seg)"); a.axis("off")

    fig.suptitle(f"Cardiac-phase statistics — {len(rows)} gated subjects "
                 f"({', '.join(dss)}; 2 flagged excluded)", fontsize=13, y=1.02)
    plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
    print("saved", OUT)

    print(f"\n{'dataset':8} {'n':>4} {'ED_frac':>9} {'ES_frac':>10} {'EF%':>10} {'EDV':>9}")
    for d in dss:
        r = [x for x in rows if x["dataset"] == d]
        edf = np.array([x["_edf"] for x in r]); esf = np.array([x["_esf"] for x in r])
        ef = np.array([x["EF_pct"] for x in r]); edv = np.array([x["EDV_mL"] for x in r])
        print(f"{d:8} {len(r):>4} {edf.mean():>6.2f} {esf.mean():>6.2f}±{esf.std():.2f} "
              f"{ef.mean():>6.1f}±{ef.std():.0f} {edv.mean():>6.0f}")
    if has_acdc:
        print("\nACDC EF by pathology:")
        for g in G_ORDER:
            v = [r["EF_pct"] for r in rows if r["dataset"] == "acdc" and r["group"] == g]
            if v:
                print(f"  {g:5} n={len(v):3} EF {np.mean(v):5.1f} ± {np.std(v):.1f}  [{min(v):.0f},{max(v):.0f}]")


if __name__ == "__main__":
    main()
