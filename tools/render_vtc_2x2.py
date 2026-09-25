"""2x2 runtime-accuracy + LV volume-time curve figure: AF row (af12) and HRV row (hrv12), one subject.

Left column = runtime vs VTC err. (paper_results_table.py --json: af12_main / hrv12_appendix).
Runtime was measured on af12 only (docs/120 §6), so the HRV row reuses the af12 runtimes -- say so in
the caption. Right column = LV curves of one subject under each rhythm (render_vtc_figure style).

    python tools/render_vtc_2x2.py --subject CMRx23_Train_P045 --out temp/vtc_2x2
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import render_runtime_accuracy as ra  # noqa: E402
from render_vtc_figure import METHODS, smooth  # noqa: E402
from render_runtime_accuracy import GRID, INK, TEXTW, paper_rc  # noqa: E402

# af12 method file -> hrv12 method file (paper_results_table.BASELINES; the rest are identical)
HRV_FILE = {"nesvor_4gpu_scatter": "nesvor_scatter", "dangi_scatter_4gpu": "dangi_scatter",
            "cinevol_motion_masked": "cinevol_masked"}
ROWS = [("AF", "af12", "af12_main"), ("HRV", "hrv12", "hrv12_appendix")]
TITLES = {"AF": "(a) Atrial fibrillation (AF)", "HRV": "(b) Heart rate variability (HRV)"}


def pareto(ax, pts, xmax):
    """Staircase through the non-dominated points (lower runtime AND lower VTC err. are better)."""
    front, best = [], np.inf
    for x, y in sorted(pts):
        if y < best:
            front.append((x, y))
            best = y
    ax.plot(*zip(*front), color="#888888", lw=1.0, ls="--", zorder=1)
    return front


def curves(cohort, subject, arm):
    out, gt = {}, None
    for m, *_ in METHODS:
        f = HRV_FILE.get(m, m) if arm == "hrv12" else m
        d = json.load(open(f"evaluation/metric_results/test/{cohort}/ef/{f}.json"))["per_subject"]
        s = next(x for x in d if x["subject"] == subject)
        out[m] = np.asarray(s["lv_curve_breath"], float)
        g = np.asarray(s["lv_curve_gt"], float)
        assert gt is None or np.allclose(gt, g), f"{f}: GT curve differs"
        gt = g
    return gt, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="cmrx2023")
    ap.add_argument("--subject", default="CMRx23_Train_P045")
    ap.add_argument("--table", default="temp/allphase_seg/paper_tables_v2.json")
    ap.add_argument("--out", default="temp/vtc_2x2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paper_rc()
    tab = json.load(open(a.table))
    runtime = {k: v["Runtime (s)"] for k, v in tab["af12_main"].items()}   # measured on af12 only

    s = 1.8                                               # drawn at 1.8x text width, fonts scaled to match
    plt.rcParams.update({k: plt.rcParams[k] * s for k in
                         ("font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize")})
    fig = plt.figure(figsize=(TEXTW * s, 2.6), layout="constrained")
    subs = fig.subfigures(1, 2, wspace=0.02)
    axs = [s.subplots(1, 2) for s in subs]               # (a) AF: runtime, VTC | (b) HRV: runtime, VTC
    ymax = 0
    for r, (name, arm, key) in enumerate(ROWS):
        rows = {k: dict(v, **{"Runtime (s)": runtime[k]}) for k, v in tab[key].items()}
        P = ra.pts(rows)
        a0, a1 = axs[r]
        subs[r].supxlabel(TITLES[name], fontsize=8 * s)
        pareto(a0, [(x, v[ra.Y]) for _, x, v in P], 2500)
        for st, x, v in P:
            ra.mark(a0, st, x, v[ra.Y], 22, 48)
        ra.log_x(a0, 0.07, 2500)
        a0.set_ylim(0, 25)
        a0.set_ylabel(ra.YLAB)
        a0.set_xlabel(ra.XLAB)

        gt, cv = curves(f"{a.source}_{arm}", a.subject, arm)
        a1.grid(True, color=GRID, linewidth=0.5, zorder=0)
        for m, label, c, dense in METHODS:
            ours = label == "Ours"
            smooth(a1, cv[m], color=c, lw=1.8 if ours else 1.0, ls="--" if dense else "-",
                   alpha=1.0 if ours else 0.55, zorder=4 if ours else 2)
        smooth(a1, gt, color=INK, lw=1.8, zorder=5)
        a1.set_ylabel("LV volume (mL)")
        a1.set_xlim(0, 1)
        a1.set_xticks(np.arange(0, 1.01, 0.2))
        ymax = max(ymax, a1.get_ylim()[1])
        a1.set_xlabel("Acquisition time (s)")
    for r in range(2):                                    # same volume scale in both rows
        axs[r][1].set_ylim(axs[r][1].get_ylim()[0], ymax)

    style = {st[0]: st for st in ra.STYLE.values()}
    hs = [Line2D([], [], color=INK, lw=1.8, label="GT")]
    for _, label, c, dense in reversed(METHODS):
        nm = label.replace("$^\\dagger$", "")
        _, _, mk, filled, _ = style[nm]
        ours = nm == "Ours"
        hs.append(Line2D([], [], color=c, lw=1.8 if ours else 1.0, ls="--" if dense else "-", marker=mk,
                         markersize=5 if ours else 4, markerfacecolor=c if filled else "white",
                         markeredgecolor="#7a2e00" if ours else c, label=label))
    leg = fig.legend(handles=hs, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                     handlelength=2.0, handletextpad=0.3, columnspacing=0.7, fontsize=6.5 * s)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_fontweight("bold")
    for ext, kw in (("pdf", {}), ("png", {"dpi": 250})):
        fig.savefig(os.path.join(a.out, f"runtime_vtc_2x2.{ext}"), bbox_inches="tight", pad_inches=0.02, **kw)


if __name__ == "__main__":
    main()
