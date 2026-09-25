"""Runtime–accuracy figure drafts for the paper (af12 test set, full 12-phase cine per subject).

y = LV volume–time curve (VTC) error (% EDV): mean per-phase |V_pred(t) - V_gt(t)| / EDV_gt. Chosen over
EF MAE because it scores all 12 phases, not just the ED/ES extremes.
Numbers come from _paper_figures/paper_results_table.py --json (the same file the paper table is built from),
so figure and table cannot drift.

    python _paper_figures/render_runtime_accuracy.py --table temp/allphase_seg/paper_tables.json --out temp/runtime_accuracy_drafts
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

TEXTW = 5.5  # ICLR text width (in)
# Okabe-Ito based + wine for SVRTK, validated all-pairs (dataviz validate_palette.js --pairs all: normal-vision
# PASS, worst CVD dE 7.6 = floor band, relieved like the low-contrast yellow/sky/pink by direct labels + distinct
# marker shapes).
STYLE = {  # name in table -> (label, color, marker, filled, dense-temporal)
    "Ours (diff1000)": ("Ours", "#D55E00", "o", True, False),
    "CiNeVol†": ("CiNeVol", "#009E73", "s", True, True),
    "Fetal CMR 4D†": ("Fetal CMR 4D", "#0072B2", "s", True, True),
    "SVRTK": ("SVRTK", "#882255", "o", False, False),
    "NiftyMIC": ("NiftyMIC", "#E69F00", "D", False, False),
    "NeSVoR": ("NeSVoR", "#CC79A7", "v", False, False),
    "Dangi": ("Dangi et al.", "#56B4E9", "^", False, False),
}
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e3"
METRICS = {"EF": ("EF MAE (%)", -1), "NCC": ("NCC", 1), "Dice": ("Dice", 1), "VTC err.": ("VTC err.", -1),
           "PSNR": ("PSNR (dB)", 1), "SSIM": ("SSIM", 1)}
Y = "VTC err."
YLAB = "VTC err. $\\downarrow$"
YLIM = (0, 24)
XLAB = "Runtime (s)"
# label offsets (points) for the VTC layout: SVRTK (167 s, 20.3) and NiftyMIC (232 s, 20.6) nearly overlap
OFFS = {"Ours": (8, 0, "left"), "CiNeVol": (-8, 0, "right"), "Fetal CMR 4D": (0, -10, "center"),
        "SVRTK": (-8, 0, "right"), "NiftyMIC": (0, -10, "center"), "NeSVoR": (8, 2, "left"),
        "Dangi et al.": (8, 0, "left")}


def paper_rc(family="sans-serif"):
    # paper figures use Arial (user decision 2026-09-25; install: docs/127 §9); math in STIX sans so
    # \mathcal{L} keeps its calligraphic form
    plt.rcParams.update({
        "font.family": family,
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def pts(tab):
    return [(STYLE[k], v["Runtime (s)"], v) for k, v in tab.items() if k in STYLE]


def mark(ax, st, x, y, size=34, ours_size=70):
    label, c, m, filled, _ = st
    ours = label == "Ours"
    ax.scatter([x], [y], s=ours_size if ours else size, marker=m, zorder=4 if ours else 3,
               facecolor=c if filled else "white", edgecolor="#7a2e00" if ours else c, linewidth=1.0 if ours else 1.3)


def log_x(ax, lo=0.07, hi=1500):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1, 10, 100, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(True, which="major", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def label_pts(ax, P, key, offs, fs=7):
    for st, x, v in P:
        dx, dy, ha = offs.get(st[0], (6, 0, "left"))
        ax.annotate(st[0] + ("$^\\dagger$" if st[4] else ""), (x, v[key]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=fs, color=INK,
                    fontweight="bold" if st[0] == "Ours" else "normal")


def legend_handles():
    hs = []
    for label, c, m, filled, dense in STYLE.values():
        hs.append(Line2D([], [], linestyle="", marker=m, markersize=7 if label == "Ours" else 5.5,
                         markerfacecolor=c if filled else "white", markeredgecolor="#7a2e00" if label == "Ours" else c,
                         markeredgewidth=1.0 if label == "Ours" else 1.3, label=label + ("$^\\dagger$" if dense else "")))
    return hs


def top_legend(obj, y=1.02):
    leg = obj.legend(handles=legend_handles(), ncol=len(STYLE), loc="lower center", bbox_to_anchor=(0.5, y),
                     frameon=False, handletextpad=0.2, columnspacing=0.9)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_fontweight("bold")


def single(P, figsize=(TEXTW, 2.3), size=34, ours_size=70):
    fig, ax = plt.subplots(figsize=figsize)
    for st, x, v in P:
        mark(ax, st, x, v[Y], size, ours_size)
    log_x(ax)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(YLAB)
    return fig, ax


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}/{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def d1_loglog_legend_top(P, out):
    fig, ax = single(P, (TEXTW, 2.4))
    ax.set_yscale("log")
    ax.set_ylim(5.5, 30)
    ax.yaxis.set_major_locator(FixedLocator([6, 8, 10, 15, 20, 30]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    top_legend(ax)
    save(fig, out, "d1_loglog_legend_top")


def d2_direct_labels(P, out, family="serif", name="d2_logx_direct_labels"):
    paper_rc(family)
    fig, ax = single(P)
    label_pts(ax, P, Y, OFFS)
    save(fig, out, name)   # "†dense temporal input" goes in the paper caption
    paper_rc("serif")


def d3_half_width(P, out):
    fig, ax = single(P, (2.75, 2.2), 26, 55)
    log_x(ax, 0.07, 2500)
    ax.set_ylim(0, 25)
    ax.set_xlabel(XLAB)
    offs = {"Ours": (6, 0, "left"), "CiNeVol": (-6, 0, "right"), "Fetal CMR 4D": (0, -9, "center"),
            "SVRTK": (-6, 0, "right"), "NiftyMIC": (0, -9, "center"), "NeSVoR": (0, 8, "center"),
            "Dangi et al.": (6, 0, "left")}
    label_pts(ax, P, Y, offs, fs=6.5)
    save(fig, out, "d3_half_width_wrapfigure")


def panels(P, out, keys, name, h=2.2):
    fig, axs = plt.subplots(1, len(keys), figsize=(TEXTW, h), constrained_layout=True)
    for ax, k in zip(axs, keys):
        lab, sgn = METRICS[k]
        for st, x, v in P:
            mark(ax, st, x, v[k], size=26, ours_size=55)
        log_x(ax)
        ax.margins(y=0.1)   # keep extreme markers off the axis line
        if len(keys) > 2:
            ax.xaxis.set_major_locator(FixedLocator([0.1, 10, 1000]))
        ax.set_ylabel(lab + (" $\\uparrow$" if sgn > 0 else " $\\downarrow$"))
        if len(keys) <= 2:
            ax.set_xlabel(XLAB)
    if len(keys) > 2:
        fig.supxlabel(XLAB, fontsize=8)
    top_legend(fig, 1.0)
    save(fig, out, name)


def d6_pareto(P, out):
    fig, ax = single(P, (TEXTW, 2.4))
    ax.axvspan(0.07, 10, color="#f4ece6", zorder=0, lw=0)
    ax.text(0.085, YLIM[1] - 0.8, "under 10 s", fontsize=6.5, color=MUTED, va="top")
    label_pts(ax, P, Y, dict(OFFS, CiNeVol=(0, 9, "center")))
    d = {st[0]: (x, v[Y]) for st, x, v in P}
    ox, oy = d["Ours"]
    cx, cy = d["CiNeVol"]
    ax.annotate("", xy=(ox * 1.5, oy + 0.9), xytext=(cx / 1.4, cy),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.7))
    ax.text(ox * 1.25, oy - 2.4, f"{cx / ox:.0f}$\\times$ faster, {cy / oy:.1f}$\\times$ lower error than CiNeVol",
            ha="left", va="center", fontsize=6.8, color=MUTED)
    top_legend(ax)
    save(fig, out, "d6_pareto_speedup")


def d7_legend_right(P, out):
    fig, ax = single(P)
    ax.legend(handles=legend_handles(), loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
              handletextpad=0.4, labelspacing=0.7, title="$^\\dagger$dense temporal input", title_fontsize=6.3)
    save(fig, out, "d7_legend_right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.out):                      # drafts only: clear the previous EF-based set
        if f.startswith("d") and f.endswith((".png", ".pdf")):
            os.remove(os.path.join(a.out, f))
    P = pts(json.load(open(a.table))["af12_main"])
    paper_rc()
    d1_loglog_legend_top(P, a.out)
    d2_direct_labels(P, a.out)
    d3_half_width(P, a.out)
    panels(P, a.out, ("VTC err.", "SSIM"), "d4_two_panel_VTC_SSIM")
    panels(P, a.out, ("VTC err.", "PSNR"), "d4_two_panel_VTC_PSNR")
    panels(P, a.out, ("NCC", "PSNR", "Dice", "VTC err."), "d5_four_panel", h=2.0)
    d6_pareto(P, a.out)
    d7_legend_right(P, a.out)
    d2_direct_labels(P, a.out, family="sans-serif", name="d8_direct_labels_sans")
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
