"""LV volume-time curve (VTC) figure for one af12 test subject: GT vs Ours vs every baseline.

Curves are the stored per-frame LV volumes (evaluation/metric_results/test/<cohort>/ef/<method>.json,
`lv_curve_gt` / `lv_curve_breath`), one value per acquisition frame. Frames are evenly spaced; the
simulator's time unit is the nominal R-R (DT = 1/12), shown as seconds assuming a nominal R-R of 1 s
(state that in the caption). Frame f covers [f/12, (f+1)/12) s and is plotted at its midpoint. Lines are drawn
with shape-preserving PCHIP interpolation through the 12 samples (no overshoot beyond neighbouring
samples). Style/palette match tools/render_runtime_accuracy.py.

    python tools/render_vtc_figure.py --cohort mnms_af12 --subject MNMs_Q4W5Z8 --out temp/vtc_figure
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

from render_runtime_accuracy import INK, GRID, TEXTW, paper_rc  # noqa: E402

# method file -> (label, color, dense-temporal); colors = render_runtime_accuracy.STYLE
METHODS = [
    ("svrtk3d_debug_scatter", "SVRTK", "#882255", False),
    ("niftymic_scatter", "NiftyMIC", "#E69F00", False),
    ("nesvor_4gpu_scatter", "NeSVoR", "#CC79A7", False),
    ("dangi_scatter_4gpu", "Dangi et al.", "#56B4E9", False),
    ("fetal_cmr_4d", "Fetal CMR 4D$^\\dagger$", "#0072B2", True),
    ("cinevol_motion_masked", "CiNeVol$^\\dagger$", "#009E73", True),
    ("vggt_final518_diff1000_ep300", "Ours", "#D55E00", False),
]


def curves(cohort, subject):
    out, gt = {}, None
    for m, *_ in METHODS:
        d = json.load(open(f"evaluation/metric_results/test/{cohort}/ef/{m}.json"))["per_subject"]
        s = next(x for x in d if x["subject"] == subject)
        out[m] = np.asarray(s["lv_curve_breath"], float)
        g = np.asarray(s["lv_curve_gt"], float)
        assert gt is None or np.allclose(gt, g), f"{m}: GT curve differs"
        gt = g
    return gt, out


def smooth(ax, y, **kw):
    t = (np.arange(len(y)) + 0.5) / len(y)
    tt = np.linspace(t[0], t[-1], 300)
    ax.plot(tt, PchipInterpolator(t, y)(tt), **kw)


def draw_vtc(ax, gt, cv, fade=0.55):
    ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
    for m, label, c, dense in METHODS:
        ours = label == "Ours"
        smooth(ax, cv[m], color=c, lw=1.8 if ours else 1.0, ls="--" if dense else "-",
               alpha=1.0 if ours else fade, zorder=4 if ours else 2, label=label)
    smooth(ax, gt, color=INK, lw=1.8, zorder=5, label="GT")
    ax.set_xlabel("Acquisition time (s)")
    ax.set_ylabel("LV volume (mL)")
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))


def combined(gt, cv, table, out, direct_labels):
    """Runtime-accuracy scatter (left, = paper runtime_accuracy.pdf data) + VTC curves (right), one
    shared top legend whose handles carry both the scatter marker and the curve line style."""
    import render_runtime_accuracy as ra
    P = ra.pts(json.load(open(table))["af12_main"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(TEXTW, 2.3), constrained_layout=True)
    for st, x, v in P:
        ra.mark(a0, st, x, v[ra.Y], 26, 55)
    ra.log_x(a0, 0.07, 2500)
    a0.set_ylim(0, 25)
    a0.set_xlabel(ra.XLAB)
    a0.set_ylabel(ra.YLAB)
    if direct_labels:
        offs = {"Ours": (6, 0, "left"), "CiNeVol": (-6, 0, "right"), "Fetal CMR 4D": (0, -9, "center"),
                "SVRTK": (-6, 0, "right"), "NiftyMIC": (0, -9, "center"), "NeSVoR": (0, 8, "center"),
                "Dangi et al.": (6, 0, "left")}
        ra.label_pts(a0, P, ra.Y, offs, fs=6.5)
    draw_vtc(a1, gt, cv)
    style = {st[0]: st for st in ra.STYLE.values()}
    hs = [Line2D([], [], color=INK, lw=1.8, label="GT")]
    for _, label, c, dense in reversed(METHODS):          # Ours first, then the table's reverse order
        name = label.replace("$^\\dagger$", "")
        _, _, mk, filled, _ = style[name]
        ours = name == "Ours"
        hs.append(Line2D([], [], color=c, lw=1.8 if ours else 1.0, ls="--" if dense else "-", marker=mk,
                         markersize=5 if ours else 4, markerfacecolor=c if filled else "white",
                         markeredgecolor="#7a2e00" if ours else c, label=label))
    leg = fig.legend(handles=hs, ncol=len(hs), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False,
                     handlelength=2.0, handletextpad=0.3, columnspacing=0.7, fontsize=6.5)
    for t in leg.get_texts():
        if t.get_text() == "Ours":
            t.set_fontweight("bold")
    name = "combined_runtime_vtc" + ("_labels" if direct_labels else "")
    fig.savefig(os.path.join(out, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight", pad_inches=0.02, dpi=250)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="cmrx2023_af12")
    ap.add_argument("--subject", default="CMRx23_Train_P045")  # paper figure; earlier draft: mnms_af12 MNMs_Q4W5Z8
    ap.add_argument("--out", default="temp/vtc_figure")
    ap.add_argument("--table", default="temp/allphase_seg/paper_tables_v2.json",
                    help="tools/paper_results_table.py --json output, for the combined runtime+VTC variants")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paper_rc()
    gt, cv = curves(a.cohort, a.subject)
    for dl in (False, True):
        combined(gt, cv, a.table, a.out, dl)

    for name, fade in (("vtc_faded", 0.55),):
        fig, ax = plt.subplots(figsize=(TEXTW * 0.6, 2.2))
        draw_vtc(ax, gt, cv, fade)
        h, l = ax.get_legend_handles_labels()
        order = [l.index("GT"), l.index("Ours")] + [i for i, s in enumerate(l) if s not in ("GT", "Ours")]
        ax.legend([h[i] for i in order], [l[i] for i in order], loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=False, handlelength=2.2)
        fig.savefig(os.path.join(a.out, f"{name}.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(a.out, f"{name}.png"), bbox_inches="tight", dpi=250)
        plt.close(fig)


if __name__ == "__main__":
    main()
