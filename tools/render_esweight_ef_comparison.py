"""Visual comparison of the ES-weight arm sweep (docs/93/94): true-EF vs predicted-EF
scatter grid + a grouped bar comparison of bias/SV-ratio/r/MAE across arms.

Read-only: only reads temp/dice_ef_sweep/mirror_full/results/_ef/*.json (same source as
compare_ef.py). Writes two PNGs to temp/dice_ef_sweep/.
"""
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

ROOT = "temp/dice_ef_sweep/mirror_full/results/_ef"
OUT_DIR = "temp/dice_ef_sweep"

# Fixed categorical order (dataviz skill reference palette), one hue per arm, held
# constant across both figures so "arm" reads as the same color everywhere.
ARMS = [
    ("noreg_ep300", "no-reg (baseline)", "#2a78d6"),      # slot 1 blue
    ("motion10_ep300", "motion10", "#eb6834"),             # slot 2 orange
    ("edes_samp_ep300", "edes_samp", "#1baf7a"),            # slot 3 aqua
    ("motion10_edes_ep300", "motion10_edes", "#eda100"),    # slot 4 yellow
    ("motion10_nohw_ep300", "motion10_nohw", "#e87ba4"),    # slot 5 magenta
    ("twophase05_ep300", "twophase05 (ref)", "#008300"),    # slot 6 green
]
BASE_ARM = "noreg_ep300"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def load(arm):
    d = json.load(open(f"{ROOT}/vggt_{arm}.json"))
    keys = ("ef_gt", "ef_breath", "edv_gt", "esv_gt", "edv_breath", "esv_breath")
    ps = [r for r in d["per_subject"] if all(r.get(k) is not None for k in keys)]
    gt = np.array([r["ef_gt"] for r in ps])
    pr = np.array([r["ef_breath"] for r in ps])
    edv_gt = np.array([r["edv_gt"] for r in ps]); esv_gt = np.array([r["esv_gt"] for r in ps])
    edv_pr = np.array([r["edv_breath"] for r in ps]); esv_pr = np.array([r["esv_breath"] for r in ps])
    sv_ratio = ((edv_pr - esv_pr) / (edv_gt - esv_gt)).mean()
    r, _ = stats.pearsonr(gt, pr)
    slope, intercept, _, _, _ = stats.linregress(gt, pr)
    mae = np.abs(pr - gt).mean()
    bias = (pr - gt).mean()
    return dict(gt=gt, pr=pr, n=len(ps), sv_ratio=sv_ratio, r=r, slope=slope,
                intercept=intercept, mae=mae, bias=bias)


data = {tag: load(tag) for tag, _, _ in ARMS}

# ─────────────────────────────── Figure 1: EF scatter grid ───────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor=SURFACE)
fig.suptitle("Predicted EF vs true EF, ES-weight arm sweep (n≈144, val)",
             fontsize=14, color=INK_PRIMARY, fontweight="bold", y=0.98)

LIM = (10, 90)
for ax, (tag, label, color) in zip(axes.flat, ARMS):
    d = data[tag]
    ax.set_facecolor(SURFACE)
    # Identity line y=x: muted reference, drawn first so it sits behind the data.
    ax.plot(LIM, LIM, "--", color=INK_MUTED, linewidth=1.5, zorder=1, label="identity (y=x)")
    # Regression fit: same role-color in every panel (not an "arm" color -- a fixed role).
    xs = np.array(LIM)
    ax.plot(xs, d["slope"] * xs + d["intercept"], "-", color="#e34948", linewidth=2,
             zorder=2, label="model fit")
    ax.scatter(d["gt"], d["pr"], s=32, color=color, alpha=0.55, edgecolors="none", zorder=3)
    ax.set_xlim(LIM); ax.set_ylim(LIM)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.set_xlabel("True EF (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel("Predicted EF (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title(label, color=color, fontsize=12, fontweight="bold", loc="left")
    stat_txt = (f"bias {d['bias']:+.1f}   slope {d['slope']:.2f}\n"
                f"r {d['r']:.3f}   MAE {d['mae']:.2f}   n={d['n']}")
    ax.text(0.97, 0.03, stat_txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=INK_SECONDARY,
            bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, lw=0.8))

handles = [mlines.Line2D([], [], color=INK_MUTED, linestyle="--", linewidth=1.5, label="identity (y=x)"),
           mlines.Line2D([], [], color="#e34948", linewidth=2, label="model fit (regression)")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           labelcolor=INK_SECONDARY, fontsize=10, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT_DIR}/esweight_ef_scatter.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

# ─────────────────────────── Figure 2: metric comparison bars ────────────────────────────
metrics = [
    ("bias", "EF bias (pred − true, pts)", "{:+.1f}"),
    ("sv_ratio", "SV-ratio (pred SV / true SV)", "{:.3f}"),
    ("r", "Per-patient EF ranking r", "{:.3f}"),
    ("mae", "EF MAE (pts)", "{:.2f}"),
]
fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5.2), facecolor=SURFACE)
fig2.suptitle("ES-weight arm sweep vs no-reg baseline (n≈144, val)",
              fontsize=14, color=INK_PRIMARY, fontweight="bold", y=1.02)

labels = [lab for _, lab, _ in ARMS]
colors = [c for _, _, c in ARMS]
x = np.arange(len(ARMS))
baseline_val = {m: data[BASE_ARM][m] for m, _, _ in metrics}

for ax, (mkey, mlabel, fmt) in zip(axes2, metrics):
    vals = [data[tag][mkey] for tag, _, _ in ARMS]
    ax.set_facecolor(SURFACE)
    bars = ax.bar(x, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(baseline_val[mkey], color=INK_MUTED, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(len(ARMS) - 0.4, baseline_val[mkey], " no-reg", color=INK_MUTED, fontsize=8,
            va="center", ha="left")
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.02 if v >= 0 else -0.02) * (max(vals) - min(vals) + 1e-9),
                fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8.5, color=INK_SECONDARY,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0), zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" (baseline)", "").replace(" (ref)", "")
                         for l in labels], rotation=35, ha="right", fontsize=8.5, color=INK_SECONDARY)
    ax.set_title(mlabel, fontsize=10.5, color=INK_PRIMARY, pad=10)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=1)

fig2.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig(f"{OUT_DIR}/esweight_metrics_comparison.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
plt.close(fig2)

print(f"wrote {OUT_DIR}/esweight_ef_scatter.png")
print(f"wrote {OUT_DIR}/esweight_metrics_comparison.png")
