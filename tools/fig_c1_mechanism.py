"""The C1 figure: gather fixes placement, and the volume doesn't care.

Three panels that carry the report's central argument:
  (a) paired plateau diffs  — what gather actually buys and costs, with significance
  (b) shared-vs-unique error — why placement can't move the score
  (c) gather advantage vs breath depth — the hypothesis that failed
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/minsukc/vggt/result/1frame_series"
FIG = f"{RES}/figs"
GOOD, BAD, DIM = "#059669", "#dc2626", "#8892a4"


def main():
    os.makedirs(FIG, exist_ok=True)
    fig = plt.figure(figsize=(15, 4.5))

    # (a) paired plateau diffs -------------------------------------------------
    ax = fig.add_subplot(1, 3, 1)
    p = json.load(open(f"{RES}/plateau_gather05_vs_no_gather.json"))["metrics"]
    show = [("val/resp/epe_dz_mm", "breathing EPE", True),
            ("val/metric/hole_frac_heart", "coverage holes", True),
            ("val/metric/recov_frac_heart", "recov_frac", False),
            ("val/psnr/motion/mean", "PSNR motion", False),
            ("val/psnr/bbox_mean", "PSNR bbox", False),
            ("val/psnr/static", "PSNR static", False)]
    labs, vals, cols = [], [], []
    for k, lab, lower_better in show:
        if k not in p:
            continue
        m = p[k]
        # normalize each diff to "% of no_gather's value" so they share an axis
        rel = 100 * m["diff"] / abs(m["B_mean"]) if m["B_mean"] else 0
        good = (m["favors"] == "A")
        labs.append(f"{lab}\n(p={m['p']:.0e})" if m["p"] < 0.05 else f"{lab}\n(n.s.)")
        vals.append(rel)
        cols.append(GOOD if good and m["p"] < 0.05 else (BAD if m["p"] < 0.05 else DIM))
    y = np.arange(len(labs))
    ax.barh(y, vals, color=cols, height=0.62)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel("gather05 − no_gather  (% of no_gather's value)")
    ax.set_title("(a) What the gather loss buys and costs\npaired over plateau ep20–38, n=19", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")
    ax.text(0.98, 0.03, "green = gather wins (p<.05)\nred = gather loses (p<.05)\ngrey = coin-flip",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555")

    # (b) shared vs unique error ----------------------------------------------
    ax = fig.add_subplot(1, 3, 2)
    d = json.load(open(f"{RES}/recon_compare.json"))
    m = d["mean"]
    bars = ["gather05\nvs GT", "no_gather\nvs GT", "gather05 vs\nno_gather"]
    vv = [m["A_vs_GT"], m["B_vs_GT"], m["A_vs_B"]]
    ax.bar(bars, vv, color=["#111111", "#d1495b", "#2563eb"], width=0.6)
    for i, v in enumerate(vv):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(m["A_vs_GT"], ls=":", color="#888", lw=1)
    ax.set_ylabel("PSNR (dB), heart ROI"); ax.set_ylim(0, max(vv) + 3)
    ax.set_title("(b) The two models make the SAME mistakes\n88% of the error is shared, 12% unique",
                 fontsize=10.5)
    ax.annotate("", xy=(2, m["A_vs_B"]), xytext=(2, m["A_vs_GT"]),
                arrowprops=dict(arrowstyle="<->", color="#2563eb", lw=1.6))
    ax.text(2.32, (m["A_vs_B"] + m["A_vs_GT"]) / 2, f"+{m['A_vs_B']-m['A_vs_GT']:.1f} dB\nthey agree with\neach other MORE\nthan with GT",
            fontsize=8, color="#2563eb", va="center")
    ax.grid(alpha=0.25, axis="y")

    # (c) the failed hypothesis ------------------------------------------------
    ax = fig.add_subplot(1, 3, 3)
    g = json.load(open(f"{RES}/gather_benefit_vs_depth.json"))
    rows = g["rows"]
    x = np.array([r["max_disp"] for r in rows])
    yb = np.array([r["gain_breath"] for r in rows])
    ax.scatter(x, yb, s=34, color="#111", alpha=0.75, edgecolors="none", label="under breathing")
    b, a = np.polyfit(x, yb, 1)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, a + b * xs, "-", color="#111", lw=1.6)
    ax.axhline(0, ls="--", color="#888", lw=1)
    ax.set_xlabel("subject's max applied breathing |SI| (mm)")
    ax.set_ylabel("gather05 − no_gather, breath PSNR (dB)")
    ax.set_title(f"(c) The hypothesis that FAILED\nno link to breath depth: r={g['r_gain_vs_maxdisp']:+.2f}, p={g['p_gain_vs_maxdisp']:.2f}",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.text(0.03, 0.03, "predicted: the splat absorbs sub-pitch\nerrors ⇒ gather should help DEEP\nbreathers. It doesn't.",
            transform=ax.transAxes, fontsize=7.5, color="#555", va="bottom")

    fig.suptitle("C1 — the gather auxiliary loss fixes through-plane placement, and the reconstruction does not care",
                 fontsize=12.5)
    fig.tight_layout()
    fig.savefig(f"{FIG}/c1_mechanism.png", dpi=125)
    plt.close(fig)
    print(f"  -> {FIG}/c1_mechanism.png")


if __name__ == "__main__":
    main()
