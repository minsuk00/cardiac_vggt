"""Figures for the one-frame ablation report. Reads only cached JSON under result/1frame_series/.

Run after tools/pull_1frame_series.py. No network, no GPU.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/minsukc/vggt/result/1frame_series"
FIG = os.path.join(RES, "figs")
H = json.load(open(os.path.join(RES, "history.json")))

ORDER = ["gather05", "no_gather", "contz", "dino_ft", "aug_moderate", "lowdiff100"]
LBL = {"gather05": "gather05 (hub)", "no_gather": "no_gather", "contz": "contz",
       "dino_ft": "dino_ft", "aug_moderate": "aug_moderate", "lowdiff100": "lowdiff100"}
C = {"gather05": "#111111", "no_gather": "#d1495b", "contz": "#7b2cbf",
     "dino_ft": "#2a9d8f", "aug_moderate": "#e07a1f", "lowdiff100": "#457b9d"}
MATCH_EP = 26


def series(v, key):
    s = H[v]["series"].get(key, [])
    return np.array([x[0] / 1000.0 for x in s]), np.array([x[1] for x in s])


def fig_trajectories():
    panels = [
        ("val/resp/epe_dz_mm", "Breathing error  EPE (mm)  ↓", None),
        ("val/resp/slope_dz", "Breathing gain  slope  →1", 1.0),
        ("val/resp/frac_deep_ignored", "Deep breaths ignored (frac)  ↓", None),
        ("val/metric/recov_frac_heart", "Heart recovery  recov_frac  ↑", None),
        ("val/psnr/bbox_mean", "PSNR bbox (dB)  ↑", None),
        ("val/psnr/static", "PSNR static (dB)  ↑", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4))
    for ax, (key, title, hline) in zip(axes.ravel(), panels):
        for v in ORDER:
            x, y = series(v, key)
            if not len(x):
                continue
            lw = 2.6 if v == "gather05" else 1.5
            ax.plot(x, y, color=C[v], lw=lw, label=LBL[v], zorder=3 if v == "gather05" else 2)
            ax.plot(x[-1], y[-1], "o", color=C[v], ms=5, zorder=4)  # each run's death epoch
        ax.axvline(MATCH_EP, ls="--", color="#888", lw=1)
        if hline:
            ax.axhline(hline, ls=":", color="#aaa", lw=1)
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("epoch"); ax.grid(alpha=0.25)
    axes[0, 0].annotate(f"epoch-matched read\n(all 6 alive)", xy=(MATCH_EP, 0.9),
                        xycoords=("data", "axes fraction"), fontsize=8, color="#666",
                        ha="right", va="top")
    axes[0, 0].legend(fontsize=8.5, loc="upper right", framealpha=0.9)
    fig.suptitle("One-frame ablation — validation trajectories (dot = run's last epoch; dashed = epoch-matched read)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/trajectories.png", dpi=125)
    plt.close(fig)


def fig_ef_noise():
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs, ys = series("gather05", "val/ef/slope")
    lo, hi = ys.min(), ys.max()
    ax.axhspan(lo, hi, color="#111111", alpha=0.09, zorder=0)
    ax.annotate(f"gather05's OWN range across epochs: {lo:.2f}–{hi:.2f}\n"
                f"(one config, one seed — this is the metric's noise)",
                xy=(0.98, hi), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=9, color="#444")
    for v in ORDER:
        x, y = series(v, "val/ef/slope")
        if not len(x):
            continue
        ax.plot(x, y, "-o", color=C[v], lw=1.6, ms=4, label=LBL[v])
    ax.axhline(1.0, ls=":", color="#aaa", lw=1)
    ax.set_xlabel("epoch"); ax.set_ylabel("EF slope (pred vs GT)")
    ax.set_title("EF slope has no power at n=29: within-run epoch-to-epoch swing exceeds every between-run difference",
                 fontsize=11)
    ax.legend(fontsize=8.5, ncol=3); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/ef_noise.png", dpi=125); plt.close(fig)


def fig_epoch_coverage():
    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    for i, v in enumerate(ORDER):
        ep = H[v]["ckpt_epoch"]
        x, _ = series(v, "val/psnr/bbox_mean")
        ax.barh(i, x[-1], color=C[v], alpha=0.85, height=0.62)
        ax.text(x[-1] + 0.6, i, f"ep{int(x[-1])}  (ckpt ep{ep})", va="center", fontsize=9)
    ax.axvline(MATCH_EP, ls="--", color="#444", lw=1.4)
    ax.text(MATCH_EP - 0.5, len(ORDER) - 0.35, "epoch-matched read", ha="right", fontsize=9, color="#444")
    ax.axvline(100, ls=":", color="#bbb"); ax.text(99, 0.1, "planned 100", ha="right", fontsize=8, color="#999")
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels([LBL[v] for v in ORDER], fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("epochs completed"); ax.set_xlim(0, 104)
    ax.set_title("All 6 runs died early, at different epochs — the confound this analysis must control", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/epoch_coverage.png", dpi=125); plt.close(fig)


def fig_zshift():
    p = os.path.join(RES, "zshift_sensitivity.json")
    if not os.path.exists(p):
        print("  (skip zshift — not computed yet)"); return
    d = json.load(open(p))
    dz, ps = np.array(d["dz_mm"]), np.array(d["psnr_db"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.plot(dz, ps, "-o", color="#111", lw=2)
    for x, lab, col in [(1.36, "gather05 EPE\n1.36 mm", "#111111"), (3.49, "no_gather EPE\n3.49 mm", "#d1495b")]:
        y = float(np.interp(x, dz, ps))
        ax.plot([x], [y], "o", ms=9, color=col, zorder=5)
        ax.annotate(lab, xy=(x, y), xytext=(x + 0.5, y + 2.5), fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.axvline(6.0, ls=":", color="#999")
    ax.text(6.1, ps.max(), "½ slice pitch", fontsize=8.5, color="#777", va="top")
    ax.set_xlabel("through-plane (z) error, mm"); ax.set_ylabel("PSNR vs unshifted GT (dB), heart ROI")
    ax.set_title(f"What a z-error actually costs at 12 mm pitch  (GT shifted vs GT, n={d['n_subj']} subjects, no model)",
                 fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG}/zshift_sensitivity.png", dpi=125); plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    fig_epoch_coverage(); print("  epoch_coverage.png")
    fig_trajectories(); print("  trajectories.png")
    fig_ef_noise(); print("  ef_noise.png")
    fig_zshift()
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
