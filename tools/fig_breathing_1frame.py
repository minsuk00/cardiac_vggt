"""Breathing figures: amplitude response, pred-vs-applied scatter + residual, clean control.

Pools per-slot predicted vs applied Dz from every subject's resp_diag.json, per model.
Run after the eval sweep. No GPU.
"""
import glob
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "/home/minsukc/vggt/result/1frame_series"
FIG = f"{RES}/figs"
BINS = [(0, 2), (2, 8), (8, 12), (12, 40)]
C = {"gather05": "#111111", "no_gather": "#d1495b", "contz": "#7b2cbf",
     "dino_ft": "#2a9d8f", "aug_moderate": "#e07a1f", "lowdiff100": "#457b9d"}


def load(dataset, variant):
    meth = f"vggt_20260715_1f_{variant}" + ("_contz" if variant == "contz" and dataset == "miitt" else "")
    P, A, CL = [], [], []
    for f in sorted(glob.glob(f"/home/minsukc/vggt/scratch/eval/{dataset}/out/*/{meth}/resp_diag.json")):
        d = json.load(open(f))
        P += d["breath"]["pred_dz_mm"]; A += d["breath"]["applied_dz_mm"]
        CL += d["clean"]["pred_dz_mm"]
    return np.array(P), np.array(A), np.array(CL)


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cmrxrecon"
    variants = sys.argv[2:] or list(C)
    data = {v: load(dataset, v) for v in variants}
    data = {v: d for v, d in data.items() if len(d[0])}
    if not data:
        print("no resp_diag data yet"); return
    os.makedirs(FIG, exist_ok=True)

    # --- amplitude response -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.9))
    ax = axes[0]
    centers = [1, 5, 10, 17]
    for v, (P, A, _) in data.items():
        ys, xs = [], []
        for (lo, hi), cx in zip(BINS, centers):
            m = (np.abs(A) >= lo) & (np.abs(A) < hi)
            if m.sum() >= 5:
                xs.append(np.abs(A[m]).mean()); ys.append(P[m].mean())
        ax.plot(xs, ys, "-o", color=C[v], lw=2, ms=5, label=v)
    lim = 20
    ax.plot([0, lim], [0, lim], "--", color="#999", lw=1.2, label="perfect (identity)")
    ax.set_xlabel("applied breathing |SI| (mm)"); ax.set_ylabel("mean predicted Δz (mm)")
    ax.set_title("Amplitude response — where breathing estimation breaks", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    ax = axes[1]
    w = 0.8 / len(data)
    for i, (v, (P, A, _)) in enumerate(data.items()):
        fr, lbl = [], []
        for (lo, hi) in BINS:
            m = (np.abs(A) >= lo) & (np.abs(A) < hi)
            fr.append(100 * P[m].mean() / np.abs(A[m]).mean() if m.sum() >= 5 and np.abs(A[m]).mean() > 0.5 else np.nan)
            lbl.append(f"{lo}–{hi}")
        ax.bar(np.arange(len(BINS)) + i * w, fr, w, color=C[v], label=v)
    ax.axhline(100, ls="--", color="#666", lw=1.2)
    ax.set_xticks(np.arange(len(BINS)) + 0.4 - w / 2); ax.set_xticklabels([f"{lo}–{hi} mm" for lo, hi in BINS])
    ax.set_ylabel("% of breathing recovered"); ax.set_xlabel("applied breathing |SI| bin")
    ax.set_title("Recovered fraction by breath depth (100% = perfect)", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25, axis="y")
    fig.suptitle(f"Breathing estimation vs breath depth — {dataset}", fontsize=12.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/amplitude_response_{dataset}.png", dpi=125); plt.close(fig)
    print(f"  amplitude_response_{dataset}.png")

    # --- scatter + residual --------------------------------------------------
    n = len(data)
    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.4), squeeze=False)
    for j, (v, (P, A, _)) in enumerate(data.items()):
        a = axes[0][j]
        a.scatter(np.abs(A), P, s=7, alpha=0.45, color=C[v], edgecolors="none")
        a.plot([0, 22], [0, 22], "--", color="#888", lw=1)
        sl = np.polyfit(np.abs(A), P, 1)[0]
        a.plot([0, 22], [0, 22 * sl], "-", color=C[v], lw=1.6)
        a.set_title(f"{v}\nslope {sl:.2f}", fontsize=9.5)
        a.set_xlim(0, 22); a.set_ylim(-3, 24); a.grid(alpha=0.2)
        if j == 0: a.set_ylabel("predicted Δz (mm)")
        b = axes[1][j]
        b.scatter(np.abs(A), P - np.abs(A), s=7, alpha=0.45, color=C[v], edgecolors="none")
        b.axhline(0, ls="--", color="#888", lw=1)
        b.axvline(12, ls=":", color="#bbb", lw=1)
        b.set_xlim(0, 22); b.set_ylim(-16, 8); b.grid(alpha=0.2)
        b.set_xlabel("applied |SI| (mm)")
        if j == 0: b.set_ylabel("residual  pred − applied (mm)")
    fig.suptitle(f"Predicted vs applied through-plane shift, per slot — {dataset}  "
                 f"(top: fit vs identity; bottom: residual, dotted = deep-breath threshold)", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{FIG}/scatter_residual_{dataset}.png", dpi=125); plt.close(fig)
    print(f"  scatter_residual_{dataset}.png")

    # --- clean negative control ---------------------------------------------
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), squeeze=False)
    for j, (v, (_, _, CL)) in enumerate(data.items()):
        a = axes[0][j]
        a.hist(CL, bins=28, color=C[v], alpha=0.8)
        a.axvline(0, ls="--", color="#666", lw=1)
        a.set_title(f"{v}\nmean|Δz| {np.abs(CL).mean():.2f}  max {np.abs(CL).max():.2f} mm", fontsize=8.5)
        a.set_xlabel("predicted Δz (mm)"); a.grid(alpha=0.2)
        if j == 0: a.set_ylabel("slots")
    fig.suptitle("Negative control: predicted Δz on CLEAN (un-breathed) input — applied is exactly 0, so any signal is invention",
                 fontsize=10.5)
    fig.tight_layout(); fig.savefig(f"{FIG}/clean_control_{dataset}.png", dpi=125); plt.close(fig)
    print(f"  clean_control_{dataset}.png")


if __name__ == "__main__":
    main()
