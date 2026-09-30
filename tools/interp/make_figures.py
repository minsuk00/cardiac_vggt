"""Figures for the interpretability section (docs/126). Reads temp/interp/* and scratch/interp/e4.
Writes figs/interp/*.pdf|png. Each panel is its own function so partial data still renders."""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

ROOT = A.ROOT
D = A.D
OUT = os.path.join(ROOT, "figs/interp")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "pdf.fonttype": 42})


def ci(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    lo, hi = A.boot_ci(x)
    return x.mean(), x.mean() - lo, hi - x.mean()


def e5_index():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/e5/s*.json"))]
    recs = [r for r in recs if r.get("dirs")]
    names = list(recs[0]["dirs"][0]["patch"])
    out = {n: [] for n in names}
    for r in recs:
        for n in names:
            v = []
            for d in r["dirs"]:
                s = (lambda q: q[2][1] - q[2][0]) if d["src"] == r["a"] else (lambda q: q[2][0] - q[2][1])
                s0, s1 = s(d["src_ncc"]), s(d["don_ncc"])
                v.append((s(d["patch"][n]["ncc"]) - s0) / (s1 - s0))
            out[n].append(np.mean(v) if r["idx"] not in A.SELECTION_IDX or not n.startswith("L3_") else np.nan)
    return out, len(recs)


def panel_layers(ax):
    idx, n = e5_index()
    L = list(range(12))
    m = [ci(idx[f"L{l}"]) for l in L]
    ax.bar(L, [x[0] * 100 for x in m], width=0.7, color=[ORANGE if l == 3 else GRAY for l in L],
           yerr=np.array([[x[1] * 100 for x in m], [x[2] * 100 for x in m]]), error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(L[::2]); ax.set_xlabel("global-attention layer patched")
    ax.set_ylabel("cardiac state transferred (%)"); ax.set_ylim(-3, 105)
    ax.set_title(f"a  Swap the reference read,\n    one layer at a time (n={n})", loc="left")
    ax.text(3, m[3][0] * 100 + 3, f"{m[3][0] * 100:.1f}%", ha="center", color=INK, fontsize=7)
    ax.text(8, 50, f"layers 18-23: {100 * np.nanmean(idx['L18_23']):.1f}%\nall but 3: {100 * np.nanmean(idx['L0_23_but3']):.1f}%",
            color=MUTED, fontsize=7, ha="center")


def panel_heads(ax):
    idx, n = e5_index()
    Hs = list(range(16))
    m = [ci(idx[f"L3_h{h}"]) for h in Hs]
    top = {2, 10, 12, 14}
    ax.bar(Hs, [x[0] * 100 for x in m], width=0.7, color=[ORANGE if h in top else GRAY for h in Hs])
    ax.set_xticks(Hs); ax.set_xlabel("layer-3 head patched (alone)")
    ax.set_ylabel("cardiac state transferred (%)")
    t4, o12 = np.nanmean(idx["L3_top4"]) * 100, np.nanmean(idx["L3_other12"]) * 100
    ax.set_title(f"b  Layer 3, one head at a time\n    (held-out n={n - 3})", loc="left")
    ax.set_xticks(Hs[::3])
    ax.text(0.98, 0.97, f"orange 4 together: {t4:.1f}%\nother 12 together: {o12:.1f}%",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=6.5)
    ax.set_ylim(-3, 40)


def e1b_vals():
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/e1b/s*.json"))]
    recs = [r for r in recs if r.get("cond")]
    conds = list(recs[0]["cond"])
    V = {c: {"G": [], "slope": [], "epe": []} for c in conds}
    for r in recs:
        for c in conds:
            m = A.e1_metrics(r, c)
            V[c]["G"].append(m["G_conly"]); V[c]["slope"].append(m["slope"]); V[c]["epe"].append(m["epe"])
    return V, len(recs)


DISS = [("full", "full"), ("ko_ref_L3", "no ref.\nread L3"),
        ("ko_refheart_L3", "no ref.\nheart L3"), ("ko_refbg_L3", "no ref.\nbkgd L3"),
        ("self_L12_17", "no cross-\nslice\nL12–17"), ("identity", "naive\nstack")]


def panel_dissociation(ax1, ax2):
    V, n = e1b_vals()
    Gf = np.nanmean(V["full"]["G"])
    x = np.arange(len(DISS))
    for ax, key, lab, col, scale in ((ax1, "G", "cardiac state transfer\n(% of full model)", BLUE, 100 / Gf),
                                     (ax2, "epe", "through-plane breathing\nerror EPE (mm)", AQUA, 1.0)):
        m = [ci(V[c][key]) for c, _ in DISS]
        ax.bar(x, [v[0] * scale for v in m], width=0.65, color=col,
               yerr=np.array([[v[1] * scale for v in m], [v[2] * scale for v in m]]), error_kw={"lw": 0.8, "ecolor": INK})
        for xi, v in zip(x, m):
            ax.text(xi, v[0] * scale + v[2] * scale + (3 if key == "G" else 0.12), f"{v[0] * scale:.0f}%" if key == "G" else f"{v[0]:.2f}",
                    ha="center", fontsize=6.5, color=INK)
        ax.set_ylabel(lab); ax.set_xticks(x); ax.set_xticklabels([l for _, l in DISS], fontsize=6.3)
    ax1.set_title(f"d  Cardiac state (n={n})", loc="left")
    ax2.set_title(f"e  Breathing correction (n={n})", loc="left")
    ax1.set_ylim(0, 118); ax2.set_ylim(0, 5.6)


def panel_locality(ax):
    recs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{D}/e7/s*.pkl"))]
    recs = [r for r in recs if r]
    prof = np.array([r["sets"][f"blk{j}"]["prof"] for r in recs for j in range(4)], float)
    bins = np.array(recs[0]["sets"]["blk0"]["bins"]) + 2.5
    m = np.nanmean(prof, 0)
    ax.plot(bins, m, color=BLUE, lw=2, marker="o", ms=4)
    ax.set_xlabel("in-plane distance from patched\nreference location (mm)")
    ax.set_ylabel("companion Δ change (mm)")
    same = np.mean([0.5 * (r["sets"]["half_L"]["d_left"] + r["sets"]["half_R"]["d_right"]) for r in recs])
    opp = np.mean([0.5 * (r["sets"]["half_L"]["d_right"] + r["sets"]["half_R"]["d_left"]) for r in recs])
    ax.set_title(f"c  The read is local\n    (n={len(recs)})", loc="left")
    ax.text(0.03, 0.05, f"half patch: same side {same:.2f} mm,\nopposite side {opp:.2f} mm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5, color=INK)
    ax.set_ylim(0, None)


BREATH = [("full", "full"), ("self_L0_5", "L0–5"), ("self_L6_11", "L6–11"), ("self_L12_17", "L12–17"),
          ("self_L18_23", "L18–23"), ("self_only", "all"), ("nbr1", "adjac.\nonly"), ("no_nbr1", "all but\nadjac."),
          ("ko_ref", "no ref.")]


def panel_breathing(ax, ax2):
    recs = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/e2/s*.json"))]
    sl = {c: [] for c, _ in BREATH}
    for r in recs:
        appl = np.array([row["disp"] for row in r["A"]])
        for c, _ in BREATH:
            pred = np.array([row["pred"][c] for row in r["A"]])
            sl[c].append(A.nan_slope(appl[..., 0][:, 1:].ravel(), pred[..., 0][:, 1:].ravel()))
    x = np.arange(len(BREATH))
    m = [ci(sl[c]) for c, _ in BREATH]
    cols = [AQUA if c == "full" else (ORANGE if c == "self_L12_17" else GRAY) for c, _ in BREATH]
    ax.bar(x, [v[0] for v in m], width=0.65, color=cols,
           yerr=np.array([[v[1] for v in m], [v[2] for v in m]]), error_kw={"lw": 0.8, "ecolor": INK})
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in BREATH], fontsize=6)
    for xi, v in zip(x, m):
        ax.text(xi, v[0] + v[2] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=6, color=INK)
    ax.set_ylabel("breathing recovered\n(slope, predicted vs applied Δz)")
    ax.set_ylim(0, 1.0)
    ax.axvspan(0.5, 5.5, color="#f0efec", zorder=-1)
    ax.text(3.0, 0.95, "no cross-slice attention in layers", ha="center", fontsize=6.3, color=MUTED)
    ax.text(7.5, 0.95, "slices visible", ha="center", fontsize=6.3, color=MUTED)
    ax.set_title(f"g  What breathing correction needs (n={len(recs)}×3 draws)", loc="left")
    # common-mode vs all-but-reference shift
    DEL = np.array([-24, -18, -12, -6, -3, 0, 3, 6, 12, 18, 24])
    for v, col, lab in (("all", GRAY, "every slice shifted by δ"),):
        ys = np.array([[np.nanmean(np.array(row)[1:, 0]) for row in r["D_"][v]] for r in recs])
        ax2.plot(DEL, np.nanmean(ys, 0), color=col, lw=2, marker="o", ms=3.5, label=lab)
    ax2.plot(DEL, np.clip(DEL, 0, None), color=INK, lw=0.8, ls="--", label="perfect (Δz≥0)")
    ax2.set_xlabel("common shift δ (mm)"); ax2.set_ylabel("mean predicted Δz (mm)")
    ax2.legend(fontsize=6, frameon=False, loc="upper left")
    ax2.set_title("h  A common shift is invisible", loc="left")


def fig_breathing():
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], wspace=0.3)
    panel_breathing(fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_breathing.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_breathing.png")


LVARMS = [("gt", "ground truth", INK, "--"), ("full", "full model", BLUE, "-"),
          ("self_L12_17", "no cross-slice attn L12–17", AQUA, "-"),
          ("ko_ref_L3", "reference read blocked (L3)", ORANGE, "-"), ("identity", "naive stack", GRAY, "-")]


def fig_lv():
    d = json.load(open(os.path.join(ROOT, "scratch/interp/e4/e4_curves.json")))
    per = d["per_subject"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1.3, 1], "wspace": 0.35})
    # example: typical on BOTH axes — full-model VTC and naive-stack swing (sum of rank distances to median)
    vt = np.array([np.mean(np.abs(np.array(p["full"]["far"]) - np.array(p["gt"]["far"]))) / max(p["gt"]["far"]) for p in per])
    sw = np.array([np.ptp(p["identity"]["far"]) / max(np.ptp(p["gt"]["far"]), 1e-9) for p in per])
    rk = lambda x: np.abs(np.argsort(np.argsort(x)) - len(x) / 2)
    ex = per[int(np.argmin(rk(vt) + rk(sw)))]
    for k, lab, col, ls in LVARMS:
        c = np.array(ex[k]["far"]) * 1.4 * 1.4 * 1e-3            # voxels -> in-plane-mm² * (pitch omitted) — relative
        ax.plot(np.arange(len(c)), c / np.array(ex["gt"]["far"]).max() / (1.4 * 1.4 * 1e-3), color=col, ls=ls, lw=1.8, label=lab)
    ax.set_xlabel("reference frame (cardiac phase)"); ax.set_ylabel("LV volume, planes ≥2 from\nreference (÷ GT max)")
    ax.legend(fontsize=5.8, frameon=False, loc="upper right", ncol=1)
    ax.set_title(f"i  LV volume–time curve (subject s{ex['idx']:03d})", loc="left")
    S = d["summary"]
    arms = [a for a in ("full", "self_L12_17", "ko_ref_L3", "identity")]
    x = np.arange(len(arms))
    m = [ci(np.array(S[a]["far"], float)[:, 2]) for a in arms]
    cols = {"full": BLUE, "self_L12_17": AQUA, "ko_ref_L3": ORANGE, "identity": GRAY}
    ax2.bar(x, [v[0] for v in m], width=0.62, color=[cols[a] for a in arms],
            yerr=np.array([[v[1] for v in m], [v[2] for v in m]]), error_kw={"lw": 0.8, "ecolor": INK})
    for xi, v in zip(x, m):
        ax2.text(xi, v[0] + v[2] + 0.03, f"{v[0]:.2f}", ha="center", fontsize=6.3)
    ax2.set_xticks(x); ax2.set_xticklabels(["full", "no cross-\nslice L12–17", "ref. read\nblocked L3", "naive\nstack"], fontsize=6.3)
    ax2.set_ylabel("LV stroke swing\n(fraction of GT)"); ax2.set_ylim(0, 1.05)
    ax2.set_title(f"j  Paper metric, nnU-Net (n={len(per)})", loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_lv.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_lv.png")


def attn_to_ref(r):
    """(24,16) companion->reference attention mass × uniform, averaged over companions WEIGHTED by
    their number of heart-ROI queries (companions without heart queries contribute nothing)."""
    st = r["stats"]["breath"]
    out = []
    for l in range(24):
        w = st[l]["cnt_h"][1:]
        out.append((st[l]["mass_h"][:, 1:, 0] * w[None]).sum(1) / max(w.sum(), 1))
    return np.stack(out) * r["S"]


def fig_real():
    d = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.1), sharey=False, gridspec_kw={"wspace": 0.25})
    names = {"MIITT_Patient_2024Jan04_Cardiomyopathy_AFib": "AF patient", "MIITT_Volunteer1": "healthy volunteer"}
    for ax, (subj, lab) in zip(axs, names.items()):
        r = d[subj]
        t = np.arange(len(r["full"]["curve_ml"])) * 0.025
        for arm, col, name in (("full", BLUE, "full model"), ("self_L12_17", AQUA, "no cross-slice attn L12–17"),
                               ("ko_ref_L3", ORANGE, "reference read blocked (L3)")):
            ax.plot(t, r[arm]["curve_ml"], color=col, lw=1.5 if arm != "self_L12_17" else 1.0,
                    ls="-" if arm != "self_L12_17" else "--", label=name)
        ax.set_xlabel("time (s)"); ax.set_title(f"{'k' if lab.startswith('AF') else 'l'}  Real RT scan, {lab}", loc="left")
        ax.text(0.99, 0.02, f"swing vs full: L3 blocked {100 * r['ko_ref_L3']['swing_rel']:.0f}%, "
                f"L12–17 {100 * r['self_L12_17']['swing_rel']:.0f}%", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6, color=INK)
    axs[0].set_ylabel("LV volume, reference\nslice excluded (mL)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=6.5, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_real.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_real.png")


def fig_features():
    """E9: where reference-phase information enters the companions' tokens (+ change heatmaps, PCA)."""
    base = os.path.join(ROOT, "scratch/interp/e9")
    recs = [r for r in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{base}/s*.pkl"))) if r]
    order = [(k, i) for i in range(24) for k in ("frame", "global")]
    x = np.arange(len(order)) / 2.0 + np.array([0.0 if k == "frame" else 0.5 for k, _ in order])
    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], hspace=0.55, wspace=0.12)
    ax = fig.add_subplot(gs[0, :])
    for g, col, lab in (("heart", ORANGE, "heart tokens"), ("bg", GRAY, "background tokens"),
                        ("special", MUTED, "special tokens")):
        m = np.array([[r["change"][o][g] for o in order] for r in recs]).mean(0)
        ax.plot(x, 100 * m, color=col, lw=2 if g == "heart" else 1.3, label=lab)
    ax.axvline(3.5, color=ORANGE, lw=0.8, ls=":")
    ax.text(3.7, 40, "global layer 3", color=ORANGE, fontsize=7)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("depth (block index; frame and global blocks interleaved)")
    ax.set_ylabel("change in companion tokens (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.set_title(f"m  Change the reference phase only: where does it show up in the OTHER slices? (n={len(recs)})", loc="left")
    # change heatmaps for one subject: companion slice, token change after global 2 / 3 / 17
    r = next(r for r in recs if "pca_tokens" in r)
    S, G, P = r["S"], 37, 5 + 37 * 37
    s = int(np.argmax(r["heart_patch"][1:].sum(1))) + 1           # companion with the largest heart
    img = r["images_a"][s].astype(np.float32)
    axs = [fig.add_subplot(gs[1, i]) for i in range(4)]
    axs[0].imshow(img, cmap="gray", vmax=np.percentile(img, 99.5)); axs[0].set_title("companion slice", fontsize=7)
    for a_, l in zip(axs[1:], (2, 3, 17)):
        fa = r["pca_tokens"][l].astype(np.float32).reshape(S, P, -1)[s, 5:]
        fb = r["pca_tokens_b"][l].astype(np.float32).reshape(S, P, -1)[s, 5:]
        d = (np.linalg.norm(fb - fa, axis=-1) / np.linalg.norm(fa, axis=-1)).reshape(G, G)
        a_.imshow(img, cmap="gray", extent=(0, G, G, 0), vmax=np.percentile(img, 99.5))
        a_.imshow(d, cmap="inferno", alpha=0.7, extent=(0, G, G, 0), vmin=0, vmax=0.8)
        a_.set_title(f"after global layer {l}", fontsize=7)
    for a_ in axs:
        a_.axis("off")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_features.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_features.png")


def fig_pca():
    """VGGT-Ω-style joint PCA (3 comps -> RGB) of companion+reference tokens at several layers."""
    base = os.path.join(ROOT, "scratch/interp/e9")
    r = next(r for r in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{base}/s*.pkl"))) if r and "pca_tokens" in r)
    S, G, P = r["S"], 37, 5 + 37 * 37
    slots = [0] + list(np.argsort(-r["heart_patch"][1:].sum(1))[:3] + 1)
    layers = [2, 3, 11, 17, 23]
    fig, axs = plt.subplots(len(slots), len(layers) + 1, figsize=(7.2, 5.2))
    for j, l in enumerate(layers):
        f = r["pca_tokens"][l].astype(np.float32).reshape(S, P, -1)[:, 5:]      # (S, G*G, C)
        X = f[slots].reshape(-1, f.shape[-1])
        X = X - X.mean(0)
        _, _, Vt = np.linalg.svd(X[::3], full_matrices=False)
        Y = X @ Vt[:3].T
        Y = (Y - np.percentile(Y, 2, 0)) / (np.percentile(Y, 98, 0) - np.percentile(Y, 2, 0) + 1e-9)
        Y = np.clip(Y, 0, 1).reshape(len(slots), G, G, 3)
        for i in range(len(slots)):
            axs[i, j + 1].imshow(Y[i]); axs[i, j + 1].axis("off")
        axs[0, j + 1].set_title(f"global {l}", fontsize=7)
    for i, s in enumerate(slots):
        im = r["images_a"][s].astype(np.float32)
        axs[i, 0].imshow(im, cmap="gray", vmax=np.percentile(im, 99.5)); axs[i, 0].axis("off")
        axs[i, 0].set_title("reference" if s == 0 else f"companion (z{r['slot_z'][s]})", fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_pca.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_pca.png")


def fig_paper():
    """Main-paper figure (ranked story, docs/126 §11): a layer sweep, b heart heatmap after L3,
    c/d crossed dissociation (4 arms), e real AF LV curve."""
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.8, wspace=1.6)
    # a: patch one layer at a time
    ax = fig.add_subplot(gs[0, 0:2])
    idx, n = e5_index()
    L = list(range(12))
    m = [np.nanmean(idx[f"L{l}"]) * 100 for l in L]
    ax.bar(L, m, width=0.7, color=[ORANGE if l == 3 else GRAY for l in L])
    ax.set_xticks(L[::2]); ax.set_xlabel("global-attention layer (of 24)")
    ax.set_ylabel("heartbeat transferred (%)"); ax.set_ylim(0, 105)
    ax.text(3, m[3] + 3, f"{m[3]:.1f}%", ha="center", fontsize=6.5)
    ax.set_title("a  Swap the reference read,\n    one layer at a time", loc="left")
    # b: where the reference phase shows up in a non-reference slice
    base = os.path.join(ROOT, "scratch/interp/e9")
    r = next(r for r in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{base}/s*.pkl"))) if r and "pca_tokens" in r)
    S, G, P = r["S"], 37, 5 + 37 * 37
    s = int(np.argmax(r["heart_patch"][1:].sum(1))) + 1
    img = r["images_a"][s].astype(np.float32)
    for k, (l, lab) in enumerate(((2, "after layer 2"), (3, "after layer 3"))):
        a_ = fig.add_subplot(gs[0, 2 + 2 * k:4 + 2 * k])
        fa = r["pca_tokens"][l].astype(np.float32).reshape(S, P, -1)[s, 5:]
        fb = r["pca_tokens_b"][l].astype(np.float32).reshape(S, P, -1)[s, 5:]
        d = (np.linalg.norm(fb - fa, axis=-1) / np.linalg.norm(fa, axis=-1)).reshape(G, G)
        a_.imshow(img, cmap="gray", extent=(0, G, G, 0), vmax=np.percentile(img, 99.5))
        a_.imshow(d, cmap="inferno", alpha=0.7, extent=(0, G, G, 0), vmin=0, vmax=0.8)
        a_.axis("off"); a_.set_title(lab, fontsize=7)
        if k == 0:
            a_.text(0.0, 1.28, "b  Change only the reference phase: effect inside a NON-reference slice",
                    transform=a_.transAxes, fontsize=8.5, fontweight="bold", ha="left")
    # c, d: crossed dissociation, 4 arms
    V, n1 = e1b_vals()
    arms = [("full", "full"), ("ko_ref_L3", "no ref.\nread\nL3"), ("self_L12_17", "no slice↔\nslice\nL12–17"),
            ("identity", "naive\nstack")]
    Gf = np.nanmean(V["full"]["G"])
    for j, (key, ylab, col, sc) in enumerate((("G", "heartbeat transfer (%)", BLUE, 100 / Gf),
                                              ("epe", "breathing error (mm)", AQUA, 1.0))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        vals = [np.nanmean(V[a][key]) * sc for a, _ in arms]
        ax.bar(range(4), vals, width=0.65, color=col)
        for xi, v in enumerate(vals):
            ax.text(xi, v + (2 if key == "G" else 0.1), f"{v:.0f}%" if key == "G" else f"{v:.2f}", ha="center", fontsize=6.3)
        ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in arms], fontsize=5.8)
        ax.set_ylabel(ylab)
        ax.set_title("c  Heartbeat" if j == 0 else "d  Breathing", loc="left")
    # e: real AF scan
    ax = fig.add_subplot(gs[1, 4:6])
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    t = np.arange(len(d8["full"]["curve_ml"])) * 0.025
    ax.plot(t, d8["full"]["curve_ml"], color=BLUE, lw=1.3, label="full")
    ax.plot(t, d8["ko_ref_L3"]["curve_ml"], color=ORANGE, lw=1.5, label="block ref. read (L3)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("LV volume (mL)")
    ax.legend(fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    ax.set_title("e  Real AF scan", loc="left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_paper_main.{ext}", dpi=220, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_paper_main.png")


def fig_paper_simple():
    """Compact 1x4 main-paper figure: a where the heartbeat is read, b heartbeat, c breathing, d real AF."""
    ARMS = [("full", "full model", INK), ("ko_ref_L3", "block reference, layer 3", ORANGE),
            ("self_L12_17", "block slice↔slice, layers 12–17", AQUA), ("identity", "naive stack", GRAY)]
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.75), gridspec_kw={"width_ratios": [1.5, 1, 1, 1.35], "wspace": 0.5})
    for a_ in axs:
        a_.tick_params(labelsize=6.5, length=2, pad=1.5)
    # a: swap the reference read in one layer at a time (E5)
    ax = axs[0]
    idx, _ = e5_index()
    keys = [f"L{l}" for l in range(12)] + ["L12_17", "L18_23"]
    xs = list(range(12)) + [13.6, 16.2]
    m = [max(np.nanmean(idx[k]) * 100, 0) for k in keys]
    ax.bar(xs, m, width=0.75, color=[ORANGE if k == "L3" else GRAY for k in keys])
    ax.text(3, m[3] + 3, f"{m[3]:.1f}%", ha="center", fontsize=6.5, color=ORANGE, fontweight="bold")
    ax.set_xticks([0, 3, 6, 9, 13.6, 16.2]); ax.set_xticklabels(["0", "3", "6", "9", "12–\n17", "18–\n23"], fontsize=6)
    ax.set_xlabel("global-attention layer", labelpad=1); ax.set_ylabel("output switches phase (%)"); ax.set_ylim(0, 110)
    ax.set_title("a  Only layer 3 matters", loc="left")
    # b, c: the four arms (E1b)
    V, _ = e1b_vals()
    Gf = np.nanmean(V["full"]["G"])
    for ax, key, sc, lab, title, fmt in ((axs[1], "G", 100 / Gf, "follows reference phase (%)", "b  Heartbeat", "{:.0f}%"),
                                         (axs[2], "epe", 1.0, "breathing error (mm)", "c  Breathing", "{:.2f}")):
        vals = [np.nanmean(V[a][key]) * sc for a, _, _ in ARMS]
        ax.bar(range(4), vals, width=0.72, color=[c for _, _, c in ARMS])
        top = max(vals)
        for xi, v in enumerate(vals):
            ax.text(xi, v + top * 0.02, fmt.format(v), ha="center", fontsize=6.3)
        ax.set_xticks([]); ax.set_ylim(0, top * 1.15); ax.set_ylabel(lab, labelpad=1)
        ax.set_title(title, loc="left")
    # d: real AF scan (E8)
    ax = axs[3]
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    t = np.arange(len(d8["full"]["curve_ml"])) * 0.025
    ax.plot(t, d8["full"]["curve_ml"], color=INK, lw=1.1)
    ax.plot(t, d8["ko_ref_L3"]["curve_ml"], color=ORANGE, lw=1.4)
    ax.set_xlabel("time (s)", labelpad=1); ax.set_ylabel("LV volume (mL)", labelpad=1)
    ax.set_title("d  Real AF scan", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in ARMS]
    fig.legend(handles, [l for _, l, _ in ARMS], loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4,
               frameon=False, fontsize=7, handlelength=1.0, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_paper_simple.{ext}", dpi=250, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_paper_simple.png")


def fig_paper_visual(sub_hb="s075", plane_hb=10, sub_br="s117", sub_attn="s090"):
    """Image-based main-paper figure: layer schematic, layer-3 attention, ED/ES images (full vs layer-3
    block), real AF LV curve, long-axis cuts (full vs slice-to-slice block in layers 12-17)."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from scipy.ndimage import gaussian_filter, zoom
    def clean(a_):
        a_.set_xticks([]); a_.set_yticks([])
        for s_ in a_.spines.values():
            s_.set_visible(False)

    fig = plt.figure(figsize=(7.0, 3.9))
    top, mid_f, bot = fig.subfigures(3, 1, height_ratios=[0.2, 1.05, 0.8], hspace=0.0)
    # ---- top: the 24 global-attention layers ----
    ax = top.add_axes([0.07, 0.0, 0.93, 1.0]); ax.set_xlim(0, 24); ax.set_ylim(0, 1.6); ax.axis("off")
    for l in range(24):
        c = ORANGE if l == 3 else (AQUA if 12 <= l <= 17 else "#e4e3de")
        ax.add_patch(Rectangle((l + 0.05, 0.05), 0.9, 0.7, color=c, lw=0))
        ax.text(l + 0.5, 0.4, str(l), ha="center", va="center", fontsize=6, color="white" if c != "#e4e3de" else MUTED)
    ax.text(3.5, 0.9, "heartbeat: read from the reference", color=ORANGE, fontsize=7.5, fontweight="bold", ha="center", va="bottom")
    ax.text(15.0, 0.9, "breathing: slices align to each other", color=AQUA, fontsize=7.5, fontweight="bold", ha="center", va="bottom")
    ax.text(-0.2, 0.4, "global-attention\nlayer (of 24)", fontsize=6, color=MUTED, ha="right", va="center")
    # ---- middle row: heartbeat ----
    mid = mid_f.add_gridspec(1, 3, width_ratios=[0.8, 2.4, 2.0], wspace=0.18, left=0.07, right=0.99, top=0.84, bottom=0.12)
    # attention: query on another slice -> layer-3 attention over the reference
    r = pickle.load(open(f"{D}/e3/{sub_attn}.pkl", "rb"))
    maps = r["maps"]["breath"]; imgs = r["images_breath"].astype(np.float32)
    k = sorted(k for k in maps if k[0] == 3 and k[1] != 0)[0]
    _, qs, qr, qc = k
    G = 37; sc = 14 * 256 / 518
    att = np.mean([maps[k][h].astype(np.float32)[0, 5:] for h in (2, 10, 12, 14)], 0).reshape(G, G)
    cy, cx = (qr + 0.5) * sc, (qc + 0.5) * sc; h = 52
    ga = mid[0].subgridspec(2, 1, hspace=0.22)
    a1, a2 = mid_f.add_subplot(ga[0]), mid_f.add_subplot(ga[1])
    a1.imshow(imgs[qs], cmap="gray", vmax=np.percentile(imgs[qs], 99.5))
    a1.plot(cx, cy, "+", color="#e34948", ms=7, mew=1.6)
    a2.imshow(imgs[0], cmap="gray", vmax=np.percentile(imgs[0], 99.5))
    a2.imshow(zoom(att, 256 / G, order=1), cmap="inferno", alpha=0.6, vmin=0, vmax=att.max())
    for a_ in (a1, a2):
        a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h); clean(a_)
    a1.set_ylabel("other slice", fontsize=6.5, labelpad=1); a2.set_ylabel("reference", fontsize=6.5, labelpad=1)
    a1.set_title("a  Layer 3", loc="left", fontsize=8)
    mid_f.add_artist(FancyArrowPatch((0.5, -0.01), (0.5, -0.21), transform=a1.transAxes, arrowstyle="-|>",
                                     mutation_scale=8, color=ORANGE, lw=1.3, clip_on=False))
    # ED/ES images: GT, full model, block layer 3
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_hb}.npz")))
    diff = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    cy, cx = np.unravel_index(np.argmax(diff), diff.shape); h = 34
    ys, xs = slice(cy - h, cy + h), slice(cx - h, cx + h)
    gi = mid[1].subgridspec(2, 3, wspace=0.04, hspace=0.06)
    cols = [("gt", "ground truth", INK), ("full", "full model", INK), ("ko_ref_L3", "block layer 3", ORANGE)]
    for j, (arm, lab, col) in enumerate(cols):
        for i, ph in enumerate(("ed", "es")):
            a_ = mid_f.add_subplot(gi[i, j])
            a_.imshow(e[f"{arm}_{ph}"][plane_hb, ys, xs].astype(np.float32), cmap="gray", vmin=0, vmax=1)
            clean(a_)
            if i == 0:
                a_.set_title(lab, fontsize=7, color=col, fontweight="bold", pad=2)
            if j == 0:
                a_.set_ylabel("reference in\n" + ("diastole" if ph == "ed" else "systole"), fontsize=6.5, labelpad=1)
            if j == 0 and i == 0:
                a_.text(0, 1.2, "b  A slice far from the reference", transform=a_.transAxes, fontsize=8,
                        fontweight="bold", ha="left", va="bottom")
    # real AF LV curve
    ax = mid_f.add_subplot(mid[2])
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    t = np.arange(len(d8["full"]["curve_ml"])) * 0.025
    ax.plot(t, d8["full"]["curve_ml"], color=INK, lw=1.0, label="full model")
    ax.plot(t, d8["ko_ref_L3"]["curve_ml"], color=ORANGE, lw=1.4, label="block layer 3")
    ax.set_xlabel("time (s)", labelpad=1); ax.set_ylabel("LV volume (mL)", labelpad=1)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.legend(fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, handlelength=1.2, columnspacing=0.8)
    ax.set_ylim(None, max(d8["full"]["curve_ml"]) * 1.2)
    ax.set_title("c  Real free-breathing AF scan", loc="left", fontsize=8)
    # ---- bottom row: breathing, long-axis cut through the stack ----
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_br}.npz")))
    diff = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    cy, cx = np.unravel_index(np.argmax(diff), diff.shape); h = 55
    f = float(e["dz"]) / 1.4
    gb = bot.add_gridspec(1, 4, wspace=0.04, left=0.07, right=0.99, top=0.84, bottom=0.02)
    cols = [("gt", "ground truth", INK), ("full", "full model", INK),
            ("self_L12_17", "block layers 12–17", AQUA), ("identity", "no correction", GRAY)]
    for j, (arm, lab, col) in enumerate(cols):
        a_ = bot.add_subplot(gb[j])
        im = e[f"{arm}_ed"][:, cy - h:cy + h, cx - 3:cx + 3].astype(np.float32).mean(2)
        a_.imshow(zoom(im, (f, 1), order=0), cmap="gray", vmin=0, vmax=1, aspect="auto")
        clean(a_)
        a_.set_title(lab, fontsize=7, color=col, fontweight="bold", pad=2)
        if j == 0:
            a_.text(0, 1.14, "d  Breathing: long-axis cut through the stacked slices", transform=a_.transAxes,
                    fontsize=8, fontweight="bold", ha="left", va="bottom")
            a_.set_ylabel("slices (base → apex)", fontsize=6.5, labelpad=1)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_paper_visual.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_paper_visual.png")


def fig_appendix(sub_attn="s090", layers=(1, 3, 8, 15), sub_hb="s075", plane_hb=10):
    """Appendix figure. a: one query point's global attention (mean of all 16 heads) over the reference and
    other slices, per layer. b: a far slice with the reference at diastole/systole under full model and the two
    blocks. c: real AF LV curve, full vs reference blocked in layer 3."""
    from scipy.ndimage import gaussian_filter, zoom

    def clean(a_):
        a_.set_xticks([]); a_.set_yticks([])
        for s_ in a_.spines.values():
            s_.set_visible(False)

    W, Hh = 7.0, 6.72
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, h):                        # inches, y measured from the top
        return fig.add_axes([x / W, (Hh - y - h) / Hh, w / W, h / Hh])

    g = 0.03
    # ---- a: attention of one point, per layer, over slices ----
    r = pickle.load(open(f"{D}/e3/{sub_attn}.pkl", "rb"))
    maps = r["maps"]["breath"]; imgs = r["images_breath"].astype(np.float32)
    qs = sorted({k[1] for k in maps if k[1] != 0})[0]
    qr, qc = next(k[2:] for k in maps if k[1] == qs)
    G = 37; sc = 14 * 256 / 518
    cy, cx = (qr + 0.5) * sc, (qc + 0.5) * sc; h = 58
    z = np.asarray(r["slot_z"])
    other = [s for s in np.argsort(np.abs(z - z[qs])) if s not in (0, qs)][0]
    cols = [(qs, "slice with\nthe point (+)"), (0, "reference\nslice"), (other, "another\nslice")]
    A = {l: maps[(l, qs, qr, qc)].astype(np.float32).mean(0) for l in layers}          # (S,1374), mean of 16 heads
    vmax = np.percentile(A[3][0, 5:], 99.5)
    ta, xa, ya = 0.8, 0.3, 0.5
    for i, l in enumerate(layers):
        for j, (s, lab) in enumerate(cols):
            a_ = ax_in(xa + j * (ta + g), ya + i * (ta + g), ta, ta)
            a_.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
            a_.imshow(zoom(A[l][s, 5:].reshape(G, G), 256 / G, order=1), cmap="inferno", alpha=0.65, vmin=0, vmax=vmax)
            if s == qs:
                a_.plot(cx, cy, "+", color="#3fd0ff", ms=6, mew=1.5)
            a_.set_xlim(cx - h, cx + h); a_.set_ylim(cy + h, cy - h); clean(a_)
            a_.text(0.03, 0.04, f"{100 * A[l][s].sum():.0f}%", transform=a_.transAxes, color="white", fontsize=6,
                    fontweight="bold" if (l == 3 and s == 0) else "normal")
            if i == 0:
                a_.set_title(lab, fontsize=7, pad=2, color=ORANGE if s == 0 else INK, fontweight="bold" if s == 0 else "normal")
            if j == 0:
                a_.set_ylabel(f"layer {l}", fontsize=7, labelpad=2, color=ORANGE if l == 3 else INK,
                              fontweight="bold" if l == 3 else "normal")
    fig.text(0.0, 1 - 0.02 / Hh, "a  Where one point (+) attends", fontsize=8.5, fontweight="bold", va="top")
    # ---- c: real AF LV curve ----
    xc = xa + 3 * ta + 2 * g + 0.5
    ax = ax_in(xc, ya, W - xc - 0.05, 4 * ta + 3 * g - 0.32)
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    t = np.arange(len(d8["full"]["curve_ml"])) * 0.025
    ax.plot(t, d8["full"]["curve_ml"], color=INK, lw=1.0, label="full model")
    ax.plot(t, d8["ko_ref_L3"]["curve_ml"], color=ORANGE, lw=1.4, label="reference blocked")
    ax.set_xlabel("time (s)", labelpad=1, fontsize=7); ax.set_ylabel("LV volume (mL)", labelpad=1, fontsize=7)
    ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.set_ylim(None, max(d8["full"]["curve_ml"]) * 1.1)
    ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.5)
    fig.text((xc - 0.5) / W, 1 - 0.02 / Hh, "c  Real AF scan: LV volume of the reconstruction", fontsize=8.5,
             fontweight="bold", va="top")
    # ---- b: heartbeat grid ----
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_hb}.npz")))
    diff = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    cy, cx = np.unravel_index(np.argmax(diff), diff.shape); h = 34
    ys, xs = slice(cy - h, cy + h), slice(cx - h, cx + h)
    rp = int(e["ref_plane"])
    yb0 = ya + 4 * ta + 3 * g + 0.14                    # panel-b title line
    yb, xb = yb0 + 0.42, 0.3
    tb = (W - xb - 4 * g) / 5
    bcols = [("ref", "reference slice\n(the only change)", ORANGE), ("identity", "another slice:\ninput", INK),
             ("full", "another slice:\nfull model", INK), ("ko_ref_L3", "reference blocked\n(layer 3)", ORANGE),
             ("self_L12_17", "slices blocked\n(layers 12–17)", AQUA)]
    for j, (arm, lab, col) in enumerate(bcols):
        for i, ph in enumerate(("ed", "es")):
            a_ = ax_in(xb + j * (tb + g), yb + i * (tb + g), tb, tb)
            im = e[f"gt_{ph}"][rp] if arm == "ref" else e[f"{arm}_{ph}"][plane_hb]
            a_.imshow(im[ys, xs].astype(np.float32), cmap="gray", vmin=0, vmax=1); clean(a_)
            if arm == "ref":
                for s_ in a_.spines.values():
                    s_.set_visible(True); s_.set_color(ORANGE); s_.set_linewidth(1.5)
            if i == 0:
                a_.set_title(lab, fontsize=7, color=col, fontweight="bold", pad=2)
            if j == 0:
                a_.set_ylabel("reference in " + ("diastole" if ph == "ed" else "systole"), fontsize=7, labelpad=2)
    fig.text(0.0, 1 - yb0 / Hh, "b  Change only the reference slice: how is another slice reconstructed?",
             fontsize=8.5, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_appendix.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_appendix.png")


def fig_attn_paper(sub="s090", layer=3, grid_q=("g14_18", "g18_14", "g18_22", "g24_18"), n_other=4):
    """Attention figure in the style of ZeroCo Fig.1 / Understanding-MV Fig.15 (rows = query points, columns =
    slices, argmax marked) plus a per-slice mass bar per row and a 180-subject per-layer profile (AVGGT Fig.2)."""
    from scipy.ndimage import zoom
    g_ = pickle.load(open(os.path.join(ROOT, f"scratch/interp/e12_grid/{sub}.pkl"), "rb"))
    n_ = pickle.load(open(os.path.join(ROOT, f"scratch/interp/e12/{sub}.pkl"), "rb"))
    rows = [(g_, next(q for q in g_["queries"] if q[2] == n), "heart") for n in grid_q]
    rows.append((n_, next(q for q in n_["queries"] if q[0] == "near" and q[2] == "body"), "outside heart"))
    S, z, imgs = g_["S"], np.asarray(g_["slot_z"]), g_["images"].astype(np.float32)
    qs = rows[0][1][1]
    others = [s for s in np.argsort(np.abs(z - z[qs])) if s not in (0, qs)][:n_other]
    cols = [qs, 0] + list(others)
    G, sc = 37, 14 * 256 / 518
    A = [rr["maps"][(layer, q[1], q[3], q[4])].astype(np.float32).mean(0) for rr, q, _ in rows]   # (S,1374) each
    vmax = np.percentile(np.stack([a[0, 5:] for a in A]), 99.9)
    hr, hc = np.mean([q[3] for _, q, _ in rows[:-1]]), np.mean([q[4] for _, q, _ in rows[:-1]])
    cy, cx, h = (hr + .5) * sc, (hc + .5) * sc, 62
    prof = np.load(os.path.join(ROOT, "scratch/interp/e3_profile.npz"))
    W, t, g = 7.0, 0.56, 0.02
    Hh = 0.42 + len(rows) * (t + g)
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    x0, y0 = 0.42, 0.34
    xb = x0 + len(cols) * (t + g) + 0.12
    for i, (rr, q, lab) in enumerate(rows):
        a = A[i]
        for j, s in enumerate(cols):
            ax = ax_in(x0 + j * (t + g), y0 + i * (t + g), t, t)
            ax.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
            if j > 0:
                m = a[s, 5:].reshape(G, G)
                ax.imshow(zoom(m, 256 / G, order=1), cmap="inferno", alpha=0.65, vmin=0, vmax=vmax)
            if j == 0:
                ax.plot((q[4] + .5) * sc, (q[3] + .5) * sc, "o", ms=4.5, mfc="#3fd0ff", mec="w", mew=0.6)
            if s == 0:
                am = np.unravel_index(np.argmax(a[0, 5:]), (G, G))
                ax.plot((am[1] + .5) * sc, (am[0] + .5) * sc, "x", color="#e34948", ms=5, mew=1.4)
                ax.plot((q[4] + .5) * sc, (q[3] + .5) * sc, "o", ms=3, mfc="none", mec="#3fd0ff", mew=0.8)
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(ORANGE); sp.set_linewidth(1.4)
            else:
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h); ax.set_xticks([]); ax.set_yticks([])
            if j > 0:
                ax.text(.04, .04, f"{100 * a[s].sum():.0f}%", transform=ax.transAxes, color="w", fontsize=5.5,
                        fontweight="bold" if s == 0 else "normal")
            if i == 0:
                ax.set_title(["query slice", "reference", "other slices", "", "", ""][j] if j < 3 else "",
                             fontsize=7, pad=2, color=ORANGE if s == 0 else INK, fontweight="bold" if s == 0 else "normal",
                             loc="left" if j == 2 else "center")
            if j == 0:
                ax.set_ylabel(lab, fontsize=6, labelpad=2)
        # per-slice attention share bar (all slices, reference first)
        ax = ax_in(xb, y0 + i * (t + g) + 0.06, 1.0, t - 0.12)
        order = [0] + [s for s in np.argsort(z) if s != 0]
        share = a.sum(1)[order] * 100
        ax.bar(range(S), share, width=0.8, color=[ORANGE] + [GRAY] * (S - 1))
        ax.axhline(100 / S, color=INK, lw=0.6, ls="--")
        ax.set_ylim(0, 60); ax.set_xticks([]); ax.tick_params(labelsize=5, length=1.5, pad=1)
        ax.set_yticks([0, 50])
        if i == 0:
            ax.set_title("share per slice (%)", fontsize=6.5, pad=2)
        if i == len(rows) - 1:
            ax.set_xlabel("ref | other slices", fontsize=5.5, labelpad=1)
    # per-layer profile over 180 subjects
    xp = xb + 1.0 + 0.42
    ax = ax_in(xp, y0 + 0.02, W - xp - 0.05, len(rows) * (t + g) - 0.42)
    L = np.arange(24)
    for key, col, lab in (("all16", INK, "all 16 heads"), ("causal4", ORANGE, "4 causal heads")):
        v = prof[key]
        ax.fill_between(L, np.percentile(v, 25, 0), np.percentile(v, 75, 0), color=col, alpha=0.18, lw=0)
        ax.plot(L, np.median(v, 0), color=col, lw=1.2, label=lab)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.set_xticks([0, 3, 8, 12, 17, 23]); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    ax.set_xlabel("global-attention layer", fontsize=6, labelpad=1)
    ax.set_ylabel("attention to reference\n(× fair share)", fontsize=6, labelpad=1)
    ax.legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.2)
    ax.text(0.97, 0.6, "median, IQR\n180 test subjects", transform=ax.transAxes, ha="right", fontsize=5.5, color=MUTED)
    fig.text(0.0, 1 - 0.03 / Hh, f"a  Where query points (●) on a non-reference slice attend, global layer {layer}",
             fontsize=7.5, fontweight="bold", va="top")
    fig.text((xp - 0.4) / W, 1 - 0.03 / Hh, "b  Attention to the reference, per layer", fontsize=7.5, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_attn_paper.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_attn_paper.png")


def fig_attn_simple(sub="s090", layer=3, grid_q=("g14_18", "g24_18")):
    """Simple attention figure. a: three examples of (point on another slice -> where it looks on the reference
    slice in layer 3). b: attention to the reference slice per layer, 180 test subjects."""
    from matplotlib.patches import FancyArrowPatch
    from scipy.ndimage import zoom
    g_ = pickle.load(open(os.path.join(ROOT, f"scratch/interp/e12_grid/{sub}.pkl"), "rb"))
    Q = [next(q for q in g_["queries"] if q[2] == n) for n in grid_q]
    imgs = g_["images"].astype(np.float32)
    qs = Q[0][1]
    G, sc = 37, 14 * 256 / 518
    A = [g_["maps"][(layer, q[1], q[3], q[4])].astype(np.float32).mean(0)[0, 5:].reshape(G, G) for q in Q]
    vmax = np.percentile(np.stack(A), 99.9)
    all_q = [q for q in g_["queries"]]
    cy = (np.mean([q[3] for q in all_q]) + .5) * sc; cx = (np.mean([q[4] for q in all_q]) + .5) * sc; h = 55
    prof = np.load(os.path.join(ROOT, "scratch/interp/e3_profile.npz"))["all16"]
    W, Hh, t, gap, pair_gap = 7.0, 1.78, 1.0, 0.22, 0.12
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    y0 = 0.5
    for k, (q, a) in enumerate(zip(Q, A)):
        x = 0.02 + k * (2 * t + gap + pair_gap)
        for j in range(2):
            ax = ax_in(x + j * (t + gap), y0, t, t)
            s = qs if j == 0 else 0
            ax.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
            if j == 0:
                ax.plot((q[4] + .5) * sc, (q[3] + .5) * sc, "o", ms=6, mfc="#3fd0ff", mec="white", mew=1)
            else:
                ax.imshow(zoom(a, 256 / G, order=1), cmap="inferno", alpha=0.7, vmin=0, vmax=vmax)
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(ORANGE); sp.set_linewidth(1.5)
            if j == 0:
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h); ax.set_xticks([]); ax.set_yticks([])
            if k == 0:
                ax.set_title("point on another slice" if j == 0 else "where it looks\non the reference slice",
                             fontsize=6.5, pad=2, color=INK if j == 0 else ORANGE)
        fig.add_artist(FancyArrowPatch(((x + t + 0.03) / W, (Hh - y0 - t / 2) / Hh), ((x + t + gap - 0.03) / W, (Hh - y0 - t / 2) / Hh),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=INK, lw=1))
    fig.text(0.0, 1 - 0.03 / Hh, f"a  Layer {layer}: a point on another slice looks at the reference heart",
             fontsize=7.5, fontweight="bold", va="top")
    # b: per-layer profile
    xb = 0.02 + len(Q) * (2 * t + gap + pair_gap) + 0.45
    ax = ax_in(xb, y0 - 0.02, W - xb - 0.05, t + 0.02)
    L = np.arange(24)
    ax.fill_between(L, np.percentile(prof, 25, 0), np.percentile(prof, 75, 0), color=ORANGE, alpha=0.2, lw=0)
    ax.plot(L, np.median(prof, 0), color=ORANGE, lw=1.3)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.text(23, 1.12, "even split", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 3, 12, 23]); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xlabel("layer", fontsize=6, labelpad=1)
    ax.set_ylabel("attention to reference\n(× even split)", fontsize=6, labelpad=1)
    fig.text((xb - 0.3) / W, 1 - 0.03 / Hh, "b  Only layer 3", fontsize=7.5, fontweight="bold", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_attn_simple.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_attn_simple.png")


def fig_cuts(sub_br="s117"):
    """Final appendix figure (2-agent debate): 2 rows x 3 columns. Columns: full / reference cut off (L3) /
    slices cut off from each other (L12-17). Row 1: real AF LV volume curve. Row 2: long-axis cut (breathing)."""
    from scipy.ndimage import gaussian_filter, zoom
    arms = [("full", "full model"), ("ko_ref_L3", "reference cut off\n(layer 3)"),
            ("self_L12_17", "slices cut off from each other\n(layers 12–17)")]
    verdict_hb = {"full": "beats", "ko_ref_L3": "flat", "self_L12_17": "beats"}
    verdict_br = {"full": "aligned", "ko_ref_L3": "aligned", "self_L12_17": "misaligned"}
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_br}.npz")))
    diff = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    cy, cx = np.unravel_index(np.argmax(diff), diff.shape); h = 55
    f = float(e["dz"]) / 1.4
    W, Hh = 7.0, 4.1
    fig = plt.figure(figsize=(W, Hh))
    x0, cw, gap = 0.62, 2.06, 0.08
    y1, h1, y2, h2 = 0.5, 1.05, 2.15, 1.6

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    curves = [d8[a]["curve_ml"] for a, _ in arms]
    lo, hi = min(map(min, curves)) - 5, max(map(max, curves)) + 5
    for j, (arm, lab) in enumerate(arms):
        x = x0 + j * (cw + gap)
        ax = ax_in(x, y1, cw, h1)
        c = d8[arm]["curve_ml"]
        ax.plot(np.arange(len(c)) * 0.025, c, color=INK, lw=1.0)
        ax.set_ylim(lo, hi); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xlabel("time (s)", fontsize=6, labelpad=1)
        if j == 0:
            ax.set_ylabel("LV volume (mL)", fontsize=6, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(lab, fontsize=7.5, fontweight="bold", pad=4)
        ax.text(0.5, -0.33, verdict_hb[arm], transform=ax.transAxes, ha="center", va="top", fontsize=8, fontweight="bold",
                color=ORANGE if verdict_hb[arm] == "flat" else INK)
        ax2 = ax_in(x + (cw - h2) / 2, y2, h2, h2)
        im = e[f"{arm}_ed"][:, cy - h:cy + h, cx - 3:cx + 3].astype(np.float32).mean(2)
        ax2.imshow(zoom(im, (f, 1), order=0), cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.text(0.5, -0.08, verdict_br[arm], transform=ax2.transAxes, ha="center", va="top", fontsize=8,
                 fontweight="bold", color=ORANGE if verdict_br[arm] == "misaligned" else INK)
    fig.text(0.0, 1 - (y1 + h1 / 2) / Hh, "Heartbeat\n(real AF scan)", fontsize=7.5, fontweight="bold", va="center", rotation=90)
    fig.text(0.0, 1 - (y2 + h2 / 2) / Hh, "Breathing\n(long-axis cut)", fontsize=7.5, fontweight="bold", va="center", rotation=90)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_cuts.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_cuts.png")


def fig_final(sub_attn="s090", grid_q=("g14_18", "g24_18"), sub_br="s117"):
    """Appendix figure. a: layer-3 attention (a point on another slice -> the reference slice's heart) and the
    per-layer attention to the reference (180 subjects). b: cutting that connection vs cutting slice-to-slice
    attention: real AF LV curve (heartbeat) and long-axis cut (breathing)."""
    from matplotlib.patches import FancyArrowPatch
    from scipy.ndimage import gaussian_filter, zoom
    W, Hh = 7.0, 4.72
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    def clean(ax):
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    # ---------- a: attention ----------
    g_ = pickle.load(open(os.path.join(ROOT, f"scratch/interp/e12_grid/{sub_attn}.pkl"), "rb"))
    Q = [next(q for q in g_["queries"] if q[2] == n) for n in grid_q]
    imgs = g_["images"].astype(np.float32); qs = Q[0][1]
    G, sc = 37, 14 * 256 / 518
    A = [g_["maps"][(3, q[1], q[3], q[4])].astype(np.float32).mean(0)[0, 5:].reshape(G, G) for q in Q]
    vmax = np.percentile(np.stack(A), 99.9)
    cy = (np.mean([q[3] for q in g_["queries"]]) + .5) * sc; cx = (np.mean([q[4] for q in g_["queries"]]) + .5) * sc; h = 55
    t, gap, pg, ya = 0.95, 0.22, 0.12, 0.36
    for k, (q, a) in enumerate(zip(Q, A)):
        x = 0.02 + k * (2 * t + gap + pg)
        for j in range(2):
            ax = ax_in(x + j * (t + gap), ya, t, t)
            s = qs if j == 0 else 0
            ax.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
            if j == 0:
                ax.plot((q[4] + .5) * sc, (q[3] + .5) * sc, "o", ms=6, mfc="#3fd0ff", mec="white", mew=1)
            else:
                ax.imshow(zoom(a, 256 / G, order=1), cmap="inferno", alpha=0.7, vmin=0, vmax=vmax)
            ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h); clean(ax)
            if j == 1:
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(ORANGE); sp.set_linewidth(1.5)
            if k == 0:
                ax.set_title("another slice" if j == 0 else "reference slice",
                             fontsize=6.5, pad=2, color=INK if j == 0 else ORANGE, fontweight="bold")
        fig.add_artist(FancyArrowPatch(((x + t + 0.03) / W, (Hh - ya - t / 2) / Hh), ((x + t + gap - 0.03) / W, (Hh - ya - t / 2) / Hh),
                                       transform=fig.transFigure, arrowstyle="-|>", mutation_scale=9, color=INK, lw=1))
    prof = np.load(os.path.join(ROOT, "scratch/interp/e3_profile.npz"))["all16"]
    xp = 0.02 + 2 * (2 * t + gap + pg) + 0.5
    ax = ax_in(xp, ya, W - xp - 0.05, t)
    L = np.arange(24)
    ax.fill_between(L, np.percentile(prof, 25, 0), np.percentile(prof, 75, 0), color=ORANGE, alpha=0.2, lw=0)
    ax.plot(L, np.median(prof, 0), color=ORANGE, lw=1.3)
    ax.axhline(1, color=MUTED, lw=0.6, ls="--")
    ax.text(23, 1.12, "even split", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    ax.set_xticks([0, 3, 12, 17, 23]); ax.set_yticks([0, 1, 2, 3, 4]); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
    ax.set_xlabel("global-attention layer", fontsize=6, labelpad=1)
    ax.set_ylabel("attention to reference\n(× even split)", fontsize=6, labelpad=1)
    ax.set_title("all 180 test subjects", fontsize=6.5, pad=2)
    fig.text(0.0, 1 - 0.03 / Hh, "a  What the model looks at: in layer 3, other slices look at the reference slice's heart",
             fontsize=8, fontweight="bold", va="top")
    # ---------- b: cuts ----------
    yb0 = ya + t + 0.22
    fig.text(0.0, 1 - yb0 / Hh, "b  What breaks when a connection is cut (test time, no retraining)",
             fontsize=8, fontweight="bold", va="top")
    arms = [("full", "full model"), ("ko_ref_L3", "cut: other slices → reference\n(layer 3)"),
            ("self_L12_17", "cut: slices ↔ each other\n(layers 12–17)")]
    vhb = {"full": "beats", "ko_ref_L3": "flat", "self_L12_17": "beats"}
    vbr = {"full": "positions correct", "ko_ref_L3": "positions correct", "self_L12_17": "positions shifted"}
    d8 = json.load(open(os.path.join(ROOT, "scratch/interp/e8/e8_scores.json")))["MIITT_Patient_2024Jan04_Cardiomyopathy_AFib"]
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_br}.npz")))
    dif = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    by, bx = np.unravel_index(np.argmax(dif), dif.shape); bh = 55
    f = float(e["dz"]) / 1.4
    x0, cw, cg = 1.12, 1.92, 0.06
    y1, h1 = yb0 + 0.45, 0.78
    y2, h2 = y1 + h1 + 0.3, 1.5
    curves = [d8[a]["curve_ml"] for a, _ in arms]
    lo, hi = min(map(min, curves)) - 5, max(map(max, curves)) + 22
    for j, (arm, lab) in enumerate(arms):
        x = x0 + j * (cw + cg)
        col = INK if arm == "full" else ORANGE if arm == "ko_ref_L3" else AQUA
        ax = ax_in(x, y1, cw, h1)
        c = d8[arm]["curve_ml"]
        ax.plot(np.arange(len(c)) * 0.025, c, color=INK, lw=1.0)
        ax.set_ylim(lo, hi); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xlabel("time (s)", fontsize=5.5, labelpad=0)
        if j > 0:
            ax.set_yticklabels([])
        ax.set_title(lab, fontsize=7, fontweight="bold", pad=3, color=col)
        ax.text(0.98, 0.97, vhb[arm], transform=ax.transAxes, ha="right", va="top", fontsize=8, fontweight="bold",
                color=ORANGE if vhb[arm] == "flat" else INK)
        im = e[f"{arm}_ed"][:, by - bh:by + bh, bx - 3:bx + 3].astype(np.float32).mean(2)   # (D, 2*bh): rows = slices
        im = zoom(im, (f, 1), order=0)
        w2 = h2 * im.shape[1] / im.shape[0]
        ax2 = ax_in(x + (cw - w2) / 2, y2, w2, h2)
        ax2.imshow(im, cmap="gray", vmin=0, vmax=1, aspect="auto"); clean(ax2)
        ax2.text(1.04, 0.5, vbr[arm].replace(" ", "\n"), transform=ax2.transAxes, ha="left", va="center", fontsize=7.5,
                 fontweight="bold", color=AQUA if arm == "self_L12_17" else INK)
        if j == 0:
            ax2.set_ylabel("stacked slices\n(base → apex)", fontsize=5.5, labelpad=1)
    fig.text(0.0, 1 - (y1 + h1 / 2) / Hh, "Heartbeat\nLV volume (mL)\nof the reconstruction\n(real AF scan)",
             fontsize=6.5, fontweight="bold", va="center")
    fig.text(0.0, 1 - (y2 + h2 / 2) / Hh, "Breathing\nlong-axis cut\nthrough the stack\n(simulated scan)",
             fontsize=6.5, fontweight="bold", va="center")
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_final.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_final.png")


def fig_A(sub="s090", q_name="g14_18", layers=(3, 8), n_other=3, accent="#6D8EAE", scale="log"):
    """Fig A: a query point (x) on a non-reference slice; its attention over the reference slice and three other
    non-reference slices, in layer 3 vs a typical other layer (mean of 16 heads, one shared colour scale; the
    overlay is transparent where attention is low so the anatomy stays visible)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.patches import FancyArrowPatch
    from scipy.ndimage import zoom
    g_ = pickle.load(open(os.path.join(ROOT, f"scratch/interp/e12_grid/{sub}.pkl"), "rb"))
    q = next(q for q in g_["queries"] if q[2] == q_name)
    imgs = g_["images"].astype(np.float32); z = np.asarray(g_["slot_z"])
    qs = q[1]
    others = [s for s in np.argsort(np.abs(z - z[qs])) if s not in (0, qs)][:n_other]
    G, sc = 37, 14 * 256 / 518
    A = {l: g_["maps"][(l, qs, q[3], q[4])].astype(np.float32).mean(0) for l in layers}     # (S,1374)
    even = 1.0 / (g_["S"] * (G * G + 5))                   # weight per token if attention were spread evenly
    vmax = np.percentile(A[layers[0]][0, 5:], 99.8)
    vmin = even / 8 if scale == "log" else 0.0
    norm = LogNorm(vmin, vmax, clip=True) if scale == "log" else Normalize(vmin, vmax, clip=True)
    cmap = plt.get_cmap("inferno" if scale == "inferno" else "turbo")     # "inferno": linear, low = transparent
    cy = (np.mean([p[3] for p in g_["queries"]]) + .5) * sc; cx = (np.mean([p[4] for p in g_["queries"]]) + .5) * sc; h = 42

    def overlay(ax, a):
        a = zoom(a.reshape(G, G), 256 / G, order=1)
        rgba = cmap(norm(np.maximum(a, 1e-12)))
        rgba[..., 3] = np.clip(a / vmax, 0, 1) ** 0.5 * 0.9 if scale == "inferno" else 0.55
        ax.imshow(rgba)

    def base(ax, s):
        ax.imshow(imgs[s], cmap="gray", vmax=np.percentile(imgs[s], 99.5))
        ax.set_xlim(cx - h, cx + h); ax.set_ylim(cy + h, cy - h); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    W, g, arrow, lab, cbw = 7.0, 0.04, 0.26, 0.22, 0.09
    n = 1 + n_other
    t = (W - arrow - lab - cbw - 0.42 - (n + 1) * g) / (n + 2)
    tq = 2 * t + g
    y0 = 0.2
    Hh = y0 + tq + 0.02
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    ax = ax_in(0, y0, tq, tq); base(ax, qs)
    ax.plot((q[4] + .5) * sc, (q[3] + .5) * sc, "x", ms=11, mew=2.6, color=accent)
    ax.set_title("query slice", fontsize=7.5, pad=2)
    xg = tq + arrow + lab
    for i, l in enumerate(layers):
        for k, s in enumerate([0] + others):
            ax = ax_in(xg + k * (t + g), y0 + i * (t + g), t, t); base(ax, s); overlay(ax, A[l][s, 5:])
            if s == 0:
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(accent); sp.set_linewidth(2)
            if i == 0 and s == 0:
                ax.set_title("reference slice", fontsize=7.5, pad=2, color=accent, fontweight="bold")
        fig.text((xg - 0.05) / W, 1 - (y0 + i * (t + g) + t / 2) / Hh, f"block {l}", fontsize=7.5, rotation=90,
                 ha="right", va="center", fontweight="bold" if l == 3 else "normal")
    fig.text((xg + (t + g) + (n_other * (t + g) - g) / 2) / W, 1 - (y0 - 0.03) / Hh, "non-reference slices",
             fontsize=7.5, ha="center", va="bottom")
    fig.add_artist(FancyArrowPatch(((tq + 0.03) / W, (Hh - y0 - tq / 2) / Hh), ((tq + arrow - 0.03) / W, (Hh - y0 - tq / 2) / Hh),
                                   transform=fig.transFigure, arrowstyle="-|>", mutation_scale=10, color=INK, lw=1.2))
    cax = ax_in(xg + n * (t + g) + 0.04, y0 + 0.1, cbw, tq - 0.2)
    cb = fig.colorbar(ScalarMappable(norm, cmap), cax=cax)
    cb.set_ticks([vmin, vmax]); cb.minorticks_off()
    cb.ax.set_yticklabels(["low", "high"], fontsize=6.5); cb.outline.set_visible(False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_A_{scale}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_A_{scale}.png")


def fig_A_curve(accent="#6D8EAE"):
    """% of a non-reference heart point's attention that lands on the reference slice, per global layer
    (median and IQR over 180 test subjects, mean of 16 heads)."""
    d = np.load(os.path.join(ROOT, "scratch/interp/e3_share.npz"))
    X = d["share"] * 100
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    L = np.arange(24)
    ax.fill_between(L, np.percentile(X, 25, 0), np.percentile(X, 75, 0), color=accent, alpha=0.25, lw=0)
    ax.plot(L, np.median(X, 0), color=accent, lw=1.6)
    ax.set_xticks([0, 3, 8, 12, 17, 23]); ax.set_xlim(-0.5, 23.5); ax.set_ylim(0, None)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.set_xlabel("layer", fontsize=7, labelpad=1)
    ax.set_ylabel("attention on\nreference slice (%)", fontsize=7, labelpad=2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_A_curve.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_A_curve.png")


def fig_A_matrix(sub="s090", layers=(3, 8), accent="#6D8EAE"):
    """Slice x slice attention matrix (one test subject, mean of 16 heads): row = slice whose heart points look,
    column = slice they look at, cell = % of that row's attention. Reference slice first, others ordered by position."""
    r = pickle.load(open(f"{D}/e3/{sub}.pkl", "rb"))
    st, S, z = r["stats"]["breath"], r["S"], np.asarray(r["slot_z"])
    order = [0] + [s for s in np.argsort(z) if s != 0]
    M = {l: st[l]["mass_h"].mean(0)[np.ix_(order, order)] * 100 for l in layers}
    vmax = max(m.max() for m in M.values())
    fig, axs = plt.subplots(1, len(layers), figsize=(1.75 * len(layers) + 0.5, 1.9), gridspec_kw={"wspace": 0.12})
    for ax, l in zip(axs, layers):
        im = ax.imshow(M[l], cmap="inferno", vmin=0, vmax=vmax)
        ax.set_xticks([0]); ax.set_xticklabels(["ref."], fontsize=6.5, color=accent, fontweight="bold")
        ax.set_yticks([0]); ax.set_yticklabels(["ref."], fontsize=6.5, color=accent, fontweight="bold")
        ax.tick_params(length=0, pad=2)
        ax.set_title(f"layer {l}", fontsize=8, pad=3, fontweight="bold" if l == 3 else "normal")
        ax.set_xlabel("slice looked at", fontsize=7, labelpad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_ylabel("slice looking", fontsize=7, labelpad=2)
    for ax in axs[1:]:
        ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=axs, fraction=0.04, pad=0.03)
    cb.ax.tick_params(labelsize=6, length=1.5); cb.outline.set_visible(False)
    cb.set_label("% of attention", fontsize=6.5, labelpad=2)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_A_matrix.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_A_matrix.png")


def hrv_curve(phase_vals, seed=0, dur=4.0, fps=12, cycle=1.0, sd=0.15, rhythm="hrv"):
    """Paper's test rhythms (appendix). Each beat's normalized duration ~ N(1, sd^2).
    hrv: the whole 12-phase cycle is time-scaled to the beat duration.
    af (sd 0.25): the cine runs at its original rate; a long beat completes the cycle then holds end-diastole
    (phase 0) for the rest; a short beat is cut off early (next beat restarts at phase 0 -- an approximation).
    In-between states linearly interpolated. Returns (t, value(t))."""
    rng = np.random.default_rng(seed)
    T = len(phase_vals)
    t = np.arange(int(round(dur * fps))) / fps
    lo, hi = (0.7, 1.3) if rhythm == "af" else (0.3, 3.0)          # AF: keep beats in a realistic 0.7-1.3x range
    edges = [0.0]
    while edges[-1] < dur:
        d = rng.normal(1, sd)
        while not lo <= d <= hi:
            d = rng.normal(1, sd)
        edges.append(edges[-1] + cycle * d)
    edges = np.array(edges)
    k = np.searchsorted(edges, t, side="right") - 1
    if rhythm == "af":
        ph = np.minimum((t - edges[k]) / cycle, 1.0) * T                     # original rate, hold at ED (T == 0)
    else:
        ph = (t - edges[k]) / (edges[k + 1] - edges[k]) * T                 # continuous phase in [0, T)
    v = np.asarray(phase_vals, float)
    return t, v[np.floor(ph).astype(int) % T] * (1 - ph % 1) + v[(np.floor(ph).astype(int) + 1) % T] * (ph % 1)


def fig_B(sub_br="s117", rhythm="hrv"):
    """Fig B, one simulated test subject for both rows. Columns full / cut layer 3 / cut layers 12-17.
    Row 1: LV volume (nnU-Net, non-reference slices) as the reference follows an HRV rhythm (paper protocol).
    Row 2: long-axis cut through the reconstructed stack (simulated breathing)."""
    from scipy.ndimage import gaussian_filter, zoom
    arms = [("full", "full model", INK), ("ko_ref_L3", "cut layer 3", "#6D8EAE"), ("self_L12_17", "cut layers 12–17", AQUA)]
    idx = int(sub_br[1:])
    e4 = next(r for r in json.load(open(os.path.join(ROOT, "scratch/interp/e4/e4_curves.json")))["per_subject"] if r["idx"] == idx)
    e = dict(np.load(os.path.join(ROOT, f"scratch/interp/e10/{sub_br}.npz")))
    vox_ml = 1.4 * 1.4 * float(e["dz"]) / 1000.0
    sd = 0.25 if rhythm == "af" else 0.15
    d8 = {a: {"curve_ml": hrv_curve(np.array(e4[a]["comp"]) * vox_ml, seed=idx, sd=sd, rhythm=rhythm)[1]}
          for a, _, _ in arms}
    dif = gaussian_filter(np.abs(e["gt_ed"].astype(float) - e["gt_es"].astype(float)).sum(0), 6)
    by, bx = np.unravel_index(np.argmax(dif), dif.shape)
    f = float(e["dz"]) / 1.4
    W, cw, cg, x0 = 7.0, 2.05, 0.05, 0.52
    D = e["full_ed"].shape[0]
    bh = 50                                                  # heart-sized in-plane crop (70 mm half-width)
    by = int(np.clip(by, bh, e["full_ed"].shape[1] - bh))
    y1, h1 = 0.2, 0.8
    y2, h2 = y1 + h1 + 0.28, cw * D * f / (2 * bh)           # true proportions, full column width
    Hh = y2 + h2 + 0.02
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    curves = [d8[a]["curve_ml"] for a, _, _ in arms]
    lo, hi = min(map(min, curves)) - 5, max(map(max, curves)) + 5
    for j, (arm, lab, col) in enumerate(arms):
        x = x0 + j * (cw + cg)
        ax = ax_in(x, y1, cw, h1)
        c = d8[arm]["curve_ml"]
        ax.plot(np.arange(len(c)) / 12, c, color=INK, lw=1.0)
        ax.set_ylim(lo, hi); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xlabel("time (s)", fontsize=6, labelpad=0)
        if j == 0:
            ax.set_ylabel("LV volume (mL)", fontsize=6, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(lab, fontsize=8, fontweight="bold", pad=3, color=col)
        im = zoom(e[f"{arm}_ed"][:, by - bh:by + bh, bx - 3:bx + 3].astype(np.float32).mean(2), (f, 1), order=0)
        ax2 = ax_in(x, y2, cw, h2)
        ax2.imshow(im, cmap="gray", vmin=0, vmax=1, aspect="auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
    fig.text(0.0, 1 - (y1 + h1 / 2) / Hh, "heartbeat", fontsize=8, fontweight="bold", va="center", rotation=90)
    fig.text(0.0, 1 - (y2 + h2 / 2) / Hh, "breathing", fontsize=8, fontweight="bold", va="center", rotation=90)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_B_{rhythm}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_B_{rhythm}.png")


def fig_B_af48(sub="CMRx24_Test_P017", cohort="cmrx2024_af48", frame=0, e15_dir="e15"):
    """Fig B from E15 (paper AF48 input, one subject). Row 1: nnU-Net LV volume of the non-reference slices
    over the 4 s AF rhythm (GT dashed). Row 2: long-axis cut through the reconstruction at `frame`."""
    import nibabel as nib
    from scipy.ndimage import zoom
    e15 = os.path.join(ROOT, "scratch/interp", e15_dir)
    man = json.load(open(os.path.join(ROOT, "scratch/eval", cohort, "out", sub, "manifest.json")))
    T, ref, dz = man["T"], int(man["scatter"]["ref_plane"]), man["dz_mm"]
    vox = 1.4 * 1.4 * dz / 1000

    def lv(arm):
        out = []
        for f in range(T):
            s = np.asarray(nib.load(os.path.join(e15, "seg", f"{arm}__t{f:02d}.nii.gz")).dataobj)
            out.append(((s == 1).sum() - (s[:, :, ref] == 1).sum()) * vox)
        return np.array(out)

    arms = [("full", "full model", INK), ("ko_ref_L3", "cut layer 3", "#6D8EAE"), ("self_L12_17", "cut layers 12–17", AQUA)]
    curves = {a: lv(a) for a, _, _ in arms}
    gt = lv("gt")
    s0 = np.asarray(nib.load(os.path.join(e15, "seg", "gt__t00.nii.gz")).dataobj)
    xs, ys, _ = np.nonzero(s0 == 1); cx, cy = int(xs.mean()), int(ys.mean())
    bh = 50
    W, cw, cg, x0 = 7.0, 2.05, 0.05, 0.52
    f_ = dz / 1.4
    y1, h1 = 0.2, 0.8
    y2, h2 = y1 + h1 + 0.28, cw * man["D"] * f_ / (2 * bh)
    Hh = y2 + h2 + 0.02
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    lo = min(min(c.min() for c in curves.values()), gt.min()) - 5
    hi = max(max(c.max() for c in curves.values()), gt.max()) + 5
    t = np.arange(T) / 12
    vmax = None
    for j, (arm, lab, col) in enumerate(arms):
        x = x0 + j * (cw + cg)
        ax = ax_in(x, y1, cw, h1)
        ax.plot(t, gt, color=GRAY, lw=0.9, ls="--")
        ax.plot(t, curves[arm], color=INK, lw=1.1)
        ax.set_ylim(lo, hi); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xlabel("time (s)", fontsize=6, labelpad=0)
        if j == 0:
            ax.set_ylabel("LV volume (mL)", fontsize=6, labelpad=1)
        else:
            ax.set_yticklabels([])
        ax.set_title(lab, fontsize=8, fontweight="bold", pad=3, color=col)
        v = np.asarray(nib.load(os.path.join(e15, arm, f"recon_t{frame:02d}.nii.gz")).dataobj, np.float32)   # (X,Y,Z)
        vmax = vmax or np.percentile(v, 99.5)
        cut = v[cx - 2:cx + 3, max(cy - bh, 0):cy + bh, :].mean(0).T                                          # (Z, Y)
        ax2 = ax_in(x, y2, cw, h2)
        ax2.imshow(zoom(cut, (f_, 1), order=0), cmap="gray", vmin=0, vmax=vmax, aspect="auto")
        ax2.set_xticks([]); ax2.set_yticks([])
        for sp in ax2.spines.values():
            sp.set_visible(False)
    fig.text(0.0, 1 - (y1 + h1 / 2) / Hh, "heartbeat", fontsize=8, fontweight="bold", va="center", rotation=90)
    fig.text(0.0, 1 - (y2 + h2 / 2) / Hh, "breathing", fontsize=8, fontweight="bold", va="center", rotation=90)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_B_af48_{sub}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", f"{OUT}/fig_B_af48.png", {a: round(float(c.max() - c.min()), 1) for a, c in curves.items()},
          "gt", round(float(gt.max() - gt.min()), 1))


def fig_B_final(sub="CMRx24_Test_P017", cohort="cmrx2024_af48", e15_dir="e15",
                eval_root="scratch/interp/eval48", frame=0, with_images=False, show_gt=True):
    """Fig B (paper engine, one subject, HRV 48 frames). Columns full / cut layer 3 / cut layers 12-17.
    Row 1: nnU-Net LV volume of the non-reference slices over 4 s (GT dashed).
    Row 2: per slice, breathing shift applied by the engine (grey bars) vs the model's correction (dots)."""
    import nibabel as nib
    e15 = os.path.join(ROOT, "scratch/interp", e15_dir)
    man = json.load(open(os.path.join(ROOT, eval_root, cohort, "out", sub, "manifest.json")))
    T, ref, dz = man["T"], int(man["scatter"]["ref_plane"]), man["dz_mm"]
    vox = 1.4 * 1.4 * dz / 1000

    def lv(arm):
        out = []
        for f in range(T):
            s = np.asarray(nib.load(os.path.join(e15, "seg", f"{arm}__t{f:02d}.nii.gz")).dataobj)
            out.append(((s == 1).sum() - (s[:, :, ref] == 1).sum()) * vox)
        return np.array(out)

    br = json.load(open(os.path.join(e15, f"breath_readout_t{frame:02d}.json")))
    order = np.argsort(br["z"])
    applied = np.asarray(br["applied_mm"])[order]
    arms = [("full", "(a) CardioStitch\n", INK), ("ko_ref_L3", "(b) w/o reference attention\n(block 3)", "#6D8EAE"),
            ("self_L12_17", "(c) w/o global attention\n(blocks 12–17)", AQUA)]
    curves = {a: lv(a) for a, _, _ in arms}
    gt = lv("gt")
    W, cw, cg, x0 = 7.0, 2.1, 0.05, 0.42
    y1, h1 = 0.2, 0.85
    y2, h2 = y1 + h1 + 0.3, 0.85
    y3, h3 = y2 + h2 + 0.32, cw * man["D"] * (dz / 1.4) / 100      # long-axis row, true proportions (100 px wide)
    Hh = (y3 + h3 + 0.02) if with_images else (y2 + h2 + 0.17)
    fig = plt.figure(figsize=(W, Hh))

    def ax_in(x, y, w, hh):
        return fig.add_axes([x / W, (Hh - y - hh) / Hh, w / W, hh / Hh])

    ref_c = list(curves.values()) + ([gt] if show_gt else [])
    lo = min(c.min() for c in ref_c) - 5
    hi = max(c.max() for c in ref_c) + 5
    t = np.arange(T) / 12
    bmax = max(applied.max(), max(np.max(br[a]) for a, _, _ in arms)) + 1.5
    for j, (arm, lab, col) in enumerate(arms):
        x = x0 + j * (cw + cg)
        ax = ax_in(x, y1, cw, h1)
        if show_gt:
            l_gt, = ax.plot(t, gt, color=GRAY, lw=0.9, ls="--")
        l_rc, = ax.plot(t, curves[arm], color=INK, lw=1.1)
        ax.set_ylim(lo, hi + (0.18 * (hi - lo) if show_gt else 0)); ax.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax.set_xlabel("time (s)", fontsize=6, labelpad=0)
        ax.set_title(lab, fontsize=7.5, pad=5, color=INK, fontweight="normal")
        ax2 = ax_in(x, y2, cw, h2)
        k = np.arange(len(applied))
        b_ap = ax2.bar(k, applied, width=0.75, color="#d6d5d0")
        l_pr, = ax2.plot(k, np.asarray(br[arm])[order], "o", ms=3.2, color=INK)
        ax2.set_ylim(-1, bmax * 1.22); ax2.set_xticks([]); ax2.tick_params(labelsize=5.5, length=1.5, pad=1)
        ax2.set_xlabel("slices (apex → base)", fontsize=6, labelpad=1)
        if j == len(arms) - 1:
            lk = dict(fontsize=6, frameon=False, ncol=2, handlelength=1.4, columnspacing=0.9, handletextpad=0.4,
                      loc="upper right", borderaxespad=0.1)
            if show_gt:
                ax.legend([l_rc, l_gt], ["recon.", "GT"], **lk)
            ax2.legend([b_ap, l_pr], ["GT shift", "predicted"], **lk)
        if j == 0:
            ax.set_ylabel("LV volume (mL)", fontsize=6, labelpad=1)
            ax2.set_ylabel("through-plane\nshift (mm)", fontsize=6, labelpad=1)
        else:
            ax.set_yticklabels([]); ax2.set_yticklabels([])
    fig.text(0.0, 1 - (y1 + h1 / 2) / Hh, "Heartbeat", fontsize=7.5, fontweight="bold", va="center",
             ha="center", rotation=90)
    fig.text(0.0, 1 - (y2 + h2 / 2) / Hh, "Breathing", fontsize=7.5, fontweight="bold", va="center",
             ha="center", rotation=90)
    if with_images:                                   # row 3: long-axis cut of the reconstruction at `frame`
        from scipy.ndimage import zoom
        s0 = np.asarray(nib.load(os.path.join(e15, "seg", "gt__t00.nii.gz")).dataobj)
        xs, ys, _ = np.nonzero(s0 == 1); cx, cy = int(xs.mean()), int(ys.mean()); bh = 50
        f_ = dz / 1.4
        vmax = None
        for j, (arm, _, _) in enumerate(arms):
            v = np.asarray(nib.load(os.path.join(e15, arm, f"recon_t{frame:02d}.nii.gz")).dataobj, np.float32)
            vmax = vmax or np.percentile(v[cx - bh:cx + bh, cy - bh:cy + bh], 99.5)
            cut = v[cx - 2:cx + 3, max(cy - bh, 0):cy + bh, :].mean(0).T
            a3 = ax_in(x0 + j * (cw + cg), y3, cw, h3)
            a3.imshow(zoom(cut, (f_, 1), order=0), cmap="gray", vmin=0, vmax=vmax, aspect="auto")
            a3.set_xticks([]); a3.set_yticks([])
            for sp in a3.spines.values():
                sp.set_visible(False)
        fig.text(0.0, 1 - (y3 + h3 / 2) / Hh, "long-axis", fontsize=8, fontweight="bold", va="center", rotation=90)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_B_final_{e15_dir.replace('e15_hrv48_', '')}.{ext}", dpi=300, bbox_inches="tight",
                    pad_inches=0.02)
    print("wrote", f"{OUT}/fig_B_final.png", {a: round(float(c.max() - c.min()), 1) for a, c in curves.items()},
          "gt", round(float(gt.max() - gt.min()), 1))


def e3_recs():
    return [r for r in (pickle.load(open(f, "rb")) for f in sorted(glob.glob(f"{D}/e3/s*.pkl"))) if r]


def panel_attn_heatmap(ax, recs):
    """Companion(heart queries)->reference attention mass per global layer x head, relative to uniform."""
    M = np.mean([attn_to_ref(r) for r in recs], 0)                    # (24,16)
    im = ax.imshow(M.T, aspect="auto", cmap="Blues", vmin=0, vmax=8, origin="lower")
    ax.set_xlabel("global-attention layer"); ax.set_ylabel("head")
    ax.set_xticks(range(0, 24, 3)); ax.set_yticks([0, 2, 10, 12, 14]); ax.tick_params(labelsize=6.5)
    for h in (2, 10, 12, 14):
        ax.add_patch(plt.Rectangle((2.5, h - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
    cb = plt.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("× uniform", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.set_title(f"f  Attention: companions → reference\n    (n={len(recs)}; orange = causal heads)", loc="left")


def panel_attn_maps(axs, recs):
    """One subject: companion query on the LV (red +), layer-3 attention over the reference slice."""
    r = next(r for r in recs if r.get("maps") and r["maps"].get("breath"))
    maps = r["maps"]["breath"]; imgs = r["images_breath"].astype(np.float32)
    keys = sorted({k[1:] for k in maps if k[0] == 3 and k[1] != 0})
    qs, qr, qc = keys[0]
    G = 37
    ax = axs[0]
    ax.imshow(imgs[qs], cmap="gray"); ax.plot((qc + 0.5) * 14 * 256 / 518, (qr + 0.5) * 14 * 256 / 518, "+", color="#e34948", ms=9, mew=1.8)
    ax.set_title("companion slice\n(query +)", fontsize=7); ax.axis("off")
    for ax, h in zip(axs[1:], (2, 12)):
        A_ = maps[(3, qs, qr, qc)][h].astype(np.float32)
        a = A_[0, 5:].reshape(G, G)
        ax.imshow(imgs[0], cmap="gray", extent=(0, G, G, 0), vmax=np.percentile(imgs[0], 99.5))
        ax.imshow(a, cmap="inferno", alpha=0.65, extent=(0, G, G, 0), vmin=0, vmax=a.max())
        ax.plot(qc + 0.5, qr + 0.5, "+", color="#1baf7a", ms=8, mew=1.5)
        ax.set_title(f"reference, L3 h{h}\n{100 * A_[0].sum():.0f}% of attention", fontsize=7); ax.axis("off")


def fig_attention():
    recs = e3_recs()
    fig = plt.figure(figsize=(7.2, 2.3))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.6, 0.25, 1, 1, 1], wspace=0.15)
    panel_attn_heatmap(fig.add_subplot(gs[0, 0]), recs)
    panel_attn_maps([fig.add_subplot(gs[0, i]) for i in (2, 3, 4)], recs)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_attention.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_attention.png")


def fig_main():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.7, wspace=0.55)
    panel_layers(fig.add_subplot(gs[0, 0]))
    panel_heads(fig.add_subplot(gs[0, 1]))
    panel_locality(fig.add_subplot(gs[0, 2]))
    g2 = gs[1, :].subgridspec(1, 2, wspace=0.28)
    panel_dissociation(fig.add_subplot(g2[0]), fig.add_subplot(g2[1]))
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_mechanism.{ext}", dpi=200, bbox_inches="tight")
    print("wrote", f"{OUT}/fig_mechanism.png")


if __name__ == "__main__":
    which = sys.argv[1:] or ["main", "attention", "breathing"]
    for w in which:
        {"main": fig_main, "attention": fig_attention, "breathing": fig_breathing, "lv": fig_lv, "real": fig_real, "features": fig_features, "pca": fig_pca, "paper": fig_paper, "simple": fig_paper_simple, "visual": fig_paper_visual, "appendix": fig_appendix, "attn": fig_attn_paper, "attn_simple": fig_attn_simple, "cuts": fig_cuts, "final": fig_final, "A": lambda: (fig_A(scale="log"), fig_A(scale="linear"), fig_A(scale="inferno")), "A_curve": fig_A_curve, "A_matrix": fig_A_matrix, "B": fig_B}[w]()
