"""Figures for the reference-slice / amplitude-vs-motion-PSNR toy-experiment."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(__file__), "..", "result", "limits_eval")
A = json.load(open(os.path.join(D, "toy_amplitude_propagation.json")))
S = A["summary"]; recs = A["records"]
# EF slopes from toy_contraction.py (scheme comparison, just re-run)
EF = {"C0\n(current:\ntarget_t index)": 0.26, "Ccov\n(index + 1\ntarget slot)": 0.36, "B\n(reference\nslice)": 0.73}

# ── Fig 1: two errors, two fates ──
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.0))
# Panel A: EF slope (reference FIXES amplitude)
labels = list(EF); vals = list(EF.values())
cols = ["#9aa0a6", "#5e97f6", "#0b8043"]
b = ax[0].bar(labels, vals, color=cols, width=0.6)
for bb, v in zip(b, vals): ax[0].text(bb.get_x()+bb.get_width()/2, v+0.02, f"{v:.2f}", ha="center", fontweight="bold")
ax[0].axhline(1.0, color="#137333", ls=":", lw=1.4); ax[0].text(0.0, 1.02, "perfect (recovers per-patient EF)", color="#137333", fontsize=8.5)
ax[0].axhline(0.0, color="#c5221f", ls=":", lw=1.4); ax[0].text(0.0, 0.03, "flat (regress-to-cohort-mean)", color="#c5221f", fontsize=8.5)
ax[0].set_ylabel("pred-EF vs true-EF slope"); ax[0].set_ylim(-0.05, 1.1)
ax[0].set_title("✓ Reference slice FIXES per-patient\ncontraction AMPLITUDE (EF)", fontsize=12)
ax[0].grid(axis="y", alpha=0.3)
# Panel B: held-out motion-residual PSNR (reference does NOT fix it)
z = S["zrand"]
labels2 = ["baseline\n(the model)", "+1-plane α\n(realistic ref)", "+global α\n(oracle ceiling)", "+per-plane α\n(regional ceiling)"]
vals2 = [z["base"], z["ref"], z["glob"], z["ppl"]]
cols2 = ["#1a73e8", "#c5221f", "#f9ab00", "#e8710a"]
b2 = ax[1].bar(labels2, vals2, color=cols2, width=0.62)
for bb, v in zip(b2, vals2):
    d = v - z["base"]
    ax[1].text(bb.get_x()+bb.get_width()/2, v+0.03, f"{v:.2f}\n({d:+.2f})", ha="center", fontweight="bold", fontsize=9)
ax[1].set_ylabel("held-out MOTION PSNR (residual, dB)"); ax[1].set_ylim(19.5, 21.2)
ax[1].set_title("✗ …but does NOT fix per-voxel held-out\nMOTION PSNR (even the oracle: +0.25 dB)", fontsize=12)
ax[1].grid(axis="y", alpha=0.3)
fig.suptitle("A target-phase REFERENCE slice fixes the contraction AMPLITUDE the model gets wrong — "
             "but the held-out per-voxel motion error is APPEARANCE PATTERN, not amplitude, so PSNR barely moves",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_amp_two_fates.png"), dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_amp_two_fates.png")

# ── Fig 2: mechanism — why the scalar can't help (low cosine), per phase ──
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
# Panel A: premise — cosine & residual
ax[0].bar(["per-plane cosine\n(M_pred vs M_gt)", "held-out residual\nafter oracle-α"], [S["cos_mean"], S["zrand"]["resid"]],
          color=["#a142f4", "#c5221f"], width=0.5)
ax[0].text(0, S["cos_mean"]+0.02, f"{S['cos_mean']:.2f}", ha="center", fontweight="bold")
ax[0].text(1, S["zrand"]["resid"]+0.02, f"{S['zrand']['resid']:.2f}", ha="center", fontweight="bold")
ax[0].set_ylim(0, 1.05); ax[0].set_ylabel("fraction")
ax[0].annotate("pattern only ~44% aligned\n⇒ a magnitude scalar can't fix it", xy=(0, S["cos_mean"]), xytext=(0.15, 0.72),
               fontsize=9, arrowprops=dict(arrowstyle="->", color="#7a1fa2"))
ax[0].annotate("98% of the held-out error\nremains after PERFECT amplitude\n⇒ only ~2% is amplitude", xy=(1, S["zrand"]["resid"]),
               xytext=(0.35, 0.40), fontsize=9, arrowprops=dict(arrowstyle="->", color="#c5221f"))
ax[0].set_title("The held-out error is appearance PATTERN, not amplitude", fontsize=11)
ax[0].grid(axis="y", alpha=0.3)
# Panel B: per-phase base vs 1-plane vs oracle
ts = sorted(set(r["t"] for r in recs))
base = [np.mean([r["zrand"]["base"] for r in recs if r["t"] == t]) for t in ts]
ref = [np.mean([r["zrand"]["ref"] for r in recs if r["t"] == t]) for t in ts]
glob = [np.mean([r["zrand"]["glob"] for r in recs if r["t"] == t]) for t in ts]
x = np.arange(len(ts)); w = 0.26
ax[1].bar(x-w, base, w, label="baseline (model)", color="#1a73e8")
ax[1].bar(x, ref, w, label="+1-plane α (realistic ref)", color="#c5221f")
ax[1].bar(x+w, glob, w, label="+global α (oracle)", color="#f9ab00")
ax[1].set_xticks(x); ax[1].set_xticklabels([f"t={t}" for t in ts])
ax[1].set_ylim(18, 22.5); ax[1].set_ylabel("held-out motion-residual PSNR (dB)")
ax[1].set_xlabel("target cardiac phase")
ax[1].set_title("Every phase: 1-plane reference HURTS, oracle ≈ flat\n(no hidden peak-systole win)", fontsize=11)
ax[1].legend(fontsize=8.5, loc="upper center"); ax[1].grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(D, "fig_amp_mechanism.png"), dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_amp_mechanism.png")
