"""Figures for the motion-correction toy-experiment report."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(REPO, "result", "limits_eval")
wc = json.load(open(os.path.join(D, "toy_warpceiling.json")))["summary"]
land = json.load(open(os.path.join(D, "toy_landscape.json")))["summary"]

# ── Fig 1: the corrected ladder — warp ceiling ≈ model ≪ oracle ──
fig, ax = plt.subplots(figsize=(8.6, 4.6))
warp_best = max(wc["rigid"], wc["lowrank_G16"], wc["lowrank_G32"], wc["free_lr5e-3"], wc["free_lr5e-3_TV"])
labels = ["identity\n(do-nothing)", "trained\nmodel", "WARP-ONLY\nCEILING\n(best optimized Δ)", "ORACLE\n(perfect placement\n= target-phase planes)"]
vals = [wc["identity_floor"], wc["model_ref"], warp_best, wc["oracle_ref"]]
cols = ["#9aa0a6", "#1a73e8", "#0b8043", "#e8710a"]
bars = ax.bar(labels, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}", ha="center", fontweight="bold", fontsize=12)
# annotate the two gaps
ax.annotate("", xy=(2, warp_best - 0.2), xytext=(1, wc["model_ref"] - 0.2),
            arrowprops=dict(arrowstyle="<->", color="#0b8043", lw=1.5))
ax.text(1.5, max(warp_best, wc["model_ref"]) + 1.2, f"motion-estimation\nheadroom ≈ {warp_best-wc['model_ref']:.1f} dB\n(tiny — model is AT the warp ceiling)",
        ha="center", color="#0b8043", fontsize=9, fontweight="bold")
ax.annotate("", xy=(3, wc["oracle_ref"] - 0.5), xytext=(2, warp_best + 0.3),
            arrowprops=dict(arrowstyle="<->", color="#c5221f", lw=2))
ax.text(2.5, 27.5, f"APPEARANCE WALL ≈ {wc['oracle_ref']-warp_best:.0f} dB\nwarp can't synthesize\ntarget-phase appearance\n(needs a learned decoder)",
        ha="center", color="#c5221f", fontsize=9.5, fontweight="bold")
ax.set_ylabel("MOTION PSNR (dB) — dynamic heart voxels", fontsize=11)
ax.set_title("The warp-only ceiling ≈ the trained model, ~14 dB below the oracle\n"
             "→ the bottleneck is the warp architecture (appearance), not motion estimation", fontsize=11)
ax.set_ylim(0, 40); ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(D, "fig_warpceiling.png"), bbox_inches="tight", dpi=120, facecolor="white"); plt.close(fig)
print("saved fig_warpceiling.png  (warp_best=%.2f, model=%.1f, oracle=%.1f)" % (warp_best, wc["model_ref"], wc["oracle_ref"]))

# ── Fig 2: warp-ceiling per Δ parameterization (rigid..free all ≈21) ──
fig, ax = plt.subplots(figsize=(7.5, 4.2))
names = ["rigid", "lowrank_G16", "lowrank_G32", "free_lr5e-3", "free_lr5e-3_TV"]
disp = ["rigid\n(3/slot)", "low-rank\nG16", "low-rank\nG32", "free\nper-pixel", "free+TV"]
ys = [wc[n] for n in names]
ax.bar(disp, ys, color="#0b8043", width=0.6)
for i, v in enumerate(ys): ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontweight="bold")
ax.axhline(wc["model_ref"], color="#1a73e8", ls="--", label="trained model 20.6")
ax.axhline(wc["identity_floor"], color="#9aa0a6", ls=":", label=f"identity {wc['identity_floor']:.1f}")
ax.axhline(wc["oracle_ref"], color="#e8710a", ls="--", label="oracle 35.0")
ax.set_ylabel("warp ceiling — motion PSNR (dB)"); ax.set_ylim(0, 38)
ax.set_title("Every Δ parameterization tops out at ~19-21 dB (≈ model)\n— the ceiling is the warp, not the Δ representation")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(D, "fig_warp_params.png"), bbox_inches="tight", dpi=120, facecolor="white"); plt.close(fig)
print("saved fig_warp_params.png")

# ── Fig 3: E0 conditioning summary (z vs x gradient; covdiv vs nocovdiv vs invwarp) ──
fig, ax = plt.subplots(figsize=(7.5, 4.2))
modes = ["z_splat", "z_splat_nocovdiv", "z_invwarp", "x_splat"]
disp = ["z\nsplat\n(current)", "z\nno-cov-div", "z\ninverse-warp", "x\nsplat\n(in-plane)"]
rises = [land[m]["rise_per_mm"] * 1000 for m in modes]
cols = ["#c5221f", "#f9ab00", "#e8710a", "#1a73e8"]
ax.bar(disp, rises, color=cols, width=0.6)
for i, v in enumerate(rises): ax.text(i, v + 0.1, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("loss gradient strength\n(L1 rise per mm × 1000)")
ax.set_title("E0: through-plane (z) gradient is ~2× weaker than in-plane (x)\ncoverage-division flattens it; inverse-warp / no-cov-div restore it")
ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(D, "fig_e0_bars.png"), bbox_inches="tight", dpi=120, facecolor="white"); plt.close(fig)
print("saved fig_e0_bars.png")
