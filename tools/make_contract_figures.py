"""Figures for the motion-PSNR contract-levers toy-experiment."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(__file__), "..", "result", "limits_eval")
L = json.load(open(os.path.join(D, "toy_contract_levers.json")))["summary"]
P = json.load(open(os.path.join(D, "toy_proximity_ceiling.json")))
Ks = [0, 1, 2, 4, L["S"]]
C3 = [L["A"][str(k)]["C3_model"] for k in Ks]
C4 = [L["A"][str(k)]["C4_heldout"] for k in Ks]
C2 = [L["A"][str(k)]["C2_id"] for k in Ks]

# ── Fig 1: Lever A — the observation leak (full rises, held-out falls) ──
fig, ax = plt.subplots(figsize=(8.8, 5.0))
x = list(range(len(Ks)))
ax.plot(x, C3, "-o", color="#1a73e8", lw=2.4, ms=8, label="C3  full-volume motion PSNR (LEAKY: scores planes where target was injected)")
ax.plot(x, C4, "-o", color="#c5221f", lw=2.4, ms=8, label="C4  held-out-z motion PSNR (leak-free: planes with NO target slice)")
ax.plot(x, C2, "--s", color="#9aa0a6", lw=1.6, ms=6, label="C2  identity-Δ splat (no model)")
for xi, v in zip(x, C3): ax.text(xi, v + 0.18, f"{v:.1f}", ha="center", color="#1a73e8", fontsize=9, fontweight="bold")
for xi, v in zip(x, C4): ax.text(xi, v - 0.5, f"{v:.1f}", ha="center", color="#c5221f", fontsize=9, fontweight="bold")
ax.annotate("more target-phase frames →\nfull PSNR rises +2 dB …", xy=(3, C3[3]), xytext=(1.2, 21.2),
            color="#1a73e8", fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#1a73e8"))
ax.annotate("… but it's an OBSERVATION LEAK:\nunmeasured planes get WORSE\n(model can't propagate)", xy=(4, C4[4]), xytext=(2.1, 13.6),
            color="#c5221f", fontsize=9.5, arrowprops=dict(arrowstyle="->", color="#c5221f"))
ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in Ks])
ax.set_xlabel("number of input slices placed AT the target cardiac phase (distinct z, fixed budget S=8)")
ax.set_ylabel("motion PSNR (dB)"); ax.set_ylim(12, 23)
ax.set_title("Lever A — adding target-phase frames only fills the planes you measure;\nthe model does NOT propagate target appearance to unobserved planes", fontsize=11)
ax.grid(alpha=0.3); ax.legend(fontsize=8.2, loc="lower left")
fig.savefig(os.path.join(D, "fig_lever_a_leak.png"), dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_lever_a_leak.png")

# ── Fig 2: Lever B + proximity ceiling — small, ceiling-capped, no retrain headroom ──
fig, ax = plt.subplots(figsize=(8.8, 5.0))
labels = ["random\n(current contract)", "near ±2", "near ±1\n(tightest)"]
model = [L["B"]["random"]["C3_model"], L["B"]["near2"]["C3_model"], L["B"]["near1"]["C3_model"]]
w = 0.36; xi = np.arange(3)
b1 = ax.bar(xi - w/2, model, w, color="#1a73e8", label="trained model (realizable now)")
# warp ceilings only measured for random & near1
ceil = [P["random"]["warp_ceiling"], None, P["near1"]["warp_ceiling"]]
xc = [0, 2]; yc = [P["random"]["warp_ceiling"], P["near1"]["warp_ceiling"]]
b2 = ax.bar(np.array(xc) + w/2, yc, w, color="#f9ab00", label="warp CEILING (best possible warp)")
for b in b1: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.06, f"{b.get_height():.1f}", ha="center", fontsize=9, fontweight="bold")
for x0, y in zip(xc, yc): ax.text(x0+w/2, y+0.06, f"{y:.1f}", ha="center", fontsize=9, fontweight="bold", color="#b06000")
ax.annotate(f"proximity buys only +{model[2]-model[0]:.1f} dB (model)\n= +{yc[1]-yc[0]:.1f} dB at the ceiling\n→ model ALREADY at ceiling, no retrain headroom",
            xy=(2, model[2]), xytext=(0.15, 21.0), fontsize=9.3,
            arrowprops=dict(arrowstyle="->", color="#444"))
ax.axhline(L["A"]["0"]["C3_model"], color="#9aa0a6", ls=":", lw=1.3)
ax.text(2.35, L["A"]["0"]["C3_model"]+0.06, "current baseline", color="#666", fontsize=8)
ax.set_xticks(xi); ax.set_xticklabels(labels)
ax.set_xlabel("input companion phases (fixed budget S=8, NO exact target frame — leak-free)")
ax.set_ylabel("motion PSNR (dB)"); ax.set_ylim(15, 22.5)
ax.set_title("Lever B — proximity sampling is real but small & ceiling-capped:\n+0.8 dB, and the model already extracts it (no retrain headroom)", fontsize=11)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=8.6, loc="upper left")
fig.savefig(os.path.join(D, "fig_lever_b_proximity.png"), dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_lever_b_proximity.png")

# ── Fig 3: verdict ladder — where every lever lands vs the wall and the oracle ──
fig, ax = plt.subplots(figsize=(9.4, 5.0))
items = [
    ("population template\n(avg heart @ t)", 14.4, "#9aa0a6"),
    ("current contract\n(random, S=8)", L["A"]["0"]["C3_model"], "#5e97f6"),
    ("+ proximity (near±1)\nMODEL", L["B"]["near1"]["C3_model"], "#1a73e8"),
    ("+ proximity\nWARP CEILING", P["near1"]["warp_ceiling"], "#f9ab00"),
    ("renderer/decoder\n(feature-splat)", L["A"]["0"]["C3_model"]+0.03, "#a142f4"),
    ("warp ceiling\n(prior, breathing)", 21.0, "#e8710a"),
    ("subject temporal\ninterp (DENSE t±1)", 28.1, "#0b8043"),
    ("ORACLE\n(perfect placement)", 35.0, "#137333"),
]
names = [i[0] for i in items]; vals = [i[1] for i in items]; cols = [i[2] for i in items]
bars = ax.barh(range(len(items)), vals, color=cols)
for b, v in zip(bars, vals): ax.text(v+0.3, b.get_y()+b.get_height()/2, f"{v:.1f}", va="center", fontsize=9, fontweight="bold")
ax.set_yticks(range(len(items))); ax.set_yticklabels(names, fontsize=8.6); ax.invert_yaxis()
ax.axvspan(L["A"]["0"]["C3_model"], 35.0, color="#fdecea", alpha=0.5, zorder=0)
ax.set_xlabel("motion PSNR (dB)"); ax.set_xlim(12, 38)
ax.set_title("Every model/renderer/sampling lever lands in a ~1 dB band near the wall;\nthe big dB lives only in DIRECTLY OBSERVING more of the target phase", fontsize=10.5)
ax.grid(axis="x", alpha=0.3)
fig.savefig(os.path.join(D, "fig_verdict_ladder.png"), dpi=120, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("saved fig_verdict_ladder.png")
