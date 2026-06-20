"""Build figures for the limitations/improvements report from the saved JSONs."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(REPO, "result", "limits_eval")
dec = json.load(open(os.path.join(D, "decomposition.json")))
imp = json.load(open(os.path.join(D, "improvements.json")))
ks = json.load(open(os.path.join(D, "kspace_singleshot.json")))

C = dict(floor="#9aa0a6", model="#1a73e8", refined="#0b8043", oracle="#e8710a",
         ceil="#c5221f", good="#188038")


def save(fig, name):
    fig.savefig(os.path.join(D, name), bbox_inches="tight", dpi=115, facecolor="white")
    plt.close(fig)
    print("saved", name)


# ── Fig 1: motion-PSNR ladder (the headline) ────────────────────────────────
fig, ax = plt.subplots(figsize=(8.2, 4.2))
names = ["identity", "model_canon", "model_refined", "oracle_perfect"]
labels = ["identity\n(do-nothing)", "model\nV_canon", "model\nV_refined", "ORACLE\n(perfect placement)"]
vals = [dec[n]["psnr_motion"] for n in names]
cols = [C["floor"], C["model"], C["refined"], C["oracle"]]
bars = ax.bar(labels, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
ax.annotate("", xy=(3, 34.9), xytext=(2, 20.6),
            arrowprops=dict(arrowstyle="<->", color=C["ceil"], lw=2))
ax.text(2.5, 28.5, f"~{dec['oracle_perfect']['psnr_motion']-dec['model_refined']['psnr_motion']:.0f} dB\nMOTION headroom\n(geometry/placement)",
        ha="center", color=C["ceil"], fontsize=10, fontweight="bold")
ax.set_ylabel("MOTION PSNR (dB) — dynamic heart voxels", fontsize=11)
ax.set_title(f"Limitation #1: the gap is MOTION ESTIMATION, not the renderer  (breathing val, n={dec['n']})",
             fontsize=11.5)
ax.set_ylim(0, 40); ax.grid(axis="y", alpha=0.3)
save(fig, "fig_ladder.png")

# ── Fig 2: sharpness decomposition (resize vs kernel) ────────────────────────
fig, ax = plt.subplots(figsize=(8.6, 4.2))
sn = ["model_refined", "oracle_perfect", "oracle_super2x", "oracle_nearest", "oracle_native256"]
sl = ["model\nV_refined", "oracle\n518→256 splat\n(current)", "oracle\n512 grid\n(super-2×)",
      "oracle\nnearest\n(no tent)", "oracle\nNATIVE-256\n(no resize)"]
sv = [dec[n]["sharp_rel"] for n in sn]
cols = [C["refined"], C["oracle"], "#f9ab00", "#a142f4", C["good"]]
bars = ax.bar(sl, sv, color=cols, width=0.66)
for b, v in zip(bars, sv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10.5, fontweight="bold")
ax.axhline(1.0, color="k", ls="--", lw=1); ax.text(4.4, 1.005, "GT", fontsize=9)
ax.set_ylabel("sharpness / GT  (in-plane gradient energy)", fontsize=11)
ax.set_title("Limitation #2: the sharpness ceiling is the 256→518→256 RESIZE, not the trilinear kernel",
             fontsize=11)
ax.set_ylim(0, 1.08); ax.grid(axis="y", alpha=0.3)
save(fig, "fig_sharpness.png")

# ── Fig 3: improvements (native splat + multi-draw) ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ns = imp["native_splat"]
ax = axes[0]
g = ["canon_518", "canon_native"]
gl = ["model splat\n(518→256, current)", "model splat\nNATIVE-256"]
sv = [ns[k]["sharp_rel"] for k in g]
bars = ax.bar(gl, sv, color=[C["oracle"], C["good"]], width=0.5)
for b, v in zip(bars, sv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylabel("sharpness / GT"); ax.set_ylim(0, 0.95); ax.grid(axis="y", alpha=0.3)
ax.set_title(f"Improvement A: native-res splat\n(model's own geometry, no retrain): "
             f"{sv[0]:.3f}→{sv[1]:.3f}", fontsize=10)

ax = axes[1]
Kgrid = sorted({int(k.split("K")[1]) for k in imp["multidraw"] if k.startswith("refined_K")})
mv = [imp["multidraw"][f"refined_K{k}"]["psnr_motion"] for k in Kgrid]
sv = [imp["multidraw"][f"refined_K{k}"]["sharp_rel"] for k in Kgrid]
ax.plot(Kgrid, mv, "o-", color=C["refined"], lw=2, label="motion PSNR")
for k, v in zip(Kgrid, mv):
    ax.text(k, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
ax.set_xlabel("K (independent scattered draws, averaged)"); ax.set_ylabel("MOTION PSNR (dB)", color=C["refined"])
ax.set_title(f"Improvement B: multi-draw ensemble\n+{mv[-1]-mv[0]:.2f} dB motion at K={Kgrid[-1]} (free, test-time)",
             fontsize=10)
ax2 = ax.twinx()
ax2.plot(Kgrid, sv, "s--", color="#a142f4", lw=1.5, label="sharpness/GT")
ax2.set_ylabel("sharpness / GT", color="#a142f4")
ax.grid(alpha=0.3)
save(fig, "fig_improvements.png")

# ── Fig 4: domain gap (k-space single-shot) ──────────────────────────────────
fig, ax = plt.subplots(figsize=(6.2, 4.2))
m = ks.get("model_R8", {})
g = ["floor\n(identity)", "model on\nCLEAN input", "model on\nSINGLE-SHOT\n(R=8 aliased)"]
v = [dec["identity"]["psnr_motion"], m.get("motion_clean", float("nan")), m.get("motion_aliased", float("nan"))]
bars = ax.bar(g, v, color=[C["floor"], C["refined"], C["ceil"]], width=0.6)
for b, x in zip(bars, v):
    ax.text(b.get_x() + b.get_width() / 2, x + 0.2, f"{x:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("MOTION PSNR (dB)")
ax.set_title(f"Limitation #4: domain gap — single-shot aliasing costs "
             f"{m.get('drop', float('nan')):.1f} dB\n(model never trained on it; back near the floor)", fontsize=10)
ax.set_ylim(0, 24); ax.grid(axis="y", alpha=0.3)
save(fig, "fig_domaingap.png")

print("\nAll figures written to", D)
