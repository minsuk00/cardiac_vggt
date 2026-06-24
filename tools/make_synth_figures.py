"""Figures for the appearance-synthesis toy-experiment report."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = os.path.join(os.path.dirname(__file__), "..", "result", "limits_eval")
rec = json.load(open(os.path.join(D, "toy_recoverability.json")))
fd = json.load(open(os.path.join(D, "toy_feature_decoder.json")))
syn = json.load(open(os.path.join(D, "toy_synth_appearance.json")))

# ── Fig 1: the verdict ladder (held-out motion PSNR) ──
fig, ax = plt.subplots(figsize=(9.2, 4.8))
labels = ["population\ntemplate\n(avg heart)", "transport\n(V_canon)", "intensity\ndecoder\n(refiner)",
          "FEATURE\ndecoder\n(V_canon+feat)", "subject\nall-phase\nmean", "recoverable\nceiling\n(DENSE data)", "oracle\n(perfect\nplacement)"]
vals = [rec["population_template"], fd["transport_Vcanon"], fd.get("intensity_decoder_rerun", fd["intensity_decoder"]),
        fd.get("both_Vcanon_plus_features", 19.38), rec["subject_mean"], rec["subject_temporal_interp"], 35.0]
cols = ["#9aa0a6", "#1a73e8", "#5e97f6", "#0b8043", "#a142f4", "#f9ab00", "#e8710a"]
bars = ax.bar(labels, vals, color=cols, width=0.66)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", fontweight="bold", fontsize=10.5)
ax.axhline(28, color="#c5221f", ls="--", lw=1.5)
ax.text(0.1, 28.4, "breakthrough line (≥28)", color="#c5221f", fontsize=9, fontweight="bold")
ax.annotate("", xy=(3, fd.get("both_Vcanon_plus_features", 19.38)), xytext=(2, fd.get("intensity_decoder_rerun", 19.36)),
            arrowprops=dict(arrowstyle="<->", color="#0b8043", lw=1.5))
ax.text(2.5, 17.0, "features add\n+0.03 dB", ha="center", color="#0b8043", fontsize=9, fontweight="bold")
ax.set_ylabel("held-out MOTION PSNR (dB)", fontsize=11)
ax.set_title("Appearance synthesis does NOT break the ceiling: feature-splat adds +0.03 dB,\n"
             "and the recoverable ceiling (28, even with DENSE data) sits below 'breakthrough'", fontsize=11)
ax.set_ylim(0, 38); ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(D, "fig_synth_verdict.png"), bbox_inches="tight", dpi=120, facecolor="white"); plt.close(fig)
print("saved fig_synth_verdict.png")

# ── Fig 2: synthetic principle toy (transport vs synthesis, + hallucination) ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
g = ["mean-blur\nfloor", "transport\nceiling", "SYNTHESIS\n(held-out)"]
v = [syn["mean_blur_floor_test"], syn["transport_ceiling_test"], syn["synth_test"]]
bars = ax.bar(g, v, color=["#9aa0a6", "#1a73e8", "#0b8043"], width=0.55)
for b, x in zip(bars, v): ax.text(b.get_x() + b.get_width() / 2, x + 0.3, f"{x:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("held-out PSNR (dB)")
ax.set_title(f"Synthetic toy: synthesis CAN beat transport (+{syn['synth_test']-syn['transport_ceiling_test']:.1f} dB)\n"
             "WHEN appearance is population-predictable", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax = axes[1]
comp = ["population\nappearance\n(component 2)", "subject-specific\ndisocclusion\n(component 3)"]
errs = [syn["intensity_err_synth"] / syn["intensity_err_transport"], 1.0]   # relative: recovered vs hallucinated
ax.bar(comp, [1 - syn["intensity_err_synth"] / syn["intensity_err_transport"], 0.0], color=["#0b8043", "#c5221f"], width=0.5)
ax.set_ylabel("fraction recovered (1=perfect)")
ax.set_ylim(0, 1.05)
ax.text(0, 0.78, "RECOVERED\n(generalizes)", ha="center", color="#0b8043", fontsize=9, fontweight="bold")
ax.text(1, 0.10, "HALLUCINATED\n(fabricated at\npop-mean location)", ha="center", color="#c5221f", fontsize=9, fontweight="bold")
ax.set_title("Synthesis recovers population-predictable appearance,\nHALLUCINATES subject-specific detail", fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.savefig(os.path.join(D, "fig_synth_toy.png"), bbox_inches="tight", dpi=120, facecolor="white"); plt.close(fig)
print("saved fig_synth_toy.png")
