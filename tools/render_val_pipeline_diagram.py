"""Diagram: how validation samples are drawn, and why V_gt is resampled."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import patches as mpatches

OUT = "/home/minsukc/vggt/result/nprlshyj_val_pipeline_diagram.png"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

fig = plt.figure(figsize=(15, 11), dpi=110)

# ───────────── Panel A: Val sampling ───────────────────────────────────────
axA = fig.add_axes([0.04, 0.55, 0.94, 0.40]); axA.set_xlim(0, 100); axA.set_ylim(0, 50); axA.axis("off")
axA.set_title("Validation sampling — deterministic but not exhaustive", fontsize=13, loc="left", pad=6)

def box(ax, x, y, w, h, text, fc="#eef", ec="#446", fontsize=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize)

def arrow(ax, x1, y1, x2, y2, text=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12, lw=1.4, color="#333"))
    if text:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.5, text, ha="center", va="bottom", fontsize=8, color="#333")

box(axA, 1, 36, 24, 10,
    "Val split: 30 subjects\n(random_8_1_1.txt)\nshuffle=False", fc="#fdf3e0")
box(axA, 1, 22, 24, 10,
    "limit_val_batches = 200\n→ seq_index = 0,1,…,199", fc="#fdf3e0")
box(axA, 1, 8, 24, 10,
    "subject = subjects[ seq_index % 30 ]\n→ each subject re-visited\n≈ 200/30 ≈ 6–7 times", fc="#fdf3e0")
arrow(axA, 13, 36, 13, 32)
arrow(axA, 13, 22, 13, 18)

box(axA, 32, 36, 32, 10,
    "rng = random.Random(\n  seq_index*1000 + hash(sub_dir)%100000)", fc="#e7f4ea")
box(axA, 32, 22, 32, 10,
    "t_sequence = [0] + shuffle(1..T-1)[:S-1]\nz_sequence = shuffle(0..Z-1)[:S]", fc="#e7f4ea")
box(axA, 32, 8, 32, 10,
    "Same seed every val epoch ⇒\nsame (t,z) picks for that seq_index.\nVal metrics reproducible run-to-run.",
    fc="#e7f4ea")
arrow(axA, 48, 36, 48, 32)
arrow(axA, 48, 22, 48, 18)

box(axA, 71, 36, 27, 10,
    "Result per val step:\nS = min(12, T, Z) random slots\nslot 0 forced to t = 0", fc="#e0eaf7")
box(axA, 71, 22, 27, 10,
    "Same 200 (subj, t-set, z-set) triples\nfor every epoch's val pass.",
    fc="#e0eaf7")
box(axA, 71, 8, 27, 10,
    "Not exhaustive: only a FROZEN\nrandom subset of (t,z) per subject.\nUnsampled (t,z) pairs are never seen.",
    fc="#e0eaf7")
arrow(axA, 84, 36, 84, 32)
arrow(axA, 84, 22, 84, 18)

# ───────────── Panel B: Why V_gt is resampled ───────────────────────────────
axB = fig.add_axes([0.04, 0.04, 0.94, 0.46]); axB.set_xlim(0, 100); axB.set_ylim(0, 50); axB.axis("off")
axB.set_title("Why V_gt is resampled onto the canonical 12×256×256 grid", fontsize=13, loc="left", pad=6)

# Native subject volume
def draw_cuboid(ax, cx, cy, dx, dy, dz_offset, label, color="#88c"):
    ax.add_patch(mpatches.Rectangle((cx, cy), dx, dy, fc=color, ec="#333", lw=1.2, alpha=0.6))
    # depth hint
    ax.add_patch(mpatches.Polygon([(cx, cy+dy), (cx+dz_offset, cy+dy+dz_offset*0.5),
                                    (cx+dx+dz_offset, cy+dy+dz_offset*0.5), (cx+dx, cy+dy)],
                                  closed=True, fc=color, ec="#333", alpha=0.4))
    ax.add_patch(mpatches.Polygon([(cx+dx, cy), (cx+dx+dz_offset, cy+dz_offset*0.5),
                                    (cx+dx+dz_offset, cy+dy+dz_offset*0.5), (cx+dx, cy+dy)],
                                  closed=True, fc=color, ec="#333", alpha=0.4))
    ax.text(cx+dx/2, cy-1.4, label, ha="center", va="top", fontsize=8)

# Subject A native
draw_cuboid(axB, 3, 28, 9, 12, 4, "Subject A native\nW=256, H=204, Z=11\nspacing (1.34, 1.34, 8.0) mm\nphysical FOV ≈ 343×273×88 mm",
            color="#a4c0e0")
# Subject B native
draw_cuboid(axB, 3, 4, 9, 14, 3, "Subject B native\nW=256, H=246, Z=8\nspacing (1.34, 1.34, 8.0) mm\nphysical FOV ≈ 343×329×64 mm",
            color="#e0bfa4")

axB.annotate("scipy.ndimage.map_coordinates\n(trilinear resample)", xy=(28, 32), xytext=(20, 32),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"),
             fontsize=8, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7d6", ec="#aa7"))
axB.annotate("", xy=(28, 12), xytext=(20, 12),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))

# Canonical cube
def draw_cube(ax, cx, cy, s, label, color="#9cc"):
    dz = 5
    ax.add_patch(mpatches.Rectangle((cx, cy), s, s, fc=color, ec="#333", lw=1.2, alpha=0.7))
    ax.add_patch(mpatches.Polygon([(cx, cy+s), (cx+dz, cy+s+dz*0.5),
                                    (cx+s+dz, cy+s+dz*0.5), (cx+s, cy+s)],
                                  closed=True, fc=color, ec="#333", alpha=0.5))
    ax.add_patch(mpatches.Polygon([(cx+s, cy), (cx+s+dz, cy+dz*0.5),
                                    (cx+s+dz, cy+s+dz*0.5), (cx+s, cy+s)],
                                  closed=True, fc=color, ec="#333", alpha=0.5))
    ax.text(cx+s/2, cy-1.4, label, ha="center", va="top", fontsize=8)

draw_cube(axB, 30, 28, 13, "Canonical grid (12×256×256)\nin per-axis normalized [-1,1]\nshared by V_canon and V_gt", color="#bce4d0")
draw_cube(axB, 30, 4,  13, "Canonical grid (12×256×256)\nin per-axis normalized [-1,1]\n(SAME shape — different mm scale)", color="#bce4d0")

axB.text(28.5, 21, "Each subject gets its own\nper-axis normalization\nso the FOV fills [-1,1]³",
         ha="left", va="center", fontsize=8, style="italic", color="#555")

# Compare-with section
axB.annotate("splat_to_volume", xy=(70, 32), xytext=(56, 32),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"),
             fontsize=8, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7d6", ec="#aa7"))
axB.annotate("splat_to_volume", xy=(70, 12), xytext=(56, 12),
             arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"),
             fontsize=8, va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff7d6", ec="#aa7"))

box(axB, 70, 26, 28, 16,
    "V_canon (12,256,256)\n  built by splatting model's\n  predicted (pos, intensity)\n\nV_gt (12,256,256)\n  built by RESAMPLING the on-disk\n  phase-0 NIfTI into same grid",
    fc="#fff", ec="#666", fontsize=8.5)
box(axB, 70, 2,  28, 16,
    "Loss = |V_canon − V_gt|·𝟙[V_gt>0]\n\nWhy resample V_gt?\n• every subject must reduce to\n  the SAME tensor shape\n• loss is voxelwise, needs same frame\n• splatting is for sparse predictions\n  V_gt is already dense → just resample",
    fc="#fff", ec="#666", fontsize=8.5)

plt.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
