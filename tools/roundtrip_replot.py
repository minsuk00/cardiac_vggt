#!/usr/bin/env python
"""Re-plot focused, READABLE panels from the cached roundtrip_diagnostic .npz arrays
(no model reload). Two figures:
  (1) reference slot (slot 0) across all 3 models, large — the "is the observed reference
      preserved?" view (input | V_canon@home | V_canon@pred | |I-home| | |I-pred|).
  (2) for each model, a few DISTINCT-z slots, large.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "result/roundtrip_miitt"
MODELS = ["control0", "gather05", "s20contz"]
COLS = ["input I", "V_canon @ home (Δ=0)", "V_canon @ pred (+Δ)",
        "|I - home|  motion@plane", "|I - pred|  round-trip"]


def load(name):
    d = np.load(os.path.join(OUT, f"{name}_arrays.npz"))
    return dict(I=d["I"].astype(np.float32), Rh=d["Rh"].astype(np.float32),
                Rp=d["Rp"].astype(np.float32), z=d["z"], frame=d["frame"],
                mae=float(d["mae"]), mae_h=float(d["mae_h"]))


def row(a, I, Rh, Rp, s, ylabel):
    errh, errp = np.abs(I[s] - Rh[s]), np.abs(I[s] - Rp[s])
    a[0].imshow(I[s], cmap="gray", vmin=0, vmax=1); a[0].set_ylabel(ylabel, fontsize=9)
    a[1].imshow(Rh[s], cmap="gray", vmin=0, vmax=1)
    a[2].imshow(Rp[s], cmap="gray", vmin=0, vmax=1)
    a[3].imshow(errh, cmap="magma", vmin=0, vmax=0.3)
    a[4].imshow(errp, cmap="magma", vmin=0, vmax=0.3)
    for aa in a:
        aa.set_xticks([]); aa.set_yticks([])


# ── (1) reference slot across models ──────────────────────────────────────────
data = {m: load(m) for m in MODELS}
fig, ax = plt.subplots(len(MODELS), 5, figsize=(15, 3.1 * len(MODELS)), dpi=110)
ax = np.atleast_2d(ax)
for r, m in enumerate(MODELS):
    d = data[m]
    row(ax[r], d["I"], d["Rh"], d["Rp"], 0,
        f"{m}\nREF z={d['z'][0]:.1f} f={int(d['frame'][0])}\nmotion@pl={d['mae_h']:.3f}")
    if r == 0:
        for c in range(5):
            ax[r][c].set_title(COLS[c], fontsize=10)
fig.suptitle("MIITT Volunteer1 — REFERENCE slot (slot 0), all 3 models", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.98])
p = os.path.join(OUT, "REF_slot_compare.png"); fig.savefig(p); plt.close(fig)
print("wrote", p)

# ── (2) per-model, a few distinct-z slots ─────────────────────────────────────
for m in MODELS:
    d = data[m]
    z = d["z"]
    seen, show = set(), []
    for i, zz in enumerate(z):
        k = round(float(zz))
        if k not in seen:
            seen.add(k); show.append(i)
        if len(show) >= 6:
            break
    fig, ax = plt.subplots(len(show), 5, figsize=(15, 3.0 * len(show)), dpi=100)
    ax = np.atleast_2d(ax)
    for r, s in enumerate(show):
        row(ax[r], d["I"], d["Rh"], d["Rp"], s,
            f"slot{s}\nz={z[s]:.1f} f={int(d['frame'][s])}" + ("\nREF" if s == 0 else ""))
        if r == 0:
            for c in range(5):
                ax[r][c].set_title(COLS[c], fontsize=10)
    fig.suptitle(f"MIITT Volunteer1 — {m}, distinct-z slots", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    p = os.path.join(OUT, f"{m}_distinctz.png"); fig.savefig(p); plt.close(fig)
    print("wrote", p)
