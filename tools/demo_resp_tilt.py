"""One-off demo: visualize the respiratory direction-jitter (tilt) logic.

Uses the SAME functions training applies (`_rotate_disp`, `reslice_volume_vec`) so
the picture is faithful. Produces result/respiratory_tilt_demo.png with:

  (A) 3D cone of sampled displacement directions around the base SI(+AP) vector,
      annotated with the literature SAX-vs-true-SI tilt range (20-45 deg).
  (B) coronal reslices: pure-D (no tilt) vs several tilted draws at the SAME |d|,
      so the through-plane content shift per draw is visible.

Run: PYTHONPATH=training:. micromamba run -n svr python tools/demo_resp_tilt.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from data.preprocess import build_data_dicts, get_canonical_transforms  # noqa: E402
from data.respiratory import _build_disp_dhw, _rotate_disp, reslice_volume_vec  # noqa: E402

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"
OUT = Path("/home/minsukc/vggt/result/respiratory_tilt_demo.png")
SUBJECT = "Train_P053"
T_FIXED = 0
D_MM = 18.0          # fixed SI magnitude for the demo
AP_RATIO = 0.35
JITTER_DEG = 30.0    # cfg.direction_jitter_deg default
N_DRAWS = 200


def load_phases(name):
    out = get_canonical_transforms()(build_data_dicts([os.path.join(DATA_ROOT, name, "sax")])[0])
    return out["phases"].squeeze(1).permute(0, 3, 2, 1).contiguous().float()  # (T,D,H,W)


def main():
    g = torch.Generator().manual_seed(0)
    base = _build_disp_dhw(torch.tensor(D_MM), torch.tensor(AP_RATIO * D_MM), "H")  # (3,) = (dD,dH,dW)

    # Sample N tilted directions exactly as sample_displacement_vectors does.
    theta = torch.rand(N_DRAWS, generator=g) * math.radians(JITTER_DEG)
    phi = torch.rand(N_DRAWS, generator=g) * (2 * math.pi)
    v = _rotate_disp(base.expand(N_DRAWS, 3), theta, phi).numpy()  # (N,3) (dD,dH,dW)
    base_np = base.numpy()

    fig = plt.figure(figsize=(15, 6.2), dpi=120)

    # ── (A) 3D cone ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    # axes: x=W(LR), y=H(AP), z=D(SI/through-plane)
    ax.scatter(v[:, 2], v[:, 1], v[:, 0], s=8, c="#1f77b4", alpha=0.45, label="tilted draws (θ~U(0,30°))")
    ax.quiver(0, 0, 0, base_np[2], base_np[1], base_np[0], color="k", lw=3,
              arrow_length_ratio=0.12, label="base SI(+AP), no tilt")
    # literature true-SI cone edge (20-45° off D), drawn as reference rings at |d|
    for ang, col in [(20, "#2ca02c"), (45, "#d62728")]:
        a = math.radians(ang)
        ring_phi = np.linspace(0, 2 * np.pi, 100)
        rz = D_MM * math.cos(a) * np.ones_like(ring_phi)       # D comp
        rr = D_MM * math.sin(a)
        ry = rr * np.cos(ring_phi); rx = rr * np.sin(ring_phi)  # spread in H/W
        ax.plot(rx, ry, rz, color=col, lw=1.6, ls="--", label=f"true-SI tilt {ang}° (lit.)")
    ax.set_xlabel("W  (LR)"); ax.set_ylabel("H  (AP)"); ax.set_zlabel("D  (SI / through-plane)")
    ax.set_title(f"(A) Sampled motion directions  |d|={D_MM:.0f}mm fixed\n"
                 "tilt = domain randomization over the UNKNOWN SAX→SI angle", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax.view_init(elev=18, azim=-60)

    # ── (B) coronal reslices: pure-D vs tilted draws ─────────────────────────
    phases = load_phases(SUBJECT)
    V = phases[T_FIXED]  # (D,H,W)
    vmax = float(V.max())
    # coronal = fix H (AP), show D (rows) x W (cols); pick mid-AP plane
    H_mid = V.shape[1] // 2

    picks = [("rest (|d|=0)", torch.zeros(3)),
             ("base (pure SI+AP)", base),
             *[(f"tilt draw {i+1}", torch.tensor(v[i])) for i in (0, 1, 2)]]
    n = len(picks)
    gs = fig.add_gridspec(n, 2, left=0.56, right=0.99, wspace=0.05, hspace=0.12)
    rest = reslice_volume_vec(V, torch.zeros(3))[:, H_mid, :].numpy()
    for r, (lbl, disp) in enumerate(picks):
        sl = reslice_volume_vec(V, disp.float())[:, H_mid, :].numpy()
        axL = fig.add_subplot(gs[r, 0])
        axL.imshow(sl, cmap="gray", vmin=0, vmax=vmax, origin="lower",
                   interpolation="nearest", aspect="auto")
        axL.set_xticks([]); axL.set_yticks([]); axL.set_ylabel(lbl, fontsize=7.5)
        if r == 0:
            axL.set_title("coronal reslice", fontsize=9)
        axR = fig.add_subplot(gs[r, 1])
        axR.imshow(sl - rest, cmap="RdBu_r", vmin=-vmax * 0.6, vmax=vmax * 0.6,
                   origin="lower", interpolation="nearest", aspect="auto")
        axR.set_xticks([]); axR.set_yticks([])
        if r == 0:
            axR.set_title("diff vs rest", fontsize=9)

    fig.suptitle("Respiratory direction-jitter (tilt) — what it samples and how it moves anatomy",
                 fontsize=12, y=0.99)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
