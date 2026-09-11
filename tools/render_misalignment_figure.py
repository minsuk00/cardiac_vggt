"""Paper figure: how misaligned the input stacks are.

Columns: clean GT | free-breathing | ungated | free-breathing + ungated | ours.
Rows: x–z | y–z through-plane reslices through the heart-mask centroid (z = SAX normal ≈ LV long axis), true aspect ratio.
Inputs come from the frozen eval bundle; the ungated columns use the slot draw recorded in
<arm>/ed_dvf.npz (slot_t per plane z), i.e. exactly what the model was fed.

  python tools/render_misalignment_figure.py --ds cmrx2023 --subject CMRx23_Train_P022 \
      --arm vggt_noreg224_ep300 --out temp/misalign/P022.png
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _misalign_common import load_case  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--gamma", type=float, default=0.7)
    ap.add_argument("--paper", action="store_true", help="no suptitle / row labels; row labels as LAX 1/2")
    a = ap.parse_args()
    case = load_case(a.ds, a.subject, a.arm, a.t)
    vols, D, dz, inplane = case["vols"], case["D"], case["dz_mm"], case["inplane_mm"]
    zc, yc, xc = case["zc"], case["yc"], case["xc"]
    y0, y1, x0, x1 = case["y0"], case["y1"], case["x0"], case["x1"]
    t_of_z, m = case["t_of_z"], case["manifest"]
    gt = vols[0][1]

    vmax = np.percentile(gt, 99.5); asp = dz / inplane
    fig, ax = plt.subplots(2, 5, figsize=(3 * 5, 2 * 2.2), squeeze=False)
    for j, (name, v) in enumerate(vols):
        v = np.clip(v / vmax, 0, 1) ** a.gamma
        panels = [(v[:, yc, x0:x1], asp, f"x–z reslice, y={yc}"),
                  (v[:, y0:y1, xc], asp, f"y–z reslice, x={xc}")]
        for i, (img, aspect, lab) in enumerate(panels):
            ax[i, j].imshow(img, cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest")
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if j == 0: ax[i, j].set_ylabel(("LAX view 1", "LAX view 2")[i] if a.paper else lab, fontsize=14 if a.paper else 10)
            if a.paper and i == 1: ax[i, j].set_xlabel(name, fontsize=14, labelpad=6)
            elif not a.paper and i == 0: ax[i, j].set_title(name, fontsize=10)
    if not a.paper: fig.suptitle(f"{a.subject}  t={a.t}  D={D} dz={dz}mm  mean|breath|={m['breath']['mean_abs_disp_mm']:.1f}mm  "
                 f"slot t per z: {[t_of_z[z] for z in range(D)]}", fontsize=9)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); fig.tight_layout(); fig.savefig(a.out, dpi=150); print("wrote", a.out)


if __name__ == "__main__":
    main()
