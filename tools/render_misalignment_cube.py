"""Paper figure (3D variant): same 5 corruption panels as render_misalignment_figure.py, drawn
as an isometric "corner cube" per panel — a high-res in-plane (SAX) lid on top, and the two
long-axis reslices as the front/right walls, all in true physical mm proportions (dz vs inplane).
The two walls are still through-plane cuts at the heart-mask centroid; placing them on the box's
outer faces is a display convention (as in radiology 3-plane cube viewers), not a claim that the
cut is literally at the box boundary.

  python tools/render_misalignment_cube.py --ds cmrx2024 --subject CMRx24_Test_P020 \
      --arm vggt_augaggr518_ep300 --out temp/misalign/fig_misalignment_P020_cube.png
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _misalign_common import load_case  # noqa: E402


def to_disp(v, vmin, vmax, gamma):
    return np.clip((v - vmin) / max(vmax - vmin, 1e-6), 0, 1) ** gamma


FRONT_COLOR, RIGHT_COLOR = "#ffcf33", "#33ccff"  # gold / cyan — guide-line & wall-border pair


def draw_cube(ax, sax_img, lax1_img, lax2_img, x_mm, y_mm, z_mm, cmap, annotate=False,
              y_cut_mm=None, x_cut_mm=None, elev=18, azim=-55):
    """sax_img (y,x), lax1_img (z,x) at Y=0, lax2_img (z,y) at X=xmax.
    annotate=True draws, on the lid, the exact rows/columns the two walls were cut from
    (y_cut_mm, x_cut_mm — physical position within the crop), color-matched to a colored
    border on each wall, so the top<->side correspondence is explicit rather than implied."""
    xs, ys, zs = np.linspace(0, x_mm, sax_img.shape[1]), np.linspace(0, y_mm, sax_img.shape[0]), \
                 np.linspace(0, z_mm, lax1_img.shape[0])
    X, Y = np.meshgrid(xs, ys); ax.plot_surface(X, Y, np.full_like(X, z_mm), rstride=1, cstride=1,
                                                 facecolors=cmap(sax_img), shade=False, zorder=3)
    X, Z = np.meshgrid(xs, zs); ax.plot_surface(X, np.zeros_like(X), Z, rstride=1, cstride=1,
                                                 facecolors=cmap(lax1_img), shade=False, zorder=1)
    Y, Z = np.meshgrid(ys, zs); ax.plot_surface(np.full_like(Y, x_mm), Y, Z, rstride=1, cstride=1,
                                                 facecolors=cmap(lax2_img), shade=False, zorder=2)
    # box edges for a crisp silhouette
    corners = np.array([[0, 0, 0], [x_mm, 0, 0], [x_mm, y_mm, 0], [0, y_mm, 0]])
    top = corners + [0, 0, z_mm]
    ec_front, ec_right = (FRONT_COLOR, RIGHT_COLOR) if annotate else ("k", "k")
    lw = 1.6 if annotate else 0.6
    ax.plot(*zip(top[0], top[1]), color=ec_front, lw=lw)   # top-front edge
    ax.plot(*zip(top[1], top[2]), color=ec_right, lw=lw)   # top-right edge
    ax.plot(*zip(top[2], top[3]), color="k", lw=0.6)
    ax.plot(*zip(top[3], top[0]), color="k", lw=0.6)
    ax.plot(*zip(corners[1], top[1]), color=ec_right, lw=lw)   # front-right vertical
    ax.plot(*zip(corners[2], top[2]), color=ec_right, lw=lw)   # right wall far edge
    ax.plot(*zip(corners[1], corners[2]), color=ec_right, lw=lw)  # right wall bottom
    ax.plot(*zip(corners[0], corners[1]), color=ec_front, lw=lw)  # front wall bottom
    if annotate:
        if y_cut_mm is not None:
            ax.plot([0, x_mm], [y_cut_mm, y_cut_mm], [z_mm, z_mm], color=FRONT_COLOR, lw=2.2, zorder=10)
        if x_cut_mm is not None:
            ax.plot([x_cut_mm, x_cut_mm], [0, y_mm], [z_mm, z_mm], color=RIGHT_COLOR, lw=2.2, zorder=10)
    ax.set_xlim(0, x_mm); ax.set_ylim(0, y_mm); ax.set_zlim(0, z_mm)
    try:
        ax.set_box_aspect((x_mm, y_mm, z_mm))
    except AttributeError:
        pass
    ax.set_axis_off(); ax.view_init(elev=elev, azim=azim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--pct-lo", type=float, default=1.0); ap.add_argument("--pct-hi", type=float, default=99.0)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--annotate", action="store_true", help="color-coded guide lines on the lid marking each wall's cut")
    ap.add_argument("--elev", type=float, default=18); ap.add_argument("--azim", type=float, default=-55)
    a = ap.parse_args()
    case = load_case(a.ds, a.subject, a.arm, a.t)
    vols, dz, inplane = case["vols"], case["dz_mm"], case["inplane_mm"]
    zc, yc, xc = case["zc"], case["yc"], case["xc"]
    y0, y1, x0, x1, z0, z1 = case["y0"], case["y1"], case["x0"], case["x1"], case["z0"], case["z1"]
    gt = vols[0][1]
    crop = gt[z0:z1, y0:y1, x0:x1]
    nz = crop[crop > 0] if (crop > 0).any() else crop.ravel()
    vmin, vmax = np.percentile(nz, a.pct_lo), np.percentile(nz, a.pct_hi)
    cmap = plt.get_cmap("gray")
    x_mm, y_mm, z_mm = (x1 - x0) * inplane, (y1 - y0) * inplane, (z1 - z0) * dz

    fig = plt.figure(figsize=(3.4 * 5, 3.6))
    for j, (name, v) in enumerate(vols):
        d = to_disp(v, vmin, vmax, a.gamma)
        sax_img = d[z1 - 1, y0:y1, x0:x1]                    # lid: topmost slice in the heart z-range
        lax1_img = d[z0:z1, yc, x0:x1]                        # front wall: x–z reslice at y=yc
        lax2_img = d[z0:z1, y0:y1, xc]                        # right wall: y–z reslice at x=xc
        ax = fig.add_subplot(1, 5, j + 1, projection="3d")
        draw_cube(ax, sax_img, lax1_img, lax2_img, x_mm, y_mm, z_mm, cmap, annotate=a.annotate,
                  y_cut_mm=(yc - y0) * inplane, x_cut_mm=(xc - x0) * inplane, elev=a.elev, azim=a.azim)
        ax.set_title(name, fontsize=13, y=0.02 if not a.paper else -0.06)
    if not a.paper:
        fig.suptitle(f"{a.subject}  t={a.t}  D={case['D']} dz={dz}mm  "
                      f"mean|breath|={case['manifest']['breath']['mean_abs_disp_mm']:.1f}mm", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.10, wspace=0.02)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170); print("wrote", a.out)


if __name__ == "__main__":
    main()
