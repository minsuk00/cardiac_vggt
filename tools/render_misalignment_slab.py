"""Paper figure (2-face variant): each panel is a SAX "lid" + ONE long-axis wall, in true
physical mm proportions. Two rows (LAX view 1 = front wall, LAX view 2 = right wall) x 5
corruption-panel columns, sharing the same _misalign_common loader/contrast logic as
render_misalignment_cube.py (which draws all 3 faces per panel instead).

  python tools/render_misalignment_slab.py --ds cmrx2024 --subject CMRx24_Test_P020 \
      --arm vggt_augaggr518_ep300 --out temp/misalign/fig_misalignment_P020_slab.png
"""
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _misalign_common import load_case  # noqa: E402
from render_misalignment_cube import to_disp, FRONT_COLOR, RIGHT_COLOR  # noqa: E402


def draw_slab(ax, sax_img, wall_img, wall_side, x_mm, y_mm, z_mm, cmap, azim=-90, elev=10,
              annotate=False, cut_mm=None):
    """wall_side: 'front' (XZ at Y=0) or 'right' (YZ at X=x_mm). azim is chosen per wall_side by
    the caller so the wall's normal points straight at the camera (front: azim=-90; right:
    azim=0) — matplotlib's default PERSPECTIVE projection draws an off-axis flat wall as a
    trapezoid even at the "right" angle, which reads as a tilt/rotation, so this also switches
    to an orthographic projection (parallel lines stay parallel) for a true undistorted rectangle.
    annotate=True draws, on the lid, the exact row/column (cut_mm) this wall was cut from,
    color-matched to the wall's border, so the top<->side correspondence is explicit."""
    color = FRONT_COLOR if wall_side == "front" else RIGHT_COLOR
    xs, ys = np.linspace(0, x_mm, sax_img.shape[1]), np.linspace(0, y_mm, sax_img.shape[0])
    zs = np.linspace(0, z_mm, wall_img.shape[0])
    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, np.full_like(X, z_mm), rstride=1, cstride=1,
                     facecolors=cmap(sax_img), shade=False, zorder=2)
    if wall_side == "front":
        X, Z = np.meshgrid(xs, zs)
        ax.plot_surface(X, np.zeros_like(X), Z, rstride=1, cstride=1,
                         facecolors=cmap(wall_img), shade=False, zorder=1)
        edges = [((0, 0, 0), (x_mm, 0, 0)), ((0, 0, z_mm), (x_mm, 0, z_mm)),
                 ((0, 0, 0), (0, 0, z_mm)), ((x_mm, 0, 0), (x_mm, 0, z_mm)),
                 ((0, 0, z_mm), (0, y_mm, z_mm)), ((x_mm, 0, z_mm), (x_mm, y_mm, z_mm)),
                 ((0, y_mm, z_mm), (x_mm, y_mm, z_mm))]
        if annotate and cut_mm is not None:
            ax.plot([0, x_mm], [cut_mm, cut_mm], [z_mm, z_mm], color=color, lw=2.2, zorder=10)
    else:
        ys_ = np.linspace(0, y_mm, wall_img.shape[1])
        zs_ = np.linspace(0, z_mm, wall_img.shape[0])
        Y, Z = np.meshgrid(ys_, zs_)
        ax.plot_surface(np.full_like(Y, x_mm), Y, Z, rstride=1, cstride=1,
                         facecolors=cmap(wall_img), shade=False, zorder=1)
        edges = [((x_mm, 0, 0), (x_mm, y_mm, 0)), ((x_mm, 0, z_mm), (x_mm, y_mm, z_mm)),
                 ((x_mm, 0, 0), (x_mm, 0, z_mm)), ((x_mm, y_mm, 0), (x_mm, y_mm, z_mm)),
                 ((x_mm, 0, z_mm), (0, 0, z_mm)), ((x_mm, y_mm, z_mm), (0, y_mm, z_mm)),
                 ((0, 0, z_mm), (0, y_mm, z_mm))]
        if annotate and cut_mm is not None:
            ax.plot([cut_mm, cut_mm], [0, y_mm], [z_mm, z_mm], color=color, lw=2.2, zorder=10)
    ec = color if annotate else "k"; lw = 1.6 if annotate else 0.6
    for p0, p1 in edges:
        ax.plot(*zip(p0, p1), color=ec, lw=lw)
    ax.set_xlim(0, x_mm); ax.set_ylim(0, y_mm); ax.set_zlim(0, z_mm)
    try:
        ax.set_box_aspect((x_mm, y_mm, z_mm))
    except AttributeError:
        pass
    ax.set_axis_off(); ax.set_proj_type("ortho"); ax.view_init(elev=elev, azim=azim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True); ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", required=True); ap.add_argument("--t", type=int, default=0)
    ap.add_argument("--out", required=True); ap.add_argument("--gamma", type=float, default=0.6)
    ap.add_argument("--pct-lo", type=float, default=1.0); ap.add_argument("--pct-hi", type=float, default=99.0)
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--annotate", action="store_true", help="color-coded guide line on the lid marking the wall's cut")
    ap.add_argument("--elev", type=float, default=10)
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

    fig = plt.figure(figsize=(3.0 * 5, 2 * 2.6))
    for j, (name, v) in enumerate(vols):
        d = to_disp(v, vmin, vmax, a.gamma)
        sax_img = d[z1 - 1, y0:y1, x0:x1]
        lax1_img = d[z0:z1, yc, x0:x1]                        # front wall: x–z reslice at y=yc
        lax2_img = d[z0:z1, y0:y1, xc]                         # right wall: y–z reslice at x=xc
        cuts = [(yc - y0) * inplane, (xc - x0) * inplane]
        for i, (wall_img, side, lab, azim) in enumerate([(lax1_img, "front", "LAX view 1", -90),
                                                          (lax2_img, "right", "LAX view 2", 0)]):
            ax = fig.add_subplot(2, 5, i * 5 + j + 1, projection="3d")
            draw_slab(ax, sax_img, wall_img, side, x_mm, y_mm, z_mm, cmap, azim=azim, elev=a.elev,
                      annotate=a.annotate, cut_mm=cuts[i])
            if j == 0:
                ax.text2D(-0.05, 0.5, lab, transform=ax.transAxes, fontsize=12,
                          rotation=90, va="center", ha="center")
            if i == 1:
                ax.set_title(name, fontsize=13, y=-0.12)
    if not a.paper:
        fig.suptitle(f"{a.subject}  t={a.t}  D={case['D']} dz={dz}mm  "
                      f"mean|breath|={case['manifest']['breath']['mean_abs_disp_mm']:.1f}mm", fontsize=9)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.98, bottom=0.08, wspace=0.02, hspace=0.02)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=170); print("wrote", a.out)


if __name__ == "__main__":
    main()
