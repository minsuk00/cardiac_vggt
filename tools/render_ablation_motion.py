"""Ablation figure: what motion each loss-ablation arm learned, on one AF12 test subject.

Columns: w/o L_obs, w/o heart term, w/o L_smooth, Full.
Rows: (1) through-plane breathing shift per slice, predicted vs true;
      (2) in-plane cardiac deformation (slice-mean shift removed) on one mid-ventricular SAX input slice;
      (3) LV volume over the cine, predicted vs GT.
All inputs are saved eval outputs (no inference). True through-plane shifts come from
_motion_epe/af12/<arm>.json (resp_diag/ed_dvf applied shifts are stale for rhythm cohorts).
Delta is saved for the first target frame only (ed_dvf.npz).

Usage: micromamba run -n svr python tools/render_ablation_motion.py [--subject cohort/id] --out <png|pdf>
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

from render_runtime_accuracy import GRID, INK, TEXTW, paper_rc

EV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation")
VOL, RES = f"{EV}/volumes", f"{EV}/metric_results/test"
ARMS = [("w/o cardiac-region term", "hw0"), ("w/o $\\mathcal{L}_{\\mathrm{obs}}$", "nogather"),
        ("w/o $\\mathcal{L}_{\\mathrm{smooth}}$", "base"), ("Full", "diff1000")]   # Table 3 row order
MM_XY = 0.5 * 255 * 1.4          # normalized -> mm (evaluation/src/engine/run_vggt.py MM_PER_NORM)
PX_PER_MM = 518 / (256 * 1.4)    # 518-grid pixels per mm
ARROW_X = 1.0                    # arrow length = true displacement


def method(a):
    return f"vggt_final518_{a}_ep300"


def ef_rows(a):
    out = {}
    for f in glob.glob(f"{RES}/*_af12/ef/{method(a)}.json"):
        for r in json.load(open(f))["per_subject"]:
            out[f"{r['cohort']}/{r['subject']}"] = r
    return out


def heart518(sd, z):
    heart = nib.load(f"{sd}/mask_heart.nii.gz").get_fdata() > 0
    return zoom(heart[:, :, z].T.astype(np.uint8), 518 / 256, order=0) > 0


def deform(d, m, raw=False):
    """Mean |in-plane delta| (mm) inside m after removing the slice-mean shift (raw: no removal); also the field."""
    f = d[..., :2].astype(np.float32) * MM_XY
    if not raw:
        f = f - f[m].mean(0)
    return np.hypot(f[..., 0], f[..., 1])[m].mean(), f


def subject_deform(sd, a):
    z = np.load(f"{sd}/{method(a)}/ed_dvf.npz")
    v = [deform(z["delta"][s], m)[0] for s in range(1, len(z["slot_z"]))
         if (m := heart518(sd, int(z["slot_z"][s]))).sum() >= 500]
    return float(np.mean(v))


def epe_demeaned(sl):
    """Per-subject through-plane EPE as in the paper: |pred - true| after removing the mean error."""
    e = np.array([v["pred"][0] - v["applied"][0] for v in sl.values()])
    return float(np.mean(np.abs(e - e.mean())))


def pick_subject(efs, epes):
    """Closest to every arm's median |EF err|, VTC err, resp. EPE and in-plane deformation
    (sum of |z|-scores). Deformation is only computed for the 30 best candidates (needs the saved fields)."""
    keys = sorted(set.intersection(*[set(e) for e in efs.values()], *[set(e) for e in epes.values()]))
    score = dict.fromkeys(keys, 0.0)
    for a, e in efs.items():
        for f in (lambda k: abs(e[k]["ef_breath"] - e[k]["ef_gt"]), lambda k: e[k]["lv_curve_nmae_breath"],
                  lambda k: epe_demeaned(epes[a][k])):
            x = np.array([f(k) for k in keys])
            for k, v in zip(keys, x):
                score[k] += abs(v - np.median(x)) / x.std()
    cand = sorted(keys, key=score.get)[:30]
    dsc = dict.fromkeys(cand, 0.0)
    for _, a in ARMS:
        vals = {k: subject_deform(f"{VOL}/{k.split('/')[0]}/out/{k.split('/')[1]}", a) for k in cand}
        x = np.array(list(vals.values()))
        for k in cand:
            dsc[k] += abs(vals[k] - np.median(x)) / x.std()
    return sorted(cand, key=lambda k: score[k] + dsc[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tight", action="store_true", help="compact layout (less white space)")
    ap.add_argument("--full-field", action="store_true",
                    help="draw the deformation over the whole slice, not only the heart ROI "
                         "(rigid-shift removal and the mm value still use the heart ROI)")
    ap.add_argument("--raw", action="store_true", help="show the raw in-plane field (no rigid-shift removal)")
    ap.add_argument("--window", choices=["slice", "heart"], default="slice",
                    help="display window top = 99.5th percentile of the whole slice or of the heart ROI")
    ap.add_argument("--gamma", type=float, default=1.0, help="display gamma (<1 brightens dark tissue)")
    a = ap.parse_args()
    efs = {k: ef_rows(k) for _, k in ARMS}
    epes = {k: json.load(open(f"{RES}/_motion_epe/af12/{method(k)}.json"))["slices"] for _, k in ARMS}
    if a.subject:
        subj = a.subject
    else:
        ranked = pick_subject(efs, epes)
        print("candidates:", ranked[:5])
        subj = ranked[0]
    cohort, sid = subj.split("/")
    sd = f"{VOL}/{cohort}/out/{sid}"

    # through-plane shift per slice (pred vs true)
    epe = {k: epes[k][subj] for _, k in ARMS}
    zs = sorted(int(z) for z in epe["diff1000"])
    true_dz = np.array([epe["diff1000"][str(z)]["applied"][0] for z in zs])

    # mid-ventricular non-reference slot: the one with the largest LV cross-section
    seg = nib.load(f"{sd}/seg_gt_3d_fullres/seg_t00.nii.gz").get_fdata()      # 1 LV, 2 MYO, 3 RV
    dvf = {k: np.load(f"{sd}/{method(k)}/ed_dvf.npz") for _, k in ARMS}
    slot_z, slot_t = dvf["diff1000"]["slot_z"].astype(int), dvf["diff1000"]["slot_t"].astype(int)
    lv_area = [(seg[:, :, slot_z[s]] == 1).sum() if s > 0 else -1 for s in range(len(slot_z))]
    s_sel = int(np.argmax(lv_area))
    z_sel, j_sel = slot_z[s_sel], slot_t[s_sel]
    m = heart518(sd, z_sel)
    inp = nib.load(f"{sd}/breath/stack_t{j_sel:02d}.nii.gz").get_fdata()[:, :, z_sel].T
    inp518 = zoom(inp, 518 / 256, order=1)
    ys, xs = np.where(m)                               # square crop centred on the heart ROI
    cy, cx = int(ys.mean()), int(xs.mean())
    h = int(max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 + 12)
    y0, y1, x0, x1 = cy - h, cy + h, cx - h, cx + h

    paper_rc()
    ORANGE, GRAYL = "#D55E00", "#b0b0b0"
    if a.tight:   # shorter plot rows sized so the square image row has no slack
        fig, ax = plt.subplots(3, 4, figsize=(TEXTW, 3.4), layout="constrained",
                               gridspec_kw={"height_ratios": [0.8, 1.0, 0.8]})
        fig.get_layout_engine().set(w_pad=0.005, h_pad=0.005, wspace=0.01, hspace=0.015)
    else:
        fig, ax = plt.subplots(3, 4, figsize=(TEXTW, 4.1), layout="constrained",
                               gridspec_kw={"height_ratios": [1, 1.15, 1]})
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.02, hspace=0.03)
    curve_gt = np.array(efs["diff1000"][subj]["lv_curve_gt"])
    fields = {k: deform(dvf[k]["delta"][s_sel], m, raw=a.raw) for _, k in ARMS}
    vmax = np.percentile(np.concatenate([np.hypot(f[..., 0], f[..., 1])[m] for _, f in fields.values()]), 98)
    frames = np.arange(len(curve_gt)) / len(curve_gt)   # s, frame = 1/12 of the nominal 1 s R-R (as in Fig. 5)
    for c, (name, k) in enumerate(ARMS):
        # (a) respiratory through-plane shift per slice: residual stems from true to predicted
        p = np.array([epe[k][str(z)]["pred"][0] for z in zs])
        a1 = ax[0, c]
        a1.vlines(zs, true_dz, p, color=GRAYL, lw=0.8, zorder=1)
        a1.scatter(zs, true_dz, s=12, facecolors="white", edgecolors="k", linewidths=0.7, zorder=2, label="True")
        a1.scatter(zs, p, s=9, color=ORANGE, zorder=3, label="Predicted")
        a1.set_title(name, fontsize=8, pad=3)
        a1.text(0.04, 0.96, f"EPE {epe_demeaned(epe[k]):.1f} mm", transform=a1.transAxes, fontsize=6.5, va="top")
        a1.set_xticks([])
        a1.grid(axis="y", color=GRID, lw=0.5)
        a1.set_axisbelow(True)
        # (b) in-plane cardiac deformation: magnitude map + arrows on the input slice
        a2 = ax[1, c]
        dmag, f = fields[k]
        # full slice drawn; the view is the heart crop widened (adjustable="datalim") to fill the column,
        # so the image edges line up with rows (a) and (c) without distorting it
        wmax = np.percentile(inp518[m] if a.window == "heart" else inp, 99.5)
        a2.imshow(np.clip(inp518 / wmax, 0, 1) ** a.gamma, cmap="gray", vmin=0, vmax=1)
        show = np.ones_like(m) if a.full_field else m
        mag = np.where(show, np.hypot(f[..., 0], f[..., 1]), np.nan)
        im = a2.imshow(mag, cmap="magma", vmin=0, vmax=vmax, alpha=0.6)
        st = 11
        yy, xx = np.mgrid[st // 2:518:st, st // 2:518:st] if a.full_field else \
            np.mgrid[y0 + st // 2:y1:st, x0 + st // 2:x1:st]
        keep = show[yy, xx]
        a2.quiver(xx[keep], yy[keep], f[yy, xx, 0][keep], f[yy, xx, 1][keep], color="white",
                  angles="xy", scale_units="xy", scale=1 / (PX_PER_MM * ARROW_X), width=0.009,
                  headwidth=3.5, headlength=3.5, headaxislength=3)
        a2.set_xlim(x0, x1); a2.set_ylim(y1, y0)
        a2.set_aspect("equal", adjustable="datalim")
        a2.text(0.04, 0.96, f"{dmag:.1f} mm", transform=a2.transAxes, fontsize=6.5, va="top", color="white")
        a2.set_xticks([]); a2.set_yticks([])
        for sp in a2.spines.values():
            sp.set_visible(False)
        # (c) LV volume over the cine
        a3 = ax[2, c]
        r = efs[k][subj]
        a3.plot(frames, curve_gt, "--", color=INK, lw=1.0, label="GT")
        a3.plot(frames, r["lv_curve_breath"], "-", color=ORANGE, lw=1.4, label="Predicted")
        a3.text(0.04, 0.04, f"EF {r['ef_breath']:.0f}%", transform=a3.transAxes, fontsize=6.5, va="bottom")
        a3.set_xticks([])
        a3.grid(axis="y", color=GRID, lw=0.5)
        a3.set_axisbelow(True)
    for row in (0, 2):
        lo = min(a_.get_ylim()[0] for a_ in ax[row]); hi = max(a_.get_ylim()[1] for a_ in ax[row])
        for c in range(4):
            ax[row, c].set_ylim(lo, hi)
            if c:
                ax[row, c].tick_params(labelleft=False)
    top = 5 * np.ceil(ax[0, 0].get_ylim()[1] / 5)       # round the shift axis up to the next 5 mm tick
    for c in range(4):
        ax[0, c].set_ylim(ax[0, c].get_ylim()[0], top + 0.4)
        ax[0, c].set_yticks(np.arange(0, top + 1, 5))
    ax[0, 0].set_ylabel("Resp. shift (mm)")
    ax[0, 0].set_xlabel("slice (apex$\\rightarrow$base)", fontsize=6.5, labelpad=1)
    ax[2, 0].set_ylabel("LV volume (mL)")
    ax[2, 0].set_xlabel("Acquisition time (s)", fontsize=6.5, labelpad=1)
    ax[1, 0].set_ylabel(" ")                         # placeholder so the row title aligns with the others
    for c in range(4):
        ax[2, c].set_xticks([0, 0.5])
        ax[2, c].tick_params(axis="x", labelsize=6, pad=1)
    fig.align_ylabels(ax[:, 0])
    from matplotlib.lines import Line2D
    hs = [(Line2D([], [], ls="none", marker="o", mfc="white", mec="k", mew=0.7, ms=4),
           Line2D([], [], ls="--", color=INK, lw=1.0)),
          (Line2D([], [], ls="none", marker="o", color=ORANGE, ms=3.5),
           Line2D([], [], ls="-", color=ORANGE, lw=1.4))]
    fig.legend(hs, ["Ground truth", "Predicted"], loc="outside lower center", ncol=2, frameon=False,
               fontsize=7, handlelength=2.2, columnspacing=1.5)
    cb = fig.colorbar(im, ax=ax[1, :], fraction=0.025, pad=0.01)
    cb.set_label("mm", fontsize=6.5, labelpad=1)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_linewidth(0.4)
    # (a)/(b)/(c): one shared x, a fixed gap left of the outermost y-label, each centred on its row.
    # Placed after the layout is final (then frozen), so the image row's narrower axes can't shift (b).
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted()
    x_lab = min(ax[r_, 0].yaxis.label.get_window_extent(rend).x0 for r_ in (0, 2))
    x_let = to_fig.transform((x_lab - 10 * fig.dpi / 72, 0))[0]
    fig.set_layout_engine("none")
    for r_, t in enumerate(["(a)", "(b)", "(c)"]):
        bb = ax[r_, 0].get_window_extent(rend)
        y = to_fig.transform((0, (bb.y0 + bb.y1) / 2))[1]
        fig.text(x_let, y, t, ha="right", va="center", fontsize=9, fontweight="bold")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("wrote", a.out, "| subject", subj, "| slice", z_sel, "| frame", j_sel)


if __name__ == "__main__":
    main()
