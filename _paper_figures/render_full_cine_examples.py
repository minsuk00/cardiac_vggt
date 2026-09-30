"""Appendix figure: full-cine qualitative examples of CardioStitch under the paper's AF protocol (af12).

One figure per subject. Columns = the 12 AF target frames. Rows = short-axis (a non-reference slice,
two planes from the reference) and long-axis (a cut through every slice): CardioStitch / target / error.
Subjects: per dataset, the best LV volume-time-curve error (VTC) among subjects in the 35-65th PSNR
percentile of that dataset (see SUBJECTS; selection printed by --select).

Reads (read-only): scratch/eval/<src>_af12/out/<subject>/{gt/gt_tXX, <METHOD>/recon_breath/vol_tXX,
mask_heart_pad10, manifest.json}; evaluation/metric_results/test/<src>_af12/{,ef/}<METHOD>.json.

    python _paper_figures/render_full_cine_examples.py [--select]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHOD = "vggt_final518_diff1000_ep300"
SRCS = ["acdc", "cmrx2023", "cmrx2024", "cmrx2025", "mnms"]
SUBJECTS = [("acdc", "ACDC_patient115"), ("cmrx2023", "CMRx23_Train_P111"), ("cmrx2024", "CMRx24_Train_P048"),
            ("cmrx2025", "CMRx25_R1val_Center001_UIH_30T_umr780_P025"), ("mnms", "MNMs_A4R4T0")]
# Paper appendix (Figs. "Full-cine reconstruction ..."): chosen by eye from the candidates, regular24, coronal-type LAX.
PAPER = [("acdc", "ACDC_patient141"), ("mnms", "MNMs_D1H2O9"), ("cmrx2023", "CMRx23_Train_P111"),
         ("cmrx2024", "CMRx24_Train_P048"), ("cmrx2025", "CMRx25_R1val_Center001_UIH_30T_umr780_P025")]
OUT = os.path.join(ROOT, "figs", "full_cine")
ERR_MAX = 0.4          # |error| colour-scale top (intensities are ~[0, 1])
SAX_OFFSET = 2         # short-axis slice = reference plane - 2 (toward apex; apex is z0), clear of the reference


def candidates(src):
    """Subjects of `src` in the 35-65th PSNR percentile, sorted by VTC error: (vtc, ef_err, psnr, subject)."""
    res = f"{ROOT}/evaluation/metric_results/test/{src}_af12"
    a = {r["subject"]: r for r in json.load(open(f"{res}/{METHOD}.json"))["per_subject"]}
    b = {r["subject"]: r for r in json.load(open(f"{res}/ef/{METHOD}.json"))["per_subject"]}
    subs = sorted(set(a) & set(b))
    p = np.array([a[s]["breath_psnr_unit_peak"] for s in subs])
    lo, hi = np.percentile(p, [35, 65])
    return sorted((b[s]["lv_curve_nmae_breath"], abs(b[s]["ef_breath"] - b[s]["ef_gt"]), ps, s)
                  for s, ps in zip(subs, p) if lo <= ps <= hi)


def select():
    for src in SRCS:
        print(src)
        for r in candidates(src)[:4]:
            print(f"   VTC {r[0]:5.2f}  EF err {r[1]:5.2f}  PSNR {r[2]:5.2f}  {r[3]}")


def load(src, sub, arm="af12"):
    d = f"{ROOT}/scratch/eval/{src}_{arm}/out/{sub}"
    man = json.load(open(f"{d}/manifest.json"))
    ref = int(man["scatter"]["ref_plane"] if "scatter" in man else man["rhythm"]["ref_plane"])
    T, dz = min(man["T"], 12), man["dz_mm"]                                      # regular24: first 12 frames = one beat
    rd = lambda p: np.asarray(nib.load(p).dataobj, dtype=np.float32)            # (X, Y, Z)
    gt = np.stack([rd(f"{d}/gt/gt_t{t:02d}.nii.gz") for t in range(T)])
    rc = np.stack([rd(f"{d}/{METHOD}/recon_breath/vol_t{t:02d}.nii.gz") for t in range(T)])
    heart = rd(f"{d}/mask_heart_pad10.nii.gz") > 0.5
    return gt, rc, heart, ref, dz


def render(src, sub, out=OUT, stem_name=None, arm="af12", lax_view="sag", pdf=False):
    gt, rc, heart, ref, dz = load(src, sub, arm)
    T, X, Y, D = gt.shape
    xs, ys, _ = np.where(heart)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) // 2 + 4
    x0, x1 = max(cx - half, 0), min(cx + half, X)
    y0, y1 = max(cy - half, 0), min(cy + half, Y)
    z_sax = ref - SAX_OFFSET if ref - SAX_OFFSET >= 0 else ref + SAX_OFFSET
    rep = max(int(round(dz / 1.4)), 1)                                           # true through-plane proportions
    vmax = float(np.percentile(gt[:, x0:x1, y0:y1], 99.5))

    def sax(v, t):                                                               # rows = y, cols = x
        return v[t, x0:x1, y0:y1, z_sax].T

    def lax(v, t):                                                               # cut at x = cx (sagittal-type); rows = z (base on top)
        return np.repeat(v[t, cx, y0:y1, ::-1].T, rep, axis=0)

    def lax2(v, t):                                                              # cut at y = cy (coronal-type); rows = z (base on top)
        return np.repeat(v[t, x0:x1, cy, ::-1].T, rep, axis=0)

    w_cell = 0.62
    h_sax = w_cell
    h_lax = w_cell * (D * rep) / (y1 - y0)
    h_lax2 = w_cell * (D * rep) / (x1 - x0)
    views = {"sag": [("LAX", lax, h_lax)], "cor": [("LAX", lax2, h_lax2)],
             "both": [("LAX 1", lax, h_lax), ("LAX 2", lax2, h_lax2)]}[lax_view]
    groups = [("SAX", sax, h_sax)] + views
    rows = [(lab, cut, vol, kind) for _, cut, _ in groups
            for lab, vol, kind in (("CardioStitch", rc, "img"), ("Target", gt, "img"), ("Error", None, "err"))]
    heights = [h for _, _, h in groups for _ in range(3)]
    gap = 0.08
    fig_w = 0.78 + T * w_cell + 0.42
    fig_h = sum(heights) + gap * (len(groups) - 1) + 0.16 + 0.1                    # top (column titles) + bottom margin
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, top = 0.78, 0.16
    y = top
    for i, (lab, cut, vol, kind) in enumerate(rows):
        if i and i % 3 == 0:
            y += gap
        for t in range(T):
            ax = fig.add_axes([(left + t * w_cell) / fig_w, 1 - (y + heights[i]) / fig_h, w_cell / fig_w, heights[i] / fig_h])
            if kind == "img":
                im = ax.imshow(cut(vol, t), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
            else:
                im = ax.imshow(np.abs(cut(rc, t) - cut(gt, t)), cmap="magma", vmin=0, vmax=ERR_MAX * vmax,
                               interpolation="nearest", aspect="auto")
                err_im = im
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("white"); s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(f"t = {t}", fontsize=6, pad=2)
        fig.text((left - 0.05) / fig_w, 1 - (y + heights[i] / 2) / fig_h, lab, ha="right", va="center", fontsize=6.5)
        y += heights[i]
    for k, (g, _, _) in enumerate(groups):
        a, b = 3 * k, 3 * k + 3
        yc = top + sum(heights[:a]) + gap * k + sum(heights[a:b]) / 2
        fig.text(0.12 / fig_w, 1 - yc / fig_h, g, rotation=90, ha="center", va="center", fontsize=7, fontweight="bold")
    cax = fig.add_axes([(left + T * w_cell + 0.08) / fig_w, 1 - (top + sum(heights[:3])) / fig_h, 0.07 / fig_w, h_sax / fig_h])
    cb = fig.colorbar(err_im, cax=cax, ticks=[0, ERR_MAX * vmax])
    cb.ax.set_yticklabels(["0", f"{ERR_MAX:.1f}"], fontsize=5)
    cb.outline.set_linewidth(0.3); cb.ax.tick_params(length=1.5, width=0.3)
    os.makedirs(out, exist_ok=True)
    stem = f"{out}/{stem_name or f'full_cine_{src}'}"
    fig.savefig(stem + ".png", dpi=200)
    if pdf:
        fig.savefig(stem + ".pdf", dpi=300)
    plt.close(fig)
    print("wrote", stem, f"ref={ref} sax_z={z_sax} D={D} dz={dz}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--more", type=int, default=0, help="also render candidates ranked 2..1+N per dataset into figs/full_cine/more/")
    ap.add_argument("--regular", action="store_true", help="render SUBJECTS from the regular24 cohort into figs/full_cine/regular/")
    ap.add_argument("--lax", default="sag", choices=["sag", "cor", "both"],
                    help="long-axis cut: sagittal-type (fixed x), coronal-type (fixed y), or both")
    ap.add_argument("--rank", type=int, default=1, help="with --regular: render the rank-k candidate per dataset (1 = SUBJECTS)")
    ap.add_argument("--src", nargs="*", help="with --regular: only these datasets")
    ap.add_argument("--paper", action="store_true", help="render PAPER subjects (regular24, coronal LAX) as PNG+PDF into figs/full_cine/paper/")
    args = ap.parse_args()
    if args.select:
        select()
    elif args.paper:
        for src, sub in PAPER:
            render(src, sub, out=os.path.join(OUT, "paper"), arm="regular24", lax_view="cor", pdf=True)
    elif args.regular:
        out = os.path.join(OUT, f"regular_{args.lax}")
        for src, sub in SUBJECTS:
            if args.src and src not in args.src:
                continue
            if args.rank > 1:
                sub = candidates(src)[args.rank - 1][3]
            render(src, sub, out=out, stem_name=f"{src}_rank{args.rank}_{sub}", arm="regular24", lax_view=args.lax)
    elif args.more:
        for src in SRCS:
            for k, (_, _, _, sub) in enumerate(candidates(src)[1:1 + args.more], start=2):
                render(src, sub, out=os.path.join(OUT, "more"), stem_name=f"{src}_rank{k}_{sub}")
    else:
        for src, sub in SUBJECTS:
            render(src, sub)
