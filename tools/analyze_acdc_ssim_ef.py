"""Analyze + plot ACDC EF for the SSIM refiner (target_t) model: predicted EF (nnU-Net on
V_refined, swept over target_t) vs GT EF (nnU-Net on the real ACDC phases, from
scratch/analysis/phase_analysis/acdc_analysis.json). Same correlation plot as the 30-val FIG 1.

Run: micromamba run -n svr python tools/analyze_acdc_ssim_ef.py
"""
import os, sys, glob, re, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEG = os.path.join(_ROOT, "scratch/analysis/phase_analysis_acdc_ssim/pred_segs")
GT_JSON = os.path.join(_ROOT, "scratch/analysis/phase_analysis/acdc_analysis.json")
OUT_PNG = os.path.join(_ROOT, "result", "acdc_ssim_ef_correlation.png")
OUT_JSON = os.path.join(_ROOT, "scratch/analysis/phase_analysis_acdc_ssim/acdc_ssim_ef.json")
VOX_ML = 1.4 * 1.4 * 12.0 / 1000.0
T = 12
LV = 1


def pred_lv_curve(subj):
    c = np.full(T, np.nan)
    for t in range(T):
        f = os.path.join(SEG, f"{subj}_t{t:02d}.nii.gz")
        if os.path.exists(f):
            c[t] = (np.asarray(nib.load(f).dataobj) == LV).sum() * VOX_ML
    return c


def main():
    gt = {r["subj"]: r["ef"] for r in json.load(open(GT_JSON))["per_subj"]}
    subjects = sorted({re.match(r"(patient\d+)_t\d{2}$", os.path.basename(s)[:-7]).group(1)
                       for s in glob.glob(os.path.join(SEG, "*.nii.gz"))
                       if re.match(r"(patient\d+)_t\d{2}$", os.path.basename(s)[:-7])})
    rows = []
    for s in subjects:
        if s not in gt:
            continue
        pc = pred_lv_curve(s)
        if np.isnan(pc).any() or (pc <= 0).any():
            continue
        rows.append(dict(subj=s, pred_ef=float((pc.max() - pc.min()) / pc.max() * 100),
                         gt_ef=float(gt[s]), pred=pc.tolist(),
                         pred_es=int(pc.argmin()), pred_ed=int(pc.argmax())))
    g = np.array([r["gt_ef"] for r in rows]); p = np.array([r["pred_ef"] for r in rows])
    slope, intc = np.polyfit(g, p, 1)
    pr = pearsonr(g, p)[0]; sp = spearmanr(g, p)[0]
    json.dump(dict(n=len(rows), slope=float(slope), intercept=float(intc),
                   pearson=float(pr), spearman=float(sp),
                   pred_ef_mean=float(p.mean()), pred_ef_std=float(p.std()),
                   gt_ef_mean=float(g.mean()), gt_ef_std=float(g.std()), rows=rows),
              open(OUT_JSON, "w"), indent=2)

    print(f"N={len(rows)} ACDC patients")
    print(f"pred-EF vs GT-EF: slope={slope:+.3f} intercept={intc:+.2f} "
          f"Pearson r={pr:+.3f} Spearman={sp:+.3f}")
    print(f"GT EF {g.mean():.1f}±{g.std():.1f}% (range {g.min():.0f}-{g.max():.0f})  |  "
          f"pred EF {p.mean():.1f}±{p.std():.1f}% (range {p.min():.0f}-{p.max():.0f})")

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.scatter(g, p, c="#9467bd", s=42, alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
    xs = np.linspace(0, 100, 50)
    ax.plot(xs, slope * xs + intc, color="#9467bd", lw=2.2,
            label=f"fit: slope={slope:+.2f}, r={pr:+.2f}")
    ax.plot(xs, xs, "--", color="0.55", lw=1, label="identity (slope 1)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("true EF (%)  [nnU-Net on real ACDC phases]")
    ax.set_ylabel("predicted EF (%)  [nnU-Net on V_refined]")
    ax.set_title(f"ACDC (N={len(rows)}) — SSIM refiner (target_t) EF\n"
                 f"Pearson r={pr:+.2f}  Spearman={sp:+.2f}  slope={slope:+.2f}", fontsize=11)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.25)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
