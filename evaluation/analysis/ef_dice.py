"""Cohort-agnostic EF + Dice for the 1-frame OOD recons, method-matched via nnU-Net Task114.

Segments the recon (clean & breath arms) AND the clean GT bundle with the SAME segmenter, so
pred-EF-vs-GT-EF isolates recon quality (pseudo-truth, docs/17/24 method). EF is a ratio
(LV_max-LV_min)/LV_max over the cardiac cycle -> the voxel-volume constant cancels, so the 12mm-pitch
caveat (docs/39) does not affect it. Dice is recon-seg vs GT-seg at the GT's ED/ES phases.

Two steps around one nnU-Net call:
  dump  <input_dir>  : copy each per-phase vol (X,Y,Z canonical, already nnU-Net-ready) -> _0000.nii.gz
                       named by INDEX (subject names contain '__'); writes ef_manifest.json.
  score <seg_dir>    : read Task114 segs -> per-subject EF(clean/breath/gt) + Dice; aggregate per cohort.

Full chain (all git-tracked; nnU-Net runs in the isolated `nnunet` env, wrapped by run_seg.sh):
  python evaluation/analysis/ef_dice.py dump  <input_dir> --method <m> --cohorts miitt ocmr acdc
  bash   evaluation/engine/run_seg.sh         <input_dir> <seg_dir>          # nnU-Net Task114 2d
  python evaluation/analysis/ef_dice.py score <seg_dir> --input <input_dir> --out <ef.json>
  python evaluation/analysis/ef_dice.py plot  <ef.json> --out <ef.png>       # EF scatter + Dice bars
"""
import argparse, glob, json, os, shutil, sys
import numpy as np
import nibabel as nib

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

LV, MYO, RV = 1, 2, 3


def method_dir(cohort, subj, method):
    """The recon method dir; contz carries a _contz suffix on OOD cohorts."""
    base = str(paths.subject_dir(cohort, subj))
    for suf in ("", "_contz"):
        d = f"{base}/{method}{suf}"
        if os.path.isdir(d):
            return d
    return None


def subjects(cohort, method):
    return [s for s in paths.subjects(cohort) if method_dir(cohort, s, method)]


def dump(args):
    os.makedirs(args.input_dir, exist_ok=True)
    manifest = []
    for cohort in args.cohorts:
        for sidx, subj in enumerate(subjects(cohort, args.method)):
            md = method_dir(cohort, subj, args.method)
            gts = sorted(glob.glob(str(paths.subject_dir(cohort, subj) / "gt" / "gt_t*.nii.gz")))
            T = len(gts)
            arms = {"gt": [str(paths.bundle_stack(cohort, subj, "gt", t)) for t in range(T)],
                    "clean": [f"{md}/recon_clean/vol_t{t:02d}.nii.gz" for t in range(T)],
                    "breath": [f"{md}/recon_breath/vol_t{t:02d}.nii.gz" for t in range(T)]}
            for arm, files in arms.items():
                for t, f in enumerate(files):
                    if not os.path.isfile(f):
                        continue
                    dst = f"{args.input_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}_0000.nii.gz"
                    shutil.copyfile(f, dst)
            manifest.append({"cohort": cohort, "sidx": sidx, "subject": subj, "T": T})
    json.dump(manifest, open(f"{args.input_dir}/ef_manifest.json", "w"), indent=2)
    print(f"dumped {len(manifest)} subjects -> {args.input_dir}")


def lv_curve(seg_dir, cohort, sidx, arm, T):
    """LV(label 1) voxel count per phase; None for any missing/empty seg."""
    curve = []
    for t in range(T):
        p = f"{seg_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}.nii.gz"
        if not os.path.isfile(p):
            return None
        curve.append(int((np.asarray(nib.load(p).dataobj) == LV).sum()))
    c = np.array(curve, float)
    return c if c.max() > 0 else None


def ef_of(curve):
    return float((curve.max() - curve.min()) / curve.max() * 100.0)


def dice(seg_dir, cohort, sidx, arm, t, gt_t, lab):
    a = np.asarray(nib.load(f"{seg_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}.nii.gz").dataobj) == lab
    b = np.asarray(nib.load(f"{seg_dir}/{cohort}__s{sidx:03d}__gt__t{gt_t:02d}.nii.gz").dataobj) == lab
    inter = np.logical_and(a, b).sum(); s = a.sum() + b.sum()
    return float(2 * inter / s) if s else float("nan")


def score(args):
    man = json.load(open(f"{args.input}/ef_manifest.json"))
    from scipy import stats
    per_cohort = {}
    rows = []
    for m in man:
        c, sidx, subj, T = m["cohort"], m["sidx"], m["subject"], m["T"]
        gt = lv_curve(args.seg_dir, c, sidx, "gt", T)
        cl = lv_curve(args.seg_dir, c, sidx, "clean", T)
        br = lv_curve(args.seg_dir, c, sidx, "breath", T)
        if gt is None:
            continue
        ed, es = int(gt.argmax()), int(gt.argmin())
        r = {"cohort": c, "subject": subj, "ef_gt": ef_of(gt),
             "ef_clean": ef_of(cl) if cl is not None else None,
             "ef_breath": ef_of(br) if br is not None else None}
        for arm, cur in [("clean", cl), ("breath", br)]:
            if cur is None:
                continue
            for name, lab in [("LV", LV), ("MYO", MYO), ("RV", RV)]:
                r[f"dice_{arm}_{name}_ED"] = dice(args.seg_dir, c, sidx, arm, ed, ed, lab)
                r[f"dice_{arm}_{name}_ES"] = dice(args.seg_dir, c, sidx, arm, es, es, lab)
        rows.append(r); per_cohort.setdefault(c, []).append(r)

    agg = {}
    for c, rs in per_cohort.items():
        d = {"n": len(rs)}
        for arm in ("clean", "breath"):
            g = np.array([x["ef_gt"] for x in rs if x.get(f"ef_{arm}") is not None])
            p = np.array([x[f"ef_{arm}"] for x in rs if x.get(f"ef_{arm}") is not None])
            if len(g) >= 3 and g.std() > 0:
                sl = float(np.polyfit(g, p, 1)[0])
                d[f"{arm}_ef_slope"] = sl
                d[f"{arm}_ef_spearman"] = float(stats.spearmanr(g, p).correlation)
                d[f"{arm}_ef_mae_pct"] = float(np.mean(np.abs(p - g)))
            for name in ("LV", "MYO", "RV"):
                for ph in ("ED", "ES"):
                    vals = [x[f"dice_{arm}_{name}_{ph}"] for x in rs
                            if f"dice_{arm}_{name}_{ph}" in x and not np.isnan(x[f"dice_{arm}_{name}_{ph}"])]
                    if vals:
                        d[f"{arm}_dice_{name}_{ph}"] = float(np.mean(vals))
        agg[c] = d
    json.dump({"aggregate": agg, "per_subject": rows}, open(args.out, "w"), indent=2)
    print(json.dumps(agg, indent=2))
    print(f"\n-> {args.out}")


def plot(args):
    """Visualize a score() JSON: per-cohort EF scatter (pred vs GT, identity + fitted slope) on top,
    Dice bars (LV/MYO/RV at ED/ES) below. clean vs breath overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open(args.input))
    agg, rows = data["aggregate"], data["per_subject"]
    cohorts = sorted(agg)
    arms = ["clean", "breath"] if args.arm == "both" else [args.arm]
    color = {"clean": "#1f77b4", "breath": "#d62728"}
    names, phases = ("LV", "MYO", "RV"), ("ED", "ES")

    nc = len(cohorts)
    fig, axes = plt.subplots(2, nc, figsize=(4.2 * nc, 7.8), squeeze=False)
    for ci, c in enumerate(cohorts):
        crows = [r for r in rows if r["cohort"] == c]
        ax = axes[0, ci]
        lo, hi = 100.0, 0.0
        for ai, arm in enumerate(arms):
            g = np.array([r["ef_gt"] for r in crows if r.get(f"ef_{arm}") is not None])
            p = np.array([r[f"ef_{arm}"] for r in crows if r.get(f"ef_{arm}") is not None])
            if not len(g):
                continue
            ax.scatter(g, p, s=22, c=color[arm], alpha=0.8, label=arm, edgecolor="none")
            lo, hi = min(lo, g.min(), p.min()), max(hi, g.max(), p.max())
            sl, sp, mae = (agg[c].get(f"{arm}_ef_slope"), agg[c].get(f"{arm}_ef_spearman"),
                           agg[c].get(f"{arm}_ef_mae_pct"))
            if sl is not None:
                xs = np.array([g.min(), g.max()])
                b = float(p.mean() - sl * g.mean())
                ax.plot(xs, sl * xs + b, c=color[arm], lw=1.4)
                ax.annotate(f"{arm}: slope {sl:.2f}, ρ {sp:.2f}, MAE {mae:.1f}%",
                            xy=(0.03, 0.95 - 0.07 * ai), xycoords="axes fraction",
                            fontsize=7.5, color=color[arm])
        pad = 0.05 * (hi - lo + 1e-6)
        lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, "--", c="0.6", lw=1, zorder=0)                 # identity
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(f"{c}  (n={agg[c]['n']})", fontsize=9)
        ax.set_xlabel("GT EF (%)", fontsize=8); ax.set_ylabel("pred EF (%)", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")

        axd = axes[1, ci]
        labels = [f"{n}\n{ph}" for n in names for ph in phases]
        x = np.arange(len(labels)); wbar = 0.8 / len(arms)
        for ai, arm in enumerate(arms):
            vals = [agg[c].get(f"{arm}_dice_{n}_{ph}", np.nan) for n in names for ph in phases]
            axd.bar(x + ai * wbar, vals, wbar, color=color[arm], label=arm, alpha=0.85)
        axd.set_xticks(x + wbar * (len(arms) - 1) / 2); axd.set_xticklabels(labels, fontsize=7)
        axd.set_ylim(0, 1); axd.set_ylabel("Dice", fontsize=8); axd.set_title(f"{c} Dice", fontsize=9)
        axd.legend(fontsize=7)

    fig.suptitle(f"EF recovery + Dice — {os.path.basename(args.input)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160); plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("input_dir")
    d.add_argument("--method", default="vggt_20260719_1f_gather05_ep99")
    d.add_argument("--cohorts", nargs="+", default=["miitt", "ocmr", "acdc"])
    s = sub.add_parser("score"); s.add_argument("seg_dir")
    s.add_argument("--input", required=True); s.add_argument("--out", required=True)
    pl = sub.add_parser("plot"); pl.add_argument("input", help="a score() output json")
    pl.add_argument("--arm", choices=["clean", "breath", "both"], default="both")
    pl.add_argument("--out", required=True)
    a = ap.parse_args()
    {"dump": dump, "score": score, "plot": plot}[a.cmd](a)
