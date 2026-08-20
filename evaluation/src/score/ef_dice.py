"""Cohort-agnostic biventricular metrics for the 1-frame recons, method-matched via nnU-Net Task114.

Segments the recon (clean & breath arms) AND the GT with the SAME segmenter, so pred-vs-GT
isolates recon quality (pseudo-truth, docs/17/24 method). Metrics per subject:
  EF (LV + RV), EDV/ESV (mL), LVM (g, MYO x 1.05 g/mL at ED), Dice + HD95 (mm) for LV/MYO/RV
  at the GT's ED/ES phases. EF is a ratio -> voxel-volume cancels (12mm-pitch caveat docs/39
  doesn't bite); EDV/ESV/LVM use the seg's own voxel volume, so they DO carry that caveat —
  method-matched vs GT, they remain fair comparisons.

Inputs come from the SCORED cines (`<arm>/cine_{clean,breath}.nii.gz` + `<subj>/cine_gt.nii.gz`,
written by score/image_metrics.py) — NOT the raw recon volumes — so segmentation sees exactly the
gauged/pose-corrected/PSF'd volume the image metrics scored, and Dice inherits the registration.

Two steps around one nnU-Net call:
  dump  <input_dir>  : slice each 4D cine into per-phase _0000.nii.gz named by INDEX (subject
                       names contain '__'); writes ef_manifest.json (records the method).
  score <seg_dir>    : read Task114 segs -> per-subject metrics; aggregate per cohort.

Full chain (all git-tracked; nnU-Net runs in the isolated `nnunet` env, wrapped by run_seg.sh):
  python evaluation/src/score/ef_dice.py dump  <input_dir> --method <m> --cohorts miitt ocmr acdc
  bash   evaluation/src/engine/run_seg.sh         <input_dir> <seg_dir>      # nnU-Net Task114 2d
  python evaluation/src/score/ef_dice.py score <seg_dir> --input <input_dir>
                                       # -> metric_results/_ef/<m>.json (per-cohort merge on re-runs)
  python evaluation/src/score/ef_dice.py plot  metric_results/_ef/<m>.json --out <ef.png>
Then re-run score/aggregate.py (or run.py) to fold the _ef file into metric_results/<ds>/<m>.json.
"""
import argparse, glob, json, os, sys
import numpy as np
import nibabel as nib

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

LV, MYO, RV = 1, 2, 3


def method_dir(cohort, subj, method):
    """The recon method dir — EXACT name only. (It used to probe a legacy `_contz` suffix,
    which made the `_ef/<method>.json` key diverge from the arm name aggregate.py joins on —
    the EF block then silently never folded. A contz arm must be dumped under its full
    literal dir name.)"""
    d = paths.arm_dir(cohort, subj, method)
    return str(d) if d.is_dir() else None


def subjects(cohort, method):
    # Same split rail as run_vggt/aggregate: a stray train/test bundle in out/ must not join
    # the citable EF cohort (paths.filter_by_split contract).
    split = os.environ.get("SPLIT", "val")
    keep, dropped = paths.filter_by_split(cohort, paths.subjects(cohort), split)
    for subj, why in dropped:
        print(f"  !! {cohort}: skipping {subj}: {why}", file=sys.stderr)
    return [s for s in keep if method_dir(cohort, s, method)]


def _dump_cine(cine_path, input_dir, cohort, sidx, arm):
    """Slice one 4D cine (X,Y,Z,T) into per-phase _0000.nii.gz for nnU-Net. Returns T or 0."""
    if not os.path.isfile(cine_path):
        return 0
    img = nib.load(str(cine_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    for t in range(vol.shape[3]):
        nib.save(nib.Nifti1Image(vol[..., t], img.affine),
                 f"{input_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}_0000.nii.gz")
    return vol.shape[3]


def dump(args):
    # sidx is an ENUMERATION index and the seg filenames carry only it — leftovers from an
    # earlier dump with a different subject set would silently attach to the wrong subject at
    # score time. Refuse a dirty dir instead of trusting the caller to have cleared it.
    if os.path.isdir(args.input_dir) and any(f.endswith(".nii.gz") for f in os.listdir(args.input_dir)):
        sys.exit(f"dump: {args.input_dir} already contains .nii.gz files from an earlier dump — "
                 f"sidx-keyed names would mis-attribute leftovers to the wrong subject. Use a fresh dir.")
    os.makedirs(args.input_dir, exist_ok=True)
    manifest, meta = [], {"method": args.method}
    for cohort in args.cohorts:
        for sidx, subj in enumerate(subjects(cohort, args.method)):
            md = method_dir(cohort, subj, args.method)
            # Segment the SCORED volumes, not the raw recons: cine_* carries the exact
            # gauge/pose/PSF treatment image_metrics scored. Missing cine => that arm was
            # never scored — skip it rather than silently falling back to raw recons.
            gt0_mtime = os.path.getmtime(paths.bundle_stack(cohort, subj, "gt", 0))
            T = _dump_cine(paths.cine_gt(cohort, subj), args.input_dir, cohort, sidx, "gt")
            if T == 0:
                print(f"  !! {cohort}/{subj}: no cine_gt.nii.gz — run score/image_metrics.py first; skipped",
                      file=sys.stderr)
                continue
            for arm in ("clean", "breath"):
                cine = f"{md}/cine_{arm}.nii.gz"
                # Freshness: a cine scored BEFORE a bundle rebuild would be segmented against
                # the rebuilt GT (the T-count guard can't see a same-T rebuild). Same mtime
                # rule image_metrics uses for cine_gt.
                if os.path.isfile(cine) and os.path.getmtime(cine) < gt0_mtime:
                    print(f"  !! {cohort}/{subj} [{arm}]: cine_{arm} is OLDER than the gt bundle "
                          f"(rebuilt since scoring?) — re-run image_metrics; arm skipped", file=sys.stderr)
                    continue
                n = _dump_cine(cine, args.input_dir, cohort, sidx, arm)
                if n not in (0, T):
                    sys.exit(f"{cohort}/{subj} [{arm}]: cine has {n} phases but GT has {T} — stale cine?")
            manifest.append({"cohort": cohort, "sidx": sidx, "subject": subj, "T": T})
    json.dump({"meta": meta, "subjects": manifest}, open(f"{args.input_dir}/ef_manifest.json", "w"), indent=2)
    print(f"dumped {len(manifest)} subjects -> {args.input_dir}")


def seg_path(seg_dir, cohort, sidx, arm, t):
    return f"{seg_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}.nii.gz"


def curve(seg_dir, cohort, sidx, arm, T, lab=LV):
    """(per-phase voxel count of `lab`, voxel volume mm^3); (None, None) for missing/empty segs."""
    counts, voxmm3 = [], None
    for t in range(T):
        p = seg_path(seg_dir, cohort, sidx, arm, t)
        if not os.path.isfile(p):
            return None, None
        img = nib.load(p)
        voxmm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
        counts.append(int((np.asarray(img.dataobj) == lab).sum()))
    c = np.array(counts, float)
    return (c, voxmm3) if c.max() > 0 else (None, None)


def ef_of(curve):
    """EF %, or None when the structure VANISHES at some phase (count 0 -> EF would read a
    clinically-impossible 100%; realistic when nnU-Net drops the RV at ES on a blurry recon).
    Callers' `is not None` gating drops the value from MAEs instead of polluting them."""
    if curve.min() <= 0:
        return None
    return float((curve.max() - curve.min()) / curve.max() * 100.0)


def vols_of(curve, voxmm3):
    """(EDV, ESV) in mL from a voxel-count curve. ESV is None when the structure vanishes
    (a 0-mL ESV is a seg failure, not physiology). Carries the voxel-volume caveat (docstring)."""
    edv = float(curve.max() * voxmm3 / 1000.0)
    esv = float(curve.min() * voxmm3 / 1000.0) if curve.min() > 0 else None
    return edv, esv


def lvm_of(seg_dir, cohort, sidx, arm, ed_t, voxmm3):
    """LV mass (g) = MYO voxels at ED x voxel volume x 1.05 g/mL. None if the seg is missing."""
    p = seg_path(seg_dir, cohort, sidx, arm, ed_t)
    if not os.path.isfile(p) or voxmm3 is None:
        return None
    n = int((np.asarray(nib.load(p).dataobj) == MYO).sum())
    return float(n * voxmm3 / 1000.0 * 1.05) if n else None


def dice(seg_dir, cohort, sidx, arm, t, gt_t, lab):
    a = np.asarray(nib.load(seg_path(seg_dir, cohort, sidx, arm, t)).dataobj) == lab
    b = np.asarray(nib.load(seg_path(seg_dir, cohort, sidx, "gt", gt_t)).dataobj) == lab
    inter = np.logical_and(a, b).sum(); s = a.sum() + b.sum()
    return float(2 * inter / s) if s else float("nan")


def hd95(seg_dir, cohort, sidx, arm, t, gt_t, lab):
    """95th-percentile symmetric surface distance (mm) between recon-seg and GT-seg.
    NaN if either mask is empty (no surface to measure)."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    ia = nib.load(seg_path(seg_dir, cohort, sidx, arm, t))
    ib = nib.load(seg_path(seg_dir, cohort, sidx, "gt", gt_t))
    a = np.asarray(ia.dataobj) == lab
    b = np.asarray(ib.dataobj) == lab
    if not a.any() or not b.any():
        return float("nan")
    spacing = ia.header.get_zooms()[:3]
    sa = a & ~binary_erosion(a)                      # surface voxels
    sb = b & ~binary_erosion(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]   # a-surface -> b-surface
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def score(args):
    man = json.load(open(f"{args.input}/ef_manifest.json"))
    # dump() writes {"meta": {...}, "subjects": [...]}; a legacy dump is a bare list.
    meta = man.get("meta", {}) if isinstance(man, dict) else {}
    subj_list = man["subjects"] if isinstance(man, dict) else man
    out = args.out or (str(paths.ef_summary(meta["method"])) if meta.get("method") else None)
    if not out:
        sys.exit("score: --out required (legacy manifest carries no method for the default path)")
    from scipy import stats
    # Leftover segs from an EARLIER dump into the same seg_dir carry sidx's from a different
    # subject enumeration — they would be silently attributed to the wrong subject. Segs must
    # postdate the manifest that names them.
    man_mtime = os.path.getmtime(f"{args.input}/ef_manifest.json")
    per_cohort = {}
    rows = []
    for m in subj_list:
        c, sidx, subj, T = m["cohort"], m["sidx"], m["subject"], m["T"]
        for f in glob.glob(seg_path(args.seg_dir, c, sidx, "*", 0).replace("t00", "t*")):
            if os.path.getmtime(f) < man_mtime:
                sys.exit(f"score: {f} predates the dump manifest — stale seg from an earlier "
                         f"dump with a different subject numbering. Re-run run_seg.sh on a fresh seg_dir.")
        gt, gtvox = curve(args.seg_dir, c, sidx, "gt", T)
        if gt is None:
            print(f"  !! {c}/{subj}: GT LV seg missing/empty — subject dropped from the EF cohort",
                  file=sys.stderr)
            continue
        ed, es = int(gt.argmax()), int(gt.argmin())
        gt_rv, _ = curve(args.seg_dir, c, sidx, "gt", T, lab=RV)
        r = {"cohort": c, "subject": subj, "ef_gt": ef_of(gt)}
        r["edv_gt"], r["esv_gt"] = vols_of(gt, gtvox)
        r["lvm_gt"] = lvm_of(args.seg_dir, c, sidx, "gt", ed, gtvox)
        if gt_rv is not None:
            r["rv_ef_gt"] = ef_of(gt_rv)
            r["rv_edv_gt"], r["rv_esv_gt"] = vols_of(gt_rv, gtvox)
        for arm in ("clean", "breath"):
            # LV, RV, and Dice/HD95 gate INDEPENDENTLY — an unsegmentable LV must not
            # suppress a valid RV, and overlap metrics only need the ED/ES segs to exist.
            cur, vox = curve(args.seg_dir, c, sidx, arm, T)
            if cur is not None:
                r[f"ef_{arm}"] = ef_of(cur)
                r[f"edv_{arm}"], r[f"esv_{arm}"] = vols_of(cur, vox)
                r[f"lvm_{arm}"] = lvm_of(args.seg_dir, c, sidx, arm, ed, vox)
            rv, rvvox = curve(args.seg_dir, c, sidx, arm, T, lab=RV)
            if rv is not None:
                r[f"rv_ef_{arm}"] = ef_of(rv)
                r[f"rv_edv_{arm}"], r[f"rv_esv_{arm}"] = vols_of(rv, rvvox)
            if all(os.path.isfile(seg_path(args.seg_dir, c, sidx, arm, t)) for t in {ed, es}):
                for name, lab in [("LV", LV), ("MYO", MYO), ("RV", RV)]:
                    r[f"dice_{arm}_{name}_ED"] = dice(args.seg_dir, c, sidx, arm, ed, ed, lab)
                    r[f"dice_{arm}_{name}_ES"] = dice(args.seg_dir, c, sidx, arm, es, es, lab)
                    r[f"hd95_{arm}_{name}_ED"] = hd95(args.seg_dir, c, sidx, arm, ed, ed, lab)
                    r[f"hd95_{arm}_{name}_ES"] = hd95(args.seg_dir, c, sidx, arm, es, es, lab)
        rows.append(r); per_cohort.setdefault(c, []).append(r)

    agg = {}
    for c, rs in per_cohort.items():
        d = {"n": len(rs)}
        for arm in ("clean", "breath"):
            # both sides may be None now (vanishing-seg guard) — pair only complete rows
            ok = [x for x in rs if x.get("ef_gt") is not None and x.get(f"ef_{arm}") is not None]
            g = np.array([x["ef_gt"] for x in ok])
            p = np.array([x[f"ef_{arm}"] for x in ok])
            if len(g) >= 3 and g.std() > 0:
                sl = float(np.polyfit(g, p, 1)[0])
                d[f"{arm}_ef_slope"] = sl
                d[f"{arm}_ef_spearman"] = float(stats.spearmanr(g, p).correlation)
                d[f"{arm}_ef_mae_pct"] = float(np.mean(np.abs(p - g)))
            # paired MAE vs the same subject's GT value, for each absolute-volume metric
            for key, unit in [("edv", "ml"), ("esv", "ml"), ("lvm", "g"),
                              ("rv_ef", "pct"), ("rv_edv", "ml"), ("rv_esv", "ml")]:
                pairs = [(x[f"{key}_gt"], x[f"{key}_{arm}"]) for x in rs
                         if x.get(f"{key}_gt") is not None and x.get(f"{key}_{arm}") is not None]
                if pairs:
                    gg, pp = map(np.array, zip(*pairs))
                    d[f"{arm}_{key}_mae_{unit}"] = float(np.mean(np.abs(pp - gg)))
            for name in ("LV", "MYO", "RV"):
                for ph in ("ED", "ES"):
                    for met in ("dice", "hd95"):
                        vals = [x[f"{met}_{arm}_{name}_{ph}"] for x in rs
                                if f"{met}_{arm}_{name}_{ph}" in x
                                and not np.isnan(x[f"{met}_{arm}_{name}_{ph}"])]
                        if vals:
                            d[f"{arm}_{met}_{name}_{ph}"] = float(np.mean(vals))
        agg[c] = d
    # NaN -> null (dice/hd95 legitimately NaN on empty masks; this is a git-tracked citable
    # file and must stay strict-JSON, same policy as aggregate.py's summaries).
    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    # MERGE with an existing file per cohort — this is ONE cross-cohort file per arm, and a
    # partial-cohort re-run (e.g. --cohorts miitt) must update only its own cohorts, not
    # silently erase the others (the next aggregate re-fold would null their EF blocks).
    prev = {}
    if os.path.isfile(out):
        try:
            prev = json.load(open(out))
        except json.JSONDecodeError:
            print(f"  !! existing {out} unreadable — replacing it wholesale", file=sys.stderr)
    kept = sorted(set(prev.get("aggregate", {})) - set(agg))
    if kept:
        print(f"  merging: keeping previous results for cohort(s) {', '.join(kept)}")
        agg = {**prev["aggregate"], **agg}
        rows = [r for r in prev.get("per_subject", []) if r.get("cohort") in kept] + rows
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(json_safe({"meta": meta, "aggregate": agg, "per_subject": rows}),
              open(out, "w"), indent=2, allow_nan=False)
    print(json.dumps(json_safe(agg), indent=2))
    print(f"\n-> {out}")


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
    d.add_argument("--method", default="vggt_augaggr224hw2_ep300")
    d.add_argument("--cohorts", nargs="+", default=list(paths.DATASETS))
    s = sub.add_parser("score"); s.add_argument("seg_dir")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default=None,
                   help="default: paths.ef_summary(<method from the dump manifest>) — the "
                        "location score/aggregate.py merges from")
    pl = sub.add_parser("plot"); pl.add_argument("input", help="a score() output json")
    pl.add_argument("--arm", choices=["clean", "breath", "both"], default="both")
    pl.add_argument("--out", required=True)
    a = ap.parse_args()
    {"dump": dump, "score": score, "plot": plot}[a.cmd](a)
