"""Cohort-agnostic EF + Dice for the 1-frame OOD recons, method-matched via nnU-Net Task114.

Segments the recon (clean & breath arms) AND the clean GT bundle with the SAME segmenter, so
pred-EF-vs-GT-EF isolates recon quality (pseudo-truth, docs/17/24 method). EF is a ratio
(LV_max-LV_min)/LV_max over the cardiac cycle -> the voxel-volume constant cancels, so the 12mm-pitch
caveat (docs/39) does not affect it. Dice is recon-seg vs GT-seg at the GT's ED/ES phases.

Two steps around one nnU-Net call:
  dump  <input_dir>  : copy each per-phase vol (X,Y,Z canonical, already nnU-Net-ready) -> _0000.nii.gz
                       named by INDEX (subject names contain '__'); writes ef_manifest.json.
  score <seg_dir>    : read Task114 segs -> per-subject EF(clean/breath/gt) + Dice; aggregate per cohort.

Usage (see sbatch/ef_dice_ood.sh for the full chain):
  python tools/ef_dice_1frame.py dump  <input_dir> --method <m> --cohorts miitt ocmr acdc
  # nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS -i <input_dir> -o <seg_dir>   (nnunet env)
  python tools/ef_dice_1frame.py score <seg_dir> --input <input_dir> --out <ef.json>
"""
import argparse, glob, json, os, shutil
import numpy as np
import nibabel as nib

E = "/home/minsukc/vggt/scratch/eval"
LV, MYO, RV = 1, 2, 3


def method_dir(cohort, subj, method):
    """The recon method dir; contz carries a _contz suffix on OOD cohorts."""
    for suf in ("", "_contz"):
        d = f"{E}/{cohort}/out/{subj}/{method}{suf}"
        if os.path.isdir(d):
            return d
    return None


def subjects(cohort, method):
    out = []
    for d in sorted(glob.glob(f"{E}/{cohort}/out/*")):
        s = os.path.basename(d)
        if method_dir(cohort, s, method):
            out.append(s)
    return out


def dump(args):
    os.makedirs(args.input_dir, exist_ok=True)
    manifest = []
    for cohort in args.cohorts:
        for sidx, subj in enumerate(subjects(cohort, args.method)):
            md = method_dir(cohort, subj, args.method)
            gts = sorted(glob.glob(f"{E}/{cohort}/out/{subj}/gt/gt_t*.nii.gz"))
            T = len(gts)
            arms = {"gt": [f"{E}/{cohort}/out/{subj}/gt/gt_t{t:02d}.nii.gz" for t in range(T)],
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("input_dir")
    d.add_argument("--method", default="vggt_20260719_1f_gather05_ep99")
    d.add_argument("--cohorts", nargs="+", default=["miitt", "ocmr", "acdc"])
    s = sub.add_parser("score"); s.add_argument("seg_dir")
    s.add_argument("--input", required=True); s.add_argument("--out", required=True)
    a = ap.parse_args()
    (dump if a.cmd == "dump" else score)(a)
