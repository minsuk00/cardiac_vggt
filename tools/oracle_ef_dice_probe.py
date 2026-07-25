"""Oracle-geometry EF/Dice gate (the debate's decisive test).

QUESTION: EF SLOPE is a free calibration bias; the real target is EF RANKING (Spearman ~0.55
for the 4wok model) and Dice — which a motion PRIOR cannot move (no new info). Does ORACLE
geometry (renderer B: subject template pooled via true elastix DVF, backward-gathered to each
phase) recover EF-ranking / Dice ABOVE the current model? This upper-bounds ANY motion/template/
geometry architecture: if even oracle geometry ≈ the model, EF/Dice are at the in-contract ceiling.

Renderers (reused from oracle_transport_probe, reference-slot ON = deployment): GT (=true phase
volume, pseudo-truth), F (identity floor), B (oracle transport), O (perfect oracle, sanity).
Score = nnU-Net Task114 LV(label1) cavity volume at ED & ES -> EF=(EDV-ESV)/EDV per subject;
Spearman(EF_renderer, EF_GT) across subjects; Dice(seg_renderer, seg_GT) at ED/ES per label.

  dump  <dir>  : render F/B/O/GT at ED&ES for all val subjects -> <dir>/{sidx}__{r}__{ph}_0000.nii.gz
  # nnUNet_predict -t 114 -m 2d -tr nnUNetTrainerV2_MMS -i <dir> -o <seg_dir>   (nnunet env)
  score <seg_dir> --input <dir>
"""
import argparse, glob, json, os
import numpy as np
import nibabel as nib

import sys
sys.path.insert(0, "vggt"); sys.path.insert(0, ".")
from tools.oracle_transport_probe import load_subject, run_case, find_ES, val_subjects  # noqa

LV, MYO, RV = 1, 2, 3
RENDERERS = ["gt", "F", "B", "O"]


def dump(args):
    os.makedirs(args.input_dir, exist_ok=True)
    subs = val_subjects(args.n)
    manifest = []
    for sidx, (sid, sax) in enumerate(subs):
        P, D, roi = load_subject(sax)
        aff = nib.load(sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))[0]).affine
        es = find_ES(P)
        phases = {"ED": 0, "ES": es}
        for ph, tt in phases.items():
            res, (VF, VB, VO, Vgt, _) = run_case(P, D, roi, tt, seed=1000 * sidx + tt, reference=True)
            vols = {"gt": Vgt, "F": VF, "B": VB, "O": VO}
            for r, v in vols.items():
                arr = v.cpu().numpy().astype(np.float32)
                dst = f"{args.input_dir}/{sidx:03d}__{r}__{ph}_0000.nii.gz"
                nib.save(nib.Nifti1Image(arr, aff), dst)
        manifest.append({"sidx": sidx, "subject": sid, "ED": 0, "ES": es})
        print(f"  [{sidx:2d}] {sid:>12} ED=0 ES={es} rendered")
    json.dump(manifest, open(f"{args.input_dir}/manifest.json", "w"), indent=2)
    print(f"dumped {len(manifest)} subjects x {len(RENDERERS)} renderers x 2 phases -> {args.input_dir}")


def lv_count(seg_dir, sidx, r, ph, lab=LV):
    p = f"{seg_dir}/{sidx:03d}__{r}__{ph}.nii.gz"
    if not os.path.isfile(p):
        return None
    s = np.asarray(nib.load(p).dataobj)
    return int((s == lab).sum())


def dice(seg_dir, sidx, r, ph, lab):
    a = f"{seg_dir}/{sidx:03d}__{r}__{ph}.nii.gz"
    b = f"{seg_dir}/{sidx:03d}__gt__{ph}.nii.gz"
    if not (os.path.isfile(a) and os.path.isfile(b)):
        return np.nan
    A = np.asarray(nib.load(a).dataobj) == lab
    B = np.asarray(nib.load(b).dataobj) == lab
    d = A.sum() + B.sum()
    return float(2 * (A & B).sum() / d) if d > 0 else np.nan


def score(args):
    from scipy import stats
    manifest = json.load(open(f"{args.input_dir}/manifest.json"))
    ef = {r: [] for r in RENDERERS}
    ef_gt = []
    dices = {r: {f"{n}_{ph}": [] for n in ["LV", "MYO", "RV"] for ph in ["ED", "ES"]}
             for r in ["F", "B", "O"]}
    for m in manifest:
        sidx = m["sidx"]
        # EF per renderer = (LV_ED - LV_ES)/LV_ED
        efs = {}
        ok = True
        for r in RENDERERS:
            ved = lv_count(args.seg_dir, sidx, r, "ED")
            ves = lv_count(args.seg_dir, sidx, r, "ES")
            if ved is None or ves is None or ved == 0:
                ok = False; efs[r] = np.nan; continue
            efs[r] = 100.0 * (ved - ves) / ved
        if not ok or not np.isfinite(efs["gt"]):
            continue
        ef_gt.append(efs["gt"])
        for r in RENDERERS:
            ef[r].append(efs[r])
        for r in ["F", "B", "O"]:
            for name, lab in [("LV", LV), ("MYO", MYO), ("RV", RV)]:
                for ph in ["ED", "ES"]:
                    dices[r][f"{name}_{ph}"].append(dice(args.seg_dir, sidx, r, ph, lab))

    ef_gt = np.array(ef_gt)
    print(f"\n=== ORACLE EF/Dice GATE — n={len(ef_gt)} subjects (reference-slot ON) ===")
    print(f"GT EF: mean {ef_gt.mean():.1f}  std {ef_gt.std():.1f}\n")
    print(f"{'renderer':>10} {'EF_mean':>8} {'EF_std':>7} {'Spearman':>9} {'Pearson':>8} {'MAE':>6}")
    for r in RENDERERS:
        p = np.array(ef[r]); mm = np.isfinite(p) & np.isfinite(ef_gt)
        if mm.sum() < 3:
            print(f"{r:>10}  (insufficient)"); continue
        sp = stats.spearmanr(ef_gt[mm], p[mm]).correlation
        pe = stats.pearsonr(ef_gt[mm], p[mm])[0]
        mae = np.abs(p[mm] - ef_gt[mm]).mean()
        print(f"{r:>10} {p[mm].mean():8.1f} {p[mm].std():7.1f} {sp:9.3f} {pe:8.3f} {mae:6.1f}")
    print(f"\n  reference: 4wok MODEL EF-Spearman = 0.552 (docs/33). "
          f"GO if oracle-B Spearman >> model; NO-GO if ~equal (EF ranking at contract ceiling).")
    print(f"\n=== Dice vs GT-seg (renderer B = oracle geometry) ===")
    print(f"{'renderer':>10} " + " ".join(f"{k:>8}" for k in ["LV_ED", "MYO_ED", "RV_ED", "LV_ES", "MYO_ES", "RV_ES"]))
    for r in ["F", "B", "O"]:
        vals = []
        for ph in ["ED", "ES"]:
            for name in ["LV", "MYO", "RV"]:
                v = [x for x in dices[r][f"{name}_{ph}"] if np.isfinite(x)]
                vals.append(np.mean(v) if v else np.nan)
        # reorder to LV_ED MYO_ED RV_ED LV_ES...
        order = [0, 1, 2, 3, 4, 5]
        print(f"{r:>10} " + " ".join(f"{vals[i]:8.3f}" for i in order))
    json.dump({"ef": ef, "ef_gt": list(ef_gt)}, open(f"{args.input_dir}/ef_dice_results.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("input_dir"); d.add_argument("--n", type=int, default=30)
    s = sub.add_parser("score"); s.add_argument("seg_dir"); s.add_argument("--input", dest="input_dir", required=True)
    args = ap.parse_args()
    (dump if args.cmd == "dump" else score)(args)
