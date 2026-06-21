"""Dice of M&Ms-nnU-Net predictions vs ACDC human GT (label conventions remapped).

Model (Task114) labels: 1=LV pool, 2=MYO, 3=RV pool.
ACDC GT labels:         1=RV pool, 2=MYO, 3=LV pool.
So per structure:  LV: pred==1 vs gt==3 | MYO: pred==2 vs gt==2 | RV: pred==3 vs gt==1.

Reports per-structure Dice averaged over cases, split by ED/ES, with std.
"""
import argparse, glob, os
import numpy as np
import nibabel as nib

# structure -> (model_label, acdc_gt_label)
STRUCTS = {"LV": (1, 3), "MYO": (2, 2), "RV": (3, 1)}


def dice(pred, gt, pl, gl):
    A, B = pred == pl, gt == gl
    s = A.sum() + B.sum()
    return float("nan") if s == 0 else 2.0 * (A & B).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    per = {ph: {s: [] for s in STRUCTS} for ph in ("ED", "ES")}
    missing = 0
    for pf in preds:
        case = os.path.basename(pf)[:-7]
        gf = os.path.join(args.gt_dir, case + ".nii.gz")
        if not os.path.exists(gf):
            missing += 1; continue
        pred = np.asarray(nib.load(pf).dataobj).astype(np.int16)
        gt = np.asarray(nib.load(gf).dataobj).astype(np.int16)
        if pred.shape != gt.shape:
            print("SHAPE MISMATCH", case, pred.shape, gt.shape); continue
        ph = "ED" if case.endswith("_ED") else "ES"
        for s, (pl, gl) in STRUCTS.items():
            d = dice(pred, gt, pl, gl)
            if not np.isnan(d):
                per[ph][s].append(d)

    print(f"\n===== ACDC Dice vs human GT {args.tag} =====")
    print(f"{'phase':6s} {'n':>4s} | " + " ".join(f"{s:>13s}" for s in STRUCTS))
    allcases = {s: [] for s in STRUCTS}
    for ph in ("ED", "ES"):
        n = max(len(per[ph][s]) for s in STRUCTS)
        cells = []
        for s in STRUCTS:
            v = per[ph][s]; allcases[s].extend(v)
            cells.append(f"{np.mean(v):.3f}±{np.std(v):.3f}" if v else "   n/a   ")
        print(f"{ph:6s} {n:4d} | " + " ".join(f"{c:>13s}" for c in cells))
    cells = [f"{np.mean(allcases[s]):.3f}±{np.std(allcases[s]):.3f}" for s in STRUCTS]
    n = max(len(allcases[s]) for s in STRUCTS)
    print(f"{'ALL':6s} {n:4d} | " + " ".join(f"{c:>13s}" for c in cells))
    mean3 = np.mean([np.mean(allcases[s]) for s in STRUCTS])
    print(f"mean over 3 structures: {mean3:.3f}   (missing GT: {missing})")


if __name__ == "__main__":
    main()
