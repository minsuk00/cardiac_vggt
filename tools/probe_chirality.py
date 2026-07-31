"""Chirality (LV<->RV left-right handedness) check across the pooled training cohort.

Analogous to tools/probe_slice_order.py (docs/58 §10a) but for the IN-PLANE axes instead of Z:
a mirror-flipped in-plane axis would place the RV on the anatomically wrong side of the LV,
teaching the model contradictory left-right anatomy. Same root cause class as the base/apex bug
(Orientationd reorders/flips array axes off each subject's affine, and CMRx/ACDC affines are NOT
real scanner geometry -- SimpleITK default / degenerate sform=qform=0, docs/58 §10a) -- so a
mirror error here would be just as invisible to a human skimming slices as the base/apex mixture
was.

Anatomical invariant used: in a normal (D-looped, situs solitus, >99.9% of humans) heart, the RV
sits ANTERIOR and to the patient's RIGHT of the LV. So for each subject: take heart_seg.nii.gz
(LV=1, RV=3, union over all T for a robust centroid), compute the LV->RV vector in raw voxel-index
space, rotate it into world RAS mm via the affine's linear part (translation cancels in a
difference of two points), and check the sign of the R component. Positive R = normal chirality.

M&Ms gets an independent cross-check: it ships its own GT segmentations
(MNMs1/*/<CODE>/<CODE>_sa_gt.nii.gz) on the RAW pre-conversion volume with a REAL oblique affine
(unlike CMRx/ACDC, see docs/58 §10a "M&Ms doubles as a labelled validation set"). If the converted
heart_seg's R-sign agrees with the GT-on-raw-affine's R-sign, that validates BOTH the converter's
in-plane axis assignment AND the detector method itself, independent of any bug shared between them.

Usage: python probe_chirality.py [--worklist PATH] [--csv OUT.csv]
"""
import argparse, glob, os, sys
import numpy as np
import nibabel as nib

ROOT = "/home/minsukc/vggt"
sys.path.insert(0, os.path.join(ROOT, "tools/nnunet_mnms_eval"))
from assemble_whs import unit_id, is_gated                             # noqa: E402

TRAINING_DATASETS = ("cmrx", "acdc_sax", "mnms_sax")


def _centroid_vec(seg4d, affine, lv_label=1, rv_label=3):
    """seg4d: (X,Y,Z,T). Returns (vec_world (R,A,S) mm, |vec_world|) or None if LV/RV missing."""
    lv = (seg4d == lv_label).any(axis=-1)
    rv = (seg4d == rv_label).any(axis=-1)
    if lv.sum() == 0 or rv.sum() == 0:
        return None
    lv_c = np.argwhere(lv).mean(axis=0)
    rv_c = np.argwhere(rv).mean(axis=0)
    vec_vox = rv_c - lv_c
    vec_world = np.asarray(affine, dtype=np.float64)[:3, :3] @ vec_vox   # translation cancels
    return vec_world, float(np.linalg.norm(vec_world))


def mnms_gt_path(code):
    hits = glob.glob(os.path.join(ROOT, "scratch/data/MNMs/MNMs1", "**", code, f"{code}_sa_gt.nii.gz"),
                      recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=os.path.join(ROOT, "scratch/data/whs/worklist.txt"))
    ap.add_argument("--csv", default=os.path.join(ROOT, "result/chirality_check/chirality.csv"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)

    # manifest, for per-source/vendor breakdown
    manifest = {}
    import csv as csvmod
    with open(os.path.join(ROOT, "training/splits/manifest.csv")) as f:
        for r in csvmod.DictReader(f):
            manifest[r["id"]] = r

    lines = [l.split() for l in open(args.worklist) if len(l.split()) == 3]
    rows = []
    n_missing_struct = 0
    for ds, regime, p in lines:
        if ds not in TRAINING_DATASETS or not is_gated(regime):
            continue
        uid, subj, out_dir = unit_id(ds, regime, p)
        seg_f = os.path.join(out_dir, "heart_seg.nii.gz")
        if not os.path.exists(seg_f):
            continue
        im = nib.load(seg_f)
        seg4d = np.asarray(im.dataobj).astype(np.uint8)
        res = _centroid_vec(seg4d, im.affine)
        if res is None:
            n_missing_struct += 1
            continue
        vec, mag = res
        r = manifest.get(subj, {})
        row = dict(unit=uid, dataset=ds, subject=subj, R=vec[0], A=vec[1], S=vec[2], mag=mag,
                   source=r.get("source", ""), vendor=r.get("vendor", ""), centre=r.get("centre", ""))

        if ds == "mnms_sax":
            gt_f = mnms_gt_path(subj.replace("MNMs_", ""))
            if gt_f is not None:
                gt_im = nib.load(gt_f)
                gt_arr = np.asarray(gt_im.dataobj).astype(np.uint8)
                gt_res = _centroid_vec(gt_arr, gt_im.affine)
                if gt_res is not None:
                    row["R_gt"] = gt_res[0][0]
                    row["gt_agrees"] = int(np.sign(vec[0]) == np.sign(gt_res[0][0]))
        rows.append(row)

    # write CSV
    cols = ["unit", "dataset", "subject", "source", "vendor", "centre", "R", "A", "S", "mag",
            "R_gt", "gt_agrees"]
    with open(args.csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})

    # summary
    print(f"{len(rows)} subjects measured, {n_missing_struct} skipped (missing LV or RV)\n")
    import collections
    by_ds = collections.defaultdict(list)
    for row in rows:
        by_ds[row["dataset"]].append(row)
    for ds, rs in sorted(by_ds.items()):
        pos = sum(1 for r in rs if r["R"] > 0)
        neg = len(rs) - pos
        mags = sorted(r["mag"] for r in rs)
        med = mags[len(mags) // 2]
        print(f"  {ds:10} n={len(rs):4}  R>0 (normal) = {pos:4}  R<0 (flipped?) = {neg:4}  "
              f"median|vec|={med:.1f}mm")

    gt_checked = [r for r in rows if r.get("gt_agrees", "") != ""]
    if gt_checked:
        agree = sum(r["gt_agrees"] for r in gt_checked)
        print(f"\n  M&Ms GT cross-check: {agree}/{len(gt_checked)} agree "
              f"(validates converter's in-plane axis assignment + this detector)")

    # per CMRx-source/vendor breakdown (mirrors the slice-order per-scanner table)
    cmrx = [r for r in rows if r["dataset"] == "cmrx"]
    if cmrx:
        print("\n  CMRx breakdown by source/vendor:")
        by_sv = collections.defaultdict(lambda: [0, 0])
        for r in cmrx:
            key = (r["source"], r["vendor"])
            by_sv[key][0 if r["R"] > 0 else 1] += 1
        for (src, ven), (pos, neg) in sorted(by_sv.items()):
            flag = "  <-- MIXED" if pos > 0 and neg > 0 and min(pos, neg) >= 3 else ""
            print(f"    {src:16} {ven:10} R>0={pos:4} R<0={neg:4}{flag}")

    outliers = [r for r in rows if r["R"] < 0]
    if outliers:
        print(f"\n  {len(outliers)} R<0 subjects (candidates for chirality flip), by |R|:")
        for r in sorted(outliers, key=lambda r: -abs(r["R"]))[:20]:
            print(f"    {r['unit']:45} R={r['R']:8.2f}  A={r['A']:8.2f}  S={r['S']:8.2f}  "
                  f"|vec|={r['mag']:.1f}")

    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
