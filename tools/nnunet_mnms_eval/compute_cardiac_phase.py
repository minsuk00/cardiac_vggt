"""Derive ED/ES + EDV/ESV/EF per GATED subject from the per-phase heart_seg siblings.

Source of truth = the sibling `heart_seg.nii.gz` (4D X,Y,Z,T, 3-class LV=1/MYO=2/RV=3) written by the
whole-heart-seg job. This is a regenerable CACHE keyed by `unit` (joins whs_manifest.csv): delete and
re-run whenever the segs change.

ED = phase with MAX LV-cavity (label 1) volume; ES = MIN. EF = (EDV-ESV)/EDV. Volumes are the full 3D
cavity (Simpson's) in native spacing -> physically-correct mL; NOT a single mid-slice area. RTFB is
excluded (frames aren't gated cardiac phases). A `unimodal_ok` column flags subjects whose LV-volume
curve doesn't cleanly contract-then-relax (a seg-quality red flag beyond the manifest's `low` flag).

Usage: python compute_cardiac_phase.py            # -> scratch/data/whs/cardiac_phase.csv
"""
import csv, os, sys
import numpy as np
import nibabel as nib

ROOT = "/home/minsukc/vggt"
WORKLIST = os.path.join(ROOT, "scratch/data/whs/worklist.txt")
MANIFEST = os.path.join(ROOT, "scratch/data/whs/whs_manifest.csv")
OUT = os.path.join(ROOT, "scratch/data/whs/cardiac_phase.csv")
sys.path.insert(0, os.path.join(ROOT, "tools/nnunet_mnms_eval"))
from assemble_whs import unit_id, is_gated                        # sibling path + gated test (reused)

MONO_OK = 0.80   # min fraction of frames moving in the physiologically-correct direction


def _acdc_cfg(out_dir):
    d = {}
    cfg = os.path.join(out_dir, "Info.cfg")
    if os.path.exists(cfg):
        for line in open(cfg):
            if ":" in line:
                k, v = line.split(":", 1)
                d[k.strip()] = v.strip()
    return d


def acdc_group(out_dir):
    """ACDC pathology (NOR/DCM/HCM/MINF/RV) from Info.cfg; '' if absent."""
    return _acdc_cfg(out_dir).get("Group", "")


def acdc_ed_es(out_dir):
    """ACDC ground-truth ED/ES as 0-indexed frame numbers from Info.cfg (Info.cfg is 1-indexed).
    Preferred over argmax/argmin for ACDC: its cine starts AND ends at ED, so argmax is ambiguous."""
    c = _acdc_cfg(out_dir)
    if "ED" in c and "ES" in c:
        return int(c["ED"]) - 1, int(c["ES"]) - 1
    return None


def circular_seg(a, b, T):
    """indices from a forward-circularly to b (inclusive)."""
    idx, i = [], a
    while True:
        idx.append(i)
        if i == b:
            return idx
        i = (i + 1) % T


def unimodality(vol):
    """Fraction of steps going the right way: ED->ES should be non-increasing, ES->ED non-decreasing.
    ~1.0 for a clean cardiac cycle; low for a noisy/broken seg. Returns (frac, ed, es)."""
    T = len(vol)
    ed, es = int(np.argmax(vol)), int(np.argmin(vol))
    if ed == es:
        return 0.0, ed, es
    con, rel = circular_seg(ed, es, T), circular_seg(es, ed, T)
    con_ok = sum(vol[con[k + 1]] <= vol[con[k]] + 1e-9 for k in range(len(con) - 1))
    rel_ok = sum(vol[rel[k + 1]] >= vol[rel[k]] - 1e-9 for k in range(len(rel) - 1))
    steps = (len(con) - 1) + (len(rel) - 1)
    return (con_ok + rel_ok) / steps if steps else 1.0, ed, es


def main():
    # manifest flag per unit (for join)
    flag = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for r in csv.DictReader(f):
                flag[r["unit"]] = r["flag"]

    rows = []
    with open(WORKLIST) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            ds, regime, path = parts
            if not is_gated(regime):
                continue
            uid, subj, out_dir = unit_id(ds, regime, path)
            seg_f = os.path.join(out_dir, "heart_seg.nii.gz")
            if not os.path.exists(seg_f):
                print("MISSING seg:", seg_f); continue
            im = nib.load(seg_f)
            seg = np.asarray(im.dataobj)
            sp = nib.affines.voxel_sizes(im.affine)
            vox_ml = float(np.prod(sp[:3]) / 1000.0)
            T = seg.shape[-1]
            vol = np.array([(seg[..., t] == 1).sum() * vox_ml for t in range(T)])  # LV cavity mL / phase
            mono, ed, es = unimodality(vol)                    # argmax/argmin + curve-quality check
            if ds == "acdc":                                   # use ACDC's ground-truth ED/ES instead
                gt = acdc_ed_es(out_dir)
                if gt is not None and gt[0] < T and gt[1] < T:
                    ed, es = gt
            edv, esv = float(vol[ed]), float(vol[es])
            ef = 100.0 * (edv - esv) / edv if edv > 0 else float("nan")
            rows.append(dict(unit=uid, dataset=ds, regime=regime, subject=subj, T=T,
                             ED=ed, ES=es, EDV_mL=round(edv, 2), ESV_mL=round(esv, 2),
                             EF_pct=round(ef, 2) if edv > 0 else "",
                             curve_mono_frac=round(mono, 3),
                             unimodal_ok=int(mono >= MONO_OK and edv > 0),
                             seg_flag=flag.get(uid, ""),
                             source=("acdc_task114" if ds == "acdc" else "task114_3d"),
                             group=(acdc_group(out_dir) if ds == "acdc" else "")))

    rows.sort(key=lambda r: (r["dataset"], r["subject"]))
    cols = ["unit", "dataset", "regime", "subject", "T", "ED", "ES", "EDV_mL", "ESV_mL",
            "EF_pct", "curve_mono_frac", "unimodal_ok", "seg_flag", "source", "group"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # summary
    print(f"{len(rows)} gated subjects -> {OUT}")
    import collections
    by = collections.defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    for ds, rs in sorted(by.items()):
        efs = [r["EF_pct"] for r in rs if r["EF_pct"] != ""]
        nbad = sum(1 for r in rs if not r["unimodal_ok"])
        nlow = sum(1 for r in rs if r["seg_flag"] == "low")
        print(f"  {ds:6} n={len(rs):3}  EF mean={np.mean(efs):5.1f} range[{min(efs):.0f},{max(efs):.0f}]  "
              f"non-unimodal={nbad}  seg_low={nlow}")
    bad = [r for r in rows if not r["unimodal_ok"]]
    if bad:
        print("  non-unimodal / suspect subjects:")
        for r in bad:
            print(f"    {r['unit']:44} EF={r['EF_pct']} mono={r['curve_mono_frac']} flag={r['seg_flag']}")


if __name__ == "__main__":
    main()
