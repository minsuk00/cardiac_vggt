"""ACDC GT-based ES-fraction analysis (no segmentation needed).

ACDC ships labeled ED/ES frames + NbFrame in each Info.cfg, and GT segs at ED & ES
(ACDC labels: 1=RV, 2=MYO, 3=LV cavity). So we read end-systole's position in the
cycle directly from ground truth:

    es_frac = (ES - ED) / NbFrame          # fraction of the R-R from ED to ES

and GT LV volumes (EDV/ESV/EF) from the ED/ES seg files. Broken down by pathology
Group (DCM/HCM/MINF/NOR/RV) to test whether disease shifts systolic timing.

Output JSON (--out_json).
"""
import argparse, glob, json, os
import numpy as np
import nibabel as nib

LV_ACDC = 3   # ACDC GT: 1=RV, 2=MYO, 3=LV cavity


def cfg(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


def lv_ml(seg_path):
    im = nib.load(seg_path)
    z = im.header.get_zooms()[:3]
    seg = np.asarray(im.dataobj)
    return float((seg == LV_ACDC).sum() * np.prod(z) / 1000.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acdc_root", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    pats = []
    for split in ("training", "testing"):
        pats += sorted(glob.glob(os.path.join(args.acdc_root, split, "patient*")))

    rows = []
    for p in pats:
        info = cfg(os.path.join(p, "Info.cfg"))
        ed, es, nb = int(info["ED"]), int(info["ES"]), int(info["NbFrame"])
        grp = info.get("Group", "?")
        pid = os.path.basename(p)
        es_frac = (es - ed) / nb
        # GT volumes at the labeled ED/ES frames
        ed_gt = os.path.join(p, f"{pid}_frame{ed:02d}_gt.nii.gz")
        es_gt = os.path.join(p, f"{pid}_frame{es:02d}_gt.nii.gz")
        edv = lv_ml(ed_gt) if os.path.exists(ed_gt) else float("nan")
        esv = lv_ml(es_gt) if os.path.exists(es_gt) else float("nan")
        ef = (edv - esv) / edv * 100.0 if edv > 0 else float("nan")
        rows.append(dict(pid=pid, group=grp, ed=ed, es=es, nb=nb,
                         es_frac=es_frac, edv=edv, esv=esv, ef=ef))

    fr = np.array([r["es_frac"] for r in rows])
    ef = np.array([r["ef"] for r in rows])
    groups = sorted({r["group"] for r in rows})
    by_group = {}
    for g in groups:
        gf = np.array([r["es_frac"] for r in rows if r["group"] == g])
        gef = np.array([r["ef"] for r in rows if r["group"] == g])
        by_group[g] = dict(n=int(len(gf)),
                           es_frac_mean=float(gf.mean()), es_frac_std=float(gf.std()),
                           es_frac_min=float(gf.min()), es_frac_max=float(gf.max()),
                           ef_mean=float(np.nanmean(gef)))

    # histogram of es_frac onto a 12-bin grid (to overlay on CMRxRecon's k/12)
    edges = np.linspace(0, 1, 13)
    hist12, _ = np.histogram(fr, bins=edges)

    out = dict(
        n=len(rows),
        es_frac_mean=float(fr.mean()), es_frac_std=float(fr.std()),
        es_frac_min=float(fr.min()), es_frac_max=float(fr.max()),
        es_frac_iqr=[float(np.percentile(fr, 25)), float(np.percentile(fr, 75))],
        ef_mean=float(np.nanmean(ef)), ef_std=float(np.nanstd(ef)),
        nbframe_min=int(min(r["nb"] for r in rows)),
        nbframe_max=int(max(r["nb"] for r in rows)),
        by_group=by_group,
        es_frac_hist12=hist12.tolist(),
        es_frac_all=fr.tolist(),
        rows=rows,
    )
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"ACDC n={out['n']}  NbFrame {out['nbframe_min']}-{out['nbframe_max']}")
    print(f"ES fraction: mean={out['es_frac_mean']:.3f} std={out['es_frac_std']:.3f} "
          f"range=[{out['es_frac_min']:.2f},{out['es_frac_max']:.2f}] IQR={[round(x,2) for x in out['es_frac_iqr']]}")
    print(f"EF: {out['ef_mean']:.1f}% +/- {out['ef_std']:.1f}%")
    print("\nby pathology group (es_frac mean+/-std, range; EF):")
    for g, d in by_group.items():
        print(f"  {g:5s} n={d['n']:3d}: es_frac {d['es_frac_mean']:.3f}+/-{d['es_frac_std']:.3f} "
              f"[{d['es_frac_min']:.2f},{d['es_frac_max']:.2f}]  EF {d['ef_mean']:.0f}%")
    print("\nes_frac on a 12-bin (k/12) grid:")
    for k in range(12):
        print(f"  [{k/12:.2f},{(k+1)/12:.2f}): {hist12[k]:3d}  {'*'*hist12[k]}")


if __name__ == "__main__":
    main()
