#!/usr/bin/env python3
"""Full scan of M&Ms-1 (MNMs1/): geometry per subject + cross-tab with CSV metadata."""
import os, csv, glob, collections
import numpy as np, nibabel as nib

ROOT = "/home/minsukc/vggt/scratch/data/MNMs/MNMs1"
CSV  = glob.glob(os.path.join(ROOT, "*.csv"))[0]

# ---- CSV metadata ----
meta = {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        meta[row["External code"]] = row
print(f"CSV rows: {len(meta)}  cols: {list(next(iter(meta.values())).keys())}\n")

# ---- walk subjects, derive split from path, read headers ----
rows = []
for sa in glob.glob(os.path.join(ROOT, "**", "*_sa.nii.gz"), recursive=True):
    rel = os.path.relpath(sa, ROOT)
    parts = rel.split(os.sep)
    code = os.path.basename(sa).replace("_sa.nii.gz", "")
    # split = first path component; sub-split (Labeled/Unlabeled) = second if present
    split = parts[0]
    sub = parts[1] if parts[0] == "Training" else ""
    has_gt = os.path.exists(sa.replace("_sa.nii.gz", "_sa_gt.nii.gz"))
    im = nib.load(sa)                       # lazy: no pixel load
    sh = im.shape
    X, Y, Z = sh[0], sh[1], sh[2]
    T = sh[3] if len(sh) > 3 else 1
    zx, zy, zz = [float(v) for v in im.header.get_zooms()[:3]]
    ax = "".join(nib.aff2axcodes(im.affine))
    m = meta.get(code, {})
    rows.append(dict(code=code, split=split, sub=sub, has_gt=has_gt,
                     X=X, Y=Y, Z=Z, T=T, zx=zx, zy=zy, zz=zz, ax=ax,
                     vendor=m.get("Vendor",""), vname=m.get("VendorName",""),
                     centre=m.get("Centre",""), path=m.get("Pathology",""),
                     ed=m.get("ED",""), es=m.get("ES",""), age=m.get("Age",""), sex=m.get("Sex","")))

n = len(rows)
print(f"Total subjects on disk (found _sa.nii.gz): {n}")
print(f"With _sa_gt: {sum(r['has_gt'] for r in rows)}   without GT: {sum(not r['has_gt'] for r in rows)}")
in_csv = sum(1 for r in rows if r['vendor'])
print(f"Matched to CSV: {in_csv}   NOT in CSV: {n-in_csv}\n")

def tab(key, rows=rows):
    c = collections.Counter(r[key] for r in rows)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))

def stats(key, rows=rows):
    v = np.array([r[key] for r in rows], float)
    return f"min={v.min():.3g} max={v.max():.3g} median={np.median(v):.3g} mean={v.mean():.3g}"

print("== SPLIT (from folder structure) ==")
for k, v in tab("split").items():
    subs = tab("sub", [r for r in rows if r["split"] == k])
    print(f"  {k:12} {v:4d}   {subs if any(subs) else ''}")
print(f"  ratio train/val/test ≈ {tab('split')}")

print("\n== VENDOR (A=Siemens B=Philips C=GE D=Canon) ==")
for k, v in tab("vendor").items(): print(f"  {k or '(none)':8} {v}")
print("\n== VENDOR × SPLIT ==")
for sp in ["Training", "Validation", "Testing"]:
    r2 = [r for r in rows if r["split"] == sp]
    print(f"  {sp:12} {tab('vendor', r2)}")

print("\n== PATHOLOGY ==")
for k, v in tab("path").items(): print(f"  {k or '(none)':8} {v}")

print("\n== CENTRE ==")
for k, v in tab("centre").items(): print(f"  centre {k or '(none)':4} {v}")

print("\n== GEOMETRY (all subjects) ==")
print(f"  T (frames) : {stats('T')}   hist={tab('T')}")
print(f"  Z (slices) : {stats('Z')}   hist={tab('Z')}")
print(f"  X in-plane : {stats('X')}")
print(f"  Y in-plane : {stats('Y')}")
print(f"  spacing x  : {stats('zx')}")
print(f"  spacing y  : {stats('zy')}")
print(f"  spacing z  : {stats('zz')}   hist={tab('zz')}")
print(f"  orientation: {tab('ax')}")

print("\n== T (frames) by vendor ==")
for vd in ["A","B","C","D"]:
    r2=[r for r in rows if r["vendor"]==vd]
    if r2: print(f"  {vd}: {stats('T', r2)}   n={len(r2)}")

print("\n== demographics ==")
ages=[float(r['age']) for r in rows if r['age'] not in ('','nan')]
if ages: print(f"  Age: min={min(ages):.0f} max={max(ages):.0f} median={np.median(ages):.0f} (n={len(ages)})")
print(f"  Sex: {tab('sex')}")
# ED/ES sanity
eds=[int(r['ed']) for r in rows if r['ed'] not in ('','nan')]
ess=[int(r['es']) for r in rows if r['es'] not in ('','nan')]
if eds: print(f"  ED frame idx: {min(eds)}..{max(eds)} (median {int(np.median(eds))});  ES: {min(ess)}..{max(ess)} (median {int(np.median(ess))})")
