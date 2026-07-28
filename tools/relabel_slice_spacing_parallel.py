"""Parallel + ATOMIC Z-spacing relabel. Same semantics as tools/relabel_slice_spacing.py
(rescale only the slice-axis column of the affine; idempotent) but:
  * ProcessPool -- the work is I/O bound on GPFS at ~1 file/s single-threaded
  * writes to a tmp file then os.replace() -- a kill can never leave a truncated NIfTI
    (that is what produced the zero-byte CMRx24_Train_P105/sax_frame_11.nii.gz)
Targets: 2024 -> 12.0 for all; 2023 -> per SUBJECT_MANIFEST.csv pitch_mm (12.0, or 10.0 for the
three 6 mm-thickness subjects).
"""
import os, csv, glob, sys
import numpy as np, nibabel as nib
from concurrent.futures import ProcessPoolExecutor

INPLANE_MAX_MM = 4.0

def _slice_axis(A):
    norms=[float(np.linalg.norm(A[:3,i])) for i in range(3)]
    big=[i for i,n in enumerate(norms) if n>INPLANE_MAX_MM]
    if len(big)!=1: raise ValueError(f"ambiguous slice axis: {norms}")
    return big[0], norms[big[0]]

def work(item):
    path,target=item
    try:
        img=nib.load(path); A=img.affine.copy(); ax,cur=_slice_axis(A)
        if abs(cur-target)<1e-4: return ("skip",path,cur,target)
        A[:3,ax]=A[:3,ax]*(target/cur)
        img.set_sform(A); img.set_qform(A)
        tmp=path+".relabeltmp.nii.gz"
        nib.save(img,tmp)
        os.replace(tmp,path)                      # atomic on the same filesystem
        return ("change",path,cur,target)
    except Exception as e:
        return ("error",path,str(e),target)

def build():
    items=[]
    r24="scratch/data/CMRxRecon2024/Cine_combined"
    for d in sorted(os.listdir(r24)):
        sx=os.path.join(r24,d,"sax")
        for f in glob.glob(os.path.join(sx,"3d_recon","sax_frame_*.nii.gz"))+[os.path.join(sx,"4d_recon.nii.gz")]:
            if os.path.exists(f): items.append((f,12.0))
    pit={r["combined_id"]:float(r["pitch_mm"]) for r in
         csv.DictReader(open("scratch/data/CMRxRecon2023/SUBJECT_MANIFEST.csv")) if r["reconstruct"]=="1"}
    r23="scratch/data/CMRxRecon2023/Cine_combined"
    for d in sorted(os.listdir(r23)):
        t=pit.get(d)
        if t is None: print(f"  WARN no manifest pitch for {d}, skipping"); continue
        sx=os.path.join(r23,d,"sax")
        for f in glob.glob(os.path.join(sx,"3d_recon","sax_frame_*.nii.gz"))+[os.path.join(sx,"4d_recon.nii.gz")]:
            if os.path.exists(f): items.append((f,t))
    return items

if __name__=="__main__":
    items=build(); print(f"[FAST RELABEL] {len(items)} files, 16 workers",flush=True)
    c={"skip":0,"change":0,"error":0}; errs=[]
    with ProcessPoolExecutor(int(os.environ.get("RELABEL_WORKERS", "16"))) as ex:
        for i,r in enumerate(ex.map(work,items,chunksize=8),1):
            c[r[0]]+=1
            if r[0]=="error": errs.append((r[1],r[2]))
            if i%500==0: print(f"  {i}/{len(items)}  {c}",flush=True)
    print(f"\nDONE  changed={c['change']} skipped={c['skip']} errors={c['error']}")
    for p,e in errs[:10]: print("  ERROR",p,e)
