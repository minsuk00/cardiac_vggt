"""Follow-up: (1) matched-filter pseudo-EDV pred-vs-GT (is the seg's -13ml EDV deficit
geometric?), (2) alignment control: CoM shift + Dice of thresholded blood pools at ES
(could pred displacement fake 'full contraction' in the GT-space ROI?),
(3) full pseudo-volume curves saved per subject."""
import json, csv, sys
import numpy as np, nibabel as nib
from scipy import ndimage as ndi
from scipy import stats

ROOT="temp/dice_ef_sweep/mirror_full/volumes"; ARM="vggt_noreg_ep300"; LV,MYO=1,2
cov=list(csv.DictReader(open("temp/dice_ef_sweep/difficulty_covariates.csv")))
err={r["subj"]:float(r["err"]) for r in cov}
worst20=set(sorted(err,key=lambda s:-err[s])[:20])
def norm01(v):
    lo,hi=np.percentile(v[v>0],[1,99.5]); return np.clip((v-lo)/max(hi-lo,1e-6),0,1)
rows=[]
for r in cov:
    subj,coh=r["subj"],r["coh"]; d=f"{ROOT}/{coh}/out/{subj}"
    try:
        seg=np.asarray(nib.load(f"{d}/heart_seg.nii.gz").dataobj)
        gt=np.asarray(nib.load(f"{d}/cine_gt.nii.gz").dataobj).astype(np.float32)
        zoom=nib.load(f"{d}/cine_gt.nii.gz").header.get_zooms()[:3]
        vox=float(np.prod(zoom))/1000.0; T=seg.shape[3]
        pred=np.stack([np.asarray(nib.load(f"{d}/{ARM}/recon_breath/vol_t{t:02d}.nii.gz").dataobj).astype(np.float32) for t in range(T)],-1)
    except Exception as e: print("skip",subj,e,file=sys.stderr); continue
    lvc=(seg==LV).sum((0,1,2)); tED,tES=int(np.argmax(lvc)),int(np.argmin(lvc))
    lvED,lvES=seg[...,tED]==LV,seg[...,tES]==LV; myoES=seg[...,tES]==MYO; myoED=seg[...,tED]==MYO
    if lvES.sum()<20 or myoES.sum()<20: continue
    gtN,prN=norm01(gt),norm01(pred)
    core=ndi.binary_erosion(lvES,iterations=2); core=core if core.sum()>10 else lvES
    coreED=ndi.binary_erosion(lvED,iterations=2); coreED=coreED if coreED.sum()>10 else lvED
    refs={}
    for tag,img,c,m in [("gtES",gtN[...,tES],core,myoES),("prES",prN[...,tES],core,myoES),
                        ("gtED",gtN[...,tED],coreED,myoED),("prED",prN[...,tED],coreED,myoED)]:
        refs[tag]=(np.median(img[c]),np.median(img[m]))
    roi=ndi.binary_dilation(lvED,iterations=2)
    def pseudo(img,blood,myo): return np.count_nonzero(img[roi]>=myo+0.5*(blood-myo))*vox
    pvES_g=pseudo(gtN[...,tES],*refs["gtES"]); pvES_p=pseudo(prN[...,tES],*refs["prES"])
    pvED_g=pseudo(gtN[...,tED],*refs["gtED"]); pvED_p=pseudo(prN[...,tED],*refs["prED"])
    # alignment: thresholded blood pools at ES, dice + CoM shift (mm, in-plane)
    bg=np.zeros_like(lvES); bp=np.zeros_like(lvES)
    bg[roi]=gtN[...,tES][roi]>=refs["gtES"][1]+0.5*(refs["gtES"][0]-refs["gtES"][1])
    bp[roi]=prN[...,tES][roi]>=refs["prES"][1]+0.5*(refs["prES"][0]-refs["prES"][1])
    inter=(bg&bp).sum(); dice=2*inter/max(bg.sum()+bp.sum(),1)
    cg,cp=ndi.center_of_mass(bg),ndi.center_of_mass(bp)
    com=float(np.hypot((cg[0]-cp[0])*zoom[0],(cg[1]-cp[1])*zoom[1])) if bg.any() and bp.any() else np.nan
    rows.append(dict(subj=subj,group="worst20" if subj in worst20 else "rest",err=err[subj],
        pvES_gt=pvES_g,pvES_pred=pvES_p,pvED_gt=pvED_g,pvED_pred=pvED_p,
        dES_ml=pvES_p-pvES_g,dED_ml=pvED_p-pvED_g,dice_bp_ES=float(dice),com_mm=com))
import pandas as pd
df=pd.DataFrame(rows)
csvout="temp/dice_ef_sweep/es_diag2_noreg.csv"; df.to_csv(csvout,index=False)
print(f"n={len(df)} -> {csvout}")
for c in ["dES_ml","dED_ml","dice_bp_ES","com_mm"]:
    print(f"{c:12s} median={df[c].median():+7.2f}  mean={df[c].mean():+7.2f}  "
          f"worst20={df[df.group=='worst20'][c].median():+7.2f}")
print("pseudo-EDV pred vs gt Wilcoxon p=%.2g"%stats.wilcoxon(df.pvED_pred,df.pvED_gt).pvalue)
print("pseudo-ESV pred vs gt Wilcoxon p=%.2g"%stats.wilcoxon(df.pvES_pred,df.pvES_gt).pvalue)
print("low-alignment subjects (dice_bp_ES<0.5): %d"%(df.dice_bp_ES<0.5).sum())
d2=df[df.dice_bp_ES>=0.5]
print("well-aligned only: dES median %+0.2f, dED median %+0.2f (n=%d)"%(d2.dES_ml.median(),d2.dED_ml.median(),len(d2)))
