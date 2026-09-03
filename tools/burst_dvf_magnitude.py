import numpy as np, glob, os, nibabel as nib, sys
from scipy.ndimage import zoom
arms=["vggt_burst5noreg224_ep300","vggt_burst5noreg224_k1_ep300","vggt_noreg224_ep300","vggt_noreg224_k5_ep300"]
srcs=sys.argv[1:] or ["cmrx2024"]
subs=[s for src in srcs for s in sorted(glob.glob(f'evaluation/volumes/{src}/out/*/'))]
def plane(vol_xyz, z, n=224):
    p=vol_xyz[:,:,z].T
    return zoom(p, n/p.shape[0], order=0)
res={a:{"heart_loc":[],"tissue_loc":[],"bg_loc":[],"heart_dz_loc":[],"const":[]} for a in arms}
for s in subs:
    if not all(os.path.exists(os.path.join(s,a,'ed_dvf.npz')) for a in arms): continue
    hm=np.asarray(nib.load(s+'mask_heart.nii.gz').dataobj)>0.5
    for a in arms:
        d=np.load(os.path.join(s,a,'ed_dvf.npz')); dl=d['delta'].astype(np.float32); zs=d['slot_z'].astype(int); ts=d['slot_t']
        dxy=dl[...,:2]*179.2; dz=dl[...,2]*90
        H=[];Tt=[];B=[];HZ=[];C=[]
        cache={}
        for i in range(len(zs)):
            t=int(ts[i])
            if t not in cache: cache[t]=np.asarray(nib.load(f"{s}breath/stack_t{t:02d}.nii.gz").dataobj)
            tissue=plane(cache[t],zs[i])>0.05; heart=plane(hm.astype(np.float32),zs[i])>0.5
            if not tissue.any(): continue
            c=dxy[i][tissue].mean(axis=0); C.append(np.linalg.norm(c))
            loc=np.linalg.norm(dxy[i]-c,axis=-1); dzl=np.abs(dz[i]-dz[i][tissue].mean())
            if heart.any(): H.append(loc[heart].mean()); HZ.append(dzl[heart].mean())
            Tt.append(loc[tissue].mean()); B.append(loc[~tissue].mean())
        for k,v in (("heart_loc",H),("tissue_loc",Tt),("bg_loc",B),("heart_dz_loc",HZ),("const",C)): res[a][k].append(np.mean(v))
print(f"n={len(res[arms[0]]['heart_loc'])} subjects, sources={srcs}; mm at ED, breath arm, medians over subjects")
print(f"{'arm':32s} {'const|dxy|':>10s} {'heart|dxy|loc':>13s} {'tissue|dxy|loc':>14s} {'bg|dxy|loc':>10s} {'heart|dz|loc':>12s}")
for a in arms:
    r=res[a]; print(f"{a:32s} {np.median(r['const']):10.2f} {np.median(r['heart_loc']):13.2f} {np.median(r['tissue_loc']):14.2f} {np.median(r['bg_loc']):10.2f} {np.median(r['heart_dz_loc']):12.2f}")
