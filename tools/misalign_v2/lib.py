"""Read-only misalignment measurement library (scratch). Never writes under scratch/data."""
import os, numpy as np, nibabel as nib
ROOT="/home/minsukc/vggt"; DATA=os.path.join(ROOT,"scratch/data")
POOLED=os.path.join(ROOT,"training/splits/pooled.txt")

def read_pooled(sections=("train","val","test")):
    out=[]; sec=None
    for line in open(POOLED):
        line=line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("[") and line.endswith("]"): sec=line[1:-1].lower(); continue
        if sec in sections: out.append((line,sec))
    return out

def load_seg(rel):
    img=nib.load(os.path.join(DATA,rel,"sax","heart_seg.nii.gz"))
    return np.asarray(img.dataobj), img.header.get_zooms()[:3]

def load_4d(rel):
    p=os.path.join(DATA,rel,"sax","4d_recon.nii.gz")
    if not os.path.exists(p):
        import glob
        fs=sorted(glob.glob(os.path.join(DATA,rel,"sax","3d_recon","sax_frame_*.nii.gz")))
        return np.stack([np.asarray(nib.load(f).dataobj,dtype=np.float32) for f in fs],-1)
    return np.asarray(nib.load(p).dataobj,dtype=np.float32)

def centroids(mask3d, min_vox=15):
    """mask3d (X,Y,Z) bool -> arrays z, xc, yc, area (voxel units)."""
    Z=mask3d.shape[2]; zs=[];xs=[];ys=[];ar=[]
    for z in range(Z):
        m=mask3d[:,:,z]; n=int(m.sum())
        if n<min_vox: continue
        ix,iy=np.nonzero(m); zs.append(z); xs.append(ix.mean()); ys.append(iy.mean()); ar.append(n)
    return np.array(zs),np.array(xs),np.array(ys),np.array(ar)

def robust_polyfit(z,y,deg,iters=4):
    w=np.ones_like(z,dtype=float); c=np.polyfit(z,y,deg,w=w)
    for _ in range(iters):
        r=y-np.polyval(c,z); mad=np.median(np.abs(r-np.median(r)))+1e-9
        cc=1.345*1.4826*mad+1e-9; w=np.clip(cc/(np.abs(r)+1e-9),0.05,1.0); c=np.polyfit(z,y,deg,w=w)
    return y-np.polyval(c,z)

def prior_metric(seg_phase,dx,dy):
    """The docs/94 metric: whole-heart centroid, deg 1/2 robust fit, max |resid| mm."""
    z,xc,yc,_=centroids(seg_phase>0)
    if len(z)<5: return np.nan
    deg=1 if len(z)<8 else 2
    rx=robust_polyfit(z.astype(float),xc,deg); ry=robust_polyfit(z.astype(float),yc,deg)
    return float(np.max(np.hypot(rx*dx,ry*dy)))
