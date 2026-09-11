import torch, torch.nn.functional as F
dev="cuda"
def ncc_shift_fast(mov, ref, mask, maxpx):
    """Same contract as metric_v2.ncc_shift but conv2d-based, cropped to the mask bbox."""
    T,H,W=mov.shape
    ys,xs=torch.nonzero(mask,as_tuple=True)
    y0,y1=max(int(ys.min())-2,0),min(int(ys.max())+3,H); x0,x1=max(int(xs.min())-2,0),min(int(xs.max())+3,W)
    m=mask[y0:y1,x0:x1].float(); n=m.sum()
    mv=mov[:,y0:y1,x0:x1]
    mu=(mv*m).sum((1,2))/n; movc=(mv-mu[:,None,None])*m; var_m=(movc**2).sum((1,2))
    refp=F.pad(ref,(maxpx,maxpx,maxpx,maxpx))[:,y0:y1+2*maxpx,x0:x1+2*maxpx]  # (T,h+2p,w+2p)
    num=F.conv2d(refp[None],movc[:,None],groups=T)[0]           # (T,2p+1,2p+1)
    S1=F.conv2d(refp[:,None],m[None,None])[:,0]; S2=F.conv2d((refp**2)[:,None],m[None,None])[:,0]
    ncc=num/torch.sqrt(var_m[:,None,None]*(S2-S1**2/n).clamp_min(0)+1e-8)
    k=2*maxpx+1; flat=ncc.view(T,-1); pk,idx=flat.max(1); iy=idx//k; ix=idx%k
    out=torch.zeros(T,2,device=mov.device)
    nc=ncc.cpu().numpy()
    for t in range(T):
        y,x=int(iy[t]),int(ix[t]); sx,sy=float(x-maxpx),float(y-maxpx)
        if 0<x<k-1:
            a,b,c=nc[t,y,x-1],nc[t,y,x],nc[t,y,x+1]; den=a-2*b+c
            if den<0: sx+=0.5*(a-c)/den
        if 0<y<k-1:
            a,b,c=nc[t,y-1,x],nc[t,y,x],nc[t,y+1,x]; den=a-2*b+c
            if den<0: sy+=0.5*(a-c)/den
        out[t,0]=float(sx); out[t,1]=float(sy)
    return out,pk
