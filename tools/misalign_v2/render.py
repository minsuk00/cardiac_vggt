"""Render a subject: SAX mosaic (ED) with LV/epi centroids, coronal+sagittal reslices through the LV
centre (physical aspect), and per-slice spike profile (seg vs image). Read-only."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import zoom

def render(rel, df, out_png, title=None):
    seg,(dx,dy,dz)=load_seg(rel); img=load_4d(rel); X,Y,Z,T=seg.shape
    d=df[df.rel==rel].sort_values("z")
    ed=img[...,0]; s0=seg[...,0]
    # LV centre: median over slices of LV centroid
    lv=s0==1; zs=[z for z in range(Z) if lv[:,:,z].sum()>15]
    cx=[np.nonzero(lv[:,:,z])[0].mean() for z in zs]; cy=[np.nonzero(lv[:,:,z])[1].mean() for z in zs]
    cxm,cym=int(np.median(cx)),int(np.median(cy))
    ncol=Z; fig=plt.figure(figsize=(1.6*ncol, 1.6*1+4.2), dpi=110)
    gs=fig.add_gridspec(2, ncol, height_ratios=[1.6,4.2])
    # crop window around LV for mosaic
    w=int(60/dx); h=int(60/dy)
    x0,x1=max(cxm-w,0),min(cxm+w,X); y0,y1=max(cym-h,0),min(cym+h,Y)
    crop=ed[x0:x1,y0:y1,:]; vmax=np.percentile(crop[crop>0],99) if (crop>0).any() else ed.max()
    for z in range(Z):
        ax=fig.add_subplot(gs[0,z]); ax.imshow(ed[x0:x1,y0:y1,z].T,cmap="gray",vmin=0,vmax=vmax,origin="lower",aspect=dy/dx)
        row=d[d.z==z]
        k=float(row.epi_spike.iloc[0]) if len(row) and np.isfinite(row.epi_spike.iloc[0]) else np.nan
        ki=float(row.img_spike.iloc[0]) if len(row) and np.isfinite(row.img_spike.iloc[0]) else np.nan
        if lv[:,:,z].sum()>15:
            ax.plot(np.nonzero(lv[:,:,z])[0].mean()-x0,np.nonzero(lv[:,:,z])[1].mean()-y0,"r+",ms=8)
        col="red" if (np.isfinite(k) and k>3) else "white"
        ax.set_title(f"z{z}\nseg {k:.1f} | img {ki:.1f}",fontsize=7,color=col); ax.axis("off")
    # reslices: coronal = fixed y=cym plane -> (X,Z); sagittal = fixed x=cxm -> (Y,Z)
    zf=dz/dx
    zrow=lambda z,f: z*(int(round(Z*f))-1)/(Z-1)  # zoom maps row z -> z*(Zout-1)/(Z-1)
    ax=fig.add_subplot(gs[1,:ncol//2]); cor=ed[x0:x1,cym,:]  # (X,Z)
    ax.imshow(zoom(cor.T,(zf,1),order=1),cmap="gray",vmin=0,vmax=vmax,origin="lower",aspect=1)
    ax.set_title("coronal reslice through LV centre (z stretched to mm)",fontsize=8)
    for z in range(Z):
        ax.axhline(zrow(z,zf),color="cyan",lw=0.3,alpha=0.5)
        row=d[d.z==z]
        if len(row) and np.isfinite(row.epi_spike_x.iloc[0]):
            ax.annotate("",xy=(cxm-x0+row.epi_spike_x.iloc[0]/dx,zrow(z,zf)),xytext=(cxm-x0,zrow(z,zf)),arrowprops=dict(arrowstyle="->",color="red" if row.epi_spike.iloc[0]>3 else "yellow",lw=1))
    ax.set_yticks([]); ax.set_xticks([])
    ax=fig.add_subplot(gs[1,ncol//2:]); sag=ed[cxm,y0:y1,:]
    ax.imshow(zoom(sag.T,(dz/dy,1),order=1),cmap="gray",vmin=0,vmax=vmax,origin="lower",aspect=1)
    ax.set_title("sagittal reslice through LV centre (arrow = phase-median epi-centroid spike; red >3mm)",fontsize=8)
    for z in range(Z):
        ax.axhline(zrow(z,dz/dy),color="cyan",lw=0.3,alpha=0.5)
        row=d[d.z==z]
        if len(row) and np.isfinite(row.epi_spike_y.iloc[0]):
            ax.annotate("",xy=(cym-y0+row.epi_spike_y.iloc[0]/dy,zrow(z,dz/dy)),xytext=(cym-y0,zrow(z,dz/dy)),arrowprops=dict(arrowstyle="->",color="red" if row.epi_spike.iloc[0]>3 else "yellow",lw=1))
    ax.set_yticks([]); ax.set_xticks([])
    fig.suptitle(title or rel,fontsize=9); fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

if __name__=="__main__":
    df=pd.read_csv(sys.argv[1]); outdir=sys.argv[2]; os.makedirs(outdir,exist_ok=True)
    for rel in sys.argv[3:]:
        render(rel,df,os.path.join(outdir,rel.split("/")[-1]+".png")); print("ok",rel)
