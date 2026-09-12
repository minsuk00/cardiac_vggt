"""F4: per-phase spike vectors for a flagged slice vs a clean slice (cross-phase consistency)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *; import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
FIG="/home/minsukc/vggt/temp/misalign_v2/figs"
def phase_kinks(rel,z):
    seg,(dx,dy,dz)=load_seg(rel); T=seg.shape[3]; out=[]
    for t in range(T):
        cs=[]
        for zz in (z-1,z,z+1):
            m=np.isin(seg[:,:,zz,t],(1,2)); ix,iy=np.nonzero(m); cs.append((ix.mean()*dx,iy.mean()*dy))
        cs=np.array(cs); out.append(cs[1]-0.5*(cs[0]+cs[2]))
    return np.array(out)
cases=[("MNMs_sax/MNMs_A1E9Q1",3,"severe (MNMs_A1E9Q1 z3)"),("ACDC_sax/ACDC_patient034",3,"real 8mm (ACDC_patient034 z3)"),("CMRxRecon2024/Cine_combined/CMRx24_Train_P025",5,"clean (CMRx24_Train_P025 z5)"),("ACDC_sax/ACDC_patient007",5,"clean, docs/94 said 32.8mm (ACDC_patient007 z5)")]
fig,axs=plt.subplots(1,4,figsize=(12,3.2),dpi=120)
for ax,(rel,z,ttl) in zip(axs,cases):
    k=phase_kinks(rel,z); ax.scatter(k[:,0],k[:,1],c=np.arange(12),cmap="twilight",s=30); ax.plot(0,0,"k+",ms=10)
    for t in range(12): ax.annotate(str(t),(k[t,0],k[t,1]),fontsize=6)
    ax.set_xlim(-30,30); ax.set_ylim(-30,30); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_title(ttl,fontsize=8); ax.set_xlabel("x offset vs neighbours (mm)")
axs[0].set_ylabel("y offset (mm)"); fig.suptitle("per-phase epicardial-centroid spike of ONE slice, all 12 cardiac phases (colour = phase)",fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/F4_phase_consistency.png"); plt.close(fig); print("ok")
