import numpy as np, sigpy as sp, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions
import cs_recon as cr
dev=sp.Device(0); xp=dev.xp
DAT="/tmp/fiss_inspect/meas_MID00227_FID105839_FISS_NR4_TR2p47_a50_24x1877.dat"
d=read_fiss_twix(DAT, verbose=False); M=d['matrix']; G=192; nc=16
kd=d['kdata'][:,:nc,:]; dirs=phyllotaxis_directions(d['n_il'],d['n_spokes_per_il'])
coord=cr.radial_coords_for(dirs,d['n_read'],G)
g=np.load('/tmp/fiss_inspect/gating.npz'); cbin_sp=g['cbin'][d['interleave']]
img_sp=~d['is_nav'] & (cbin_sp>=0)
mps=cr.estimate_coil_maps(kd[img_sp],coord[img_sp],G,dev)
cf=sp.to_device(coord[img_sp].reshape(-1,3),dev)
dcf=sp.to_device((np.linalg.norm(coord[img_sp].reshape(-1,3),axis=1)**2).astype(np.float32),dev)
acc=xp.zeros((G,G,G),dtype=xp.complex64)
for c in range(nc):
    k=sp.to_device(kd[img_sp][:,c,:].reshape(-1).astype(np.complex64),dev)
    acc+=xp.conj(mps[c])*sp.nufft_adjoint(k*dcf,cf,oshape=(G,G,G))
stat=xp.abs(acc).get()
prod=np.load('/home/minsukc/vggt/scratch/data/fiss/recon_preview/cs_v01_prod.npz')['resp0']  #(25,192^3) full fov
pmean=prod.mean(0)  # time-average ~ static
# same transverse slice through heart
z=96  # mid of 192
fig,axs=plt.subplots(1,3,figsize=(15,5))
for ax,im,t in zip(axs,[stat[:,:,z],pmean[:,:,z],prod[0][:,:,z]],
   ['STATIC all-spokes (ceiling)','CS time-avg (LLR)','CS phase0 (LLR)']):
    ax.imshow(im.T,cmap='gray',vmax=np.percentile(im,99.5),origin='lower'); ax.set_title(t); ax.axis('off')
plt.tight_layout(); plt.savefig('/home/minsukc/vggt/scratch/data/fiss/recon_preview/ceiling_compare.png',dpi=100)
# sharpness proxy (gradient energy) heart region
def sharp(im): 
    c=im[60:130,60:130]; return float(np.abs(np.gradient(c)).mean()/ (c.mean()+1e-9))
print('sharpness static %.3f  CS-mean %.3f  CS-phase0 %.3f'%(sharp(stat[:,:,z]),sharp(pmean[:,:,z]),sharp(prod[0][:,:,z])))
print('saved ceiling_compare.png')
