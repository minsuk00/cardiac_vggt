import numpy as np, sigpy as sp, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions
import cs_recon as cr
dev=sp.Device(0); xp=dev.xp
DAT="/tmp/fiss_inspect/meas_MID00227_FID105839_FISS_NR4_TR2p47_a50_24x1877.dat"
d=read_fiss_twix(DAT, verbose=False); M=d['matrix']; G=192; nc=16
kd=d['kdata'][:,:nc,:]
dirs=phyllotaxis_directions(d['n_il'],d['n_spokes_per_il'])
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
vol=xp.abs(acc).get()
lo=(G-M)//2; volc=vol[lo:lo+M,lo:lo+M,lo:lo+M]   # crop to heart
cz=M//2
fig,axs=plt.subplots(1,3,figsize=(13,4.5)); vmax=np.percentile(volc,99.5)
for ax,sl,t in zip(axs,[volc[:,:,cz],volc[:,M//2,:],volc[M//2,:,:]],['transverse','coronal','sagittal']):
    ax.imshow(sl.T,cmap='gray',vmax=vmax,origin='lower'); ax.set_title('static grid192->crop112 '+t); ax.axis('off')
plt.tight_layout(); plt.savefig('/home/minsukc/vggt/scratch/data/fiss/recon_preview/diag_static192.png',dpi=100)
print('saved diag_static192.png; vol range',float(volc.min()),float(volc.max()))
