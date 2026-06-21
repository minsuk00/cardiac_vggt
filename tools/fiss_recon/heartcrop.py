import numpy as np, sigpy as sp, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions
import cs_recon as cr
dev=sp.Device(0); xp=dev.xp
# static ceiling at 192 (recompute, save vol)
DAT="/tmp/fiss_inspect/meas_MID00227_FID105839_FISS_NR4_TR2p47_a50_24x1877.dat"
d=read_fiss_twix(DAT,verbose=False); G=192; nc=16
kd=d['kdata'][:,:nc,:]; dirs=phyllotaxis_directions(d['n_il'],d['n_spokes_per_il'])
coord=cr.radial_coords_for(dirs,d['n_read'],G)
g=np.load('/tmp/fiss_inspect/gating.npz'); cbin=g['cbin'][d['interleave']]; img_sp=~d['is_nav']&(cbin>=0)
mps=cr.estimate_coil_maps(kd[img_sp],coord[img_sp],G,dev)
cf=sp.to_device(coord[img_sp].reshape(-1,3),dev); dcf=sp.to_device((np.linalg.norm(coord[img_sp].reshape(-1,3),axis=1)**2).astype(np.float32),dev)
acc=xp.zeros((G,G,G),dtype=xp.complex64)
for c in range(nc):
    k=sp.to_device(kd[img_sp][:,c,:].reshape(-1).astype(np.complex64),dev); acc+=xp.conj(mps[c])*sp.nufft_adjoint(k*dcf,cf,oshape=(G,G,G))
stat=xp.abs(acc).get()
fin=np.load('/home/minsukc/vggt/scratch/data/fiss/recon_preview/cs_v01_final.npz')['resp0']  # (25,192^3)
# heart crop: central region around heart; transverse slice z=96, crop 60..130
z=96; cr_=slice(58,132)
fig,axs=plt.subplots(1,3,figsize=(15,5.3))
for ax,im,t in zip(axs,[stat[cr_,cr_,z],fin.mean(0)[cr_,cr_,z],fin[0][cr_,cr_,z]],
   ['STATIC all-spokes (ceiling)','FINAL CS time-avg','FINAL CS phase0']):
    ax.imshow(im.T,cmap='gray',vmax=np.percentile(im,99.5),origin='lower');ax.set_title(t);ax.axis('off')
plt.tight_layout();plt.savefig('/home/minsukc/vggt/scratch/data/fiss/recon_preview/heartcrop_compare.png',dpi=110)
# also a heart-cropped beating gif
import matplotlib.animation as anim
cine=fin[:,cr_,cr_,z]; vmax=np.percentile(cine,99)
figg,axg=plt.subplots(figsize=(4.5,4.5));axg.axis('off')
im0=axg.imshow(cine[0].T,cmap='gray',vmax=vmax,origin='lower')
def u(i): im0.set_data(cine[i].T);axg.set_title(f'heart phase {i}/25');return[im0]
anim.FuncAnimation(figg,u,frames=25,interval=110).save('/home/minsukc/vggt/scratch/data/fiss/recon_preview/cs_v01_final_heart.gif',writer=anim.PillowWriter(fps=9))
print('saved heartcrop_compare.png + cs_v01_final_heart.gif')
