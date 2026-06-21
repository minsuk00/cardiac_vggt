"""Full-FOV cines (whole thorax) from a full_fov cs_recon .npz.

Saves transverse + coronal beating-heart GIFs over the whole 192^3 FOV (not the
heart crop), plus a 25-phase montage. Run:
  micromamba run -n fiss-recon python view_fullfov.py <npz> [--resp 0] [--tag X]
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim

ap = argparse.ArgumentParser()
ap.add_argument('npz')
ap.add_argument('--resp', type=int, default=0)
ap.add_argument('--tag', default='fullfov')
ap.add_argument('--outdir', default='/home/minsukc/vggt/scratch/data/fiss/recon_preview')
args = ap.parse_args()

v = np.load(args.npz)[f'resp{args.resp}']        # (n_card, X, Y, Z) full FOV
ncard, M = v.shape[0], v.shape[1]
cz, cy = M // 2, M // 2
vmax = np.percentile(v, 99.5)


def gif(stack, name, title):
    fig, ax = plt.subplots(figsize=(5, 5)); ax.axis('off')
    im = ax.imshow(stack[0].T, cmap='gray', vmax=vmax, origin='lower')
    def u(i):
        im.set_data(stack[i].T); ax.set_title(f'{title}  phase {i}/{ncard}'); return [im]
    a = anim.FuncAnimation(fig, u, frames=ncard, interval=110, blit=False)
    p = os.path.join(args.outdir, name)
    a.save(p, writer=anim.PillowWriter(fps=9)); plt.close(); print('saved', p)


gif(v[:, :, :, cz], f'{args.tag}_transverse_full.gif', 'transverse (full FOV)')
gif(v[:, :, cy, :], f'{args.tag}_coronal_full.gif', 'coronal (full FOV)')

# montage of all phases, transverse mid-slice, full FOV
ncol = 5; nrow = int(np.ceil(ncard / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2.2, nrow * 2.2))
axs = np.atleast_2d(axs)
for i in range(nrow * ncol):
    ax = axs[i // ncol, i % ncol]; ax.axis('off')
    if i < ncard:
        ax.imshow(v[i][:, :, cz].T, cmap='gray', vmax=vmax, origin='lower')
        ax.set_title(f't{i}', fontsize=8)
fig.suptitle(f'{args.tag} full-FOV transverse, 25 cardiac phases')
plt.tight_layout()
p = os.path.join(args.outdir, f'{args.tag}_montage_full.png')
plt.savefig(p, dpi=110); plt.close(); print('saved', p)
