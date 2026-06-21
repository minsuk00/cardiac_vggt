"""Render a reconstructed FISS 5D recon: per-phase montage + beating-heart GIF.

Input: the .npz from cs_recon.py (keys resp{r} -> (n_card, M, M, M) magnitude).
Run: micromamba run -n fiss-recon python view_recon.py <recon.npz> [--resp 0] [--outdir DIR]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--resp', type=int, default=0)
    ap.add_argument('--outdir', default='/home/minsukc/vggt/scratch/data/fiss/recon_preview')
    ap.add_argument('--tag', default=None)
    args = ap.parse_args()
    tag = args.tag or os.path.splitext(os.path.basename(args.npz))[0]

    v = np.load(args.npz)[f'resp{args.resp}']           # (n_card, X, Y, Z)
    ncard, M = v.shape[0], v.shape[1]
    # pick the most cardiac-dynamic central transverse (fixed-Z) slice
    zr = range(M // 3, 2 * M // 3)
    zc = list(zr)[int(np.argmax([v[:, M//4:3*M//4, M//4:3*M//4, z].var(0).mean()
                                 for z in zr]))]
    crop = slice(M // 5, 4 * M // 5)
    cine = v[:, crop, crop, zc]                          # (n_card, x, y)
    vmax = np.percentile(cine, 99.5)

    ncol = 5; nrow = int(np.ceil(ncard / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axs = np.atleast_2d(axs)
    for i in range(nrow * ncol):
        ax = axs[i // ncol, i % ncol]; ax.axis('off')
        if i < ncard:
            ax.imshow(cine[i].T, cmap='gray', vmax=vmax, origin='lower')
            ax.set_title(f't{i}', fontsize=8)
    fig.suptitle(f'{tag} resp{args.resp}: {ncard} cardiac phases (Z={zc})')
    plt.tight_layout()
    mont = os.path.join(args.outdir, f'{tag}_resp{args.resp}_montage.png')
    plt.savefig(mont, dpi=110); plt.close()

    fig, ax = plt.subplots(figsize=(4, 4)); ax.axis('off')
    im = ax.imshow(cine[0].T, cmap='gray', vmax=vmax, origin='lower')

    def upd(i):
        im.set_data(cine[i].T); ax.set_title(f'cardiac phase {i}/{ncard}'); return [im]
    a = anim.FuncAnimation(fig, upd, frames=ncard, interval=120, blit=False)
    gif = os.path.join(args.outdir, f'{tag}_resp{args.resp}.gif')
    a.save(gif, writer=anim.PillowWriter(fps=8)); plt.close()
    print('saved', mont, gif, '| dynamic Z =', zc)


if __name__ == '__main__':
    main()
