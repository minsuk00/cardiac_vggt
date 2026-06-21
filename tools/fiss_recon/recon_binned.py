"""Motion-resolved gridding recon per (cardiac[,respiratory]) bin.

Validates self-gating: a diastole-vs-systole pair should show the LV cavity
change size. Per-bin gridding is heavily undersampled (~hundreds of spokes for
112^3) -> streaky; this is the qualitative motion check, not the final image
(that's the CS recon). All 23 imaging spokes of an interleave inherit the
interleave's cardiac/resp bin (navigator phase; interleave spans ~59 ms ~= 1
cardiac bin at this HR).

Run: micromamba run -n fiss-recon python recon_binned.py <dat> --cbins 0 12 --coils 10
"""
import argparse
import numpy as np
import sigpy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions, radial_coords, radial_density


def grid_spokes(kd_sel, coord_sel, dcf, G, M, nc):
    """kd_sel (nspoke,coil,read), coord_sel (nspoke,read,3) -> RSS vol (M^3)."""
    lo = (G - M) // 2
    cf = coord_sel.reshape(-1, 3)
    w = np.broadcast_to(dcf[None, :], kd_sel.shape[::2]).reshape(-1)
    vol = np.zeros((M, M, M))
    for c in range(nc):
        ksp = (kd_sel[:, c, :].reshape(-1) * w).astype(np.complex64)
        img = sp.nufft_adjoint(ksp, cf, oshape=(G, G, G))
        vol += np.abs(img[lo:lo+M, lo:lo+M, lo:lo+M]) ** 2
    return np.sqrt(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dat')
    ap.add_argument('--gating', default='/tmp/fiss_inspect/gating.npz')
    ap.add_argument('--cbins', type=int, nargs='+', default=[0, 12])
    ap.add_argument('--width', type=int, default=0,
                    help='pool cardiac bins within +-width (mod ncard) of each center')
    ap.add_argument('--ncard', type=int, default=25)
    ap.add_argument('--rbin', type=int, default=-1, help='-1 = pool all resp')
    ap.add_argument('--coils', type=int, default=10)
    ap.add_argument('--out', default='/tmp/fiss_inspect/binned.png')
    args = ap.parse_args()

    d = read_fiss_twix(args.dat)
    M, G, nc = d['matrix'], d['n_read'], min(args.coils, d['n_coil'])
    g = np.load(args.gating)
    cbin_il, rbin_il = g['cbin'], g['rbin']              # per-interleave

    dirs = phyllotaxis_directions(d['n_il'], d['n_spokes_per_il'])
    coord = radial_coords(dirs, d['n_read'])             # (spoke, read, 3)
    dcf = radial_density(d['n_read'], M)
    # Hann apodization on the outer readout (tames undersampling noise for the
    # qualitative check; the CS recon won't need it).
    n = np.arange(d['n_read'])
    dcf = dcf * (0.5 + 0.5 * np.cos(np.pi * (n - d['n_read']/2) / (d['n_read']/2)))
    il = d['interleave']; isnav = d['is_nav']
    # per-spoke bins from per-interleave assignment
    cbin_sp = cbin_il[il]; rbin_sp = rbin_il[il]

    vols = []
    for cb in args.cbins:
        centers = [(cb + k) % args.ncard for k in range(-args.width, args.width + 1)]
        sel = np.isin(cbin_sp, centers) & (~isnav)
        if args.rbin >= 0:
            sel &= (rbin_sp == args.rbin)
        nsp = int(sel.sum())
        print(f"cardiac bin {cb}: {nsp} spokes ({nsp/23:.0f} interleaves), gridding...")
        vols.append(grid_spokes(d['kdata'][sel], coord[sel], dcf, G, M, nc))
    vols = np.array(vols)

    cz = M // 2
    n = len(args.cbins)
    fig, axs = plt.subplots(2, n, figsize=(4*n, 8))
    vmax = np.percentile(vols, 99)
    axs = np.atleast_2d(axs)
    for j, cb in enumerate(args.cbins):
        axs[0, j].imshow(vols[j][:, :, cz].T, cmap='gray', vmax=vmax, origin='lower')
        axs[0, j].set_title(f'cardiac bin {cb}  (transverse)'); axs[0, j].axis('off')
        axs[1, j].imshow(vols[j][:, M//2, :].T, cmap='gray', vmax=vmax, origin='lower')
        axs[1, j].set_title(f'cardiac bin {cb}  (coronal)'); axs[1, j].axis('off')
    plt.tight_layout(); plt.savefig(args.out, dpi=95)
    print('saved', args.out)
    np.save(args.out.replace('.png', '_vols.npy'), vols.astype(np.float32))


if __name__ == '__main__':
    main()
