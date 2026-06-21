"""Single static-frame gridding recon — validates the phyllotaxis trajectory.

Grids ALL spokes as one frame (45048 spokes >> Nyquist for 112^3) via
density-compensated adjoint NUFFT, RSS coil-combines, and saves 3 orthogonal
centre slices. A recognizable thorax => trajectory orientation/scaling correct.

Run: micromamba run -n fiss-recon python tools/fiss_recon/recon_frame.py <dat> [--coils 8]
"""
import argparse
import numpy as np
import sigpy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions, radial_coords, radial_density


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dat')
    ap.add_argument('--coils', type=int, default=8, help='use first N coils (speed)')
    ap.add_argument('--out', default='/tmp/fiss_inspect/frame_recon.png')
    args = ap.parse_args()

    d = read_fiss_twix(args.dat)
    M = d['matrix']
    G = d['n_read']                                  # oversampled grid (=2*M)
    lo = (G - M) // 2                                # crop offset
    nc = min(args.coils, d['n_coil'])

    # exclude SI navigator spokes (r==0): 1877 identical +z spokes that
    # over-sample the kz axis and streak the image. They are gating-only.
    img_sp = ~d['is_nav']
    kd = d['kdata'][img_sp][:, :nc, :]               # (spoke', coil, read)
    dirs = phyllotaxis_directions(d['n_il'], d['n_spokes_per_il'])[img_sp]
    coord = radial_coords(dirs, d['n_read'])         # (spoke', read, 3), span +-G/2
    dcf = radial_density(d['n_read'], M)             # (read,)  ~ kr^2
    # mild Hann apodization on the outer readout to tame gridding noise (the
    # CS recon won't need this; it's only for the sanity image).
    n = np.arange(d['n_read'])
    dcf = dcf * (0.5 + 0.5 * np.cos(np.pi * (n - d['n_read'] / 2) / (d['n_read'] / 2)))

    # flatten samples; apply density compensation
    coord_f = coord.reshape(-1, 3)                   # (spoke*read, 3)
    w = np.broadcast_to(dcf[None, :], (kd.shape[0], d['n_read'])).reshape(-1)
    print(f"gridding {coord_f.shape[0]} samples x {nc} coils -> {G}^3, crop {M}^3 ...")

    vol = np.zeros((M, M, M), dtype=np.float64)
    for c in range(nc):
        ksp = (kd[:, c, :].reshape(-1) * w).astype(np.complex64)
        img = sp.nufft_adjoint(ksp, coord_f, oshape=(G, G, G))
        img = img[lo:lo + M, lo:lo + M, lo:lo + M]   # crop oversampled FOV
        vol += np.abs(img) ** 2
        print(f"  coil {c} done")
    vol = np.sqrt(vol)

    # orthogonal centre slices
    cz, cy, cx = [s // 2 for s in vol.shape]
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    vmax = np.percentile(vol, 99.5)
    for ax, sl, name in zip(
        axs, [vol[cx, :, :], vol[:, cy, :], vol[:, :, cz]],
        ['axis0=cx (sag?)', 'axis1=cy (cor?)', 'axis2=cz (tra?)']):
        ax.imshow(sl.T, cmap='gray', vmax=vmax, origin='lower')
        ax.set_title(name); ax.axis('off')
    fig.suptitle('FISS single-frame gridding recon (trajectory validation)')
    plt.tight_layout(); plt.savefig(args.out, dpi=95)
    print('saved', args.out)
    np.save(args.out.replace('.png', '_vol.npy'), vol.astype(np.float32))


if __name__ == '__main__':
    main()
