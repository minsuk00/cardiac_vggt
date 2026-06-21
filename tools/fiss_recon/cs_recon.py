"""Motion-resolved compressed-sensing recon for FISS free-running data.

Per respiratory bin, reconstruct a 25-cardiac-phase 4D volume at 112^3 by
solving (PDHG, sigpy, GPU):
    min_x  0.5 * sum_i || Sense_i(mps) x_i - y_i ||^2
           + lam_s * TV_spatial(x) + lam_t * TV_cardiac(x)
where i indexes cardiac bins and x is (n_card, M, M, M) complex.

Coil maps (ESPIRiT) are estimated once from all motion-averaged imaging spokes.
Readout 2x oversampling is removed (crop 224->112 in projection space) so the
solve runs at the true 112^3 matrix instead of the 224^3 gridding FOV.

Run on a GPU node (A40): micromamba run -n fiss-recon python cs_recon.py <dat> [opts]
"""
import argparse
import numpy as np
import sigpy as sp
import sigpy.mri
from sigpy import backend

from twix_io import read_fiss_twix
from trajectory import phyllotaxis_directions


# sigpy's Embed._apply (adjoint of Slice) hardcodes np.zeros -> breaks on GPU.
# Patch it to use the input's array backend so Slice works with cupy arrays.
def _embed_apply_device(self, input):
    xp = backend.get_array_module(input)
    output = xp.zeros(self.oshape, dtype=input.dtype)
    output[self.idx] = input
    return output


sp.linop.Embed._apply = _embed_apply_device


def radial_coords_for(dirs, n_read, grid):
    """Per-sample k-space coords for a recon grid of size `grid`.

    The full readout (k_max) maps to the grid Nyquist (+-grid/2), preserving the
    native 2 mm resolution while setting the recon FOV = grid * 2 mm. `grid` must
    be large enough to contain the whole thorax (~360 mm => grid >= 180); recon is
    then cropped to the central matrix^3 (heart). Reconstructing at the nominal
    matrix (112, 220 mm) would alias the body onto the heart.
    """
    kr = ((np.arange(n_read) - n_read / 2.0) / n_read) * grid   # span +-grid/2
    return (dirs[:, None, :] * kr[None, :, None]).astype(np.float32)


def estimate_coil_maps(kdata, coord, G, device, lp_frac=0.09):
    """Smooth-division coil maps (n_coil,G,G,G) at the full recon-grid size G.

    Grid each coil to a complex image, low-pass it (keep central k fraction
    `lp_frac`), then S_c = LP(C_c) / sqrt(sum_c |LP(C_c)|^2). Robust for radial
    (no ESPIRiT calibration / k-space-centering pitfalls). A loose support mask
    (morphologically closed) zeros only true background, not interior pixels.
    """
    xp = device.xp
    nc = kdata.shape[1]
    cf = sp.to_device(coord.reshape(-1, 3), device)
    dcf = sp.to_device((np.linalg.norm(coord.reshape(-1, 3), axis=1)**2
                        ).astype(np.float32), device)
    w = max(8, int(G * lp_frac)); lo = G // 2 - w // 2
    win = xp.zeros((G, G, G), dtype=xp.float32)
    win[lo:lo+w, lo:lo+w, lo:lo+w] = 1.0   # centered: matches sp.fft convention
    C = xp.zeros((nc, G, G, G), dtype=xp.complex64)
    for c in range(nc):
        k = sp.to_device(kdata[:, c, :].reshape(-1).astype(np.complex64), device)
        img = sp.nufft_adjoint(k * dcf, cf, oshape=(G, G, G))
        C[c] = sp.ifft(sp.fft(img, axes=(0, 1, 2)) * win, axes=(0, 1, 2))
    rss = xp.sqrt((xp.abs(C) ** 2).sum(0))
    # support mask, with interior holes filled on the host (cupy lacks
    # binary_fill_holes) so coil maps don't punch black blobs into the recon
    from scipy import ndimage as ndi
    m = sp.to_device(rss, sp.cpu_device) > 0.02 * float(rss.max())
    m = ndi.binary_closing(m, iterations=2)
    m = ndi.binary_fill_holes(m)
    mask = sp.to_device(m.astype(np.float32), device)
    return (C / (rss + 1e-8) * mask).astype(xp.complex64)


class LocalLowRank(sp.prox.Prox):
    """Locally-low-rank prox over the cardiac dimension (the paper's regularizer).

    Partition the volume into b^3 spatial blocks; for each block form a
    (n_card x b^3) Casorati matrix and singular-value soft-threshold it. Unlike
    GLOBAL low-rank, in a small block the static background does not dwarf the
    cardiac-motion components, so LLR separates motion from noise -> clean AND
    moving. A cycling block-shift each call avoids blocking artifacts.

    Threshold is relative to each block's largest singular value (lamda ~ 0.05,
    matching the paper's LRtWeight); absolute thresholds are unusable because the
    data scale and block size both shift the singular-value magnitudes.
    """
    def __init__(self, ishape, lamda, block=8):
        self.lamda = lamda
        self.block = block
        self._call = 0
        super().__init__(ishape)

    def _prox(self, alpha, input):
        xp = backend.get_array_module(input)
        T, X, Y, Z = self.shape
        b = self.block
        sh = (self._call * 3) % b            # cycling shift -> no fixed block grid
        self._call += 1
        x = xp.roll(input, (sh, sh, sh), axis=(1, 2, 3))
        nx, ny, nz = X // b, Y // b, Z // b
        xb = (x.reshape(T, nx, b, ny, b, nz, b)
              .transpose(1, 3, 5, 0, 2, 4, 6)
              .reshape(nx * ny * nz, T, b * b * b))       # (nblk, T, P)
        gram = xb @ xb.conj().transpose(0, 2, 1)          # (nblk, T, T)
        w, U = xp.linalg.eigh(gram)
        s = xp.sqrt(xp.maximum(w, 0)) + 1e-12
        factor = xp.maximum(1.0 - self.lamda * s[:, -1:] / s, 0.0)  # (nblk, T)
        D = (U * factor[:, None, :]) @ U.conj().transpose(0, 2, 1)
        xb = D @ xb
        x = (xb.reshape(nx, ny, nz, T, b, b, b)
             .transpose(3, 0, 4, 1, 5, 2, 6).reshape(T, X, Y, Z))
        x = xp.roll(x, (-sh, -sh, -sh), axis=(1, 2, 3))
        return x.astype(input.dtype)


class TemporalLowRank(sp.prox.Prox):
    """SVT prox on the temporal Casorati matrix (n_card x n_voxel).

    Pools SNR across all cardiac phases: the static anatomy is rank-1 (shared by
    all ~10k spokes), motion lives in a few extra singular components. This is the
    right denoiser for free-running cine -- TV cannot share across phases.
    """
    def __init__(self, ishape, lamda):
        self.lamda = lamda
        super().__init__(ishape)

    def _prox(self, alpha, input):
        # SVT on the wide (T x N) Casorati matrix via the small T x T Gram matrix
        # (cuSOLVER gesvd needs m>=n, and the full SVD of a T x 7M matrix is huge).
        # The threshold is RELATIVE to the largest singular value (lamda is a
        # fraction in ~[0.02,0.15]); an absolute threshold is meaningless because
        # s ~ sqrt(N) ~ thousands. SVT(X) = U diag(max(s-t,0)/s) U^H X.
        xp = backend.get_array_module(input)
        T = self.shape[0]
        X = input.reshape(T, -1)
        gram = X @ X.conj().T                              # (T, T) Hermitian
        w, U = xp.linalg.eigh(gram)                        # w = s^2 >= 0
        s = xp.sqrt(xp.maximum(w, 0)) + 1e-12
        t = self.lamda * float(s.max())                    # relative threshold
        factor = xp.maximum(1.0 - t / s, 0.0)              # s_thr / s
        D = (U * factor[None, :]) @ U.conj().T            # (T,T)
        return (D @ X).reshape(self.shape).astype(input.dtype)


def cs_recon_resp(kdata, coord, mps, cbin_sp, sel_base, n_card, M, device,
                  lam_lr=0.05, lam_tvt=0.01, dcf_pow=0.5, iters=80, reg='llr',
                  block=8, full_fov=False):
    """Reconstruct one respiratory bin -> (n_card, M, M, M) complex (on host).

    Temporal global low-rank (lam_lr) is the primary regularizer; optional light
    spatial wavelet (lam_s). The data term is weighted by |kr|^dcf_pow as a mild
    density-compensation preconditioner (dcf_pow=0.5 balances convergence vs
    noise; 1.0 over-fits noisy outer k-space, 0.0 converges slowly).
    """
    xp = device.xp
    G = mps.shape[1]                                      # recon grid (=n_read)
    lo = (G - M) // 2                                     # central-crop offset
    ishape = (n_card, G, G, G)
    ncoil = mps.shape[0]
    blocks, ys = [], []
    for cb in range(n_card):
        sel = sel_base & (cbin_sp == cb)
        c_i = sp.to_device(coord[sel].reshape(-1, 3), device)
        wsq = sp.to_device(
            (np.linalg.norm(coord[sel].reshape(-1, 3), axis=1) ** dcf_pow
             ).astype(np.float32), device)
        W = sp.linop.Multiply((ncoil, c_i.shape[0]), wsq[None, :])
        S = sp.mri.linop.Sense(mps, coord=c_i)           # (G,G,G)->(coil,nsamp)
        blocks.append(W * S * sp.linop.Slice(ishape, (cb,)))
        yk = sp.to_device(
            kdata[sel].transpose(1, 0, 2).reshape(ncoil, -1).astype(np.complex64),
            device)
        ys.append((yk * wsq[None, :]).ravel())
    A = sp.linop.Vstack(blocks)                          # ishape (n_card,G,G,G)
    y = xp.concatenate(ys).reshape(A.oshape)

    # normalize so the adjoint recon peaks ~1 -> lam_* are data-scale-free
    scale = float(xp.abs(A.H(y)).max()) + 1e-12
    y = y / scale

    lr_prox = (LocalLowRank(ishape, lam_lr, block=block) if reg in ('llr', 'llrtv')
               else TemporalLowRank(ishape, lam_lr))
    if reg == 'llrtv' and lam_tvt > 0:
        # paper recipe: LLR(cardiac) + TVt(cardiac), no spatial TV. Combined via
        # PDHG with G=[I; grad_t] and a stacked prox [LLR ; soft-thresh(TVt)].
        FDt = sp.linop.FiniteDifference(ishape, axes=(0,))
        Greg = sp.linop.Vstack([sp.linop.Identity(ishape), FDt])
        proxg = sp.prox.Stack([lr_prox, sp.prox.L1Reg(FDt.oshape, lam_tvt)])
        x = sp.app.LinearLeastSquares(
            A, y, x=xp.zeros(ishape, dtype=xp.complex64),
            G=Greg, proxg=proxg, max_iter=iters, show_pbar=True).run()
    else:
        x = sp.app.LinearLeastSquares(
            A, y, x=xp.zeros(ishape, dtype=xp.complex64),
            proxg=lr_prox, max_iter=iters, accelerate=True, show_pbar=True).run()
    if not full_fov:
        x = x[:, lo:lo+M, lo:lo+M, lo:lo+M]              # crop central FOV (heart)
    return sp.to_device(x * scale, sp.cpu_device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dat')
    ap.add_argument('--gating', default='/tmp/fiss_inspect/gating.npz')
    ap.add_argument('--ncard', type=int, default=25)
    ap.add_argument('--rbins', type=int, nargs='+', default=[0])
    ap.add_argument('--coils', type=int, default=20)
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--lam_lr', type=float, default=0.05, help='low-rank weight (rel. to top SV; paper LRt=0.05)')
    ap.add_argument('--lam_tvt', type=float, default=0.01, help='cardiac TV weight (paper TVt=0.01)')
    ap.add_argument('--reg', choices=['llr', 'glr', 'llrtv'], default='llr', help='llrtv = paper LLR+TVt')
    ap.add_argument('--block', type=int, default=8, help='LLR spatial block size (must divide grid)')
    ap.add_argument('--full_fov', action='store_true', help='output full recon grid (no heart crop)')
    ap.add_argument('--dcf_pow', type=float, default=0.5, help='data |kr|^p weighting (0=none,1=full dcf)')
    ap.add_argument('--grid', type=int, default=192, help='recon grid (>=180 to hold the thorax)')
    ap.add_argument('--out', default='/tmp/fiss_inspect/cs_recon.npy')
    args = ap.parse_args()

    dev = sp.Device(0)
    d = read_fiss_twix(args.dat)
    M = d['matrix']                                          # output crop (heart FOV)
    G = args.grid                                            # recon grid (full FOV)
    nc = min(args.coils, d['n_coil'])
    kd = d['kdata'][:, :nc, :]                               # full readout, no de-os
    dirs = phyllotaxis_directions(d['n_il'], d['n_spokes_per_il'])
    coord = radial_coords_for(dirs, d['n_read'], G)          # span +-G/2
    g = np.load(args.gating)
    cbin_sp = g['cbin'][d['interleave']]
    rbin_sp = g['rbin'][d['interleave']]
    img_sp = ~d['is_nav'] & (cbin_sp >= 0)

    print(f"estimating coil maps at {G}^3 ...")
    mps = estimate_coil_maps(kd[img_sp], coord[img_sp], G, dev)

    out = {}
    for rb in args.rbins:
        sel_base = img_sp & (rbin_sp == rb)
        print(f"CS recon resp bin {rb}: {int(sel_base.sum())} spokes")
        vol = cs_recon_resp(kd, coord, mps, cbin_sp, sel_base, args.ncard, M, dev,
                            lam_lr=args.lam_lr, lam_tvt=args.lam_tvt, dcf_pow=args.dcf_pow,
                            iters=args.iters, reg=args.reg, block=args.block,
                            full_fov=args.full_fov)
        out[f'resp{rb}'] = np.abs(vol).astype(np.float32)
    np.savez(args.out.replace('.npy', '.npz'), **out)
    print('saved', args.out.replace('.npy', '.npz'))


if __name__ == '__main__':
    main()
