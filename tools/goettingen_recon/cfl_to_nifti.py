#!/usr/bin/env python
"""Convert a BART .cfl/.hdr RT-NLINV reconstruction to a magnitude NIfTI cine.

BART dim layout for these recons: dim0=X, dim1=Y, dim10=T(frames), dim13=Z(slices);
all other dims are singleton. Output NIfTI is 4D [X, Y, Z, T] magnitude (float32).

The raw-mode recon lives on a 1.5x-oversampled grid (240 for a 160 matrix); by default we
center-crop back to the nominal 160 matrix (--matrix 160; pass 0 to keep the full grid).
"""
import argparse, sys
import numpy as np


def readcfl(name):
    name = name[:-4] if name.endswith('.cfl') else name
    with open(name + '.hdr') as f:
        f.readline()                      # "# Dimensions"
        dims = [int(x) for x in f.readline().split()]
    n = int(np.prod(dims))
    with open(name + '.cfl', 'rb') as f:
        data = np.fromfile(f, dtype=np.complex64, count=n)
    return data.reshape(dims, order='F')   # BART is column-major


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cfl')
    ap.add_argument('out_nii')
    ap.add_argument('--matrix', type=int, default=160,
                    help='center-crop X,Y to this size (0 = keep full grid)')
    ap.add_argument('--spacing', type=float, nargs=3, default=[1.6, 1.6, 6.0],
                    help='voxel spacing mm (X Y Z); FOV 256mm/160 = 1.6mm, slice 6mm')
    args = ap.parse_args()

    vol = readcfl(args.cfl)               # complex, full BART dim list
    dims = vol.shape
    X, Y = dims[0], dims[1]
    T = dims[10] if len(dims) > 10 else 1
    Z = dims[13] if len(dims) > 13 else 1
    # collapse to [X, Y, Z, T]
    vol = vol.reshape(X, Y, -1, order='F')             # squeeze middle singletons via order
    # robust: move by explicit axes instead
    vol = readcfl(args.cfl)
    arr = np.abs(vol).astype(np.float32)
    arr = np.moveaxis(arr, [0, 1, 13, 10], [0, 1, 2, 3]) if arr.ndim > 13 else arr
    arr = np.squeeze(arr)
    if arr.ndim == 3:                                   # single slice -> [X,Y,T]
        arr = arr[:, :, None, :] if arr.shape[-1] == T else arr[..., None]
    # now arr should be [X, Y, Z, T]
    if arr.ndim != 4:
        print(f'WARN unexpected ndim {arr.ndim}, shape {arr.shape}', file=sys.stderr)

    if args.matrix and arr.shape[0] > args.matrix:
        m = args.matrix
        x0 = (arr.shape[0] - m) // 2
        y0 = (arr.shape[1] - m) // 2
        arr = arr[x0:x0 + m, y0:y0 + m]

    import nibabel as nib
    aff = np.diag(args.spacing + [1.0])
    nib.save(nib.Nifti1Image(arr, aff), args.out_nii)
    print(f'wrote {args.out_nii} shape={arr.shape} (X,Y,Z,T) range=[{arr.min():.3g},{arr.max():.3g}]')


if __name__ == '__main__':
    main()
