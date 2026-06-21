"""Analytic spiral-phyllotaxis trajectory for the FISS free-running acquisition.

Piccini et al., MRM 2011 ("Spiral phyllotaxis"). For projection r within
interleave i (N = I*R total):
    n      = i + r*I
    phi(n) = n * golden_angle          (azimuth about z)
    theta(n) = (pi/2) * sqrt(n / N)     (polar from +z pole; sqrt => uniform density)
The first spoke of each interleave (r==0) is FORCED to exact SI (+z) as the
self-gating navigator (Di Sopra 2019 free-running modification).

Coordinates are returned in 1/FOV-normalised k-space units scaled to a matrix
of `matrix` (sigpy NUFFT convention: coords in [-matrix/2, matrix/2]).
The exact orientation/handedness convention is VALIDATED empirically with a
single-frame gridding recon (recon_frame.py), not assumed.
"""
import numpy as np

GOLDEN_ANGLE = np.deg2rad(137.50776405)  # 2*pi*(1 - 1/phi)


def phyllotaxis_directions(n_il, n_spokes_per_il, force_si_nav=True):
    """Unit direction (x,y,z) per spoke, shape (n_il*n_spokes_per_il, 3).

    Spoke ordering matches twix_io: interleave-major (i outer, r inner).
    """
    I, R = n_il, n_spokes_per_il
    N = I * R
    i = np.repeat(np.arange(I), R)
    r = np.tile(np.arange(R), I)
    n = i + r * I                      # phyllotaxis running index
    theta = (np.pi / 2.0) * np.sqrt(n / N)
    phi = n * GOLDEN_ANGLE
    dirs = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1).astype(np.float32)
    if force_si_nav:
        dirs[r == 0] = np.array([0., 0., 1.], dtype=np.float32)
    return dirs


def radial_coords(dirs, n_read):
    """Spoke directions -> per-sample k-space coords for sigpy NUFFT.

    Returns (n_spoke, n_read, 3) in sample units spanning [-n_read/2, n_read/2].
    Because the readout is 2x oversampled, these coords naturally span an
    oversampled grid of size n_read (= 2*matrix). Recon onto an (n_read)^3 grid
    then crop the central matrix^3 -> the acquired (2x) FOV is reconstructed and
    the outer half cropped, which removes FOV wrap-around aliasing.
    Each spoke is a full diameter through k-space centre.
    """
    kr = (np.arange(n_read) - n_read / 2.0)               # sample units
    coords = dirs[:, None, :] * kr[None, :, None]          # (spoke, read, 3)
    return coords.astype(np.float32)


def radial_density(n_read, matrix):
    """1-D radial density-compensation weight along the readout.

    3D radial: each k-shell of radius kr has surface area ~ kr^2, so the
    per-sample density compensation goes as kr^2 (NOT |kr|, which is the 2D
    case). The spiral-phyllotaxis sqrt-polar law makes the *angular* density
    uniform, so a radial-only kr^2 weight is appropriate.
    """
    kr = np.abs((np.arange(n_read) - n_read / 2.0) / n_read) * matrix
    w = kr ** 2
    w[kr == 0] = (0.5 / n_read) ** 2   # small non-zero weight at DC
    return w.astype(np.float32)
