"""Read a FISS free-running Siemens TWIX .dat into k-space + per-spoke indexing.

Run env: `fiss-recon` (twixtools + numpy). Reading needs no GPU.

The free-running FISS acquisition (Zenodo 13868462) is 3D radial spiral
phyllotaxis: I interleaves x R spokes each, the first spoke of every interleave
(`TAGFLAG1`) oriented along SI as a self-gating navigator. See
docs/14_fiss_5d_reconstruction.md for the full data characterization.
"""
import numpy as np
import twixtools


def read_fiss_twix(path, verbose=True):
    """Return a dict with k-space and per-spoke metadata.

    keys:
      kdata    : complex64 (n_spoke, n_coil, n_read)
      interleave : int (n_spoke,)  index i of the interleave each spoke belongs to
      spoke_in_il: int (n_spoke,)  index r within the interleave (0 = SI navigator)
      is_nav   : bool (n_spoke,)   True for r==0 (TAGFLAG1)
      n_il, n_spokes_per_il, n_coil, n_read : ints
      quaternion : (4,) slab orientation (constant across spokes)
      slab_pos_mm: (3,) slab center offset (dSag,dCor,dTra)
      fov_mm, matrix, res_mm : floats/ints from the protocol header
    """
    twix = twixtools.read_twix(path, parse_pmu=False, parse_geometry=True,
                               verbose=False)
    meas = twix[-1]
    img = [m for m in meas['mdb'] if m.is_image_scan()]
    n_spoke = len(img)
    n_coil, n_read = img[0].data.shape

    kdata = np.empty((n_spoke, n_coil, n_read), dtype=np.complex64)
    is_nav = np.zeros(n_spoke, dtype=bool)
    for s, m in enumerate(img):
        kdata[s] = m.data
        if 'TAGFLAG1' in m.get_active_flags():
            is_nav[s] = True

    nav_pos = np.flatnonzero(is_nav)
    n_il = len(nav_pos)
    assert n_spoke % n_il == 0, f"{n_spoke} spokes not divisible by {n_il} interleaves"
    R = n_spoke // n_il
    # spokes are stored in acquisition order: interleave-major, spoke-within-minor
    interleave = np.repeat(np.arange(n_il), R)
    spoke_in_il = np.tile(np.arange(R), n_il)
    # sanity: every nav must be the r==0 of its interleave
    assert np.array_equal(nav_pos, np.arange(n_il) * R), \
        "navigator spokes are not at the start of each interleave block"

    my = meas['hdr']['MeasYaps']
    sl = my['sSliceArray']['asSlice'][0]
    pos = sl.get('sPosition', {})
    base = my['sKSpace']['lBaseResolution']
    fov = sl['dReadoutFOV']
    out = dict(
        kdata=kdata, interleave=interleave, spoke_in_il=spoke_in_il, is_nav=is_nav,
        n_il=n_il, n_spokes_per_il=R, n_coil=n_coil, n_read=n_read,
        quaternion=np.asarray(img[0].mdh.SliceData.Quaternion, dtype=np.float32),
        slab_pos_mm=np.array([pos.get('dSag', 0.), pos.get('dCor', 0.),
                              pos.get('dTra', 0.)], dtype=np.float32),
        fov_mm=float(fov), matrix=int(base), res_mm=float(fov) / int(base),
    )
    if verbose:
        print(f"FISS twix: {n_spoke} spokes = {n_il} il x {R}, "
              f"{n_coil} coils, {n_read} read; matrix {base}, "
              f"FOV {fov:.0f}mm, res {out['res_mm']:.2f}mm")
    return out


if __name__ == "__main__":
    import sys
    d = read_fiss_twix(sys.argv[1])
    print({k: (v.shape if hasattr(v, 'shape') else v)
           for k, v in d.items() if k != 'kdata'})
