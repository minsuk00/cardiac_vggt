#!/usr/bin/env python3
"""Export one MIITT real-time volunteer into a fetal_cmr_4d recon directory.

fetal_cmr_4d normally starts from ktrecon (Philips ReconFrame) output. MIITT is
already reconstructed, so we inject NIfTIs at the pipeline's post-ktrecon entry
point (the `data/` dir) and auto-generate the masks that would otherwise be drawn
by hand in MITK.

Produces, under <RECONDIR>:
    data/s01_rlt_ab.nii.gz   real-time magnitude stack, 4D (X,Y,Z=13,T=180)
    data/s01_dc_ab.nii.gz    temporal-mean "DC"/static image, 3D (X,Y,Z=13)
    mask/s01_mask_heart.nii.gz   per-slice heart ROI, 3D (X,Y,Z=13)
    mask/mask_chest.nii.gz       3D recon-FOV mask (dilated heart hull)
    cardsync/                     (empty, created for later stages)

Heart localisation is fully automatic: temporal standard deviation across the 180
free-breathing frames peaks at the beating blood pool, giving a clean heart ROI
with no manual segmentation.

Usage:
    python baselines/fetal_cmr_4d/export_miitt.py Volunteer1 [Volunteer2 ...]
"""
import sys
import os
import numpy as np
import nibabel as nib
from scipy import ndimage

MIITT_NIFTI = "/home/minsukc/MIITT/nifti/{vol}/realtime/sax/4d_recon.nii.gz"
RECON_ROOT = "/home/minsukc/vggt/scratch/fetal_cmr_4d/recon"

# heart localisation via CARDIAC-BAND spectral power. Free-breathing puts high
# temporal *variance* everywhere (chest wall, diaphragm, liver), so raw variance
# can't isolate the heart. But cardiac motion (~0.7-2 Hz) is higher-frequency
# than respiration (~0.2-0.4 Hz); at 40 Hz sampling over 4.5 s they separate
# cleanly. We sum FFT power in the cardiac band -> the beating heart lights up,
# body respiratory motion is suppressed -> smooth, peak, 3D region-grow.
FRAME_DT_S = 0.025        # 25 ms/frame (MIITT RT)
CARD_BAND_HZ = (0.7, 2.2)  # ~42-132 bpm
SMOOTH_SIGMA_VOX = 2.0     # in-plane Gaussian on the cardiac-power map
PEAK_FRAC = 0.15           # threshold as a fraction of the smoothed 3D peak value
BORDER_MARGIN_VOX = 16     # exclude FOV-edge aliasing/wrap artifacts from peak-finding
HEART_DILATE_ITER = 3      # ~voxels; on 2.6 mm grid ≈ 8 mm halo around heart
CHEST_DILATE_ITER = 8      # recon-FOV hull, generous


def cardiac_power_map(rt):
    """rt: (X,Y,Z,T) -> (X,Y,Z) FFT power summed over the cardiac frequency band."""
    X, Y, Z, T = rt.shape
    sig = rt - rt.mean(axis=3, keepdims=True)      # remove DC
    F = np.fft.rfft(sig, axis=3)
    p = (F.real ** 2 + F.imag ** 2)                # (X,Y,Z,nfreq)
    freq = np.fft.rfftfreq(T, d=FRAME_DT_S)
    band = (freq >= CARD_BAND_HZ[0]) & (freq <= CARD_BAND_HZ[1])
    return p[:, :, :, band].sum(axis=3)


def temporal_std_heart_mask(rt):
    """rt: (X,Y,Z,T) magnitude -> (X,Y,Z) boolean heart ROI.

    Global peak of smoothed cardiac-band power fixes the heart column (px,py);
    each slice is then thresholded independently at PEAK_FRAC of the *global*
    peak and the component nearest (px,py) is kept. This recovers the heart on
    every slice where cardiac signal is present (not just those 3D-connected to
    the peak) while staying anchored on the heart, rejecting off-heart blobs.
    """
    X, Y, Z, T = rt.shape
    cp = cardiac_power_map(rt)
    sm = ndimage.gaussian_filter(cp, sigma=(SMOOTH_SIGMA_VOX, SMOOTH_SIGMA_VOX, 0))
    # find the heart peak away from the FOV border (edge aliasing/wrap artifacts
    # can otherwise out-peak the heart, e.g. MIITT Volunteer3 at x=125/128)
    b = BORDER_MARGIN_VOX
    smc = np.zeros_like(sm)
    smc[b:X - b, b:Y - b, :] = sm[b:X - b, b:Y - b, :]
    px, py, pz = np.unravel_index(np.argmax(smc), smc.shape)
    thr = PEAK_FRAC * sm[px, py, pz]
    m = sm >= thr
    # keep the single 3D connected component containing the heart peak
    lab, _ = ndimage.label(m)
    m = lab == lab[px, py, pz]
    mask = np.zeros((X, Y, Z), dtype=bool)
    for z in range(Z):
        sl = m[:, :, z]
        if sl.any():
            sl = ndimage.binary_fill_holes(sl)
            sl = ndimage.binary_dilation(sl, iterations=HEART_DILATE_ITER)
        mask[:, :, z] = sl
    return mask


def export(vol):
    src = MIITT_NIFTI.format(vol=vol)
    im = nib.load(src)
    rt = im.get_fdata().astype(np.float32)  # (X,Y,Z,T)
    affine = im.affine
    X, Y, Z, T = rt.shape
    print(f"[{vol}] RT {rt.shape}  spacing {tuple(round(z,3) for z in im.header.get_zooms())}")

    recondir = os.path.join(RECON_ROOT, vol)
    datadir = os.path.join(recondir, "data")
    maskdir = os.path.join(recondir, "mask")
    for d in (datadir, maskdir, os.path.join(recondir, "cardsync")):
        os.makedirs(d, exist_ok=True)

    # --- real-time 4D stack (magnitude) ---
    rlt = nib.Nifti1Image(rt, affine)
    rlt.header.set_zooms(im.header.get_zooms())
    nib.save(rlt, os.path.join(datadir, "s01_rlt_ab.nii.gz"))

    # --- DC / static temporal mean ---
    dc = rt.mean(axis=3)  # (X,Y,Z)
    dcimg = nib.Nifti1Image(dc.astype(np.float32), affine)
    dcimg.header.set_zooms(im.header.get_zooms()[:3])
    nib.save(dcimg, os.path.join(datadir, "s01_dc_ab.nii.gz"))

    # --- heart mask (per-slice ROI via temporal variance) ---
    heart = temporal_std_heart_mask(rt)
    print(f"[{vol}] heart mask voxels/slice: "
          f"{[int(heart[:,:,z].sum()) for z in range(Z)]}")
    himg = nib.Nifti1Image(heart.astype(np.uint8), affine)
    himg.header.set_zooms(im.header.get_zooms()[:3])
    nib.save(himg, os.path.join(maskdir, "s01_mask_heart.nii.gz"))

    # --- chest mask (generous recon-FOV hull) ---
    chest = ndimage.binary_dilation(heart, iterations=CHEST_DILATE_ITER)
    chest = ndimage.binary_fill_holes(chest)
    cimg = nib.Nifti1Image(chest.astype(np.uint8), affine)
    cimg.header.set_zooms(im.header.get_zooms()[:3])
    nib.save(cimg, os.path.join(maskdir, "mask_chest.nii.gz"))

    print(f"[{vol}] wrote -> {recondir}")


if __name__ == "__main__":
    vols = sys.argv[1:] or ["Volunteer1"]
    for v in vols:
        export(v)
