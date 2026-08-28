"""Causal test dump: GT ES frame per subject in 3 appearance arms (geometry identical):
  ctrl    - untouched GT ES frame
  matched - in-plane Gaussian blur, sigma calibrated per subject so the gradient energy
            in the heart ROI matches the pred (vggt_noreg_ep300) ES frame's
  sig10   - fixed sigma=1.0 px (dose-response point)
Writes nnU-Net inputs <cohort>__s<idx>__<arm>_0000.nii.gz + manifest. New dirs only."""
import csv, json, os, sys
import numpy as np, nibabel as nib
from scipy import ndimage as ndi

ROOT = "temp/dice_ef_sweep/mirror_full/volumes"; ARM = "vggt_noreg_ep300"; LV, MYO = 1, 2
OUT = "temp/dice_ef_sweep/es_blurtest/in"
assert not (os.path.isdir(OUT) and os.listdir(OUT)), "fresh dir required"
os.makedirs(OUT, exist_ok=True)

cov = list(csv.DictReader(open("temp/dice_ef_sweep/difficulty_covariates.csv")))
def norm01(v):
    lo, hi = np.percentile(v[v > 0], [1, 99.5]); return np.clip((v-lo)/max(hi-lo,1e-6),0,1)
def grad_energy(img, roi):
    gx, gy = np.gradient(img, axis=(0,1)); return float(np.hypot(gx,gy)[roi].mean())

manifest = []
for sidx, r in enumerate(cov):
    subj, coh = r["subj"], r["coh"]; d = f"{ROOT}/{coh}/out/{subj}"
    try:
        segimg = nib.load(f"{d}/heart_seg.nii.gz"); seg = np.asarray(segimg.dataobj)
        gtimg = nib.load(f"{d}/cine_gt.nii.gz")
        gt = np.asarray(gtimg.dataobj).astype(np.float32)
        lvc = (seg == LV).sum((0,1,2)); tES = int(np.argmin(lvc)); tED = int(np.argmax(lvc))
        if (seg[..., tES] == LV).sum() < 20: raise ValueError("tiny ES seg")
        pred = np.asarray(nib.load(f"{d}/{ARM}/recon_breath/vol_t{tES:02d}.nii.gz").dataobj).astype(np.float32)
    except Exception as e:
        print("skip", subj, e, file=sys.stderr); continue
    roi = ndi.binary_dilation(seg[..., tED] == LV, iterations=2)
    g = gt[..., tES]; gN, pN = norm01(gt)[..., tES], norm01(pred[..., None])[..., 0]
    target = grad_energy(pN, roi) / max(grad_energy(gN, roi), 1e-6)
    # calibrate in-plane sigma so blurred-GT gradient energy ratio == target
    sig = 0.0
    if target < 1.0:
        sigmas = np.linspace(0.2, 3.0, 15)
        ratios = [grad_energy(ndi.gaussian_filter(gN, (s, s, 0)), roi) /
                  max(grad_energy(gN, roi), 1e-6) for s in sigmas]
        sig = float(sigmas[int(np.argmin(np.abs(np.array(ratios) - target)))])
    aff = gtimg.affine
    for arm, vol in [("ctrl", g),
                     ("matched", ndi.gaussian_filter(g, (sig, sig, 0))),
                     ("sig10", ndi.gaussian_filter(g, (1.0, 1.0, 0)))]:
        nib.save(nib.Nifti1Image(vol.astype(np.float32), aff),
                 f"{OUT}/{coh}__s{sidx:03d}__{arm}_0000.nii.gz")
    manifest.append(dict(cohort=coh, sidx=sidx, subject=subj, tES=tES,
                         sigma_matched=sig, grad_ratio_target=round(target, 3),
                         vox_ml=float(np.prod(gtimg.header.get_zooms()[:3]))/1000.0,
                         esv_gtseg_ml=float((seg[..., tES] == LV).sum()) *
                                      float(np.prod(gtimg.header.get_zooms()[:3]))/1000.0))
json.dump(manifest, open(f"{OUT}/blur_manifest.json", "w"), indent=1)
sig = np.array([m["sigma_matched"] for m in manifest])
print(f"dumped {len(manifest)} subjects x3 arms -> {OUT}")
print("matched sigma quartiles:", np.percentile(sig, [10,25,50,75,90]).round(2))
