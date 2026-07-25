"""How many dB does a through-plane (z) error actually COST at 12 mm slice pitch?

Motivation (H1): at matched epoch, no_gather has 2.6x the breathing EPE of gather05 (3.49 vs
1.36 mm) and ignores 35% of deep breaths, yet recov_frac_heart / psnr_motion / psnr_bbox are
identical to ~0.04 dB. Either the outcome metrics are broken, or a few mm of z-error is simply
worth ~nothing at 12 mm pitch.

This measures the conversion factor directly and model-free: take the real GT volume, translate it
by dz mm along z, and score it against the unshifted GT with the SAME PSNR/ROI the harness uses.
That gives dB-vs-mm. If 3.5 mm costs ~0.1 dB, H1 is mechanistically confirmed and the "gather
buys nothing" reading is about the METRIC's resolution, not the model.

Uses the frozen bundle GT (no model, no GPU).
"""
import glob
import json
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import shift as ndshift

ROOT = "/home/minsukc/vggt/scratch/eval/cmrxrecon/out"
SPACING_Z_MM = 12.0
DZS = [0.5, 1.0, 1.36, 2.0, 3.0, 3.49, 5.0, 6.0, 8.0, 12.0]  # incl. the two models' actual EPEs


def psnr(a, b, m):
    mse = float(((a[m] - b[m]) ** 2).mean())
    peak = float(b[m].max())
    return float(10 * np.log10(peak ** 2 / max(mse, 1e-10)))


def main():
    subs = sorted(glob.glob(f"{ROOT}/*/gt/gt_t00.nii.gz"))
    out = {str(d): [] for d in DZS}
    for f in subs:
        subj = f.split("/")[-3]
        sd = os.path.dirname(os.path.dirname(f))
        gt = np.asarray(nib.load(f).dataobj, dtype=np.float32)          # (X,Y,Z)
        heart = np.asarray(nib.load(os.path.join(sd, "mask_heart.nii.gz")).dataobj) > 0.5
        fov = np.asarray(nib.load(os.path.join(sd, "mask.nii.gz")).dataobj) > 0.5
        m = heart & fov
        if not m.any():
            continue
        for dz in DZS:
            # translate along z by dz mm (= dz/12 voxels), trilinear — the same interpolation
            # regime the splat uses when it places a slice off-plane.
            g = ndshift(gt, (0.0, 0.0, dz / SPACING_Z_MM), order=1, mode="nearest")
            out[str(dz)].append(psnr(g, gt, m))
    print(f"z-shift sensitivity on {len(subs)} CMRx subjects (heart&FOV ROI, ED phase)")
    print(f"{'dz mm':>7s} {'PSNR dB':>9s} {'  <- cost vs perfect':>22s}")
    ref = None
    rows = []
    for dz in DZS:
        v = float(np.mean(out[str(dz)]))
        rows.append((dz, v))
        print(f"{dz:7.2f} {v:9.2f}")
    od = "/home/minsukc/vggt/result/1frame_series"
    os.makedirs(od, exist_ok=True)
    json.dump({"dz_mm": [r[0] for r in rows], "psnr_db": [r[1] for r in rows],
               "n_subj": len(subs), "note": "GT shifted by dz vs unshifted GT, heart&FOV ROI"},
              open(os.path.join(od, "zshift_sensitivity.json"), "w"), indent=1)
    print(f"\n-> {od}/zshift_sensitivity.json")


if __name__ == "__main__":
    main()
