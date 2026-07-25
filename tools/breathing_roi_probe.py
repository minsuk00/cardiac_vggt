"""Is the breathing metric measuring the region that actually gets scored?

The docs/38 breathing metric (training/loss.py:632) gates predicted Dz on `img_int > 0.05` --
i.e. it averages Dz over the WHOLE in-FOV slice: chest wall, lungs, liver, heart. But PSNR /
recov_frac are scored on the HEART. So a model could place the heart correctly and the periphery
badly (or vice versa) and the two numbers would disagree by construction.

That matters because at matched epoch, no_gather has 2.6x the FOV-averaged breathing EPE of
gather05 (3.49 vs 1.36 mm) on IDENTICAL val breathing, yet identical heart PSNR -- and an
independent GT-shift test says a 2 mm z-error should cost several dB. So either the placement
difference is not where the score looks, or something else is going on.

This recomputes per-slot predicted Dz under two ROIs from the SAME ed_dvf.npz:
  * FOV   : img>0.05, reproducing the trainer metric
  * HEART : the frozen bundle's mask_heart at that slot's z-plane
and compares slope / EPE under each.
"""
import glob
import json
import os
import sys
import numpy as np
import nibabel as nib

MM_PER_NORM_Z = (12 - 1) / 2.0 * 12.0  # 66 mm, matches loss.py:624


def upsample_mask(m2d, size=518):
    """(256,256) canonical plane mask -> (518,518) input-slice grid (nearest, matches the
    bilinear image resize closely enough for a region gate)."""
    idx = (np.arange(size) / (size - 1) * (m2d.shape[0] - 1)).round().astype(int)
    return m2d[np.ix_(idx, idx)]


def probe(dataset, method):
    root = f"/home/minsukc/vggt/scratch/eval/{dataset}/out"
    rows = []
    for f in sorted(glob.glob(f"{root}/*/{method}/ed_dvf.npz")):
        d = os.path.dirname(f)
        subj = f.split("/")[-3]
        sd = os.path.dirname(d)
        z = np.load(f)
        delta, slot_z, appl = z["delta"], z["slot_z"], z["applied_disp_mm"]
        dz_mm = delta[..., 2].astype(np.float32) * MM_PER_NORM_Z          # (S,518,518)
        heart = np.asarray(nib.load(os.path.join(sd, "mask_heart.nii.gz")).dataobj) > 0.5  # (X,Y,Z)
        # ed_dvf has no images; reconstruct the FOV gate from the bundle's FOV mask instead.
        fovm = os.path.join(sd, "mask.nii.gz")
        fovm = fovm if os.path.exists(fovm) else os.path.join(sd, "mask_fov.nii.gz")
        fov = np.asarray(nib.load(fovm).dataobj) > 0.5
        for s in range(dz_mm.shape[0]):
            zp = int(slot_z[s])
            if zp >= heart.shape[2]:
                continue
            hm = upsample_mask(heart[:, :, zp].T)   # (X,Y)->(Y,X) to match slice layout
            fm = upsample_mask(fov[:, :, zp].T)
            if fm.sum() < 100:
                continue
            rec = {"subject": subj, "slot": s, "z": zp, "applied": float(appl[s, 0]),
                   "pred_fov": float(dz_mm[s][fm].mean())}
            if hm.sum() >= 100:
                rec["pred_heart"] = float(dz_mm[s][hm].mean())
                rec["heart_std"] = float(dz_mm[s][hm].std())
            rec["fov_std"] = float(dz_mm[s][fm].std())
            rows.append(rec)
    return rows


def fit(x, y):
    if len(x) < 3 or np.std(x) < 1e-6:
        return None, None
    return float(np.polyfit(x, y, 1)[0]), float(np.corrcoef(x, y)[0, 1])


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cmrxrecon"
    out = {}
    for method in sys.argv[2:]:
        rows = [r for r in probe(dataset, method) if "pred_heart" in r]
        if not rows:
            print(f"{method}: no data"); continue
        a = np.array([r["applied"] for r in rows])
        pf = np.array([r["pred_fov"] for r in rows])
        ph = np.array([r["pred_heart"] for r in rows])
        sf, cf = fit(a, pf); sh, ch = fit(a, ph)
        out[method] = {"n_slots": len(rows),
                       "fov":   {"slope": sf, "corr": cf, "epe_mm": float(np.abs(pf - a).mean())},
                       "heart": {"slope": sh, "corr": ch, "epe_mm": float(np.abs(ph - a).mean())},
                       "within_heart_std_mm": float(np.mean([r["heart_std"] for r in rows])),
                       "within_fov_std_mm": float(np.mean([r["fov_std"] for r in rows]))}
        o = out[method]
        print(f"\n=== {method}  ({o['n_slots']} slots, {dataset})")
        print(f"  FOV-gated   (what the metric measures): slope {sf:.3f}  corr {cf:.3f}  EPE {o['fov']['epe_mm']:.2f} mm")
        print(f"  HEART-gated (what the score cares about): slope {sh:.3f}  corr {ch:.3f}  EPE {o['heart']['epe_mm']:.2f} mm")
        print(f"  within-slot Dz std: heart {o['within_heart_std_mm']:.2f} mm | fov {o['within_fov_std_mm']:.2f} mm")
    p = f"/home/minsukc/vggt/result/1frame_series/roi_probe_{dataset}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\n-> {p}")


if __name__ == "__main__":
    main()
