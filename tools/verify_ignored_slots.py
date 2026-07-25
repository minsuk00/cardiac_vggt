"""Are the "ignored" slots a model failure, or ill-posed slices?

An adversarial review claims the 8 slots where the hub predicts Dz~0 despite a big applied shift are
all the LAST plane of the stack, heart-free, and mostly EVACUATED by breathing (the reslice pulls
from z + d/12, past the last content plane, into zeros) -- i.e. ill-posed, not a model failure.

This checks that independently, for BOTH models, from the frozen bundles.
"""
import glob
import json
import os
import numpy as np
import nibabel as nib

ROOT = "/home/minsukc/vggt/scratch/eval/cmrxrecon/out"


def probe(method):
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/*/{method}/resp_diag.json")):
        subj = f.split("/")[-3]
        sd = f"{ROOT}/{subj}"
        rd = json.load(open(f))
        npz = np.load(f"{ROOT}/{subj}/{method}/ed_dvf.npz")
        slot_z = npz["slot_z"]
        heart = np.asarray(nib.load(f"{sd}/mask_heart.nii.gz").dataobj) > 0.5   # (X,Y,Z)
        # the breathing-corrupted input stack actually fed to the model, at ED
        stack = np.asarray(nib.load(f"{sd}/breath/stack_t00.nii.gz").dataobj, dtype=np.float32)
        clean = np.asarray(nib.load(f"{sd}/clean/stack_t00.nii.gz").dataobj, dtype=np.float32)
        D = heart.shape[2]
        zs_with_content = [z for z in range(D) if heart[:, :, z].any()]
        pred = rd["breath"]["pred_dz_mm"]; appl = rd["breath"]["applied_dz_mm"]
        n = min(len(pred), len(slot_z))
        for s in range(n):
            z = int(slot_z[s])
            if z >= D:
                continue
            rows.append({
                "subject": subj, "z": z, "z_norm": z / max(D - 1, 1),
                "pred": pred[s], "appl": abs(appl[s]),
                "heart_frac": float(heart[:, :, z].mean()),
                "fov_frac_breath": float((stack[:, :, z] > 0.05).mean()),
                "fov_frac_clean": float((clean[:, :, z] > 0.05).mean()),
                "z_is_last_content": z == (max(zs_with_content) if zs_with_content else -1),
            })
    return rows


def report(method, rows):
    big = [r for r in rows if r["appl"] >= 5]
    ign = [r for r in big if abs(r["pred"]) < 2]
    trk = [r for r in big if abs(r["pred"]) >= 2]
    print(f"\n=== {method}")
    print(f"  slots with applied >= 5 mm: {len(big)}   ignored(|pred|<2): {len(ign)} ({100*len(ign)/max(len(big),1):.0f}%)")
    if ign:
        print(f"    ignored : z_norm {np.mean([r['z_norm'] for r in ign]):.2f}  "
              f"heart_frac {np.mean([r['heart_frac'] for r in ign]):.4f}  "
              f"fov_breath {np.mean([r['fov_frac_breath'] for r in ign]):.3f}  "
              f"(clean was {np.mean([r['fov_frac_clean'] for r in ign]):.3f})  "
              f"last-content-plane {sum(r['z_is_last_content'] for r in ign)}/{len(ign)}")
    if trk:
        print(f"    tracked : z_norm {np.mean([r['z_norm'] for r in trk]):.2f}  "
              f"heart_frac {np.mean([r['heart_frac'] for r in trk]):.4f}  "
              f"fov_breath {np.mean([r['fov_frac_breath'] for r in trk]):.3f}")
    # THE decisive split: restrict to content-bearing slots
    cb = [r for r in big if r["fov_frac_breath"] >= 0.15]
    cbi = [r for r in cb if abs(r["pred"]) < 2]
    print(f"  CONTENT-BEARING big breaths (fov_breath>=0.15): {len(cb)}   ignored: {len(cbi)} "
          f"({100*len(cbi)/max(len(cb),1):.0f}%)")
    low = [r for r in big if r["fov_frac_breath"] < 0.15]
    lowi = [r for r in low if abs(r["pred"]) < 2]
    print(f"  LOW-CONTENT big breaths      (fov_breath< 0.15): {len(low)}   ignored: {len(lowi)} "
          f"({100*len(lowi)/max(len(low),1):.0f}%)")
    # aggregate fit on content-bearing slots only
    cba = [r for r in rows if r["fov_frac_breath"] >= 0.15]
    if len(cba) > 5:
        a = np.array([r["appl"] for r in cba]); p = np.array([r["pred"] for r in cba])
        print(f"  fit on ALL content-bearing slots (n={len(cba)}/{len(rows)}): "
              f"slope {np.polyfit(a,p,1)[0]:.3f}  corr {np.corrcoef(a,p)[0,1]:.3f}  "
              f"EPE {np.abs(p-a).mean():.2f} mm")


if __name__ == "__main__":
    for m in ["vggt_20260715_1f_gather05", "vggt_20260715_1f_no_gather"]:
        rows = probe(m)
        if rows:
            report(m, rows)
            json.dump(rows, open(f"/home/minsukc/vggt/result/1frame_series/slots_{m}.json", "w"))
