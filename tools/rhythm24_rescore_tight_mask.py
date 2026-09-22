"""Re-score existing scored cines inside the TIGHT heart mask (mask_heart.nii.gz) instead of the
padded scoring ROI (mask_heart_pad10). READ-ONLY on every result; writes one json to temp/.

Why: the scorer registers each recon (6-DOF) and then scores inside a FIXED padded mask. A recon
that ends at the mask edge (NeSVoR, SVRTK, Fetal, a zeroed CiNeVol) is dragged out from under the
mask by its own pose, so a rim strip is scored against nothing; VGGT (full FOV) never pays that.
The tight mask sits >= 10 mm inside the padded one in-plane (checked on 6 subjects), farther than
any fitted shift, so no method's rim is scored there. Same slices as the padded mask.

Uses the standing metric functions (evaluation/src/score/image_metrics.py: psnr_unit_peak, ssim,
ncc) on the standing scored cines (<arm>/cine_breath.nii.gz = exactly what metrics.json scored),
so the ONLY change is the mask.

    PYTHONPATH=training:. python tools/rhythm24_rescore_tight_mask.py af24 [--arms ...] [--json out]
"""
import argparse
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "evaluation"), os.path.join(ROOT, "evaluation", "src", "score")]
import paths                                                       # noqa: E402
from image_metrics import psnr_unit_peak, ssim, ncc               # noqa: E402

SOURCES = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms")
DEFAULT_ARMS = ("fetal_cmr_4d", "cinevol", "vggt_final518_base_ep300", "vggt_final518_diff1000_ep300")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    per = {m: {} for m in a.arms}                      # arm -> (ds, subj) -> (psnr, ssim, ncc)
    for src in SOURCES:
        ds = f"{src}_{a.arm}"
        keep, _ = paths.filter_by_split(ds, paths.subjects(ds), "test")
        for s in keep:
            gt_img = nib.load(str(paths.cine_gt(ds, s)))
            gt = np.asarray(gt_img.dataobj, np.float32)
            tight = np.asarray(nib.load(str(paths.heart_mask(ds, s))).dataobj) > 0.5
            content = np.asarray(nib.load(str(paths.fov_mask(ds, s))).dataobj) > 0.5
            mask = tight & content
            for m in a.arms:
                p = paths.arm_dir(ds, s, m) / "cine_breath.nii.gz"
                if not p.is_file():
                    continue
                v = np.asarray(nib.load(str(p)).dataobj, np.float32)
                assert v.shape == gt.shape, (ds, s, m, v.shape, gt.shape)
                T = v.shape[3]
                per[m][(ds, s)] = (float(np.nanmean([psnr_unit_peak(v[..., t], gt[..., t], mask) for t in range(T)])),
                                   float(np.nanmean([ssim(v[..., t], gt[..., t], mask) for t in range(T)])),
                                   float(np.nanmean([ncc(v[..., t], gt[..., t], mask) for t in range(T)])))
            print(f"{ds}/{s}: done", flush=True)
    keys = sorted(set.intersection(*[set(d) for d in per.values()]))
    out = {"arm": a.arm, "mask": "mask_heart (tight) & FOV", "n_paired": len(keys), "arms": {}, "paired_vs_cinevol": {}}
    print(f"\n=== {a.arm}: TIGHT heart mask, paired n={len(keys)} ===")
    print(f"{'arm':30s} {'PSNR':>7s} {'SSIM':>7s} {'NCC':>7s}")
    for m in a.arms:
        v = np.array([per[m][k] for k in keys])
        out["arms"][m] = dict(zip(("psnr", "ssim", "ncc"), v.mean(0).tolist()))
        print(f"{m:30s} {v[:,0].mean():7.2f} {v[:,1].mean():7.3f} {v[:,2].mean():7.3f}")
    if "cinevol" in per:
        c = np.array([per["cinevol"][k] for k in keys])
        for m in a.arms:
            if m == "cinevol":
                continue
            o = np.array([per[m][k] for k in keys]); d = c - o
            res = {}
            for i, nm in enumerate(("psnr", "ssim", "ncc")):
                res[nm] = {"cinevol_minus": float(d[:, i].mean()), "p": float(wilcoxon(d[:, i]).pvalue),
                           "cinevol_better_n": int((d[:, i] > 0).sum())}
            out["paired_vs_cinevol"][m] = res
            print(f"CiNeVol - {m:28s}: " + "  ".join(f"{nm} {res[nm]['cinevol_minus']:+.3f} (p={res[nm]['p']:.1e}, CiNeVol better {res[nm]['cinevol_better_n']}/{len(keys)})" for nm in res))
    if a.json:
        if os.path.exists(a.json):
            sys.exit(f"REFUSING: {a.json} exists")
        json.dump(out, open(a.json, "w"), indent=1); print("wrote", a.json)


if __name__ == "__main__":
    main()
