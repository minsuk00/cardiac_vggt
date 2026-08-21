"""Identity Dice floors — what an UNCORRECTED input scores against the target-phase GT.

Two model-free references (pure label arithmetic on the frozen bundle; docs/86):

  cardiac : each z-plane's GT seg taken at that slice's own random cardiac phase
            (the name-seeded one-frame-per-slice draw), no breathing -> the
            breath-hold do-nothing volume.
  full    : the same, plus each plane's frozen breathing shift from the manifest
            -> the actual do-nothing input every arm receives.

An arm's Dice minus the `full` floor is its total motion-correction credit; the
cardiac->full gap is breathing's added damage. Measured values (144 val subjects):
cardiac 0.826 ED / 0.699 ES, full 0.735 / 0.636; arms score 0.87-0.89 / 0.80-0.84.

Inputs (all read-only): heart_seg.nii.gz (Task114-on-GT, docs/39), manifest.json
(`breath.disp_dhw_mm`), and the recorded per-slice phase draw slot_t/slot_z from any
arm's ed_dvf.npz (the draw is name-seeded, hence identical across arms; no model
inference is involved). ED/ES = argmax/argmin of the GT LV voxel-count curve, the
same convention as ef_dice.py.

Writes metric_results/_floors/dice_identity.json and NOTHING else; refuses to
overwrite an existing output without --force.

Run:  micromamba run -n svr python evaluation/src/score/dice_floors.py
"""
import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import paths                                                          # noqa: E402

LV = 1                       # Task114 labels (matches ef_dice.py)
INPLANE_MM = 1.4


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_path = paths.RESULTS / "_floors" / "dice_identity.json"
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} exists — pass --force to regenerate (values are deterministic).")

    per_subject, missing = [], 0
    for man_p in sorted(glob.glob(str(paths.VOLUMES / "*" / "out" / "*" / "manifest.json"))):
        man = json.load(open(man_p))
        if man.get("split") != "val":
            continue
        sd = os.path.dirname(man_p)
        ds, subj = man_p.split("/")[-4], os.path.basename(sd)
        seg_p = os.path.join(sd, "heart_seg.nii.gz")
        dvfs = sorted(glob.glob(os.path.join(sd, "*", "ed_dvf.npz")))
        if not (os.path.exists(seg_p) and dvfs):
            missing += 1
            continue
        seg = np.transpose(np.asarray(nib.load(seg_p).dataobj), (3, 2, 1, 0))  # (T,D,H,W)
        T, D, H, W = seg.shape
        dvf = np.load(dvfs[0])
        if np.any(np.abs(dvf["slot_z"] - np.round(dvf["slot_z"])) > 1e-6):
            sys.exit(f"{dvfs[0]}: fractional slot_z (continuous-z arm) — pick an integer-z arm.")
        t_of = {int(round(float(z))): int(t) for z, t in zip(dvf["slot_z"], dvf["slot_t"])}
        disp = np.asarray(man["breath"]["disp_dhw_mm"])                # (D,3) dZ,dY,dX mm
        counts = (seg == LV).reshape(T, -1).sum(1)
        if counts.max() == 0:
            missing += 1
            continue
        # Slot 0 is the reference slice: the sweep RE-EXTRACTS it at each queried target phase
        # (run_vggt.py reconstruct), so the true do-nothing input always carries a correct
        # target-phase slice at z_mid — score that plane at tt, not at the recorded phase 0.
        ref_plane = int(round(float(dvf["slot_z"][0])))
        row = {"dataset": ds, "subject": subj}
        for tag, tt in [("ED", int(counts.argmax())), ("ES", int(counts.argmin()))]:
            gt = seg[tt] == LV
            card = np.zeros_like(gt)
            full = np.zeros_like(gt)
            for z in range(D):
                t_in = tt if z == ref_plane else t_of.get(z, tt)
                card[z] = seg[t_in, z] == LV
                dZ, dY, dX = disp[z]
                zs = int(round(z + dZ / man["dz_mm"]))
                if 0 <= zs < D:
                    # The builder breathes by SAMPLING at +disp (out[y,x] = V[y+dY, x+dX]),
                    # so reproducing it with np.roll needs the NEGATIVE shift — verified
                    # against real breathed bundle pixels on 15/15 subjects (docs/86).
                    full[z] = np.roll(np.roll(seg[t_in, zs] == LV,
                                              -int(round(dY / INPLANE_MM)), 0),
                                      -int(round(dX / INPLANE_MM)), 1)
            row[f"cardiac_{tag}"] = dice(card, gt)
            row[f"full_{tag}"] = dice(full, gt)
        per_subject.append(row)

    agg = {}
    for ds in sorted({r["dataset"] for r in per_subject}):
        rs = [r for r in per_subject if r["dataset"] == ds]
        agg[ds] = {"n": len(rs), **{k: float(np.nanmean([r[k] for r in rs]))
                                    for k in ("cardiac_ED", "cardiac_ES", "full_ED", "full_ES")}}

    out = {"meta": {
        "what": "Identity Dice floors: LV Dice of the UNCORRECTED input vs target-phase GT. "
                "'cardiac' = each plane's GT seg at its own randomly-drawn cardiac phase "
                "(no breathing); 'full' = the same plus the frozen per-plane breathing shift "
                "= the actual do-nothing input. Model-free (label arithmetic only). "
                "An arm's Dice minus 'full' is its motion-correction credit. See docs/86.",
        "ed_es": "argmax/argmin of GT LV voxel-count curve (ef_dice.py convention)",
        "sources": "heart_seg.nii.gz + manifest.breath.disp_dhw_mm + slot_t/slot_z from any "
                   "arm's ed_dvf.npz (name-seeded draw, identical across arms)",
        "generator": "evaluation/src/score/dice_floors.py"},
        "aggregate": agg, "per_subject": per_subject}
    os.makedirs(out_path.parent, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"{len(per_subject)} subjects ({missing} skipped, no seg/draw) -> {out_path}")
    for ds, a in agg.items():
        print(f"  {ds:10} cardiac {a['cardiac_ED']:.3f}/{a['cardiac_ES']:.3f}  "
              f"full {a['full_ED']:.3f}/{a['full_ES']:.3f}  (n={a['n']})")


if __name__ == "__main__":
    main()
