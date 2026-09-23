#!/usr/bin/env python
"""Breathing-motion EPE for the VGGT arms on a 24-frame rhythm cohort, against the TRUE applied shift.

WHY THIS EXISTS: run_vggt.py's resp_diag.json scores predicted through-plane dz against
manifest["breath"]["disp_dhw_mm"] -- one frozen displacement per plane. The rhythm cohorts copy that
block verbatim from the SOURCE bundle but breathe FRAME-WISE (build_af_bundle.simulate:
r = r0[z] + f*DT/T_BREATH), so plane z's real shift depends on which frame VGGT was fed. The stored
epe/slope/corr are therefore against a stale reference (measured on CMRx24_Test_P017: slope 0.02,
corr 0.03) and must not be reported. This recomputes them:

  predicted dz  = resp_diag.json breath.pred_dz_mm            (per slot, ED pass, unchanged)
  applied  dz   = common.applied(man, (z, j))[dz]  with z = ed_dvf.slot_z, j = ed_dvf.slot_t
                  (u*amp*lujan((r0[z] + j*DT/T_BREATH) % 1), or r0[z] alone on frozen-breath arms)

resp_diag drops blank slots (FOV gate: no pixel > 0.05) without recording which; they are re-found
from the input slices and only they are dropped. A subject is skipped only if that recount fails to
match resp_diag's slot count.

Usage:  PYTHONPATH=training:. python evaluation/src/analysis/motion_epe/vggt.py af24 [--json out.json]
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve()
ROOT = str(next(p for p in HERE.parents if (p / "evaluation").is_dir()))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
sys.path.insert(0, str(HERE.parent))
import common                                                            # noqa: E402

EVAL = os.path.join(ROOT, "scratch", "eval")


def subject_row(arm_dir, man):
    b = json.load(open(os.path.join(arm_dir, "resp_diag.json"))).get("breath") or {}
    z = np.load(os.path.join(arm_dir, "ed_dvf.npz"))
    pred = np.asarray(b.get("pred_dz_mm", []), float)
    # resp_diag drops blank slots (no pixel > 0.05) without saying which; re-find them from the input
    # slice (breath/stack_t{slot_t}[slot_z], the image VGGT got) and align the rest in slot order.
    sd = os.path.dirname(arm_dir)
    slots = [(int(round(sz)), int(j)) for sz, j in zip(z["slot_z"], z["slot_t"])]
    slots = [(sz, j) for sz, j in slots
             if (np.asarray(nib.load(os.path.join(sd, "breath", f"stack_t{j:02d}.nii.gz")).dataobj)[:, :, sz] > 0.05).any()]
    if pred.size == 0 or pred.size != len(slots):
        return None
    appl = common.applied(man, slots)[:, 0]             # dz; honours frozen_breath + the manifest's DT/T_BREATH
    err = pred - appl
    dm = float(np.abs(err - err.mean()).mean())
    # "epe" = demeaned, raw in "epe_raw" -- same convention as common.score / cinevol.py
    row = {"epe": dm, "epe_demeaned": dm, "epe_raw": float(np.abs(err).mean()),
           "stale_epe": b.get("epe_dz_mm", np.nan) if b.get("epe_dz_mm") is not None else np.nan,
           "applied_abs": float(np.abs(appl).mean())}
    if appl.std() > 1e-6:
        row["slope"] = float(np.polyfit(appl, pred, 1)[0]); row["corr"] = float(np.corrcoef(appl, pred)[0, 1])
    row["slices"] = {sz: {"applied": [float(q)], "pred": [float(p)]} for (sz, _), q, p in zip(slots, appl, pred)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--json", default=None)
    ap.add_argument("--slices-dir", default=None, help="per-slice dump <dir>/<arm>.json per VGGT arm, for common_set.py")
    a = ap.parse_args()
    out = {}
    print(f"=== {a.arm}: breathing-motion EPE vs the TRUE frame-wise shift (mm, through-plane) ===")
    print(f"{'method':34}{'n':>5}{'skip':>6}{'EPE':>8}{'demeaned':>10}{'slope':>8}{'corr':>7}{'|applied|':>11}{'stale EPE':>11}")
    for m in sorted({p.split("/")[-2] for p in glob.glob(f"{EVAL}/*_{a.arm}/out/*/vggt_*/resp_diag.json")}):
        rows, skip, slices = [], 0, {}
        for p in glob.glob(f"{EVAL}/*_{a.arm}/out/*/{m}/resp_diag.json"):
            d = os.path.dirname(p)
            man = json.load(open(os.path.join(d, "..", "manifest.json")))
            r = subject_row(d, man)
            if r is None:
                skip += 1
            else:
                parts = p.split("/")                      # .../<ds>/out/<subject>/<arm>/resp_diag.json
                slices[f"{parts[-5]}/{parts[-3]}"] = r.pop("slices")
                rows.append(r)
        if a.slices_dir:
            common.write_slices(os.path.join(a.slices_dir, f"{m}.json"), m, ["dz"], slices)
        mean = lambda k: float(np.nanmean([r.get(k, np.nan) for r in rows]))      # noqa: E731
        out[m] = {k: mean(k) for k in ("epe", "epe_demeaned", "epe_raw", "slope", "corr", "applied_abs", "stale_epe")}
        out[m].update(n=len(rows), skipped=skip)
        o = out[m]
        print(f"{m:34}{o['n']:5d}{skip:6d}{o['epe_raw']:8.2f}{o['epe_demeaned']:10.2f}{o['slope']:8.2f}{o['corr']:7.2f}"
              f"{o['applied_abs']:11.2f}{o['stale_epe']:11.2f}")
    if a.json:
        common.write_json(a.json, out)      # merge: the file also holds the other methods' rows


if __name__ == "__main__":
    main()
