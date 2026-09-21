#!/usr/bin/env python
"""Breathing-motion EPE for the VGGT arms on a 24-frame rhythm cohort, against the TRUE applied shift.

WHY THIS EXISTS: run_vggt.py's resp_diag.json scores predicted through-plane dz against
manifest["breath"]["disp_dhw_mm"] -- one frozen displacement per plane. The rhythm cohorts copy that
block verbatim from the SOURCE bundle but breathe FRAME-WISE (build_af_bundle.simulate:
r = r0[z] + f*DT/T_BREATH), so plane z's real shift depends on which frame VGGT was fed. The stored
epe/slope/corr are therefore against a stale reference (measured on CMRx24_Test_P017: slope 0.02,
corr 0.03) and must not be reported. This recomputes them:

  predicted dz  = resp_diag.json breath.pred_dz_mm            (per slot, ED pass, unchanged)
  applied  dz   = u*amp*lujan((r0[z] + j*DT/T_BREATH) % 1)[0]  with z = ed_dvf.slot_z, j = ed_dvf.slot_t

A subject is skipped when resp_diag dropped a slot through its FOV gate (slot order no longer
recoverable from the json alone). Fetal CMR 4D predicts no per-slice dz, so it has no such metric.

Usage:  PYTHONPATH=training:. python tools/rhythm24_resp_epe.py af24 [--json out.json]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "tools")]
from build_af_bundle import DT, T_BREATH, breathing_model                 # noqa: E402
from data.respiratory import lujan_displacement                          # noqa: E402

EVAL = os.path.join(ROOT, "scratch", "eval")


def subject_row(arm_dir, man):
    b = json.load(open(os.path.join(arm_dir, "resp_diag.json"))).get("breath") or {}
    z = np.load(os.path.join(arm_dir, "ed_dvf.npz"))
    pred = np.asarray(b.get("pred_dz_mm", []), float)
    if pred.size == 0 or pred.size != z["slot_z"].size:
        return None
    u, amp, r0, n, _ = breathing_model(man)
    appl = np.array([(u * amp * float(lujan_displacement(float((r0[int(round(sz))] + int(j) * DT / T_BREATH) % 1.0),
                                                         1.0, n=n)))[0]
                     for sz, j in zip(z["slot_z"], z["slot_t"])], float)
    err = pred - appl
    row = {"epe": float(np.abs(err).mean()), "epe_demeaned": float(np.abs(err - err.mean()).mean()),
           "stale_epe": b.get("epe_dz_mm"), "applied_abs": float(np.abs(appl).mean())}
    if appl.std() > 1e-6:
        row["slope"] = float(np.polyfit(appl, pred, 1)[0]); row["corr"] = float(np.corrcoef(appl, pred)[0, 1])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {}
    print(f"=== {a.arm}: breathing-motion EPE vs the TRUE frame-wise shift (mm, through-plane) ===")
    print(f"{'method':34}{'n':>5}{'skip':>6}{'EPE':>8}{'demeaned':>10}{'slope':>8}{'corr':>7}{'|applied|':>11}{'stale EPE':>11}")
    for m in sorted({p.split("/")[-2] for p in glob.glob(f"{EVAL}/*_{a.arm}/out/*/vggt_*/resp_diag.json")}):
        rows, skip = [], 0
        for p in glob.glob(f"{EVAL}/*_{a.arm}/out/*/{m}/resp_diag.json"):
            d = os.path.dirname(p)
            man = json.load(open(os.path.join(d, "..", "manifest.json")))
            r = subject_row(d, man)
            if r is None:
                skip += 1
            else:
                rows.append(r)
        mean = lambda k: float(np.nanmean([r.get(k, np.nan) for r in rows]))      # noqa: E731
        out[m] = {k: mean(k) for k in ("epe", "epe_demeaned", "slope", "corr", "applied_abs", "stale_epe")}
        out[m].update(n=len(rows), skipped=skip)
        o = out[m]
        print(f"{m:34}{o['n']:5d}{skip:6d}{o['epe']:8.2f}{o['epe_demeaned']:10.2f}{o['slope']:8.2f}{o['corr']:7.2f}"
              f"{o['applied_abs']:11.2f}{o['stale_epe']:11.2f}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
