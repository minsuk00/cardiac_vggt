#!/usr/bin/env python
"""Verify the self-gate on a 24-frame (`*24`) bundle, end to end, with no GPU.

The gate's 24-frame behaviour has a failure mode that CANNOT crash: if the ramp modulus follows
the frame count instead of `n_cardphase`, every frame gets its own output bin, the
two-images-per-bin averaging the whole design exists to create silently disappears, and every
downstream artefact still looks well-formed. So it needs a check, and the check needs to be shown
to fire.

Method: synthesise the nnU-Net segmentations instead of running the net -- for each (slice, frame)
write an LV blob whose voxel count follows the TRUE LV volume at that frame's simulated cardiac
position, taken from the bundle's own rhythm manifest. `detect_ed` then has a known right answer,
and the rest of the real `assemble` runs untouched.

Checks
  1  gate.json records n_cardphase and beats_in_window from the bundle
  2  cardphase.txt holds exactly D * T thetas
  3  theta[z][f] == theta[z][f + NCP]   <- THE mechanism: the ramp wraps, frames share a bin
  4  theta takes exactly NCP distinct values (not T)
  5  stack4d time pixdim == RR / NCP, not RR / T (sets the engine's temporal PSF width)
  6  theta[0][0] == 0 (slice 0 is re-rolled ED-first for the -cardphase count token)
  7  every slice's detected ED equals the argmax of the LV curve we planted

Usage:
    PYTHONPATH=training:. python tools/check_gate_24.py --source cmrx2024 \
        --subject CMRx24_Test_P017 --arm af24 [--work-tag check24]
"""
import argparse
import json
import os
import subprocess
import sys

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "training"), ROOT, os.path.join(ROOT, "evaluation")]
import paths                                                       # noqa: E402

GATE = os.path.join(ROOT, "evaluation/src/engine/fetal4d_gate.py")


def plant_segs(ds, subj, work_tag, split):
    """Write synthetic nnU-Net output whose LV area follows the bundle's own truth. -> (D,T) area."""
    man = json.load(open(paths.manifest(ds, subj)))
    T, D, NCP = int(man["T"]), int(man["D"]), int(man.get("n_cardphase", man["T"]))
    src = paths.subject_dir(man["source"], subj)
    segs = sorted((src / "seg_gt").glob("seg_t*.nii.gz"))
    vol = np.array([(np.asarray(nib.load(str(f)).dataobj) == 1).sum() for f in segs], float)
    pos = np.asarray(man["rhythm"]["pos_per_plane"], float)          # (D, T)
    area = np.interp(pos % NCP, np.arange(NCP + 1), np.r_[vol, vol[0]])   # (D, T) true LV volume

    ref = nib.load(str(paths.bundle_stack(ds, subj, "rolled", 0)))
    shape = ref.shape                                                # (X, Y, Z)
    out = paths.VOLUMES / "_fetal4d_gate" / work_tag / "seg"
    out.mkdir(parents=True, exist_ok=True)
    # Scale the true volumes into voxel counts that fit one X-Y plane, preserving ORDER (which is
    # all detect_ed's argmax depends on) -- the absolute counts are irrelevant to the gate.
    lo, hi = area.min(), area.max()
    nvox = np.round(4 + (area - lo) / max(hi - lo, 1e-9) * (shape[0] * shape[1] // 4 - 4)).astype(int)
    for f in range(T):
        a = np.zeros(shape, dtype=np.uint8)
        for z in range(D):
            flat = a[:, :, z].reshape(-1)
            flat[:nvox[z, f]] = 1
        nib.save(nib.Nifti1Image(a, ref.affine), str(out / f"{ds}__{subj}__f{f:02d}.nii.gz"))
    return man, T, D, NCP, area


def run(stage, ds, subj, work_tag, split):
    cmd = [sys.executable, GATE, stage, "--split", split, "--sources", ds,
           "--subjects", subj, "--work-tag", work_tag]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": f"training:{ROOT}"})
    if r.returncode:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        sys.exit(f"{stage} failed")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--arm", default="af24")
    ap.add_argument("--split", default="test")
    ap.add_argument("--work-tag", default="check24")
    a = ap.parse_args()
    ds = f"{a.source}_{a.arm}"

    fails = []

    def rep(name, ok, detail=""):
        fails.append(name) if not ok else None
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    run("dump", ds, a.subject, a.work_tag, a.split)
    man, T, D, NCP, area = plant_segs(ds, a.subject, a.work_tag, a.split)
    run("assemble", ds, a.subject, a.work_tag, a.split)

    gd = paths.subject_dir(ds, a.subject) / "fetal_cmr_4d" / "gate"
    meta = json.load(open(gd / "gate.json"))
    th = np.array(open(gd / "cardphase.txt").read().split(), float)

    print(f"\n=== {ds} / {a.subject}   T={T} D={D} n_cardphase={NCP} ===")
    rep("1 gate.json records n_cardphase / beats_in_window",
        meta.get("n_cardphase") == NCP and meta.get("beats_in_window") == T // NCP,
        f"n_cardphase={meta.get('n_cardphase')} beats={meta.get('beats_in_window')}")
    rep("2 cardphase.txt has D*T thetas", th.size == D * T, f"{th.size} vs {D * T}")
    if th.size != D * T:
        sys.exit(1)
    theta = th.reshape(D, T)
    wrap = max(float(np.abs(theta[:, f] - theta[:, f + NCP]).max())
               for f in range(T - NCP)) if T > NCP else 0.0
    rep("3 theta[f] == theta[f+NCP]  (ramp wraps; frames share a bin)", wrap < 1e-9,
        f"max|diff|={wrap:.2e}")
    nuniq = len(np.unique(np.round(theta, 6)))
    rep("4 theta takes exactly NCP distinct values", nuniq == NCP, f"{nuniq} distinct, want {NCP}")
    dt = float(nib.load(str(gd / "stack4d.nii.gz")).header.get_zooms()[3])
    rep("5 stack4d time pixdim == RR/NCP", abs(dt - 1.0 / NCP) < 1e-6,
        f"dt={dt:.6f}s  RR/NCP={1.0 / NCP:.6f}  RR/T={1.0 / T:.6f}")
    rep("6 theta[0][0] == 0 (slice 0 ED-first)", abs(theta[0, 0]) < 1e-9, f"{theta[0, 0]:.2e}")
    want = {z: int(np.argmax(area[z])) for z in range(D)}
    got = {p["z"]: p["ed_frame"] for p in meta["per_slice"]}
    bad = [z for z in want if got.get(z) != want[z]]
    rep("7 detected ED == planted argmax, every slice", not bad,
        f"{D - len(bad)}/{D} slices" + (f"  first bad z={bad[0]}: got {got.get(bad[0])} "
                                        f"want {want[bad[0]]}" if bad else ""))

    print(f"\n{'ALL CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
