"""Write <subj>/scatter/stack_t{k} for a rhythm cohort (e.g. af12): the SAME-INPUT stack the classical
scatter baselines (SVRTK / NiftyMIC / NeSVoR / Dangi, INPUT=scatter) read -- exactly what VGGT sees
when queried at frame k. Plane z = breath/stack_t{phase_per_plane[z]}[z]; the reference plane =
breath/stack_t{k}[ref]. Same assembly as build_inputs/pooled.add_scatter, but it reuses the frame
draw build_af_bundle already wrote into manifest["scatter"] (pooled's version re-draws it, and
asserts the plain cohort's seed). Pure copy of existing slices; skips subjects that already have a
complete scatter/, and writes via a temp dir so a partial run never leaves a half-built one.

Usage:  PYTHONPATH=training:. python tools/build_af_scatter.py af12 [--check]
"""
import argparse
import glob
import json
import os
import shutil

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build(sd):
    man = json.load(open(os.path.join(sd, "manifest.json")))
    T, D = man["T"], man["D"]
    ppp, ref = man["scatter"]["phase_per_plane"], int(man["scatter"]["ref_plane"])
    dst = os.path.join(sd, "scatter")
    if len(glob.glob(os.path.join(dst, "stack_t*.nii.gz"))) == T:
        return "exists"
    breath = [nib.load(os.path.join(sd, "breath", f"stack_t{t:02d}.nii.gz")) for t in range(T)]
    arrs = [np.asarray(im.dataobj, dtype=np.float32) for im in breath]
    tmp = f"{dst}.tmp{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    for k in range(T):
        out = np.empty_like(arrs[0])
        for z in range(D):
            out[:, :, z] = arrs[k if z == ref else ppp[z]][:, :, z]
        nib.save(nib.Nifti1Image(out, breath[0].affine), os.path.join(tmp, f"stack_t{k:02d}.nii.gz"))
    if os.path.isdir(dst):          # incomplete leftover from an earlier crash
        shutil.rmtree(dst)
    os.replace(tmp, dst)
    return "built"


def check(sd):
    """Every scatter/stack_t{k} plane is byte-identical to the breath/ frame it claims to be."""
    man = json.load(open(os.path.join(sd, "manifest.json")))
    ppp, ref = man["scatter"]["phase_per_plane"], int(man["scatter"]["ref_plane"])
    L = lambda d, k: np.asarray(nib.load(os.path.join(sd, d, f"stack_t{k:02d}.nii.gz")).dataobj, np.float32)  # noqa: E731
    br = [L("breath", t) for t in range(man["T"])]
    for k in range(man["T"]):
        s = L("scatter", k)
        for z in range(man["D"]):
            if not np.array_equal(s[..., z], br[k if z == ref else ppp[z]][..., z]):
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    subs = sorted(glob.glob(os.path.join(ROOT, "scratch", "eval", f"*_{a.arm}", "out", "*", "manifest.json")))
    counts = {}
    for m in subs:
        sd = os.path.dirname(m)
        r = ("ok" if check(sd) else "MISMATCH") if a.check else build(sd)
        counts[r] = counts.get(r, 0) + 1
        if r == "MISMATCH":
            print("MISMATCH", sd)
    print(f"{a.arm}: {len(subs)} subjects {counts}")


if __name__ == "__main__":
    main()
