"""Verify the output of `tools/convert_to_sax_layout.py` (docs/58 A2).

The conversion claims to be **lossless**: a pure axis permute + flip, no interpolation. This
checks that claim against the source files rather than trusting it, plus the structural
invariants the training loader depends on.

Per subject:
  1. exactly 12 `sax_frame_{tt}.nii.gz`
  2. affine is axis-aligned (all off-diagonal terms zero) -> `Orientationd(LPS)` is a no-op
  3. voxel spacing and pitch match what `convert_meta.json` recorded
  4. **voxel data is BIT-IDENTICAL** to the source 4D frame after the recorded permute+flip

Then it FAULT-INJECTS: it re-runs check 4 comparing frame 0 against the source data for frame 1,
which must report a mismatch. A verifier that has not been shown to fail on a broken input is
worthless (`docs/58` §9), so this runs every time and the result is printed.

Usage:
    python tools/verify_sax_conversion.py 'scratch/data/ACDC_sax/*/sax'
    python tools/verify_sax_conversion.py 'scratch/data/MNMs_sax/*/sax' --frames 0 3 6 9
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_sax_layout import DATA, apply_reframe  # noqa: E402


def _perm_from_meta(m):
    return [1, 0, 2] if m["reframe"]["swapped_inplane"] else [0, 1, 2]


def check_subject(sax, frames):
    """Returns a list of problem strings (empty == clean)."""
    errs = []
    meta_p = os.path.join(sax, "convert_meta.json")
    if not os.path.exists(meta_p):
        return ["no convert_meta.json"]
    m = json.load(open(meta_p))

    fs = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
    if len(fs) != 12:
        errs.append(f"{len(fs)} frames (expected 12)")
    if not fs:
        return errs

    im = nib.load(fs[0])
    R = im.affine[:3, :3]
    off = float(np.abs(R - np.diag(np.diag(R))).max())
    if off > 1e-9:
        errs.append(f"affine not axis-aligned (max off-diagonal {off:.2e})")
    sp = [abs(float(z)) for z in im.header.get_zooms()[:3]]
    if not np.allclose(sp, m["out_spacing_xyz"], atol=1e-4):
        errs.append(f"spacing {sp} != meta {m['out_spacing_xyz']}")
    if abs(sp[2] - m["pitch_mm"]) > 1e-4:
        errs.append(f"pitch {sp[2]} != meta {m['pitch_mm']}")

    arr = np.asarray(nib.load(os.path.join(DATA, m["source_file"])).dataobj)
    perm = _perm_from_meta(m)
    for j in frames:
        want = apply_reframe(arr[..., m["frame_indices_native"][j]], perm, m["reframe"]["flips"])
        got = np.asarray(nib.load(os.path.join(sax, "3d_recon", f"sax_frame_{j:02d}.nii.gz")).dataobj)
        if got.shape != want.shape:
            errs.append(f"f{j} shape {got.shape} != {want.shape}")
        elif not np.array_equal(got, want):
            d = float(np.abs(got.astype(np.float64) - want).max())
            errs.append(f"f{j} VOXELS DIFFER (max |d| {d})")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", help="glob for the sax dirs, e.g. 'scratch/data/ACDC_sax/*/sax'")
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 6],
                    help="which of the 12 frames to byte-compare (default: 0 and 6)")
    args = ap.parse_args()

    dirs = sorted(glob.glob(args.pattern))
    if not dirs:
        print(f"nothing matched {args.pattern}")
        return 1
    print(f"checking {len(dirs)} subjects, byte-comparing frames {args.frames}")

    bad = 0
    for sax in dirs:
        errs = check_subject(sax, args.frames)
        if errs:
            bad += 1
            print(f"  FAIL {os.path.basename(os.path.dirname(sax))}: {'; '.join(errs)}")
    print(f"{len(dirs) - bad} ok, {bad} bad")

    # Fault injection — the byte-comparison must FAIL when pointed at the wrong source frame.
    m = json.load(open(os.path.join(dirs[0], "convert_meta.json")))
    arr = np.asarray(nib.load(os.path.join(DATA, m["source_file"])).dataobj)
    wrong = apply_reframe(arr[..., m["frame_indices_native"][1]],
                          _perm_from_meta(m), m["reframe"]["flips"])
    got = np.asarray(nib.load(os.path.join(dirs[0], "3d_recon", "sax_frame_00.nii.gz")).dataobj)
    fired = not np.array_equal(got, wrong)
    print(f"fault-inject (frame 0 vs frame 1's source): mismatch detected = {fired}"
          f"  <- must be True, else the check is inert")
    return 1 if (bad or not fired) else 0


if __name__ == "__main__":
    raise SystemExit(main())
