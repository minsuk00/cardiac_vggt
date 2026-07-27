#!/usr/bin/env python
"""One-time on-disk relabel of the slice-axis (Z) spacing in our reconstructed cine NIfTIs.

WHY: our k-space->NIfTI recons stamped Z spacing = slice THICKNESS (8 mm), not the true
center-to-center PITCH. This rewrites ONLY the affine's slice-axis spacing to the true pitch
so the geometry is physically honest (the canonical pipeline uses 12 mm; see docs/27).

  CMRx : 8 -> 12 mm   (8 mm thickness + 4 mm gap, CMRxRecon2024 protocol; constant)
  OCMR : 8 -> per-subject measured pitch = mean consecutive distance of
          meta.json `slice_positions_mm` (9.6 or 10.0 mm)  [honesty only — the OCMR
          eval path reads meta positions directly, not the affine]

These NIfTIs are OUR reconstructions (regenerable from raw k-space), not pristine source.
Voxel data is NEVER touched — only the affine header (verified byte-identical via set_sform).
ACDC / Göttingen are left alone (their affines are already honest).

Usage:
  python tools/relabel_slice_spacing.py --dataset both --dry-run   # show, change nothing
  python tools/relabel_slice_spacing.py --dataset both             # do it (in place)
  python tools/relabel_slice_spacing.py --dataset both --revert    # restore Z -> 8 mm
"""
import argparse
import glob
import json
import os

import nibabel as nib
import numpy as np

CMRX_ROOT = "scratch/data/CMRxRecon2024/Cine_combined"
CMRX2023_ROOT = "scratch/data/CMRxRecon2023/Cine_combined"
OCMR_ROOT = "scratch/data/ocmr/recon"
ORIG_Z_MM = 8.0          # both CMRx and OCMR were stamped with the 8 mm thickness
CMRX_TRUE_MM = 12.0      # CMRx true pitch (8 mm thickness + 4 mm gap; documented for 2023)
CMRX2023_SIXMM_MM = 10.0 # 2023 6 mm-thickness variants: 6 + the documented 4 mm gap. ASSUMPTION.
CMRX2023_SIXMM_ORIG = 6.0  # ...and their stamped source value, for a correct --revert
INPLANE_MAX_MM = 4.0     # in-plane spacing is <= ~2.2 mm; slice axis is the only col with norm > this


def _slice_axis(affine):
    """Index of the through-plane (slice) axis = the unique spatial column with norm > INPLANE_MAX_MM.

    Errors if not exactly one such column (refuses to guess on an unexpected affine)."""
    norms = [float(np.linalg.norm(affine[:3, i])) for i in range(3)]
    big = [i for i, n in enumerate(norms) if n > INPLANE_MAX_MM]
    if len(big) != 1:
        raise ValueError(f"expected exactly one slice axis (norm > {INPLANE_MAX_MM}), got {norms}")
    return big[0], norms[big[0]]


def _relabel_file(path, target_mm, dry, revert, orig_mm=ORIG_Z_MM):
    """orig_mm is per-file, not the global ORIG_Z_MM: the 2023 6 mm-thickness variants were
    stamped 6.0, so a --revert that assumed 8.0 would silently corrupt them."""
    img = nib.load(path)
    A = img.affine.copy()
    axis, cur = _slice_axis(A)
    want = orig_mm if revert else target_mm
    if abs(cur - want) < 1e-4:
        return ("skip", cur, want)                 # already at target (idempotent)
    if not dry:
        A[:3, axis] = A[:3, axis] * (want / cur)   # rescale only the slice column
        img.set_sform(A)
        img.set_qform(A)
        nib.save(img, path)
    return ("change", cur, want)


def _cmrx_files():
    fs = sorted(glob.glob(os.path.join(CMRX_ROOT, "*/sax/3d_recon/sax_frame_*.nii.gz")))
    fs += sorted(glob.glob(os.path.join(CMRX_ROOT, "*/sax/4d_recon.nii.gz")))
    return [(f, CMRX_TRUE_MM) for f in fs]


def _cmrx2023_files():
    """CMRxRecon2023 — PER-SUBJECT target, unlike 2024's single constant.

    Standard protocol (259 subjects): thickness 8 mm + gap 4 mm = 12.0 mm pitch. Unlike 2024 this
    is DOCUMENTED, on the challenge website's Task 1 page ("slice thickness 8.0 mm, and slice gap
    4.0 mm"), and it agrees with an independent LV-caliper measurement.

    ⚠️ The 6 mm-thickness protocol variants get 10.0 mm, which is an ASSUMPTION, not a measurement:
    it takes the documented 4 mm gap to be a protocol constant (6 + 4). It could not be measured --
    only one of those subjects carries the LAX label the caliper method needs, and that subject's
    two duplicate copies disagreed by 2 mm (9.71 vs 11.64). A 50%-distance-factor reading would give
    9 mm, and the caliper reading was actually closest to 12 mm. Flagged in the 2023 README.
    """
    six_mm_ids = {"CMRx23_Train_P040", "CMRx23_Train_P046", "CMRx23_Test_P116"}
    out = []
    for f in sorted(glob.glob(os.path.join(CMRX2023_ROOT, "*/sax/3d_recon/sax_frame_*.nii.gz"))) + \
             sorted(glob.glob(os.path.join(CMRX2023_ROOT, "*/sax/4d_recon.nii.gz"))):
        subj = os.path.relpath(f, CMRX2023_ROOT).split(os.sep)[0]
        if subj in six_mm_ids:
            out.append((f, CMRX2023_SIXMM_MM, CMRX2023_SIXMM_ORIG))
        else:
            out.append((f, CMRX_TRUE_MM, ORIG_Z_MM))
    return out


def _ocmr_files():
    out = []
    for f in sorted(glob.glob(os.path.join(OCMR_ROOT, "**/sax_cine.nii.gz"), recursive=True)):
        if "_failed" in f:                          # skip excluded failed recons
            continue
        mf = os.path.join(os.path.dirname(f), "meta.json")
        if not os.path.exists(mf):
            print(f"  WARN no meta.json, skipping {f}")
            continue
        pos = np.asarray(json.load(open(mf)).get("slice_positions_mm", []), dtype=np.float64)
        if len(pos) < 2:
            print(f"  WARN <2 slice positions, skipping {f}")
            continue
        pitch = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).mean())
        out.append((f, round(pitch, 4)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cmrx", "cmrx2023", "ocmr", "both"], default="both",
                    help="'both' = cmrx(2024) + ocmr, the historical default. cmrx2023 is opt-in.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true", help="restore each file's ORIGINAL stamped spacing")
    args = ap.parse_args()

    # entries are (path, target_mm) or (path, target_mm, orig_mm); orig defaults to ORIG_Z_MM
    work = []
    if args.dataset in ("cmrx", "both"):
        work += _cmrx_files()
    if args.dataset == "cmrx2023":
        work += _cmrx2023_files()
    if args.dataset in ("ocmr", "both"):
        work += _ocmr_files()
    work = [(w + (ORIG_Z_MM,))[:3] if len(w) == 2 else w for w in work]

    mode = "DRY-RUN" if args.dry_run else ("REVERT" if args.revert else "RELABEL")
    print(f"[{mode}] {len(work)} files\n")
    changed = skipped = 0
    samples = []
    for path, target, orig in work:
        action, cur, want = _relabel_file(path, target, args.dry_run, args.revert, orig)
        if action == "change":
            changed += 1
            if len(samples) < 12:
                samples.append(f"  {cur:.2f} -> {want:.2f}  {path}")
        else:
            skipped += 1
    for s in samples:
        print(s)
    if changed > len(samples):
        print(f"  … and {changed - len(samples)} more")
    print(f"\n{mode}: {changed} would change, {skipped} already-at-target/skipped"
          if args.dry_run else f"\n{mode}: {changed} changed, {skipped} already-at-target/skipped")


if __name__ == "__main__":
    main()
