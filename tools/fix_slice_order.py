"""Standardise SAX slice order to APEX-AT-z0 across the pooled cohort.

See `docs/58` §10a. Sources disagree about which anatomical end is stored first: CMRxRecon2023/24
and ACDC are (almost) uniformly base-first, M&Ms is apex-first, and CMRxRecon2025 is genuinely
mixed *within one scanner*. `respiratory.py` applies a one-sided breathing displacement along the
array's D axis with **no anatomical anchor**, so the storage order alone decides whether the
simulated heart moves inferiorly (physiological) or superiorly (backwards). Under base-first it is
backwards, which was true for 893 of 1343 subjects.

The fix is `np.flip(vol, axis=2)` on every base-first subject, so afterwards **every** subject has
apex at z0 and index increasing toward the base (= superior).

Design decisions (docs/58 §10a):

* **Driven per subject from a decision file**, never from a per-source rule — CMRx2025-UIH is
  ~50/50 within one scanner, so a blanket rule would corrupt half of it. Default driver is
  `result/slice_order_check/slice_order_decisions.csv` (columns `subject, order, flip, ...`),
  produced by `render_slice_order_check.py` + human adjudication of the ambiguous tail.
* **The affine is left untouched.** Verified for all four sources: every file already carries
  axcodes LPS with a POSITIVE z scale, i.e. it declares `+z = Superior`. The base is superior, so
  an array honouring that header has apex at z0. Base-first subjects therefore contradict their
  own header today, and flipping the ARRAY is exactly what makes the header honest.
  ⚠️ Do NOT "fix" this by editing the affine instead: `Orientationd(axcodes="LPS")` in
  `preprocess.py` reorders axes by what the affine says, so declaring `+z = Inferior` would make
  MONAI silently flip every one of those subjects straight back, with no error.
* **`convert_meta.json`'s `reframe.flips[2]` IS toggled** for flipped ACDC/M&Ms subjects.
  `verify_sax_conversion.py` proves each converted file is bit-identical to the original source
  *after applying the recorded flips*; changing the data without changing the record would
  silently break that losslessness proof. CMRx has no `convert_meta.json` (reconstructed, not
  converted) so 742 of the 893 need no metadata change.
* **All six file types flip together**, all on array axis 2, so image / seg / ROI / canonical stay
  mutually consistent: the 12 `3d_recon` frames, `4d_recon`, `heart_seg`, `heart_roi`,
  `heart_seg_canonical`, `heart_roi_canonical`.
* **Preflight** asserts the canonical files' z axis runs the SAME direction as the native files'
  before anything is written (see `preflight_zdir`). If `Orientationd` were ever a non-no-op for
  some subject, flipping both on axis 2 would silently de-synchronise them.
* **Atomic** (tmp file + `os.replace`), fully reversible via
  `scratch/data/_provenance/slice_order_fix.json`.

⚠️ **Revert ordering.** docs/56's `fix_slice_roll.py` already rewrote these same arrays and has its
own sidecar. The two must be undone **flip first, then roll** — reverting the roll underneath an
applied flip would corrupt the stack.

⚠️ After applying, the monai `cache_signature()` must change (or the cache be deleted):
`PersistentDataset` keys on input PATHS, not contents, so an existing cache silently serves
pre-flip volumes.

Usage:
    python tools/fix_slice_order.py                      # dry run + preflight
    python tools/fix_slice_order.py --fault-inject       # prove --verify actually fires
    python tools/fix_slice_order.py --apply
    python tools/fix_slice_order.py --verify             # expect 0 base-first remaining
    python tools/fix_slice_order.py --revert --apply
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
DECISIONS = os.path.join(ROOT, "result/slice_order_check/slice_order_decisions.csv")
PROV = os.path.join(DATA, "_provenance", "slice_order_fix.json")

SAX_GLOBS = [
    os.path.join(DATA, "CMRxRecon202*/Cine_combined/*/sax"),
    os.path.join(DATA, "ACDC_sax/*/sax"),
    os.path.join(DATA, "MNMs_sax/*/sax"),
]

# every one of these has z on ARRAY AXIS 2:
#   (X, Y, Z)  (X, Y, Z, T)  (256, 256, D)  (256, 256, D, 12)
FILE_NAMES = ["4d_recon.nii.gz", "heart_seg.nii.gz", "heart_roi.nii.gz",
              "heart_seg_canonical.nii.gz", "heart_roi_canonical.nii.gz"]
Z_AXIS = 2


def find_sax_dirs():
    out = {}
    for pat in SAX_GLOBS:
        for d in glob.glob(pat):
            out[os.path.basename(os.path.dirname(d))] = d
    return out


def subject_files(sax):
    fs = sorted(glob.glob(os.path.join(sax, "3d_recon", "sax_frame_*.nii.gz")))
    fs += [os.path.join(sax, n) for n in FILE_NAMES]
    return [f for f in fs if os.path.exists(f)]


def flip_file(path):
    """np.flip along z, written atomically. Idempotent-by-involution: flipping twice restores."""
    img = nib.load(path)
    data = np.asarray(img.dataobj)
    if data.ndim < 3:
        raise ValueError(f"{path}: ndim={data.ndim}, expected >=3")
    flipped = np.ascontiguousarray(np.flip(data, axis=Z_AXIS))
    out = nib.Nifti1Image(flipped, img.affine, img.header)
    tmp = path[: -len(".nii.gz")] + ".tmp_flip.nii.gz"      # keep .nii.gz: nibabel sniffs the ext
    nib.save(out, tmp)
    os.replace(tmp, path)
    return path


def flip_subject(sax):
    for f in subject_files(sax):
        flip_file(f)
    toggle_convert_meta(sax)
    return sax


def toggle_convert_meta(sax):
    """Keep `verify_sax_conversion.py`'s bit-exactness proof true after the data changed."""
    p = os.path.join(sax, "convert_meta.json")
    if not os.path.exists(p):
        return False                                        # CMRx: nothing to keep in sync
    m = json.load(open(p))
    new = not m["reframe"]["flips"][Z_AXIS]
    m["reframe"]["flips"][Z_AXIS] = new
    # `flip_subject` is its own inverse, so this runs on --revert too. Record the RESULTING value
    # rather than "applied/reverted" so the appended log is self-describing either way.
    m.setdefault("post_conversion_fixes", []).append(
        {"fix": "slice_order_apex_at_z0", "doc": "docs/58 §10a", "axis": Z_AXIS,
         "reframe_flips_z_now": new,
         "note": "array flipped on z; reframe.flips[2] toggled so verify_sax_conversion stays valid"})
    tmp = p + ".tmp"
    json.dump(m, open(tmp, "w"), indent=1)
    os.replace(tmp, p)
    return True


# ── verification helpers ─────────────────────────────────────────────────────────────────────

def _profile(path, ed_only=True):
    a = np.asarray(nib.load(path).dataobj)
    if a.ndim == 4 and ed_only:
        a = a[..., 0]
    return np.array([(a[..., z] > 0).sum() for z in range(a.shape[Z_AXIS])], dtype=float)


def preflight_zdir(sax):
    """Do the canonical files run the same z direction as the native ones?

    Both are flipped on axis 2, so if `Orientationd` had ever reversed z for some subject the two
    would de-synchronise silently. Correlate the per-plane labeled-voxel profiles of `heart_seg`
    (native) and `heart_seg_canonical`; a NEGATIVE correlation means reversed.
    -> (subject, status, corr)
    """
    subj = os.path.basename(os.path.dirname(sax))
    nat = os.path.join(sax, "heart_seg.nii.gz")
    can = os.path.join(sax, "heart_seg_canonical.nii.gz")
    if not (os.path.exists(nat) and os.path.exists(can)):
        return subj, "missing", float("nan")
    pn, pc = _profile(nat), _profile(can)
    if pn.shape != pc.shape:
        return subj, f"shape {pn.shape} vs {pc.shape}", float("nan")
    if pn.std() == 0 or pc.std() == 0:
        return subj, "flat", float("nan")
    c = float(np.corrcoef(pn, pc)[0, 1])
    return subj, ("ok" if c > 0 else "REVERSED"), c


def detect_order(sax):
    """Re-run the adopted f1+f2 detector on whatever is currently on disk."""
    from render_slice_order_check import features
    p = os.path.join(sax, "heart_seg.nii.gz")
    if not os.path.exists(p):
        return os.path.basename(os.path.dirname(sax)), None
    f = features(p)
    return os.path.basename(os.path.dirname(sax)), (f["order"] if f else None)


def load_decisions(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", default=DECISIONS)
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--verify", action="store_true", help="re-detect on disk; no writes")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--fault-inject", default="",
                    help="flip ONE named subject back and confirm --verify catches exactly it")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="debug: only touch the first N subjects")
    ap.add_argument("--force", action="store_true",
                    help="override the already-applied guard (see the re-run guard in main)")
    a = ap.parse_args()

    sax_dirs = find_sax_dirs()
    print(f"found {len(sax_dirs)} sax dirs")

    # ── verify ────────────────────────────────────────────────────────────────────────────────
    if a.verify:
        dirs = sorted(sax_dirs.values())
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            res = list(ex.map(detect_order, dirs, chunksize=8))
        base = [s for s, o in res if o == "base-first"]
        apex = sum(1 for _, o in res if o == "apex-first")
        und = sum(1 for _, o in res if o is None)
        print(f"on disk now: apex-first {apex}   base-first {len(base)}   undetermined {und}")
        if base:
            print("  STILL BASE-FIRST (expected 0 after --apply):")
            for s in base[:40]:
                print("   ", s)
        return 0 if not base else 1

    # ── revert ────────────────────────────────────────────────────────────────────────────────
    if a.revert:
        if not os.path.exists(PROV):
            print(f"no sidecar at {PROV}, nothing to revert")
            return 1
        rec = json.load(open(PROV))
        subs = rec["flipped_subjects"]
        print(f"reverting {len(subs)} subjects" + ("" if a.apply else "   [DRY RUN]"))
        if a.apply:
            dirs = [sax_dirs[s] for s in subs if s in sax_dirs]
            with ProcessPoolExecutor(max_workers=a.workers) as ex:
                for i, _ in enumerate(ex.map(flip_subject, dirs, chunksize=4)):
                    if (i + 1) % 200 == 0:
                        print(f"  {i+1}/{len(dirs)}", flush=True)
            os.replace(PROV, PROV + ".reverted")
            print("reverted; sidecar renamed to .reverted")
        return 0

    # ── fault injection: prove --verify fires ────────────────────────────────────────────────
    if a.fault_inject:
        s = a.fault_inject
        if s not in sax_dirs:
            print(f"unknown subject {s}")
            return 1
        print(f"FAULT INJECT: flipping {s} back, then re-detecting it")
        flip_subject(sax_dirs[s])
        _, order = detect_order(sax_dirs[s])
        print(f"  detector now reports: {order}   (expect 'base-first' => the check CAN fire)")
        print(f"  restoring {s}")
        flip_subject(sax_dirs[s])
        _, order2 = detect_order(sax_dirs[s])
        print(f"  detector after restore: {order2}  (expect 'apex-first')")
        return 0 if (order == "base-first" and order2 == "apex-first") else 1

    # ── plan ──────────────────────────────────────────────────────────────────────────────────
    rows = load_decisions(a.decisions)
    todo = [r["subject"] for r in rows if r["flip"] == "yes"]
    missing = [s for s in todo if s not in sax_dirs]
    if missing:
        print(f"ERROR: {len(missing)} decided subjects not found on disk, e.g. {missing[:5]}")
        return 1
    if a.limit:
        todo = todo[: a.limit]

    by_src = {}
    for r in rows:
        if r["flip"] == "yes":
            by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"decisions: {a.decisions}")
    print(f"  flip {len(todo)} subjects  {by_src}")
    nfiles = sum(len(subject_files(sax_dirs[s])) for s in todo)
    nmeta = sum(os.path.exists(os.path.join(sax_dirs[s], "convert_meta.json")) for s in todo)
    print(f"  {nfiles} nifti files, {nmeta} convert_meta.json to keep in sync")

    # ── preflight ─────────────────────────────────────────────────────────────────────────────
    print("\npreflight: canonical z-direction must match native ...", flush=True)
    dirs = [sax_dirs[s] for s in todo]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        pf = list(ex.map(preflight_zdir, dirs, chunksize=8))
    bad = [(s, st, c) for s, st, c in pf if st == "REVERSED"]
    other = [(s, st, c) for s, st, c in pf if st not in ("ok", "REVERSED")]
    print(f"  ok {sum(st=='ok' for _, st, _ in pf)}   REVERSED {len(bad)}   "
          f"unusable {len(other)}")
    for s, st, c in other[:10]:
        print(f"    unusable {s}: {st}")
    if bad:
        print("  ABORT — canonical and native z run opposite directions for:")
        for s, _, c in bad[:20]:
            print(f"    {s}  corr={c:+.3f}")
        return 1
    if a.preflight_only:
        return 0

    if not a.apply:
        print("\n[DRY RUN — pass --apply to write]")
        return 0

    # ── re-run guard ─────────────────────────────────────────────────────────────────────────
    # `flip_subject` is its own INVERSE, which makes a second --apply destructive, not idempotent:
    # it would flip all 893 subjects back to base-first, re-toggle the 151 convert_meta.json, and
    # overwrite the sidecar with a record that then falsely asserts the fix is applied — with
    # output byte-identical to a successful first run. Nothing else catches this: the decisions CSV
    # is static, and preflight_zdir is provably blind (Pearson r is invariant when BOTH profiles are
    # reversed). Found by the /prove-it audit, docs/58 §10b.
    if os.path.exists(PROV) and not a.force:
        print(f"\nREFUSING: {PROV} exists — the fix is already applied.")
        print("  Re-applying would FLIP EVERYTHING BACK (the operation is its own inverse).")
        print("  Run --verify to check on-disk state, --revert --apply to undo, or --force to override.")
        return 1

    # ── apply ─────────────────────────────────────────────────────────────────────────────────
    print(f"\nflipping {len(todo)} subjects ...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, _ in enumerate(ex.map(flip_subject, dirs, chunksize=4)):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(dirs)}", flush=True)

    os.makedirs(os.path.dirname(PROV), exist_ok=True)
    json.dump({"doc": "docs/58 §10a",
               "operation": "np.flip(axis=2) -> standardise slice order to APEX at z0",
               "affine": "UNCHANGED (already LPS with +z = Superior for all sources)",
               "convert_meta": "reframe.flips[2] toggled for ACDC/M&Ms so "
                               "verify_sax_conversion.py stays valid",
               "revert_order": "undo THIS fix before docs/56 slice_roll_fix.json",
               "decisions_file": os.path.relpath(a.decisions, ROOT),
               "counts": by_src,
               "n_subjects": len(todo), "n_files": nfiles,
               "files_per_subject": ["3d_recon/sax_frame_*.nii.gz"] + FILE_NAMES,
               "flipped_subjects": todo},
              open(PROV, "w"), indent=1)
    print(f"wrote {PROV}")
    print("\nDONE. Next: bump cache_signature() (monai keys on paths, not contents), "
          "then re-run with --verify")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
