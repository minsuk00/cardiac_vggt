"""Capture / compare the canonical preprocessing output across an env change.

The `docs/49` bar for a dependency upgrade is: the float32 canonical cube must be
**byte-identical** across a subject set spanning every distinct geometric path
(all (X,Y) spacings + all Z crop/pad/identity paths). This script implements that
check as a two-phase capture-then-compare so it works across an env mutation:

    # BEFORE the upgrade
    python tools/env_parity_preprocess.py capture --out ref_numpy1.npz
    # ... change the env ...
    # AFTER
    python tools/env_parity_preprocess.py compare --ref ref_numpy1.npz

`capture` stores, per subject, the float32 phases cube + the content mask, plus an
env stamp. `compare` RECOMPUTES both from the real NIfTIs and requires bitwise
equality (np.array_equal + sha256 of the raw bytes).

Hardened after a prove-it review: it now refuses a vacuous pass. An empty subject
set, a lost subject, a key-set mismatch between ref and recompute, or a reference
captured in the SAME env are all FATAL rather than silently reported as N/N.

Used for the numpy 1.26.4 -> 2.x migration (2026-07-24); reusable for any future
torch/monai/numpy bump.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "training")
sys.path.insert(0, ".")

from data.preprocess import (  # noqa: E402
    build_data_dicts,
    get_canonical_transforms,
)

DATA_ROOT = "/home/minsukc/vggt/scratch/data/CMRxRecon2024/Cine_combined"


def pick_subjects(limit=None):
    """Subjects spanning every distinct (X,Y) spacing and every Z path.

    Mirrors docs/49's spanning set: the geometry of the resample is what a
    library change could perturb, so cover each distinct geometric path once.
    """
    import nibabel as nib

    sax_dirs = sorted(glob.glob(os.path.join(DATA_ROOT, "*", "sax")))
    seen_xy, seen_z, chosen = set(), set(), []
    for d in sax_dirs:
        p0 = os.path.join(d, "3d_recon", "sax_frame_00.nii.gz")
        if not os.path.exists(p0):
            print(f"  WARNING: no phase-00 file, skipping {d}")
            continue
        try:
            h = nib.load(p0)
            zx, zy, zz = (round(float(v), 4) for v in h.header.get_zooms()[:3])
            H, Z = int(h.shape[1]), int(h.shape[2])
        except Exception as e:
            # Loud: a silently-dropped subject quietly shrinks coverage while the
            # success ratio still reads 100%.
            print(f"  WARNING: header read failed ({type(e).__name__}), skipping {d}")
            continue
        if abs(zz - 12.0) > 1e-3:
            # Keying the Z path on shape[2] is only valid when the on-disk Z pitch
            # is the canonical 12 mm (docs/27 relabel). Otherwise crop/pad/identity
            # would be mis-keyed silently.
            print(f"  WARNING: zz={zz} != 12.0, Z-path keying unreliable for {d}")
        # Key on (zx, zy, H) — NOT (zx, zy). ResizeWithPadOrCropd decides crop vs
        # pad per axis from native_extent = size * spacing, and H varies in
        # {162,204,246} INDEPENDENTLY of zy, so two subjects sharing zy can take
        # opposite Y branches. Keying on spacing alone leaves one branch untested.
        # (X needs no such term: W is always 256, so X extent is a function of zx.)
        key_xy, key_z = (zx, zy, H), Z
        if key_xy in seen_xy and key_z in seen_z:
            continue
        seen_xy.add(key_xy)
        seen_z.add(key_z)
        chosen.append(d)
        if limit and len(chosen) >= limit:
            print(f"  WARNING: --limit {limit} reached; the set is NO LONGER spanning")
            break
    return chosen


def run_one(sax_dir, transforms):
    """Return the float32 canonical outputs for one subject."""
    dd = build_data_dicts([sax_dir])[0]
    out = transforms(dd)
    phases = out["phases"]
    mask = out["content_mask"]
    phases = phases.as_tensor() if hasattr(phases, "as_tensor") else phases
    mask = mask.as_tensor() if hasattr(mask, "as_tensor") else mask
    return (phases.float().cpu().numpy().astype(np.float32),
            mask.float().cpu().numpy().astype(np.float32))


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def env_stamp() -> str:
    """Identify the env, so `compare` can refuse a same-env (vacuous) reference."""
    import monai
    return f"numpy={np.__version__} torch={torch.__version__} monai={monai.__version__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--out", default="/tmp/env_parity_ref.npz")
    ap.add_argument("--ref", default="/tmp/env_parity_ref.npz")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"numpy {np.__version__} | torch {torch.__version__}")
    import monai
    print(f"monai {monai.__version__}")

    # float32 pre-CastToTyped, exactly as docs/49 measured it
    transforms = get_canonical_transforms(storage_dtype=torch.float32)
    subs = pick_subjects(args.limit)
    label = "spanning" if not args.limit else "TRUNCATED (--limit, NOT spanning)"
    print(f"{label} subject set: {len(subs)} subjects\n")

    if not subs:
        sys.exit("FATAL: no subjects found — check DATA_ROOT is mounted. Refusing "
                 "to report a vacuous pass.")

    results = {}
    for i, d in enumerate(subs):
        name = os.path.basename(os.path.dirname(d)) + "_" + os.path.basename(d)
        # A subject that raises here is EXACTLY the regression this tool exists to
        # catch (a new numpy/monai breaking the pipeline). Never swallow it — a
        # skipped subject would drop out of both the numerator and denominator and
        # still report a clean N/N.
        ph, mk = run_one(d, transforms)
        results[name + "|phases"] = ph
        results[name + "|mask"] = mk
        print(f"  [{i+1}/{len(subs)}] {name}: phases{ph.shape} "
              f"sha {digest(ph)[:16]} | mask sha {digest(mk)[:16]}")

    assert len(results) == 2 * len(subs), (
        f"expected {2*len(subs)} arrays, got {len(results)} — a subject was lost")

    if args.mode == "capture":
        np.savez_compressed(args.out, _env=np.array(env_stamp()), **results)
        print(f"\ncaptured {len(results)//2} subjects -> {args.out}")
        print(f"env stamp: {env_stamp()}")
        return

    ref = np.load(args.ref)

    # The reference must come from a DIFFERENT env, else we are comparing an env
    # against itself and any result is meaningless.
    ref_env = str(ref["_env"]) if "_env" in ref.files else "<none recorded>"
    print(f"\nref env : {ref_env}\nthis env: {env_stamp()}")
    if ref_env == env_stamp():
        sys.exit("FATAL: reference was captured in THIS SAME env — the comparison "
                 "would be vacuous. Re-capture under the old env.")
    if ref_env == "<none recorded>":
        print("WARNING: reference has no env stamp (pre-hardening capture); "
              "cannot prove it came from a different env.")

    # Both directions must match, or a shrunken/expanded set reports a false 100%.
    ref_keys = {k for k in ref.files if k != "_env"}
    if ref_keys != set(results):
        only_ref = sorted(ref_keys - set(results))
        only_new = sorted(set(results) - ref_keys)
        sys.exit(f"FATAL: key-set mismatch — {len(only_ref)} only in ref "
                 f"{only_ref[:3]}, {len(only_new)} only in new {only_new[:3]}. "
                 "The two runs did not select the same subjects.")

    n_ok = n_bad = 0
    bad = []
    for k in sorted(results):
        a, b = ref[k], results[k]
        if a.shape != b.shape:
            n_bad += 1
            bad.append((k, f"shape {a.shape} vs {b.shape}"))
            continue
        if np.array_equal(a, b) and digest(a) == digest(b):
            n_ok += 1
        else:
            n_bad += 1
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            bad.append((k, f"max|d|={d.max():.3e} mean|d|={d.mean():.3e}"))

    print(f"\n{'='*60}")
    print(f"BYTE-IDENTICAL: {n_ok}/{n_ok+n_bad} arrays")
    for k, why in bad:
        print(f"  DIFFERS: {k}  {why}")
    print(f"{'='*60}")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
