#!/usr/bin/env python
"""Convert the OCMR gated (ECG breath-hold) SAX stacks into the canonical 12-phase layout.

Output per subject (mirrors `ACDC_sax/`, `MNMs_sax/`, `MIITT_sax/`):

    scratch/data/OCMR_sax/OCMR_<series>/sax/3d_recon/sax_frame_{00..11}.nii.gz
                                           /sax/heart_seg.nii.gz             (X,Y,Z,12)
                                           /sax/heart_roi.nii.gz             (X,Y,Z)   native union
                                           /sax/heart_seg_canonical.nii.gz   (256,256,D,12)
                                           /sax/heart_roi_canonical.nii.gz   (256,256,D)
                                           /sax/convert_meta.json

The SOURCE tree (`scratch/data/ocmr/recon/gated/`) is opened READ-ONLY and never written to.

This is `convert_miitt_to_12phase.py` with two differences, both forced by the data:

1. **The source is one 4-D cine, not per-frame files.** MIITT could `shutil.copy2` its frames;
   here each of the 12 picked frames is sliced out of `sax_cine.nii.gz` (X,Y,Z,T) and re-saved
   with the SOURCE affine. No resample, no interpolation, no re-framing — OCMR already ships a
   clean axis-aligned LPS affine (asserted in `check_geometry`, raises rather than warns).

2. **Spacing genuinely varies**, so MIITT's fixed `EXPECTED_ZOOMS` equality check becomes a
   range check. Measured over the 8 gated SAX stacks: in-plane 1.98–2.25 mm, pitch 7.8–10.0 mm.
   The pitch is the TRUE centre-to-centre distance (median of consecutive acquisition `position`
   fields, `scratch/data/ocmr/README.md` §3) — not the slab thickness — which is what the
   native-z design (docs/58) turns into the reconstruction grid, so it must not be re-derived.

Frame selection is the shared rule, imported verbatim from `convert_to_sax_layout.pick_frames`:
`native_idx(j) = (ed + round(j*T/12)) % T`, nearest native frame, NO temporal interpolation
(`V_gt` is the supervision target, and blended targets teach blur). ED therefore lands exactly on
output frame 0. ED comes from `scratch/data/whs/cardiac_phase.csv` (`dataset == "ocmr"` gated SAX
rows) and is never assumed 0 — `fs_0063` has ED=1 and `fs_0074` has ED=18, and the `% T` wrap is
what makes those work. 182 native frames across the 8 subjects become 96.

`heart_seg_canonical` / `heart_roi_canonical` are REQUIRED, not optional: `training/loss.py`
raises when a sample has no `heart_roi_canonical` and `loss.volume.heart_weight > 0`. They are
free here — OCMR already ships the persisted native Task114 seg on the SAME affine as the cine,
so `assemble_whs.build_canonical_siblings` rebuilds both with no GPU and no nnU-Net rerun.

`vendor` is left EMPTY on purpose. The attributes CSV records a scanner CODE (`30pris`, `15sola`)
whose expansion is not documented anywhere in this repo, so it is stored verbatim as
`scanner_code` rather than guessed into a vendor string — these subjects drop out of
vendor-stratified analyses instead of polluting them. All 8 are `sub == "vol"` (volunteers) in
`ocmr_data_attributes.csv`, hence `group = "NOR"`.

**Slice order is standardised to apex-at-z0 IN THIS CONVERTER**, unlike the ACDC/M&Ms/MIITT ones.
OCMR is mixed — measured 3 apex-first, 5 base-first — so it needs the same per-subject flip the
rest of the cohort got on 2026-07-31 (docs/58 §10a). The decision comes per subject from the
adopted f1+f2 detector (`render_slice_order_check.features`) run on the ED frame of the SELECTED
12-phase seg, and an undetermined subject RAISES rather than being guessed. The flip is
`np.flip(axis=2)` on the images and the seg, applied BEFORE `heart_roi` / the canonical siblings
are derived, so every file is mutually consistent by construction. **The affine is deliberately
left untouched** — every OCMR file already declares LPS with +z = Superior, so flipping the array
is what makes the header honest; editing the affine instead would make `Orientationd` in
`preprocess.py` silently flip it straight back.

Why here and not via `tools/fix_slice_order.py`: that tool holds a single cohort-wide provenance
sidecar (`scratch/data/_provenance/slice_order_fix.json`, 893 subjects) and refuses to re-run
without `--force`, which would overwrite that revert record. This tree is regenerable from
read-only source in ~2 min, so it needs no separate sidecar — `convert_meta.json` records the
flip in `reframe.flips[2]` and a `slice_order` block.

Usage:
    python tools/convert_ocmr_to_12phase.py                  # dry run, all 8 subjects
    python tools/convert_ocmr_to_12phase.py --apply
    python tools/convert_ocmr_to_12phase.py --subjects fs_0012_3T --apply
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np

# Derived from __file__, not hardcoded, so the converter behaves identically when run from a git
# worktree (where a hardcoded /home/minsukc/vggt would import the OTHER tree's helpers).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "scratch/data")
SRC_GLOB = os.path.join(DATA, "ocmr", "recon", "gated", "exam_*", "sax__*")
OUT_ROOT = os.path.join(DATA, "OCMR_sax")
CARDIAC_CSV = os.path.join(DATA, "whs", "cardiac_phase.csv")
ATTRS_CSV = os.path.join(DATA, "ocmr", "ocmr_data_attributes.csv")
WHS_ROWS = os.path.join(DATA, "whs", "rows")
ID_PREFIX = "OCMR_"

sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "nnunet_mnms_eval"))
from convert_to_sax_layout import NUM_PHASES, pick_frames                    # noqa: E402
from assemble_whs import LOW_LABELED, build_canonical_siblings, build_roi    # noqa: E402
from render_slice_order_check import features as slice_order_features        # noqa: E402

Z_AXIS = 2   # (X, Y, Z[, T]) for every array this converter writes; matches fix_slice_order.py

# Plausibility bounds, not protocol constants: OCMR spacing varies per exam (see docstring).
# These exist to catch a mis-parsed header, not to pin a value.
INPLANE_MM_RANGE = (1.0, 4.0)
PITCH_MM_RANGE = (5.0, 15.0)

GROUP_BY_SUB = {"vol": "NOR"}     # `sub` column of ocmr_data_attributes.csv; all 8 gated SAX are vol
VENDOR = ""                       # scanner code is recorded instead — see module docstring
CENTRE = ""


def read_ed_es():
    """{series: (T_native, ed, es, unit, subject_path)} for the gated OCMR **SAX** units.

    `cardiac_phase.csv`'s `subject` for OCMR is the recon-relative path
    (`gated/exam_fs_0012/sax__fs_0012_3T`), so the series name is its last path component with
    the `sax__` prefix stripped. LAX units are skipped: they are single-slice and have no stack.
    """
    out = {}
    with open(CARDIAC_CSV) as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "ocmr" or row["regime"] != "gated":
                continue
            leaf = row["subject"].rstrip("/").split("/")[-1]
            if not leaf.startswith("sax__"):
                continue
            out[leaf[len("sax__"):]] = (int(row["T"]), int(row["ED"]), int(row["ES"]),
                                        row["unit"], row["subject"])
    if not out:
        raise SystemExit(f"no gated SAX dataset=='ocmr' rows in {CARDIAC_CSV}")
    return out


def read_attrs():
    """{series: {scn, sub, ...}} from ocmr_data_attributes.csv (keyed on the .h5 basename)."""
    out = {}
    with open(ATTRS_CSV) as f:
        for row in csv.DictReader(f):
            name = (row.get("file name") or "").strip()
            if name.endswith(".h5"):
                out[name[:-3]] = row
    return out


def find_sources():
    """{series: sax_dir} for every gated SAX stack on disk."""
    out = {}
    for d in sorted(glob.glob(SRC_GLOB)):
        series = os.path.basename(d)[len("sax__"):]
        if series in out:
            raise ValueError(f"duplicate series name {series}: {out[series]} and {d}")
        out[series] = d
    return out


def check_geometry(series, im):
    """Assert the source really is LPS + axis-aligned + SAX-shaped spacing. Raises, never warns."""
    ax = "".join(nib.aff2axcodes(im.affine))
    if ax != "LPS":
        raise ValueError(f"{series}: axcodes {ax} != LPS — reorient the ARRAY (never the affine)")
    A = im.affine[:3, :3]
    if not np.allclose(A, np.diag(np.diag(A)), atol=1e-6):
        raise ValueError(f"{series}: affine is not axis-aligned:\n{A}")
    zooms = tuple(float(z) for z in im.header.get_zooms()[:3])
    for z in zooms[:2]:
        if not INPLANE_MM_RANGE[0] <= z <= INPLANE_MM_RANGE[1]:
            raise ValueError(f"{series}: in-plane spacing {zooms} outside {INPLANE_MM_RANGE}")
    if not PITCH_MM_RANGE[0] <= zooms[2] <= PITCH_MM_RANGE[1]:
        raise ValueError(f"{series}: slice pitch {zooms[2]} outside {PITCH_MM_RANGE}")
    if not zooms[2] > max(zooms[0], zooms[1]):
        raise ValueError(f"{series}: zooms={zooms} — axis 2 is not the slice axis")
    return zooms


def convert_one(series, src_dir, T_csv, ed, es, attrs, apply, in_mm=6.0):
    out_id = f"{ID_PREFIX}{series}"
    cine_f = os.path.join(src_dir, "sax_cine.nii.gz")
    seg_f = os.path.join(src_dir, "heart_seg.nii.gz")
    for p in (cine_f, seg_f):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{series}: missing {p}")

    cine_im = nib.load(cine_f)
    if cine_im.ndim != 4:
        raise ValueError(f"{series}: sax_cine is {cine_im.shape}, expected 4D")
    T = cine_im.shape[3]
    if T != T_csv:
        raise ValueError(f"{series}: cine T={T} but cardiac_phase.csv says T={T_csv}")
    if not (0 <= ed < T and 0 <= es < T):
        raise ValueError(f"{series}: ED={ed}/ES={es} out of range for T={T}")

    seg_im = nib.load(seg_f)
    seg4d = np.asarray(seg_im.dataobj).astype(np.uint8)              # (X, Y, Z, T)
    # The seg must sit on the SAME grid as the images — build_canonical_siblings resamples the seg
    # with the image's canonical transform, so a mismatch here would silently misalign the ROI
    # against the volume it is meant to weight.
    if seg4d.shape != cine_im.shape:
        raise ValueError(f"{series}: heart_seg {seg4d.shape} != cine {cine_im.shape}")
    if not np.allclose(seg_im.affine, cine_im.affine, atol=1e-6):
        raise ValueError(f"{series}: heart_seg affine does not match the cine")

    zooms = check_geometry(series, cine_im)
    idx = pick_frames(T, ed)
    # ES on the 12-frame grid: nearest sampled j to ES's fractional position. Advisory only —
    # `cardiac_phase.csv` is what the EF sweep actually reads.
    es_j = int(round(((es - ed) % T) / float(T) * NUM_PHASES)) % NUM_PHASES

    # Slice order, decided per subject on the ED frame of the SELECTED seg (frame 0 == ED by
    # construction). Undetermined raises: a silently mis-ordered stack reverses the simulated
    # breathing direction with no error (docs/58 §10a).
    seg12 = np.ascontiguousarray(seg4d[..., idx])
    so = slice_order_features(seg12[..., 0])
    if so is None or so["order"] is None:
        raise ValueError(f"{series}: slice order undetermined "
                         f"({'unusable seg' if so is None else 'f1/f2 disagree'}) — adjudicate "
                         f"with tools/render_slice_order_check.py before converting")
    flip_z = so["order"] == "base-first"

    a = attrs.get(series, {})
    meta = {
        "id": out_id,
        "source_file": os.path.relpath(cine_f, DATA),
        "native_T": T, "ed_native": ed, "es_native": es,
        "frame_indices_native": idx, "es_frame_on_12grid_advisory": es_j,
        "native_zooms_xyz": list(zooms), "out_spacing_xyz": list(zooms),
        "out_shape_xyz": list(cine_im.shape[:3]), "pitch_mm": zooms[2],
        "reframe": {"src_axcodes": "LPS", "swapped_inplane": False,
                    "flips": [False, False, bool(flip_z)],
                    "note": "OCMR ships a clean axis-aligned LPS affine; the only change is the "
                            "z flip that standardises slice order to apex-at-z0"},
        "slice_order": {"src_order": so["order"], "flipped_z": bool(flip_z),
                        "detector": "render_slice_order_check.features (f1+f2, docs/58 §10a)",
                        "f1_total": round(float(so["f1_total"]), 4),
                        "f2_cavity": round(float(so["f2_cavity"]), 4),
                        "n_labeled_planes": int(so["n_labeled"])},
        "converter": os.path.basename(__file__), "num_phases": NUM_PHASES,
        "arm": "gated", "roi_in_mm": in_mm,
        "group": GROUP_BY_SUB.get(a.get("sub", ""), ""), "vendor": VENDOR, "centre": CENTRE,
        "scanner_code": a.get("scn", ""), "fov_flag": a.get("fov", ""),
    }
    if not apply:
        return meta, None

    sax = os.path.join(OUT_ROOT, out_id, "sax")
    rec = os.path.join(sax, "3d_recon")
    os.makedirs(rec, exist_ok=True)
    arr = np.asarray(cine_im.dataobj)                                # (X, Y, Z, T), source dtype
    for j, t in enumerate(idx):
        v = arr[..., t]
        if flip_z:
            v = np.flip(v, axis=Z_AXIS)
        # pid in the tmp name so two concurrent converters can never collide on it.
        tmp = os.path.join(rec, f".sax_frame_{j:02d}.{os.getpid()}.tmp.nii.gz")
        nib.save(nib.Nifti1Image(np.ascontiguousarray(v), cine_im.affine), tmp)   # write, then rename
        os.replace(tmp, os.path.join(rec, f"sax_frame_{j:02d}.nii.gz"))

    # Everything below derives from seg12, so flipping it HERE keeps heart_seg / heart_roi / both
    # canonical siblings mutually consistent with the images by construction.
    if flip_z:
        seg12 = np.ascontiguousarray(np.flip(seg12, axis=Z_AXIS))
    nib.save(nib.Nifti1Image(seg12, seg_im.affine), os.path.join(sax, "heart_seg.nii.gz"))

    # Native-space ROI: REBUILT from the 12 selected frames, not copied from the source (whose
    # union is over all T). Same call assemble_whs.py makes for every other gated unit, so
    # `roi_vox` in the whs manifest row means the same thing across sources.
    union = (seg12 > 0).any(axis=-1).astype(np.uint8)
    roi = build_roi(union, zooms, in_mm=in_mm, z_extend=1)
    nib.save(nib.Nifti1Image(roi.astype(np.uint8), seg_im.affine),
             os.path.join(sax, "heart_roi.nii.gz"))

    cseg, croi, cspacing = build_canonical_siblings(seg12, seg_im.affine, in_mm=in_mm)
    caffine = np.diag([*cspacing, 1.0])
    nib.save(nib.Nifti1Image(cseg, caffine), os.path.join(sax, "heart_seg_canonical.nii.gz"))
    nib.save(nib.Nifti1Image(croi.astype(np.uint8), caffine),
             os.path.join(sax, "heart_roi_canonical.nii.gz"))

    meta["canonical_shape_xyz"] = list(cseg.shape[:3])
    meta["canonical_spacing_xyz"] = [float(x) for x in cspacing]
    meta["roi_voxels"] = int(croi.sum())
    with open(os.path.join(sax, "convert_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    write_whs_row(out_id, seg12, zooms, int(roi.sum()))
    return meta, sax


def write_whs_row(out_id, seg12, zooms, roi_vox):
    """One `scratch/data/whs/rows/<unit>.csv` line, byte-compatible with assemble_whs.py's.

    Same columns, same `flag` rule (mean labeled z-planes per frame < LOW_LABELED -> "low"), so
    these units join whs_manifest.csv — and therefore reach `cardiac_phase.csv`'s `seg_flag`
    column — exactly like the cmrx/acdc_sax/mnms_sax/miitt_sax units. That matters:
    `ef_eval.compute_ef_metrics` silently SKIPS any subject whose seg_flag is not "ok", and the
    pre-conversion OCMR rows have a BLANK seg_flag (they never had a whs_manifest row at all), so
    without this they would contribute nothing to the EF metric while looking fine everywhere else.
    """
    X, Y, Z, T = seg12.shape
    n_lab = [int((seg12[..., t] > 0).any(axis=(0, 1)).sum()) for t in range(T)]
    mean_lab = float(np.mean(n_lab))
    lv, myo, rv = (int((seg12 == c).sum()) for c in (1, 2, 3))
    flag = "low" if mean_lab < LOW_LABELED else "ok"
    row = [out_id, "ocmr_sax", "gated", out_id, f"{X}x{Y}x{Z}x{T}",
           "x".join(f"{s:.2f}" for s in zooms), str(T), f"{mean_lab:.2f}",
           str(lv), str(myo), str(rv), str(roi_vox), flag]
    os.makedirs(WHS_ROWS, exist_ok=True)
    with open(os.path.join(WHS_ROWS, out_id + ".csv"), "w") as f:
        f.write(",".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated series names, e.g. fs_0012_3T "
                                                  "(default: every gated SAX ocmr row)")
    ap.add_argument("--in_mm", type=float, default=6.0, help="ROI dilation (mm), matches assemble_whs")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = ap.parse_args()

    ed_es = read_ed_es()
    srcs = find_sources()
    attrs = read_attrs()
    wanted = [s.strip() for s in args.subjects.split(",") if s.strip()] or sorted(ed_es)
    missing = [s for s in wanted if s not in ed_es]
    if missing:
        raise SystemExit(f"no gated SAX cardiac_phase.csv row for: {missing}")
    missing = [s for s in wanted if s not in srcs]
    if missing:
        raise SystemExit(f"no recon dir under {SRC_GLOB} for: {missing}")

    n_ok = n_fail = 0
    for series in wanted:
        T, ed, es, _unit, _subj = ed_es[series]
        try:
            meta, sax = convert_one(series, srcs[series], T, ed, es, attrs, args.apply,
                                    in_mm=args.in_mm)
        except Exception as e:  # noqa: BLE001 — one bad subject must not hide the other 7
            print(f"  FAIL {series}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        n_ok += 1
        print(f"  {meta['id']:<20s} T={meta['native_T']:>2d} ED={meta['ed_native']:>2d} "
              f"ES={meta['es_native']:>2d}->j{meta['es_frame_on_12grid_advisory']:<2d} "
              f"shape={meta['out_shape_xyz']} dz={meta['pitch_mm']:.2f} "
              f"{meta['slice_order']['src_order']:<11s}"
              f"{'FLIP-Z' if meta['slice_order']['flipped_z'] else '      '} "
              f"idx={meta['frame_indices_native']}"
              + (f"  canon={meta['canonical_shape_xyz']} roi={meta['roi_voxels']}" if sax else ""))
    print(f"{'wrote' if args.apply else 'dry-run'}: ok={n_ok} fail={n_fail} -> {OUT_ROOT}")
    if not args.apply:
        print("(dry run — pass --apply to write)")


if __name__ == "__main__":
    main()
