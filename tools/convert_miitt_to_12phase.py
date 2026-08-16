#!/usr/bin/env python
"""Convert the MIITT gated (ECG breath-hold) cine into the canonical 12-phase training layout.

Output per subject (mirrors `ACDC_sax/`, `MNMs_sax/`, i.e. `CMRxRecon*/Cine_combined/<ID>/sax/`):

    scratch/data/MIITT_sax/MIITT_<subj>/sax/3d_recon/sax_frame_{00..11}.nii.gz
                                           /sax/heart_seg.nii.gz             (X,Y,Z,12)
                                           /sax/heart_roi.nii.gz             (X,Y,Z)   native union
                                           /sax/heart_seg_canonical.nii.gz   (256,256,D,12)
                                           /sax/heart_roi_canonical.nii.gz   (256,256,D)
                                           /sax/convert_meta.json

The SOURCE tree (`scratch/data/MIITT/`) is opened READ-ONLY and never written to.

Why this converter is so much thinner than `convert_to_sax_layout.py` (ACDC / M&Ms):

1. **Only job is picking 12 of 30 frames.** `tools/convert_miitt_to_nifti.py` already wrote one
   3D NIfTI per gated cardiac phase in exactly this layout, so the image conversion is a
   selection + rename, not a resample. Same rule as the other sources — reused verbatim from
   `convert_to_sax_layout.pick_frames`: `native_idx(j) = (ed + round(j*T/12)) % T`, nearest
   native frame, NO temporal interpolation (`V_gt` is the supervision target, and blended
   targets teach blur). ED therefore lands exactly on output frame 0.

2. **No re-framing.** MIITT is already stamped with a clean axis-aligned LPS affine and the true
   protocol spacing (1.5 x 1.5 in-plane, dz = 8 mm thickness + 2 mm gap = 10 mm), so there is no
   permute/flip to plan. That is ASSERTED here rather than assumed — if a future MIITT drop is
   not LPS-diagonal this raises instead of silently mis-orienting the heart (a mis-oriented
   heart still looks like a heart and degrades the anatomical prior with no crash).

3. **Canonical heart siblings come for free.** MIITT already ships the persisted native-space
   nnU-Net (Task114) `heart_seg.nii.gz` (X,Y,Z,30) on the SAME affine as the images, so
   `assemble_whs.build_canonical_siblings` rebuilds `heart_seg_canonical` / `heart_roi_canonical`
   on the (256,256,D) native-z grid with no GPU and no nnU-Net rerun. These are REQUIRED, not
   optional: `loss.volume.heart_weight > 0` raises when a sample has no `heart_roi_canonical`
   (training/loss.py), and every shipped pooled config sets it.

ED/ES come from `scratch/data/whs/cardiac_phase.csv` (`dataset == "miitt"` rows, native 30-frame
indices) — never assumed 0: 10 of the 13 MIITT subjects have ED = 28 or 29, and the `% T` wrap in
`pick_frames` is what makes that work.

⚠️ Slice order: verify apex-at-z0 per subject with `tools/render_slice_order_check.py` AFTER
running this (the source frame 0 is not ED, so the check is only valid on the converted tree).
This converter deliberately does NOT flip — a wrong ordering must be fixed deliberately, per
subject, never by a per-source rule (docs/56, docs/58 sec 10a).

Usage:
    python tools/convert_miitt_to_12phase.py                 # dry run, all 13 subjects
    python tools/convert_miitt_to_12phase.py --apply
    python tools/convert_miitt_to_12phase.py --subjects Volunteer1,Volunteer2 --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np

ROOT = "/home/minsukc/vggt"
DATA = os.path.join(ROOT, "scratch/data")
SRC_ROOT = os.path.join(DATA, "MIITT", "nifti")
OUT_ROOT = os.path.join(DATA, "MIITT_sax")
CARDIAC_CSV = os.path.join(DATA, "whs", "cardiac_phase.csv")
ID_PREFIX = "MIITT_"

sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tools", "nnunet_mnms_eval"))
from convert_to_sax_layout import NUM_PHASES, pick_frames                    # noqa: E402
from assemble_whs import LOW_LABELED, build_canonical_siblings, build_roi    # noqa: E402

# Expected native geometry (tools/convert_miitt_to_nifti.py, real values from J. Hamilton).
EXPECTED_ZOOMS = (1.5, 1.5, 10.0)
ZOOM_TOL = 1e-3

WHS_ROWS = os.path.join(DATA, "whs", "rows")   # per-unit whs_manifest rows (assemble_whs convention)

# Cohort labels. `group` uses the SAME vocabulary as the other converted sources, because
# tools/build_manifest.py maps it with `pathology_label = "healthy" if group == "NOR" else
# "diseased"` — a novel string here would silently label a healthy volunteer diseased.
# vendor/centre: MIITT ships no scanner metadata (not in convert_miitt_to_nifti.py, no README),
# so vendor stays EMPTY rather than guessed — these subjects simply drop out of vendor-stratified
# analyses instead of polluting them.
GROUPS = {
    "Patient_2023Sep23_ARVC": "ARV",
    "Patient_2024Feb08_HCM": "HCM",
    "Patient_2024Jan04_Cardiomyopathy_AFib": "Other",   # subtype not specified by the source
}
DEFAULT_GROUP = "NOR"           # the 10 volunteers
VENDOR = ""
CENTRE = "UMich"


def read_ed_es():
    """{subject: (T_native, ed, es)} for the gated MIITT units."""
    out = {}
    with open(CARDIAC_CSV) as f:
        for row in csv.DictReader(f):
            if row["dataset"] == "miitt":
                out[row["subject"]] = (int(row["T"]), int(row["ED"]), int(row["ES"]))
    if not out:
        raise SystemExit(f"no dataset=='miitt' rows in {CARDIAC_CSV}")
    return out


def check_geometry(subj, im):
    """Assert the source really is LPS + axis-aligned + protocol spacing. Raises, never warns."""
    ax = "".join(nib.aff2axcodes(im.affine))
    if ax != "LPS":
        raise ValueError(f"{subj}: axcodes {ax} != LPS — reorient the ARRAY (never the affine)")
    A = im.affine[:3, :3]
    if not np.allclose(A, np.diag(np.diag(A)), atol=1e-6):
        raise ValueError(f"{subj}: affine is not axis-aligned:\n{A}")
    zooms = tuple(float(z) for z in im.header.get_zooms()[:3])
    if not np.allclose(zooms, EXPECTED_ZOOMS, atol=ZOOM_TOL):
        raise ValueError(f"{subj}: zooms {zooms} != expected {EXPECTED_ZOOMS}")
    if not zooms[2] > max(zooms[0], zooms[1]):
        raise ValueError(f"{subj}: zooms={zooms} — axis 2 is not the slice axis")
    return zooms


def convert_one(subj, T_csv, ed, es, apply, in_mm=6.0):
    out_id = f"{ID_PREFIX}{subj}"
    src_sax = os.path.join(SRC_ROOT, subj, "gated", "sax")
    src_rec = os.path.join(src_sax, "3d_recon")
    seg_f = os.path.join(src_sax, "heart_seg.nii.gz")
    roi_f = os.path.join(src_sax, "heart_roi.nii.gz")
    for p in (src_rec, seg_f, roi_f):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{subj}: missing {p}")

    seg_im = nib.load(seg_f)
    seg4d = np.asarray(seg_im.dataobj).astype(np.uint8)              # (X, Y, Z, T)
    if seg4d.ndim != 4:
        raise ValueError(f"{subj}: heart_seg is {seg4d.shape}, expected 4D")
    T = seg4d.shape[3]
    if T != T_csv:
        raise ValueError(f"{subj}: heart_seg T={T} but cardiac_phase.csv says T={T_csv}")
    if not (0 <= ed < T and 0 <= es < T):
        raise ValueError(f"{subj}: ED={ed}/ES={es} out of range for T={T}")

    frames = sorted(os.listdir(src_rec))
    if len(frames) != T:
        raise ValueError(f"{subj}: {len(frames)} frame files but heart_seg T={T}")

    idx = pick_frames(T, ed)
    # ES on the 12-frame grid: nearest sampled j to ES's fractional position. Advisory only —
    # `cardiac_phase.csv` is what the EF sweep actually reads.
    es_j = int(round(((es - ed) % T) / float(T) * NUM_PHASES)) % NUM_PHASES

    src_frames = [os.path.join(src_rec, f"sax_frame_{t:02d}.nii.gz") for t in idx]
    im0 = nib.load(src_frames[0])
    zooms = check_geometry(subj, im0)
    # The seg must sit on the SAME grid as the images — build_canonical_siblings resamples the
    # seg with the image's canonical transform, so a mismatch here would silently misalign the
    # ROI against the volume it is supposed to weight.
    if not np.allclose(seg_im.affine, im0.affine, atol=1e-6) or seg4d.shape[:3] != im0.shape[:3]:
        raise ValueError(f"{subj}: heart_seg grid {seg4d.shape[:3]} / affine does not match the image")

    meta = {
        "id": out_id,
        "source_file": os.path.relpath(src_rec, DATA),
        "native_T": T, "ed_native": ed, "es_native": es,
        "frame_indices_native": idx, "es_frame_on_12grid_advisory": es_j,
        "native_zooms_xyz": list(zooms), "out_spacing_xyz": list(zooms),
        "out_shape_xyz": list(im0.shape[:3]), "pitch_mm": zooms[2],
        "reframe": {"src_axcodes": "LPS", "swapped_inplane": False, "flips": [False, False, False],
                    "note": "MIITT ships a clean axis-aligned LPS affine; no re-framing needed"},
        "converter": os.path.basename(__file__), "num_phases": NUM_PHASES,
        "arm": "gated", "roi_in_mm": in_mm,
        "group": GROUPS.get(subj, DEFAULT_GROUP), "vendor": VENDOR, "centre": CENTRE,
    }
    if not apply:
        return meta, None

    sax = os.path.join(OUT_ROOT, out_id, "sax")
    rec = os.path.join(sax, "3d_recon")
    os.makedirs(rec, exist_ok=True)
    for j, src in enumerate(src_frames):
        # Byte copy: the source frame already has the exact affine/dtype/shape we want, so
        # re-encoding through nibabel could only lose something.
        shutil.copy2(src, os.path.join(rec, f"sax_frame_{j:02d}.nii.gz"))

    seg12 = np.ascontiguousarray(seg4d[..., idx])
    nib.save(nib.Nifti1Image(seg12, seg_im.affine), os.path.join(sax, "heart_seg.nii.gz"))

    # Native-space ROI: REBUILT from the 12 selected frames, not copied from the source (whose
    # union is over all 30). Same call assemble_whs.py makes for every other gated unit, so
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
    column — exactly like the cmrx/acdc_sax/mnms_sax units. That matters: `ef_eval.compute_ef_metrics`
    silently SKIPS any subject whose seg_flag is not "ok", so a missing row would leave these
    subjects contributing nothing to the EF metric while looking fine everywhere else.
    """
    X, Y, Z, T = seg12.shape
    n_lab = [int((seg12[..., t] > 0).any(axis=(0, 1)).sum()) for t in range(T)]
    mean_lab = float(np.mean(n_lab))
    lv, myo, rv = (int((seg12 == c).sum()) for c in (1, 2, 3))
    flag = "low" if mean_lab < LOW_LABELED else "ok"
    row = [out_id, "miitt_sax", "gated", out_id, f"{X}x{Y}x{Z}x{T}",
           "x".join(f"{s:.2f}" for s in zooms), str(T), f"{mean_lab:.2f}",
           str(lv), str(myo), str(rv), str(roi_vox), flag]
    os.makedirs(WHS_ROWS, exist_ok=True)
    with open(os.path.join(WHS_ROWS, out_id + ".csv"), "w") as f:
        f.write(",".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="", help="comma-separated source subject names "
                                                  "(default: every dataset=='miitt' row)")
    ap.add_argument("--in_mm", type=float, default=6.0, help="ROI dilation (mm), matches assemble_whs")
    ap.add_argument("--apply", action="store_true", help="write; otherwise dry-run")
    args = ap.parse_args()

    ed_es = read_ed_es()
    wanted = [s.strip() for s in args.subjects.split(",") if s.strip()] or sorted(ed_es)
    missing = [s for s in wanted if s not in ed_es]
    if missing:
        raise SystemExit(f"no cardiac_phase.csv row for: {missing}")

    n_ok = n_fail = 0
    for subj in wanted:
        T, ed, es = ed_es[subj]
        try:
            meta, sax = convert_one(subj, T, ed, es, args.apply, in_mm=args.in_mm)
        except Exception as e:  # noqa: BLE001 — one bad subject must not hide the other 12
            print(f"  FAIL {subj}: {type(e).__name__}: {e}")
            n_fail += 1
            continue
        n_ok += 1
        print(f"  {meta['id']:<40s} T={meta['native_T']} ED={meta['ed_native']:>2d} "
              f"ES={meta['es_native']:>2d}->j{meta['es_frame_on_12grid_advisory']:<2d} "
              f"shape={meta['out_shape_xyz']} idx={meta['frame_indices_native']}"
              + (f"  canon={meta['canonical_shape_xyz']} roi={meta['roi_voxels']}" if sax else ""))
    print(f"{'wrote' if args.apply else 'dry-run'}: ok={n_ok} fail={n_fail} -> {OUT_ROOT}")
    if not args.apply:
        print("(dry run — pass --apply to write)")


if __name__ == "__main__":
    main()
