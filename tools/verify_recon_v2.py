"""Full-cohort verification of the v2 CMRxRecon recons (2023 / 2024 / 2025).

Every subject is FULLY decompressed and read -- there is no sampling and no
header-only shortcut -- so this catches truncated gzip streams, all-zero slices,
NaNs, and 3d/4d disagreement, not just the metadata.

Per subject it checks:

  STRUCTURE  4d_recon.nii.gz present; exactly 12 frames sax_frame_00..11;
             cine_sax_info.csv present; no leftover *.relabeltmp* files.
  DATA       4d loads fully (=> gzip stream intact), dtype float32, ndim 4, T==12,
             all values finite, and EVERY (z, t) plane has some nonzero signal
             (this is the check that would have caught the all-zero-plane recon).
  CONSISTENCY  each 3d frame k loads fully, shape/dtype/affine match the 4d, and
             its voxels equal 4d[..., k].
  GEOMETRY   axcodes == LPS; slice-axis spacing == the expected pitch; in-plane
             spacing == FOV / ReconMatrix from the subject's own cine_sax_info.csv;
             grid shape == (ReconMatrix_X, ReconMatrix_Y, SliceNum); affine finite
             and non-singular; sform/qform codes set.

Expected slice pitch comes from an authoritative per-year source, never a guess:
  2023 -> SUBJECT_MANIFEST.csv  pitch_mm   (12.0, or 10.0 for the 6 mm subjects)
  2024 -> 12.0 for every subject
  2025 -> recon_report.json     pitch_mm   (12.0 / 10.0; `pitch_provisional`
          subjects are still checked, but counted separately in the summary
          because their expected value is itself an assumption)

WHAT THIS DOES NOT PROVE: that the source CSV metadata is itself correct (the open
Philips FOVy question), that the 2025 provisional pitches are right, or that the
reconstruction is scientifically better than v1 -- only that what is on disk is
internally complete, uncorrupted, and consistent with its stated metadata.

Usage:
    micromamba run -n svr python -u tools/verify_recon_v2.py --workers 4 \
        --out scratch/data/recon_v2_verification_full.json
    # smoke first:
    micromamba run -n svr python -u tools/verify_recon_v2.py --limit 3
"""
import argparse, csv, glob, json, os, sys, collections
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import nibabel as nib

ROOT = "scratch/data"
INPLANE_MAX_MM = 4.0          # any axis whose column norm exceeds this is the slice axis
N_FRAMES = 12


def slice_axis(affine):
    norms = [float(np.linalg.norm(affine[:3, i])) for i in range(3)]
    big = [i for i, n in enumerate(norms) if n > INPLANE_MAX_MM]
    if len(big) != 1:
        raise ValueError(f"ambiguous slice axis, column norms {norms}")
    return big[0], norms


def read_info_csv(path):
    out = {}
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) >= 2:
                out[row[0].strip()] = row[1].strip()
    return out


def check_subject(task):
    """One subject, fully read. Returns (subject, list_of_problems, info_dict)."""
    year, subj, sax, expected_pitch, pitch_provisional, v1_sax, expected_inplane = task
    problems = []
    info = {"subject": subj, "year": year, "expected_pitch": expected_pitch,
            "pitch_provisional": pitch_provisional}

    def bad(check, msg):
        problems.append({"subject": subj, "year": year, "check": check, "detail": msg})

    try:
        # ---------- STRUCTURE ----------
        p4 = os.path.join(sax, "4d_recon.nii.gz")
        frames = [os.path.join(sax, "3d_recon", f"sax_frame_{k:02d}.nii.gz") for k in range(N_FRAMES)]
        missing = [os.path.basename(p) for p in [p4] + frames if not os.path.exists(p)]
        if missing:
            bad("structure", f"missing files: {missing}")
            return subj, problems, info
        empty = [os.path.basename(p) for p in [p4] + frames if os.path.getsize(p) == 0]
        if empty:
            bad("structure", f"zero-byte files: {empty}")
            return subj, problems, info
        extra = sorted(set(glob.glob(os.path.join(sax, "3d_recon", "*.nii.gz"))) - set(frames))
        if extra:
            bad("structure", f"unexpected files in 3d_recon: {[os.path.basename(e) for e in extra]}")
        stray = glob.glob(os.path.join(sax, "**", "*relabeltmp*"), recursive=True)
        if stray:
            bad("structure", f"leftover relabel tmp files: {[os.path.basename(s) for s in stray]}")
        csv_path = os.path.join(sax, "cine_sax_info.csv")
        if not os.path.exists(csv_path):
            bad("structure", "cine_sax_info.csv missing")

        info["sizes_mb"] = round(sum(os.path.getsize(p) for p in [p4] + frames) / 1e6, 2)
        info["mtime_4d"] = os.path.getmtime(p4)
        info["mtime_min"] = min(os.path.getmtime(p) for p in [p4] + frames)

        # ---------- 4D DATA (full decompress) ----------
        img4 = nib.load(p4)
        A4 = np.asarray(img4.affine, dtype=np.float64)
        vol4 = np.asanyarray(img4.dataobj)
        info["shape_4d"] = tuple(int(x) for x in vol4.shape)
        info["dtype_4d"] = str(img4.get_data_dtype())

        if img4.get_data_dtype() != np.dtype(np.float32):
            bad("dtype", f"4d dtype {img4.get_data_dtype()}, expected float32")
        if vol4.ndim != 4:
            bad("shape", f"4d ndim {vol4.ndim}, expected 4")
            return subj, problems, info
        if vol4.shape[3] != N_FRAMES:
            bad("shape", f"4d has {vol4.shape[3]} phases, expected {N_FRAMES}")
        if not np.isfinite(vol4).all():
            n_nan = int(np.isnan(vol4).sum()); n_inf = int(np.isinf(vol4).sum())
            bad("finite", f"4d has {n_nan} NaN and {n_inf} Inf voxels")

        # every (z, t) plane must carry signal -- the all-zero-slice failure mode
        plane_max = np.abs(vol4).max(axis=(0, 1))                     # (Z, T)
        dead = np.argwhere(plane_max <= 0)
        if dead.size:
            bad("zero_plane", f"{len(dead)} all-zero (z,t) planes, first 10: "
                              f"{[(int(z), int(t)) for z, t in dead[:10]]}")
        info["n_zero_planes"] = int(dead.shape[0])

        # adjacent cardiac phases must actually differ -- two identical neighbours mean the
        # temporal resample duplicated a frame instead of interpolating it
        if vol4.shape[3] == N_FRAMES:
            dup = [k for k in range(N_FRAMES)
                   if np.array_equal(vol4[..., k], vol4[..., (k + 1) % N_FRAMES])]
            if dup:
                bad("temporal", f"phases identical to their successor at k={dup} "
                                f"(temporal resample duplicated frames)")
            info["n_duplicate_phases"] = len(dup)
        info["intensity"] = {"min": float(vol4.min()), "max": float(vol4.max()),
                             "mean": float(vol4.mean()),
                             "nonzero_frac": round(float((vol4 != 0).mean()), 4)}

        # ---------- 3D FRAMES: full read + equality against the 4d ----------
        max_abs_diff = 0.0
        for k, fp in enumerate(frames):
            img3 = nib.load(fp)
            vol3 = np.asanyarray(img3.dataobj)
            if tuple(vol3.shape) != tuple(vol4.shape[:3]):
                bad("consistency", f"frame {k:02d} shape {vol3.shape} != 4d spatial {vol4.shape[:3]}")
                continue
            if img3.get_data_dtype() != np.dtype(np.float32):
                bad("dtype", f"frame {k:02d} dtype {img3.get_data_dtype()}, expected float32")
            if not np.allclose(np.asarray(img3.affine, dtype=np.float64), A4, atol=1e-4, rtol=0):
                bad("consistency", f"frame {k:02d} affine differs from the 4d affine")
            d = float(np.max(np.abs(vol3.astype(np.float64) - vol4[..., k].astype(np.float64))))
            max_abs_diff = max(max_abs_diff, d)
        info["max_abs_diff_3d_vs_4d"] = max_abs_diff
        scale = max(abs(info["intensity"]["max"]), 1e-12)
        if max_abs_diff > 1e-5 * scale:
            bad("consistency", f"3d frames disagree with the 4d volume, max|diff|={max_abs_diff:.6g} "
                               f"(volume max {scale:.6g})")

        # ---------- GEOMETRY ----------
        if not np.isfinite(A4).all():
            bad("affine", "4d affine contains non-finite entries")
        elif abs(float(np.linalg.det(A4[:3, :3]))) < 1e-9:
            bad("affine", "4d affine direction matrix is singular")
        axcodes = "".join(nib.aff2axcodes(A4))
        info["axcodes"] = axcodes
        if axcodes != "LPS":
            bad("orientation", f"axcodes {axcodes}, expected LPS")
        sform_code = int(img4.header["sform_code"]); qform_code = int(img4.header["qform_code"])
        info["sform_code"], info["qform_code"] = sform_code, qform_code
        if sform_code == 0 and qform_code == 0:
            bad("affine", "both sform_code and qform_code are 0 (no valid spatial frame)")

        ax, norms = slice_axis(A4)
        pitch = norms[ax]
        inplane = sorted(norms[i] for i in range(3) if i != ax)
        info["pitch_mm"] = round(pitch, 4)
        info["inplane_mm"] = [round(v, 4) for v in inplane]
        if expected_pitch is None:
            bad("pitch", "no authoritative expected pitch for this subject")
        elif abs(pitch - expected_pitch) > 1e-3:
            bad("pitch", f"slice spacing {pitch:.4f} mm, expected {expected_pitch:.4f} mm")

        if os.path.exists(csv_path):
            meta = read_info_csv(csv_path)
            info["thickness_csv"] = meta.get("SliceThickness", "")
            try:
                rx, ry = int(float(meta["ReconMatrix_X"])), int(float(meta["ReconMatrix_Y"]))
                fx, fy = float(meta["FOVx"]), float(meta["FOVy"])
                # `FOVx / ReconMatrix_X` is the pixel size ONLY when ReconMatrix_X happens to equal
                # the acquired readout base (nx/2). True for every Siemens subject, FALSE for the
                # 2025 Philips (nx=304, rx=256) -- see docs/55. So where the recon records its own
                # derived pixel size (2025 `recon_report.json`), trust that; the CSV rule is a
                # fallback for 2023/2024, which are all Siemens.
                if expected_inplane is not None:
                    exp_inplane = sorted(expected_inplane)
                    src = "recon_report.pixel_mm"
                else:
                    exp_inplane = sorted([fx / rx, fy / ry])
                    src = "FOV/ReconMatrix"
                info["inplane_expected_mm"] = [round(v, 4) for v in exp_inplane]
                if not np.allclose(inplane, exp_inplane, rtol=3e-3, atol=0):
                    bad("inplane", f"in-plane spacing {inplane} != {src} {exp_inplane}")
                if (int(vol4.shape[0]), int(vol4.shape[1])) != (rx, ry):
                    bad("grid", f"in-plane grid {vol4.shape[:2]} != ReconMatrix ({rx}, {ry})")
            except (KeyError, ValueError, ZeroDivisionError) as e:
                bad("inplane", f"cannot derive expected in-plane spacing: {e}")
            try:
                nz = int(float(meta["SliceNum"]))
                if int(vol4.shape[2]) != nz:
                    bad("grid", f"{vol4.shape[2]} slices on disk != SliceNum {nz} in csv")
            except (KeyError, ValueError):
                bad("grid", "SliceNum missing from cine_sax_info.csv")

        # ---------- PROVENANCE: this volume must NOT be the archived v1 ----------
        # Every other check passes happily on a stale v1 leftover -- it is a perfectly
        # valid NIfTI. Bit-identity with the v1 archive is unambiguous proof that the
        # subject was never re-reconstructed with the fixed ESPIRiT.
        if v1_sax:
            p1 = os.path.join(v1_sax, "4d_recon.nii.gz")
            if not os.path.exists(p1):
                info["v1_compare"] = "absent"
            else:
                v1 = np.asanyarray(nib.load(p1).dataobj)
                if tuple(v1.shape) != tuple(vol4.shape):
                    bad("provenance", f"v1 shape {v1.shape} != v2 shape {vol4.shape}")
                else:
                    d = float(np.max(np.abs(v1.astype(np.float64) - vol4.astype(np.float64))))
                    denom = max(float(np.abs(v1).max()), 1e-12)
                    info["v1_compare"] = "identical" if d == 0.0 else "differs"
                    info["v1_rel_maxdiff"] = d / denom
                    if d == 0.0:
                        bad("provenance", "4d is BIT-IDENTICAL to the archived v1 volume -- "
                                          "this subject was not re-reconstructed")

    except Exception as e:
        bad("exception", f"{type(e).__name__}: {e}")
    return subj, problems, info


def expected_pitch_map(year, subjects):
    """subject_id -> (pitch_mm, provisional_flag). Authoritative per-year source.

    Returns a PLAIN dict on purpose: a defaultdict would look populated but silently
    yield None through `.get(s, default)` and `s not in map`, turning the pitch check
    into a vacuous "no expected pitch" instead of a real comparison.
    """
    if year == "2024":
        return {s: (12.0, False) for s in subjects}   # 2024 is 12.0 mm for every subject
    if year == "2023":
        path = os.path.join(ROOT, "CMRxRecon2023", "SUBJECT_MANIFEST.csv")
        with open(path) as fh:
            return {r["combined_id"]: (float(r["pitch_mm"]), False)
                    for r in csv.DictReader(fh) if r["reconstruct"] == "1"}
    if year == "2025":
        path = os.path.join(ROOT, "CMRxRecon2025", "recon_report.json")
        with open(path) as fh:
            return {r["cid"]: (float(r["pitch_mm"]), bool(r.get("pitch_provisional", False)))
                    for r in json.load(fh)}
    raise ValueError(year)


# 2025 was 360 until 2026-07-27, when the one confirmed duplicate
# (CMRx25_R2val_Center004_UIH_15T_umr680_P006) was moved to _archive/ -- see that year's
# DUPLICATES.txt. 2023/2024 were already deduplicated before their volumes were built.
EXPECTED_COUNTS = {"2023": 196, "2024": 294, "2025": 359}
PREFIX = {"2023": "CMRx23_", "2024": "CMRx24_", "2025": "CMRx25_"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", default=["2023", "2024", "2025"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: N subjects per year")
    ap.add_argument("--compare-v1", action="store_true",
                    help="also assert each 4d differs from the archived v1 volume "
                         "(proves the subject was actually re-reconstructed; doubles the reads)")
    ap.add_argument("--out", default="scratch/data/recon_v2_verification_full.json")
    args = ap.parse_args()

    report, all_problems = {}, []
    for year in args.years:
        base = os.path.join(ROOT, f"CMRxRecon{year}", "Cine_combined")
        dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        pitches = expected_pitch_map(year, dirs)

        cohort_problems = []
        bad_prefix = [d for d in dirs if not d.startswith(PREFIX[year])]
        if bad_prefix:
            cohort_problems.append({"year": year, "check": "cohort",
                                    "detail": f"{len(bad_prefix)} dirs without the {PREFIX[year]} "
                                              f"prefix: {bad_prefix[:10]}"})
        no_recon = [d for d in dirs
                    if not os.path.exists(os.path.join(base, d, "sax", "4d_recon.nii.gz"))]
        if no_recon:
            cohort_problems.append({"year": year, "check": "cohort",
                                    "detail": f"{len(no_recon)} dirs with no 4d_recon: {no_recon[:10]}"})
        subjects = [d for d in dirs if d not in set(no_recon)]
        if not args.limit and len(subjects) != EXPECTED_COUNTS[year]:
            cohort_problems.append({"year": year, "check": "cohort",
                                    "detail": f"{len(subjects)} reconstructed subjects, "
                                              f"expected {EXPECTED_COUNTS[year]}"})
        no_pitch = [s for s in subjects if s not in pitches]
        if no_pitch:
            cohort_problems.append({"year": year, "check": "cohort",
                                    "detail": f"{len(no_pitch)} subjects absent from the pitch "
                                              f"source: {no_pitch[:10]}"})
        if args.limit:
            subjects = subjects[:args.limit]

        v1_base = os.path.join(ROOT, f"CMRxRecon{year}_recon_v1_espirit_imagedomain")
        use_v1 = args.compare_v1 and os.path.isdir(v1_base)
        if args.compare_v1 and not use_v1:
            cohort_problems.append({"year": year, "check": "cohort",
                                    "detail": f"v1 archive {v1_base} not found"})
        # 2025 records its own derived pixel size per subject; 2023/2024 do not (CSV rule applies).
        inplane_map = {}
        if year == "2025":
            with open(os.path.join(ROOT, "CMRxRecon2025", "recon_report.json")) as fh:
                inplane_map = {r["cid"]: [float(v) for v in r["pixel_mm"]] for r in json.load(fh)
                               if r.get("pixel_mm")}
        tasks = [(year, s, os.path.join(base, s, "sax"),
                  pitches.get(s, (None, False))[0], pitches.get(s, (None, False))[1],
                  os.path.join(v1_base, s, "sax") if use_v1 else None,
                  inplane_map.get(s))
                 for s in subjects]
        print(f"[{year}] {len(tasks)} subjects, full read of {len(tasks) * 13} files", flush=True)

        problems, infos = list(cohort_problems), []
        with ProcessPoolExecutor(args.workers) as ex:
            for i, (subj, probs, info) in enumerate(ex.map(check_subject, tasks, chunksize=1), 1):
                problems.extend(probs)
                infos.append(info)
                if i % 25 == 0:
                    print(f"  [{year}] {i}/{len(tasks)}  problems={len(problems)}", flush=True)

        pitch_hist = collections.Counter(i.get("pitch_mm") for i in infos)
        ax_hist = collections.Counter(i.get("axcodes") for i in infos)
        shape_hist = collections.Counter(str(i.get("shape_4d")) for i in infos)
        report[year] = {
            "n_subjects": len(infos),
            "n_files_read": len(infos) * 13,
            "n_problems": len(problems),
            "pitch_histogram": {str(k): v for k, v in sorted(pitch_hist.items(), key=lambda x: str(x[0]))},
            "n_pitch_provisional": sum(1 for i in infos if i.get("pitch_provisional")),
            "axcodes_histogram": dict(ax_hist),
            "n_distinct_shapes": len(shape_hist),
            "total_gb": round(sum(i.get("sizes_mb", 0) for i in infos) / 1000, 2),
            "n_zero_planes_total": sum(i.get("n_zero_planes", 0) for i in infos),
            "n_subjects_with_duplicate_phases": sum(1 for i in infos if i.get("n_duplicate_phases")),
            "v1_compare": dict(collections.Counter(i.get("v1_compare", "not_checked") for i in infos)),
            "v1_rel_maxdiff_min": min([i["v1_rel_maxdiff"] for i in infos
                                       if "v1_rel_maxdiff" in i] or [None]),
            "max_abs_diff_3d_vs_4d": max([i.get("max_abs_diff_3d_vs_4d", 0.0) for i in infos] or [0.0]),
            "oldest_mtime": min([i.get("mtime_min", 0) for i in infos] or [0]),
            "problems": problems[:300],
            "subjects": infos,
        }
        all_problems.extend(problems)
        print(f"[{year}] DONE  problems={len(problems)}  pitch={dict(pitch_hist)}  "
              f"axcodes={dict(ax_hist)}  zero_planes={report[year]['n_zero_planes_total']}", flush=True)
        for p in problems[:15]:
            print("    ", p, flush=True)

    report["TOTAL_PROBLEMS"] = len(all_problems)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nTOTAL PROBLEMS: {len(all_problems)}  ->  {args.out}")
    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
