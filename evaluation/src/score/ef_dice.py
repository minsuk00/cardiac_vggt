"""Cohort-agnostic biventricular metrics for the 1-frame recons, method-matched via nnU-Net Task114.

Segments the recon (clean & breath arms) AND the GT with the SAME segmenter, so pred-vs-GT
isolates recon quality (pseudo-truth, docs/17/24 method). Metrics per subject:
  EF (LV + RV), EDV/ESV (mL), LVM (g, MYO x 1.05 g/mL at ED), Dice + HD95 (mm) for LV/MYO/RV
  at the GT's ED/ES phases. EF is a ratio -> voxel-volume cancels (12mm-pitch caveat docs/39
  doesn't bite); EDV/ESV/LVM use the seg's own voxel volume, so they DO carry that caveat —
  method-matched vs GT, they remain fair comparisons.

Inputs come from the SCORED cines (`<arm>/cine_{clean,breath}.nii.gz` + `<subj>/cine_gt.nii.gz`,
written by score/image_metrics.py) — NOT the raw recon volumes — so segmentation sees exactly the
gauged/pose-corrected/PSF'd volume the image metrics scored, and Dice inherits the registration.

Two steps around one nnU-Net call:
  dump  <input_dir>  : slice each 4D cine into per-phase _0000.nii.gz named by INDEX (subject
                       names contain '__'); writes ef_manifest.json (records the method).
  score <seg_dir>    : read Task114 segs -> per-subject metrics; aggregate per cohort.

Full chain (all git-tracked; nnU-Net runs in the isolated `nnunet` env, wrapped by run_seg.sh):
  python evaluation/src/score/ef_dice.py dump  <input_dir> --method <m> --cohorts miitt ocmr acdc
  bash   evaluation/src/engine/run_seg.sh         <input_dir> <seg_dir>      # nnU-Net Task114 2d
  python evaluation/src/score/ef_dice.py score <seg_dir> --input <input_dir>
                                       # -> metric_results/<split>/_ef/<m>.json (per-cohort merge on re-runs)
  python evaluation/src/score/ef_dice.py plot  metric_results/<split>/_ef/<m>.json --out <ef.png>
Then re-run score/aggregate.py (or run.py) to fold the _ef file into metric_results/<split>/<ds>/<m>.json.
<split> = $SPLIT (default val) at dump time, recorded in the manifest so score writes the same split.
"""
import argparse, glob, json, os, sys, uuid
import numpy as np
import nibabel as nib

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import paths  # noqa: E402

LV, MYO, RV = 1, 2, 3


def method_dir(cohort, subj, method):
    """The recon method dir — EXACT name only. (It used to probe a legacy `_contz` suffix,
    which made the `_ef/<method>.json` key diverge from the arm name aggregate.py joins on —
    the EF block then silently never folded. A contz arm must be dumped under its full
    literal dir name.)"""
    d = paths.arm_dir(cohort, subj, method)
    return str(d) if d.is_dir() else None


def subjects(cohort, method):
    # Same split rail as run_vggt/aggregate: a stray train/test bundle in out/ must not join
    # the citable EF cohort (paths.filter_by_split contract).
    split = os.environ.get("SPLIT", "val")
    keep, dropped = paths.filter_by_split(cohort, paths.subjects(cohort), split)
    for subj, why in dropped:
        print(f"  !! {cohort}: skipping {subj}: {why}", file=sys.stderr)
    return [s for s in keep if method_dir(cohort, s, method)]


def _json_field(path, key):
    """One field of a JSON file; None if the file is missing/unreadable or lacks the key."""
    try:
        return json.load(open(path)).get(key)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def _read_text(path):
    try:
        return open(path).read().strip()
    except OSError:
        return None


def _dump_cine(cine_path, input_dir, cohort, sidx, arm, roi):
    """Slice one 4D cine (X,Y,Z,T) into per-phase _0000.nii.gz for nnU-Net, multiplied by the
    heart ROI `roi` (X,Y,Z bool). Returns T or 0.

    Every volume — GT, VGGT, and the classical baselines — is cropped to the SAME heart ROI
    before segmentation. The baselines reconstruct only inside that ROI (their published
    protocol), and nnU-Net is measurably worse on a black-exterior crop than on a full FOV
    (ACDC vs expert labels: LV/MYO/RV ED Dice 0.952/0.871/0.918 -> 0.936/0.818/0.868, with
    tail failures) — so segmenting GT/VGGT full-FOV would hand them a segmenter advantage
    unrelated to reconstruction. Cropping all arms identically makes the seg metrics fair;
    absolute Dice/EF are therefore lower than full-FOV literature numbers for every method."""
    if not os.path.isfile(cine_path):
        return 0
    img = nib.load(str(cine_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if roi.shape != vol.shape[:3]:
        sys.exit(f"{cine_path}: heart ROI {roi.shape} does not match cine {vol.shape[:3]}")
    for t in range(vol.shape[3]):
        nib.save(nib.Nifti1Image(vol[..., t] * roi, img.affine),
                 f"{input_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}_0000.nii.gz")
    return vol.shape[3]


def _heart_roi(cohort, subj):
    """The SEGMENTATION crop = the +10 mm padded heart ROI (docs/104 §4 / docs/107), not the
    tight image-metric ROI: the tight crop inflates nnU-Net's ES LV read on reconstructions."""
    p = paths.heart_mask_pad(cohort, subj)
    if not os.path.isfile(p):
        sys.exit(f"{cohort}/{subj}: no {paths.HEART_MASK_PAD} — seg metrics are ROI-cropped for every "
                 f"arm and cannot be computed without it (tools/build_padded_heart_mask.py)")
    return np.asarray(nib.load(str(p)).dataobj) > 0.5


def _roi_sha(cohort, subj):
    return paths.file_sha256(paths.heart_mask_pad(cohort, subj))


# Which nnU-Net configuration this process scores with. MUST match run_seg.sh's SEG_CFG default,
# and score() refuses a seg_dir stamped with anything else.
SEG_CONFIG = os.environ.get("SEG_CFG", "3d_fullres")


def _gt_seg_cached(cohort, subj, gt_sha):
    """True when the GT seg cache holds segs of THIS gt bundle, cropped by THIS heart ROI, from
    THIS segmenter config.

    Content-keyed on gt_sha256 AND the ROI's sha (an ROI regenerated under an unchanged GT must
    not reuse segs cropped by the old one) AND seg_config. The segmenter was NOT part of this key
    historically, and that is the trap: flipping run_seg.sh to 3d_fullres would otherwise leave
    every subject with an existing cache "cached", so 3D predictions would be scored against 2D
    GT segs with no error and no warning. A cache written before this field existed is 2d by
    definition, so an absent field reads as "2d" and the pre-existing 2d path stays bit-identical.
    """
    src = paths.seg_gt_dir(cohort, subj, SEG_CONFIG) / "src.json"
    return (_json_field(src, "gt_sha256") == gt_sha
            and _json_field(src, "roi_sha256") == _roi_sha(cohort, subj)
            and (_json_field(src, "seg_config") or "2d") == SEG_CONFIG)


def dump(args):
    # sidx is an ENUMERATION index and the seg filenames carry only it — leftovers from an
    # earlier dump with a different subject set would silently attach to the wrong subject at
    # score time. Refuse a dirty dir instead of trusting the caller to have cleared it.
    if os.path.isdir(args.input_dir) and any(f.endswith(".nii.gz") for f in os.listdir(args.input_dir)):
        sys.exit(f"dump: {args.input_dir} already contains .nii.gz files from an earlier dump — "
                 f"sidx-keyed names would mis-attribute leftovers to the wrong subject. Use a fresh dir.")
    os.makedirs(args.input_dir, exist_ok=True)
    # dump_id ties a seg_dir to THIS dump: run_seg.sh copies it to <seg_dir>/ef_dump_id and score()
    # refuses a seg_dir whose id differs (stale segs from an earlier dump with a different sidx
    # numbering). Content-keyed, not mtime-keyed — see the freshness note below.
    manifest, meta = [], {"method": args.method, "split": os.environ.get("SPLIT", "val"),
                          "dump_id": uuid.uuid4().hex}
    if args.gt_only:
        # Pre-warm <subject>/seg_gt/ for every subject in the split BEFORE any arm exists
        # (GT is method-independent). Writes cine_gt itself (normally image_metrics does) and
        # dumps only the cropped GT; `score --gt-only` then populates the cache.
        import image_metrics
        meta["method"] = "__gt_only__"
        for cohort in args.cohorts:
            split = os.environ.get("SPLIT", "val")
            keep, _ = paths.filter_by_split(cohort, paths.subjects(cohort), split)
            for sidx, subj in enumerate(keep):
                gt_sha = image_metrics.ensure_cine_gt(cohort, subj)
                if _gt_seg_cached(cohort, subj, gt_sha):
                    continue
                T = _dump_cine(paths.cine_gt(cohort, subj), args.input_dir, cohort, sidx,
                               "gt", _heart_roi(cohort, subj))
                manifest.append({"cohort": cohort, "sidx": sidx, "subject": subj, "T": T,
                                 "gt_sha256": gt_sha, "gt_cached": False})
        json.dump({"meta": meta, "subjects": manifest}, open(f"{args.input_dir}/ef_manifest.json", "w"), indent=2)
        print(f"dumped GT for {len(manifest)} uncached subjects -> {args.input_dir}")
        return
    for cohort in args.cohorts:
        for sidx, subj in enumerate(subjects(cohort, args.method)):
            md = method_dir(cohort, subj, args.method)
            # Segment the SCORED volumes, not the raw recons: cine_* carries the exact
            # gauge/pose/PSF treatment image_metrics scored. Missing cine => that arm was
            # never scored — skip it rather than silently falling back to raw recons.
            # Freshness is CONTENT-keyed (paths.gt_sha256 of gt_t00 vs the hash image_metrics
            # recorded when it wrote the cine), never mtime-keyed: GPFS purge-avoidance `touch`es
            # rewrite every mtime. A cine scored BEFORE a bundle rebuild would otherwise be
            # segmented against the rebuilt GT (the T-count guard can't see a same-T rebuild).
            gt_sha = paths.gt_sha256(cohort, subj)
            if _json_field(paths.cine_gt_src(cohort, subj), "gt_sha256") != gt_sha:
                print(f"  !! {cohort}/{subj}: cine_gt.nii.gz is missing or from a different gt bundle "
                      f"(rebuilt since scoring?) — re-run score/image_metrics.py; skipped", file=sys.stderr)
                continue
            roi = _heart_roi(cohort, subj)
            # GT is method-independent: segment it once per gt bundle. score() fills
            # <subject>/seg_gt/ from the first seg_dir that has it; later dumps skip GT and
            # score() copies the cached segs back into the seg_dir under this dump's sidx.
            gt_cached = _gt_seg_cached(cohort, subj, gt_sha)
            if gt_cached:
                T = int(_json_field(paths.seg_gt_dir(cohort, subj, SEG_CONFIG) / "src.json", "T"))
            else:
                T = _dump_cine(paths.cine_gt(cohort, subj), args.input_dir, cohort, sidx, "gt", roi)
            if T == 0:
                print(f"  !! {cohort}/{subj}: no cine_gt.nii.gz — run score/image_metrics.py first; skipped",
                      file=sys.stderr)
                continue
            arm_gt_sha = _json_field(f"{md}/metrics.json", "gt_sha256")
            for arm in ("clean", "breath"):
                cine = f"{md}/cine_{arm}.nii.gz"
                if os.path.isfile(cine) and arm_gt_sha != gt_sha:
                    print(f"  !! {cohort}/{subj} [{arm}]: cine_{arm} was scored against a different gt bundle "
                          f"(rebuilt since scoring?) — re-run image_metrics; arm skipped", file=sys.stderr)
                    continue
                n = _dump_cine(cine, args.input_dir, cohort, sidx, arm, roi)
                if n not in (0, T):
                    sys.exit(f"{cohort}/{subj} [{arm}]: cine has {n} phases but GT has {T} — stale cine?")
            manifest.append({"cohort": cohort, "sidx": sidx, "subject": subj, "T": T,
                             "gt_sha256": gt_sha, "gt_cached": gt_cached})
    json.dump({"meta": meta, "subjects": manifest}, open(f"{args.input_dir}/ef_manifest.json", "w"), indent=2)
    print(f"dumped {len(manifest)} subjects -> {args.input_dir}")


def seg_path(seg_dir, cohort, sidx, arm, t):
    return f"{seg_dir}/{cohort}__s{sidx:03d}__{arm}__t{t:02d}.nii.gz"


def curve(seg_dir, cohort, sidx, arm, T, lab=LV):
    """(per-phase voxel count of `lab`, voxel volume mm^3); (None, None) for missing/empty segs."""
    counts, voxmm3 = [], None
    for t in range(T):
        p = seg_path(seg_dir, cohort, sidx, arm, t)
        if not os.path.isfile(p):
            return None, None
        img = nib.load(p)
        voxmm3 = float(abs(np.linalg.det(img.affine[:3, :3])))
        counts.append(int((np.asarray(img.dataobj) == lab).sum()))
    c = np.array(counts, float)
    return (c, voxmm3) if c.max() > 0 else (None, None)


def ef_of(curve):
    """EF %, or None when the structure VANISHES at some phase (count 0 -> EF would read a
    clinically-impossible 100%; realistic when nnU-Net drops the RV at ES on a blurry recon).
    Callers' `is not None` gating drops the value from MAEs instead of polluting them."""
    if curve.min() <= 0:
        return None
    return float((curve.max() - curve.min()) / curve.max() * 100.0)


def vols_of(curve, voxmm3):
    """(EDV, ESV) in mL from a voxel-count curve. ESV is None when the structure vanishes
    (a 0-mL ESV is a seg failure, not physiology). Carries the voxel-volume caveat (docstring)."""
    edv = float(curve.max() * voxmm3 / 1000.0)
    esv = float(curve.min() * voxmm3 / 1000.0) if curve.min() > 0 else None
    return edv, esv


def lvm_of(seg_dir, cohort, sidx, arm, ed_t, voxmm3):
    """LV mass (g) = MYO voxels at ED x voxel volume x 1.05 g/mL. None if the seg is missing."""
    p = seg_path(seg_dir, cohort, sidx, arm, ed_t)
    if not os.path.isfile(p) or voxmm3 is None:
        return None
    n = int((np.asarray(nib.load(p).dataobj) == MYO).sum())
    return float(n * voxmm3 / 1000.0 * 1.05) if n else None


def dice(seg_dir, cohort, sidx, arm, t, gt_t, lab):
    a = np.asarray(nib.load(seg_path(seg_dir, cohort, sidx, arm, t)).dataobj) == lab
    b = np.asarray(nib.load(seg_path(seg_dir, cohort, sidx, "gt", gt_t)).dataobj) == lab
    inter = np.logical_and(a, b).sum(); s = a.sum() + b.sum()
    return float(2 * inter / s) if s else float("nan")


def hd95(seg_dir, cohort, sidx, arm, t, gt_t, lab):
    """95th-percentile symmetric surface distance (mm) between recon-seg and GT-seg.
    NaN if either mask is empty (no surface to measure)."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    ia = nib.load(seg_path(seg_dir, cohort, sidx, arm, t))
    ib = nib.load(seg_path(seg_dir, cohort, sidx, "gt", gt_t))
    a = np.asarray(ia.dataobj) == lab
    b = np.asarray(ib.dataobj) == lab
    if not a.any() or not b.any():
        return float("nan")
    spacing = ia.header.get_zooms()[:3]
    sa = a & ~binary_erosion(a)                      # surface voxels
    sb = b & ~binary_erosion(b)
    da = distance_transform_edt(~sb, sampling=spacing)[sa]   # a-surface -> b-surface
    db = distance_transform_edt(~sa, sampling=spacing)[sb]
    return float(np.percentile(np.concatenate([da, db]), 95))


def _sync_gt_seg_cache(seg_dir, cohort, sidx, subj, T, m):
    """GT segs flow between <seg_dir> (sidx-named, per dump) and <subject>/seg_gt/ (the cache).
    Cached at dump time -> copy the cache INTO seg_dir so curve/dice/hd95 find them unchanged.
    Not cached -> this seg_dir just segmented GT: populate the cache (src.json written LAST, so
    a partial copy is never mistaken for a valid cache). Legacy manifests without gt_sha256
    are left alone (GT was dumped, nothing to sync)."""
    import shutil
    gt_sha = m.get("gt_sha256")
    if not gt_sha:
        return
    cache = paths.seg_gt_dir(cohort, subj, SEG_CONFIG)
    if m.get("gt_cached"):
        for t in range(T):
            dst = seg_path(seg_dir, cohort, sidx, "gt", t)
            if not os.path.isfile(dst):
                shutil.copyfile(cache / f"seg_t{t:02d}.nii.gz", dst)
        return
    if _gt_seg_cached(cohort, subj, gt_sha):
        return
    srcs = [seg_path(seg_dir, cohort, sidx, "gt", t) for t in range(T)]
    if not all(os.path.isfile(p) for p in srcs):
        return                                   # GT seg incomplete here — score() reports it
    # tmp + os.replace: two arms of one subject scoring in parallel both populate the cache;
    # a reader must never see a half-copied seg (identical bytes, so the last replace is benign).
    os.makedirs(cache, exist_ok=True)
    for t, p in enumerate(srcs):
        tmp = cache / f".seg_t{t:02d}.{os.getpid()}.tmp.nii.gz"
        shutil.copyfile(p, tmp); os.replace(tmp, cache / f"seg_t{t:02d}.nii.gz")
    tmp = cache / f".src.{os.getpid()}.tmp.json"
    json.dump({"gt_sha256": gt_sha, "roi_sha256": _roi_sha(cohort, subj), "T": T,
               "crop": paths.HEART_MASK_PAD, "seg_config": SEG_CONFIG},
              open(tmp, "w")); os.replace(tmp, cache / "src.json")


def score(args):
    # FAIL CLOSED on the segmenter. run_seg.sh stamps the config it predicted with; if this
    # process would score those segs under a different one, every GT-vs-prediction comparison
    # below is cross-segmenter and silently meaningless. An unstamped seg_dir predates the stamp
    # and is 2d by definition.
    stamp = os.path.join(args.seg_dir, "nnunet_config")
    got = open(stamp).read().strip() if os.path.isfile(stamp) else "2d"
    if got != SEG_CONFIG:
        sys.exit(f"score: {args.seg_dir} was segmented with '{got}' but SEG_CFG is "
                 f"'{SEG_CONFIG}' — scoring these against a '{SEG_CONFIG}' GT cache would be "
                 f"cross-segmenter. Re-run run_seg.sh with SEG_CFG={SEG_CONFIG}, or set "
                 f"SEG_CFG={got} to score this seg_dir as it stands.")
    man = json.load(open(f"{args.input}/ef_manifest.json"))
    # dump() writes {"meta": {...}, "subjects": [...]}; a legacy dump is a bare list.
    meta = man.get("meta", {}) if isinstance(man, dict) else {}
    subj_list = man["subjects"] if isinstance(man, dict) else man
    out = args.out or (str(paths.ef_summary(meta["method"], meta["split"])) if meta.get("method") else None)
    if not out and meta.get("method") != "__gt_only__":
        sys.exit("score: --out required (legacy manifest carries no method for the default path)")
    from scipy import stats
    # Leftover segs from an EARLIER dump into the same seg_dir carry sidx's from a different
    # subject enumeration — they would be silently attributed to the wrong subject. The seg_dir
    # must carry THIS dump's id (run_seg.sh writes <seg_dir>/ef_dump_id from the manifest and
    # refuses a dir stamped with another id). Content-keyed, not mtime-keyed — a legacy manifest
    # without dump_id cannot be verified and is refused outright.
    dump_id = meta.get("dump_id")
    if not dump_id:
        sys.exit("score: manifest carries no dump_id (legacy dump) — re-run `ef_dice.py dump` on a fresh dir.")
    seg_id = _read_text(f"{args.seg_dir}/ef_dump_id")
    if seg_id != dump_id:
        sys.exit(f"score: {args.seg_dir}/ef_dump_id is {seg_id!r} but the manifest's dump_id is {dump_id!r} — "
                 f"segs are from another dump (or run_seg.sh was bypassed). Re-run run_seg.sh on a fresh seg_dir.")
    if meta.get("method") == "__gt_only__":
        # `dump --gt-only` manifest: only populate <subject>/seg_gt/, no arm metrics.
        n = 0
        for m in subj_list:
            c, sidx, subj, T = m["cohort"], m["sidx"], m["subject"], m["T"]
            _sync_gt_seg_cache(args.seg_dir, c, sidx, subj, T, m)
            n += _gt_seg_cached(c, subj, m["gt_sha256"])
        print(f"seg_gt cache populated for {n}/{len(subj_list)} subjects")
        return
    per_cohort = {}
    rows = []
    for m in subj_list:
        c, sidx, subj, T = m["cohort"], m["sidx"], m["subject"], m["T"]
        _sync_gt_seg_cache(args.seg_dir, c, sidx, subj, T, m)
        gt, gtvox = curve(args.seg_dir, c, sidx, "gt", T)
        if gt is None:
            print(f"  !! {c}/{subj}: GT LV seg missing/empty — subject dropped from the EF cohort",
                  file=sys.stderr)
            continue
        ed, es = int(gt.argmax()), int(gt.argmin())
        gt_rv, _ = curve(args.seg_dir, c, sidx, "gt", T, lab=RV)
        r = {"cohort": c, "subject": subj, "ef_gt": ef_of(gt)}
        r["edv_gt"], r["esv_gt"] = vols_of(gt, gtvox)
        r["lvm_gt"] = lvm_of(args.seg_dir, c, sidx, "gt", ed, gtvox)
        # Per-frame volume curves (mL), kept so any curve metric can be recomputed without re-seg.
        r["lv_curve_gt"] = (gt * gtvox / 1000.0).tolist()
        if gt_rv is not None:
            r["rv_ef_gt"] = ef_of(gt_rv)
            r["rv_edv_gt"], r["rv_esv_gt"] = vols_of(gt_rv, gtvox)
            r["rv_curve_gt"] = (gt_rv * gtvox / 1000.0).tolist()
        for arm in ("clean", "breath"):
            # LV, RV, and Dice/HD95 gate INDEPENDENTLY — an unsegmentable LV must not
            # suppress a valid RV, and overlap metrics only need the ED/ES segs to exist.
            cur, vox = curve(args.seg_dir, c, sidx, arm, T)
            if cur is not None:
                r[f"ef_{arm}"] = ef_of(cur)
                r[f"edv_{arm}"], r[f"esv_{arm}"] = vols_of(cur, vox)
                r[f"lvm_{arm}"] = lvm_of(args.seg_dir, c, sidx, arm, ed, vox)
                # Volume-curve error: frame-by-frame |pred - GT| (same phase index), mean over
                # frames, / GT EDV -> % of EDV. Defined even when the LV vanishes (EF undefined).
                lv_ml = cur * vox / 1000.0
                r[f"lv_curve_{arm}"] = lv_ml.tolist()
                r[f"lv_curve_nmae_{arm}"] = float(np.mean(np.abs(lv_ml - gt * gtvox / 1000.0))
                                                  / r["edv_gt"] * 100.0)
            rv, rvvox = curve(args.seg_dir, c, sidx, arm, T, lab=RV)
            if rv is not None:
                r[f"rv_ef_{arm}"] = ef_of(rv)
                r[f"rv_edv_{arm}"], r[f"rv_esv_{arm}"] = vols_of(rv, rvvox)
                rv_ml = rv * rvvox / 1000.0
                r[f"rv_curve_{arm}"] = rv_ml.tolist()
                if gt_rv is not None:
                    r[f"rv_curve_nmae_{arm}"] = float(np.mean(np.abs(rv_ml - gt_rv * gtvox / 1000.0))
                                                      / r["rv_edv_gt"] * 100.0)
            if all(os.path.isfile(seg_path(args.seg_dir, c, sidx, arm, t)) for t in {ed, es}):
                for name, lab in [("LV", LV), ("MYO", MYO), ("RV", RV)]:
                    r[f"dice_{arm}_{name}_ED"] = dice(args.seg_dir, c, sidx, arm, ed, ed, lab)
                    r[f"dice_{arm}_{name}_ES"] = dice(args.seg_dir, c, sidx, arm, es, es, lab)
                    r[f"hd95_{arm}_{name}_ED"] = hd95(args.seg_dir, c, sidx, arm, ed, ed, lab)
                    r[f"hd95_{arm}_{name}_ES"] = hd95(args.seg_dir, c, sidx, arm, es, es, lab)
        rows.append(r); per_cohort.setdefault(c, []).append(r)

    agg = {}
    for c, rs in per_cohort.items():
        d = {"n": len(rs)}
        for arm in ("clean", "breath"):
            # both sides may be None now (vanishing-seg guard) — pair only complete rows
            ok = [x for x in rs if x.get("ef_gt") is not None and x.get(f"ef_{arm}") is not None]
            g = np.array([x["ef_gt"] for x in ok])
            p = np.array([x[f"ef_{arm}"] for x in ok])
            if len(g) >= 3 and g.std() > 0:
                sl = float(np.polyfit(g, p, 1)[0])
                d[f"{arm}_ef_slope"] = sl
                d[f"{arm}_ef_spearman"] = float(stats.spearmanr(g, p).correlation)
                d[f"{arm}_ef_mae_pct"] = float(np.mean(np.abs(p - g)))
            # paired MAE vs the same subject's GT value, for each absolute-volume metric
            for key, unit in [("edv", "ml"), ("esv", "ml"), ("lvm", "g"),
                              ("rv_ef", "pct"), ("rv_edv", "ml"), ("rv_esv", "ml")]:
                pairs = [(x[f"{key}_gt"], x[f"{key}_{arm}"]) for x in rs
                         if x.get(f"{key}_gt") is not None and x.get(f"{key}_{arm}") is not None]
                if pairs:
                    gg, pp = map(np.array, zip(*pairs))
                    d[f"{arm}_{key}_mae_{unit}"] = float(np.mean(np.abs(pp - gg)))
            for key in ("lv_curve_nmae", "rv_curve_nmae"):
                vals = [x[f"{key}_{arm}"] for x in rs if x.get(f"{key}_{arm}") is not None]
                if vals:
                    d[f"{arm}_{key}_pct"] = float(np.mean(vals))
            for name in ("LV", "MYO", "RV"):
                for ph in ("ED", "ES"):
                    for met in ("dice", "hd95"):
                        vals = [x[f"{met}_{arm}_{name}_{ph}"] for x in rs
                                if f"{met}_{arm}_{name}_{ph}" in x
                                and not np.isnan(x[f"{met}_{arm}_{name}_{ph}"])]
                        if vals:
                            d[f"{arm}_{met}_{name}_{ph}"] = float(np.mean(vals))
        agg[c] = d
    # NaN -> null (dice/hd95 legitimately NaN on empty masks; this is a git-tracked citable
    # file and must stay strict-JSON, same policy as aggregate.py's summaries).
    def json_safe(o):
        if isinstance(o, dict):
            return {k: json_safe(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [json_safe(v) for v in o]
        if isinstance(o, float) and not np.isfinite(o):
            return None
        return o

    # MERGE with an existing file per cohort — this is ONE cross-cohort file per arm, and a
    # partial-cohort re-run (e.g. --cohorts miitt) must update only its own cohorts, not
    # silently erase the others (the next aggregate re-fold would null their EF blocks).
    prev = {}
    if os.path.isfile(out):
        try:
            prev = json.load(open(out))
        except json.JSONDecodeError:
            print(f"  !! existing {out} unreadable — replacing it wholesale", file=sys.stderr)
    kept = sorted(set(prev.get("aggregate", {})) - set(agg))
    if kept:
        print(f"  merging: keeping previous results for cohort(s) {', '.join(kept)}")
        agg = {**prev["aggregate"], **agg}
        rows = [r for r in prev.get("per_subject", []) if r.get("cohort") in kept] + rows
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(json_safe({"meta": meta, "aggregate": agg, "per_subject": rows}),
              open(out, "w"), indent=2, allow_nan=False)
    print(json.dumps(json_safe(agg), indent=2))
    print(f"\n-> {out}")


def plot(args):
    """Visualize a score() JSON: per-cohort EF scatter (pred vs GT, identity + fitted slope) on top,
    Dice bars (LV/MYO/RV at ED/ES) below. clean vs breath overlaid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.load(open(args.input))
    agg, rows = data["aggregate"], data["per_subject"]
    cohorts = sorted(agg)
    arms = ["clean", "breath"] if args.arm == "both" else [args.arm]
    color = {"clean": "#1f77b4", "breath": "#d62728"}
    names, phases = ("LV", "MYO", "RV"), ("ED", "ES")

    nc = len(cohorts)
    fig, axes = plt.subplots(2, nc, figsize=(4.2 * nc, 7.8), squeeze=False)
    for ci, c in enumerate(cohorts):
        crows = [r for r in rows if r["cohort"] == c]
        ax = axes[0, ci]
        lo, hi = 100.0, 0.0
        for ai, arm in enumerate(arms):
            g = np.array([r["ef_gt"] for r in crows if r.get(f"ef_{arm}") is not None])
            p = np.array([r[f"ef_{arm}"] for r in crows if r.get(f"ef_{arm}") is not None])
            if not len(g):
                continue
            ax.scatter(g, p, s=22, c=color[arm], alpha=0.8, label=arm, edgecolor="none")
            lo, hi = min(lo, g.min(), p.min()), max(hi, g.max(), p.max())
            sl, sp, mae = (agg[c].get(f"{arm}_ef_slope"), agg[c].get(f"{arm}_ef_spearman"),
                           agg[c].get(f"{arm}_ef_mae_pct"))
            if sl is not None:
                xs = np.array([g.min(), g.max()])
                b = float(p.mean() - sl * g.mean())
                ax.plot(xs, sl * xs + b, c=color[arm], lw=1.4)
                ax.annotate(f"{arm}: slope {sl:.2f}, ρ {sp:.2f}, MAE {mae:.1f}%",
                            xy=(0.03, 0.95 - 0.07 * ai), xycoords="axes fraction",
                            fontsize=7.5, color=color[arm])
        pad = 0.05 * (hi - lo + 1e-6)
        lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, "--", c="0.6", lw=1, zorder=0)                 # identity
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_title(f"{c}  (n={agg[c]['n']})", fontsize=9)
        ax.set_xlabel("GT EF (%)", fontsize=8); ax.set_ylabel("pred EF (%)", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")

        axd = axes[1, ci]
        labels = [f"{n}\n{ph}" for n in names for ph in phases]
        x = np.arange(len(labels)); wbar = 0.8 / len(arms)
        for ai, arm in enumerate(arms):
            vals = [agg[c].get(f"{arm}_dice_{n}_{ph}", np.nan) for n in names for ph in phases]
            axd.bar(x + ai * wbar, vals, wbar, color=color[arm], label=arm, alpha=0.85)
        axd.set_xticks(x + wbar * (len(arms) - 1) / 2); axd.set_xticklabels(labels, fontsize=7)
        axd.set_ylim(0, 1); axd.set_ylabel("Dice", fontsize=8); axd.set_title(f"{c} Dice", fontsize=9)
        axd.legend(fontsize=7)

    fig.suptitle(f"EF recovery + Dice — {os.path.basename(args.input)}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=160); plt.close(fig)
    print(f"-> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("input_dir")
    d.add_argument("--method", default="vggt_augaggr224hw2_ep300")
    d.add_argument("--cohorts", nargs="+", default=list(paths.DATASETS))
    d.add_argument("--gt-only", action="store_true",
                   help="dump only the cropped GT for every split subject lacking a seg_gt cache "
                        "(no arm needed); pair with `score` on the resulting seg_dir to fill the cache")
    s = sub.add_parser("score"); s.add_argument("seg_dir")
    s.add_argument("--input", required=True)
    s.add_argument("--out", default=None,
                   help="default: paths.ef_summary(<method>, <split> from the dump manifest) — the "
                        "location score/aggregate.py merges from")
    pl = sub.add_parser("plot"); pl.add_argument("input", help="a score() output json")
    pl.add_argument("--arm", choices=["clean", "breath", "both"], default="both")
    pl.add_argument("--out", required=True)
    a = ap.parse_args()
    {"dump": dump, "score": score, "plot": plot}[a.cmd](a)
