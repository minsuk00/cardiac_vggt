"""Single source of truth for evaluation paths + arm naming.

The heavy data lives on GPFS, symlinked in at ``evaluation/volumes`` (subject-major):

    volumes/<dataset>/out/<subject>/
        manifest.json                         # per-subject bundle spec (T, spacing, breath disp)
        gt/gt_t{00..T-1}.nii.gz               # unshifted target phases
        clean/stack_t{00..T-1}.nii.gz         # frozen clean input stacks
        breath/stack_t{00..T-1}.nii.gz        # frozen breathing-corrupted input stacks
        mask.nii.gz | mask_fov.nii.gz         # FOV mask (name varies by dataset)
        mask_heart.nii.gz  heart_seg*.nii.gz  # heart ROI / segmentation
        <arm>/                                # one dir per method (svrtk3d, nesvor, vggt_*)
            recon_clean/vol_t{00..T-1}.nii.gz
            recon_breath/vol_t{00..T-1}.nii.gz
            metrics.json  timing.json  resp_diag.json  ed_dvf.npz
            metadata.json  provenance.txt

Every path/naming convention the harness uses is built HERE, so a layout change is a
one-function edit instead of a hunt across run_vggt.py / src/score/*.py
and ~15 tools/ scripts. Import standalone:

    import sys; sys.path.insert(0, "<repo>/evaluation"); import paths
    for arm in paths.arms("cmrx2024"):
        for subj in paths.subjects("cmrx2024"):
            v = paths.recon("cmrx2024", subj, arm, "clean", 0)
"""
import json
import os
from pathlib import Path

EVAL_ROOT = Path(__file__).resolve().parent
VOLUMES = EVAL_ROOT / "volumes"          # -> GPFS (subject-major PRECIOUS data: recons/metrics)
CHECKPOINTS = EVAL_ROOT / "checkpoints"  # -> GPFS (copied ckpts per arm)
RESULTS = EVAL_ROOT / "metric_results"       # git-tracked cohort summaries
FIGURES = EVAL_ROOT / "comparison_figures"   # -> GPFS (subject-major DISPOSABLE figures; rm-safe)

# One dir per POOLED SOURCE (was: 4 dirs split by in-dist vs "OOD"). That distinction is gone —
# ACDC and M&Ms are in the training pool now, and every source here is gated + breathing-simulated,
# so they differ by provenance, not by regime. Keys match `build_inputs/pooled.py`'s --source.
DATASETS = ("cmrx2023", "cmrx2024", "cmrx2025", "acdc", "mnms", "miitt", "ocmr")

# Rhythm-arm cohorts (docs/110): `<source>_<arm>` holds the SAME subjects re-simulated under an
# irregular cardiac rhythm, built by tools/build_af_bundle.py. They are deliberately NOT in
# DATASETS, because six call sites do `default=list(paths.DATASETS)` (score/run.py, ef_dice.py,
# run_baselines.py, fetal4d_gate.py, run_dangi.py, tools/build_padded_heart_mask.py) — adding them
# there would silently widen every bare cohort sweep in the repo, including the live campaign's.
# Use ALL_DATASETS for argparse `choices=` only, so these cohorts are opt-in by name.
# The `*24` arms are the final paper run: same rhythms, 24 frames per slice (2 nominal beats)
# instead of 12. They are SEPARATE cohort names on purpose — rebuilding in place would destroy the
# 12-frame bundles and recons that docs/111–112 rest on, and those are not regenerable from git.
RHYTHM_ARMS = ("regular_frozen", "regular", "hrv", "af", "af_rvr", "af_pause",
               "regular_frozen24", "regular24", "hrv24", "af24",
               "af12",   # af24's first 12 frames per slice, on the current simulator (tools/build_af12_from_af24.py)
               "hrv12")  # hrv24's first 12 frames per slice, same builder (--rhythm hrv)
EXTRA_DATASETS = tuple(f"{d}_{a}" for d in DATASETS for a in RHYTHM_ARMS)
ALL_DATASETS = DATASETS + EXTRA_DATASETS

VARIANTS = ("clean", "breath")           # the two recon conditions (both in one metrics.json)
BUNDLE_DIRS = ("gt", "clean", "breath", "scatter", "rolled")  # input-bundle subdirs; NOT arms
# input-bundle phase-stack filename prefix per subdir: gt/ -> gt_t*, clean|breath|scatter|rolled/ -> stack_t*
# rolled/ = breath/ with each plane's 12 phases circularly shifted by a frozen random roll (unknown
# per-slice phase, for the self-gating Fetal CMR 4D arm; built by pooled.py add_rolled).
_STACK_PREFIX = {"gt": "gt", "clean": "stack", "breath": "stack", "scatter": "stack", "rolled": "stack"}


# --- roots -----------------------------------------------------------------
def dataset_root(dataset):
    """The subject-major cohort root:  volumes/<dataset>/out ."""
    return VOLUMES / dataset / "out"


def subject_dir(dataset, subject):
    return dataset_root(dataset) / subject


def arm_dir(dataset, subject, arm):
    return subject_dir(dataset, subject) / arm


# --- enumeration (arm-style iteration over subject-major disk) --------------
def subjects(dataset):
    """Built subjects = subject dirs that carry a manifest.json (sorted)."""
    root = dataset_root(dataset)
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and (d / "manifest.json").is_file())


def filter_by_split(dataset, subject_list, split):
    """Partition `subject_list` into (keep, dropped) by each bundle's own `manifest["split"]`.

    A bundle is a directory; anything that lands under `<source>/out/` joins the cohort just by
    existing. That is a real failure mode, not a hypothetical: a build of a TEST or TRAIN subject
    into the same dir would otherwise be reconstructed, scored and AVERAGED IN silently — the
    bundle dir is not split-keyed. The builder records `split` in
    every manifest, so every consumer that defines a cohort must honour it. `dropped` is a list of
    (subject, reason) so the caller can report what it excluded.
    """
    keep, dropped = [], []
    for s in subject_list:
        try:
            m = json.load(open(manifest(dataset, s)))
        except (json.JSONDecodeError, OSError):
            dropped.append((s, "unreadable manifest")); continue
        # No default: `m.get("split", split)` would fail OPEN, keeping an unlabelled bundle for ANY
        # requested split — the exact silent-averaging this function exists to prevent. Every
        # manifest the builder writes carries the key (verified across all 144 on disk), so a
        # missing one means a hand-made or pre-split bundle and must be dropped, not trusted.
        if m.get("split") != split:
            dropped.append((s, f"built for split '{m.get('split')}'")); continue
        keep.append(s)
    return keep, dropped


def arms(dataset, subject=None):
    """Method/arm folder names. For one subject if given, else the union across all
    subjects. Excludes the input-bundle dirs (gt/clean/breath)."""
    def _arms_in(subj):
        sd = subject_dir(dataset, subj)
        if not sd.is_dir():
            return set()
        # A real method arm has recon_{clean,breath}; the positive filter excludes stray dirs
        # (aborted runs, __pycache__, scratch) that would otherwise become phantom arms.
        return {d.name for d in sd.iterdir()
                if d.is_dir() and d.name not in BUNDLE_DIRS
                and ((d / "recon_clean").is_dir() or (d / "recon_breath").is_dir())}

    if subject is not None:
        return sorted(_arms_in(subject))
    out = set()
    for subj in subjects(dataset):
        out |= _arms_in(subj)
    return sorted(out)


# --- recon volumes ---------------------------------------------------------
def recon(dataset, subject, arm, variant, phase):
    """One predicted phase volume.  variant in {'clean','breath'}."""
    assert variant in VARIANTS, variant
    return arm_dir(dataset, subject, arm) / f"recon_{variant}" / f"vol_t{phase:02d}.nii.gz"


def recon_dir(dataset, subject, arm, variant):
    return arm_dir(dataset, subject, arm) / f"recon_{variant}"


def recon_stamp(dataset, subject, arm, variant):
    """PER-VARIANT identity of the run that wrote `recon_<variant>/`.

    `metadata.json` is per ARM, one file, rewritten every run — so it cannot tell you that
    `recon_clean/` is older than `recon_breath/`. That gap is reachable by the shipped driver's
    own default: re-running an arm with `--arms breath` (the default) leaves the previous run's
    `recon_clean/` in place, the scorer discovers variants by `.is_dir()` and scores it, and
    `cost_psnr = clean - breath` then subtracts two DIFFERENT checkpoints. No crash, no warning.
    One stamp per variant makes that detectable; score/image_metrics.py compares them.
    """
    return recon_dir(dataset, subject, arm, variant) / "stamp.json"


# --- input bundle ----------------------------------------------------------
def manifest(dataset, subject):
    return subject_dir(dataset, subject) / "manifest.json"


def bundle_stack(dataset, subject, kind, phase):
    """One input-bundle phase stack.  kind in {'gt','clean','breath','scatter'} (gt/ uses the
    gt_t* prefix; the others use stack_t*). scatter/ = the same-input stack VGGT sees (built
    from breath/ by build_inputs/pooled.py add_scatter)."""
    assert kind in BUNDLE_DIRS, kind
    return subject_dir(dataset, subject) / kind / f"{_STACK_PREFIX[kind]}_t{phase:02d}.nii.gz"


def fov_mask(dataset, subject):
    """FOV mask — name is 'mask.nii.gz' (cmrxrecon) or 'mask_fov.nii.gz' (OOD); resolve
    whichever exists, preferring the plain name."""
    sd = subject_dir(dataset, subject)
    for name in ("mask.nii.gz", "mask_fov.nii.gz"):
        if (sd / name).is_file():
            return sd / name
    raise FileNotFoundError(f"no FOV mask (mask.nii.gz / mask_fov.nii.gz) under {sd}")


def heart_mask(dataset, subject):
    """Tight heart ROI (seg union + 6 mm in-plane + z±1). Since docs/107 it is only the seed
    `heart_mask_pad` is built from; the image-metric scoring ROI is `heart_mask_pad` ∩ FOV."""
    return subject_dir(dataset, subject) / "mask_heart.nii.gz"


HEART_MASK_PAD = "mask_heart_pad10.nii.gz"


def heart_mask_pad(dataset, subject):
    """`heart_mask` dilated +10 mm in-plane (z-extent unchanged), clamped to the FOV
    (tools/build_padded_heart_mask.py). Used for the SEGMENTATION crop (ef_dice) and as the
    classical baselines' reconstruction mask — the tight crop biases nnU-Net's ES LV read on
    reconstructions by ~+5 pp EF MAE (docs/104 §4); the pad removes it (docs/107)."""
    return subject_dir(dataset, subject) / HEART_MASK_PAD


# --- per-arm artifacts -----------------------------------------------------
def metrics(dataset, subject, arm):
    return arm_dir(dataset, subject, arm) / "metrics.json"


def cine(dataset, subject, arm, variant):
    """4D canonical-grid cine of a method's recon as scored by src/score/image_metrics.py
    (post gauge/pose/PSF) — what viz.py and the seg chain consume."""
    assert variant in VARIANTS, variant
    return arm_dir(dataset, subject, arm) / f"cine_{variant}.nii.gz"


def cine_gt(dataset, subject):
    """Shared 4D GT cine (method-independent). image_metrics.py writes it only if absent."""
    return subject_dir(dataset, subject) / "cine_gt.nii.gz"


def seg_gt_dir(dataset, subject, seg_config="2d"):
    """Cache of the GT cine's nnU-Net segs (seg_t{00..T-1}.nii.gz + src.json {"gt_sha256", "crop",
    "seg_config"}), written by ef_dice.py score and reused by later dumps so GT is segmented once,
    not per arm.

    KEYED ON THE SEGMENTER CONFIG, and `2d` keeps the historical bare `seg_gt/` path. Two reasons
    it cannot be one shared dir:
      - the cache key never included the segmenter, so flipping run_seg.sh to 3d_fullres would
        leave every existing subject "cached" and score 3D predictions against 2D GT segs, with
        no error and no warning;
      - `seg_gt/` has a consumer that is NOT a metric. tools/build_af_bundle.py reads its LV
        volume curve for the AF volume-matched resume, so overwriting it would change the
        simulated AF INPUT and invalidate every bundle and reconstruction built on it. (That
        caller hardcodes the "seg_gt" basename rather than coming through here, so it is
        insulated either way -- keep it that way.)
    """
    sub = "seg_gt" if seg_config == "2d" else f"seg_gt_{seg_config}"
    return subject_dir(dataset, subject) / sub


def cine_gt_src(dataset, subject):
    """Sidecar recording WHICH gt bundle `cine_gt.nii.gz` was derived from ({"gt_sha256": ...}).
    Freshness is content-keyed, never mtime-keyed: GPFS purge-avoidance `touch`es rewrite every
    mtime and must not make a cine look stale (or fresh)."""
    return subject_dir(dataset, subject) / "cine_gt.src.json"


def gt_sha256(dataset, subject):
    """Content id of the gt bundle = sha256 of gt_t00 (the same single file the old mtime rule
    keyed on; a bundle rebuild rewrites every phase, so phase 0 stands for the set)."""
    return file_sha256(bundle_stack(dataset, subject, "gt", 0))


def file_sha256(path, chunk=1 << 22):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def ckpt_fingerprint(path, edge=64 << 20):
    """Content id of a checkpoint: "v2:<size>:<sha256 of first+last 64 MiB>[:16]". Replaces the
    legacy "<size>:<int(mtime)>" (which any `touch` invalidated). Head+tail rather than the full
    ~9 GB file: a training checkpoint differs from every other in its tensor bytes at both ends, and
    a full GPFS read would cost ~1-2 min per invocation. None if unreadable."""
    import hashlib
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read(edge))
            if size > edge:
                fh.seek(max(edge, size - edge))
                h.update(fh.read(edge))
        return f"v2:{size}:{h.hexdigest()[:16]}"
    except OSError:
        return None


def same_fingerprint(a, b):
    """Equality across fingerprint formats: two v2 (or two legacy) ids compare directly; a legacy
    vs v2 pair cannot be compared -> None (caller falls back to path identity)."""
    if not a or not b:
        return None
    va, vb = str(a).startswith("v2:"), str(b).startswith("v2:")
    if va != vb:
        return None
    return a == b


def metadata(dataset, subject, arm):
    return arm_dir(dataset, subject, arm) / "metadata.json"


def resp_diag(dataset, subject, arm):
    return arm_dir(dataset, subject, arm) / "resp_diag.json"


# --- analysis figures ------------------------------------------------------
# Per-arm renders (gif_*, panel_dvf.png) live IN the arm dir (arm_dir) beside the recons they
# depict — the whole arm dir is the keep/delete unit. Only figures that belong to NO single arm
# go to the separate FIGURES tree (on GPFS, off /home): cross-arm compares + cohort summaries.
def panel_dvf(dataset, subject, arm):
    """Per-arm predicted-Δz panel, co-located with the gifs: volumes/<ds>/out/<subj>/<arm>/panel_dvf.png."""
    return arm_dir(dataset, subject, arm) / "panel_dvf.png"


def compare_dir(dataset, subject):
    """Cross-arm figures for one subject (compare_*.gif): comparison_figures/<ds>/<subject>/_compare/.
    Leading '_' => never mistaken for an arm; compare spans arms so it owns no single one."""
    return FIGURES / dataset / subject / "_compare"


def cohort_fig_dir(dataset):
    """Cohort-level figures (EF scatter, per-arm breathing summaries): comparison_figures/<ds>/."""
    return FIGURES / dataset


# --- cohort summary --------------------------------------------------------
def summary(dataset, arm, split):
    """Git-tracked cohort summary (the citable numbers). Split-keyed, no default: the val and
    test runs of one arm must never write the same file."""
    return RESULTS / split / dataset / f"{arm}.json"


def ef_summary(arm, split):
    """The EF/Dice chain's output for one arm (ef_dice.py score; ALL cohorts in one file —
    the chain runs cross-cohort around a single nnU-Net call). src/score/aggregate.py reads
    this to fold the biventricular block into each dataset's summary. Split-keyed like summary()."""
    return RESULTS / split / "_ef" / f"{arm}.json"


def legacy_summary(dataset, arm):
    """Where aggregate.py historically wrote the cohort summary (GPFS, beside the subject
    dirs). Kept for back-compat reads during migration."""
    return dataset_root(dataset) / f"{arm}_summary.json"


# --- arm naming (the ONE place the vggt method string is built) ------------
def canonical_arm(model_name, date=None, continuous_z=False):
    """Build a VGGT arm name from its identity slug. WRITE-SIDE ONLY — use this to name a
    NEW run. Do NOT use it to reconstruct an existing arm name for lookup; enumerate with
    ``arms()`` instead, which reads the real dir names on disk.

    Guards the historical doubling bug: driver scripts sometimes passed a model_name that
    already contained 'contz' *and* set continuous_z=True, so the old inline
    ``f"vggt_{date}_{model_name}" + ("_contz" if continuous_z else "")`` appended a second
    '_contz' (only on OOD cohorts). Here the contz marker is added at most once. As a
    consequence this does NOT reproduce the legacy doubled ``vggt_..._contz_contz`` OOD
    dirs still on disk — those are reached only by enumeration, never rebuilt here.

    An already-'vggt_'-prefixed model_name is accepted and never re-prefixed (with or
    without date). date is optional and legacy: going forward the arm is a bare slug
    (date/epoch/scheme live in MODELS.md), but passing date reproduces the old
    ``vggt_<date>_<model>`` form.
    """
    core = model_name[len("vggt_"):] if model_name.startswith("vggt_") else model_name
    stem = f"vggt_{date}_{core}" if date is not None else f"vggt_{core}"
    if continuous_z and "contz" not in stem:
        stem += "_contz"
    return stem
