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
one-function edit instead of a hunt across run_vggt.py / assemble_and_gif.py / aggregate.py
and ~15 tools/ scripts. Import standalone:

    import sys; sys.path.insert(0, "<repo>/evaluation"); import paths
    for arm in paths.arms("cmrx2024"):
        for subj in paths.subjects("cmrx2024"):
            v = paths.recon("cmrx2024", subj, arm, "clean", 0)
"""
import json
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
VARIANTS = ("clean", "breath")           # the two recon conditions (both in one metrics.json)
BUNDLE_DIRS = ("gt", "clean", "breath")  # input-bundle subdirs; NOT arms
# input-bundle phase-stack filename prefix per subdir: gt/ -> gt_t*, clean|breath/ -> stack_t*
_STACK_PREFIX = {"gt": "gt", "clean": "stack", "breath": "stack"}


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
    into the same dir would otherwise be reconstructed, scored and AVERAGED IN silently — neither
    the bundle dir nor `metric_results/<ds>/<arm>.json` is split-keyed. The builder records `split` in
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
    One stamp per variant makes that detectable; assemble_and_gif compares them.
    """
    return recon_dir(dataset, subject, arm, variant) / "stamp.json"


# --- input bundle ----------------------------------------------------------
def manifest(dataset, subject):
    return subject_dir(dataset, subject) / "manifest.json"


def bundle_stack(dataset, subject, kind, phase):
    """One input-bundle phase stack.  kind in {'gt','clean','breath'} (gt/ uses the
    gt_t* prefix; clean/ and breath/ use stack_t*)."""
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
    return subject_dir(dataset, subject) / "mask_heart.nii.gz"


# --- per-arm artifacts -----------------------------------------------------
def metrics(dataset, subject, arm):
    return arm_dir(dataset, subject, arm) / "metrics.json"


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
def summary(dataset, arm):
    """Git-tracked cohort summary (the citable numbers)."""
    return RESULTS / dataset / f"{arm}.json"


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
