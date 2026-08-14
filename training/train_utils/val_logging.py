"""Pure helpers shared by `trainer.py` and `trainer_viz.py` for validation logging (docs/60).

Everything here is a free function with no trainer state: subject identity, cohort
selection, plane selection, and the respiratory off-slab statistics. Keeping them out of
the trainer keeps both call sites short and makes them directly unit-testable
(`tests/test_val_logging_helpers.py`).
"""

import csv
import logging
import os

# Number of z-planes rendered in the cardiac filmstrip / GIF. Shared by the renderer and
# the GIF tiler — they reshape on it, so it must be one constant, not two literals.
N_FILM_PLANES = 5


# ── subject identity ────────────────────────────────────────────────────────
def subject_id(subject_path):
    """Subject id for a `MRIDataset.subjects` entry.

    `_find_subjects` builds `<data_root>/<split-file line>/sax`, so the id is the PARENT
    directory, not the basename — `basename` is always the literal "sax". Matches the
    `id` column of `manifest.csv` and the `subject` column of `cardiac_phase.csv`.
    """
    return os.path.basename(os.path.dirname(os.path.normpath(str(subject_path))))


def subject_source(subject_path):
    """Source-dataset tag from the path alone: ACDC / CMRx23 / CMRx24 / CMRx25 / MNMs.

    Every pooled subject id is prefixed with its origin by the converters, so this works
    for both `pooled.txt` layouts (`ACDC_sax/<id>` and `CMRxRecon2023/Cine_combined/<id>`).
    """
    sid = subject_id(subject_path)
    return sid.split("_")[0] if "_" in sid else sid


def pick_one_index_per_source(subjects, max_picks=8):
    """First index of each source, in first-appearance order.

    Deterministic (no RNG) so visual panels track the SAME subjects across epochs and runs.
    Replaces a hardcoded index list, which under the sorted pooled split selected only the
    two smallest sources.
    """
    seen, picks = set(), []
    for i, path in enumerate(subjects):
        src = subject_source(path)
        if src not in seen:
            seen.add(src)
            picks.append(i)
            if len(picks) >= max_picks:
                break
    return tuple(picks)


def pick_visual_indices(subjects, vendor_by_subject):
    """Deterministic source/vendor coverage for validation visual panels."""
    vendor_targets = {
        "CMRx25": {"Siemens", "UIH", "Philips"},
        "MNMs": {"Siemens", "GE", "Canon"},
    }
    seen, picks = set(), []
    for i, path in enumerate(subjects):
        src = subject_source(path)
        if src in vendor_targets:
            vendor = vendor_by_subject.get(subject_id(path))
            if vendor not in vendor_targets[src]:
                continue
            key = (src, vendor)
        else:
            key = (src, None)
        if key not in seen:
            seen.add(key)
            picks.append(i)
    return tuple(picks)


def seq_index_to_subject(mri_ds, seq_index):
    """(subject_id, source) for a val `seq_index`, mirroring `MRIDataset.get_data`.

    `get_data` returns `seq_name` but not `subj_idx`, and `seq_name` flattens the rel-path
    with underscores so the source token is not recoverable from it consistently. Returns
    (None, None) if anything is missing.
    """
    try:
        vt = getattr(mri_ds, "val_targets", None)
        subj_idx = vt[seq_index % len(vt)][0] if vt else seq_index % len(mri_ds.subjects)
        path = mri_ds.subjects[int(subj_idx)]
        return subject_id(path), subject_source(path)
    except Exception:
        return None, None


def load_subject_groups(split_file, column):
    """{subject_id: value} for one `manifest.csv` column, e.g. "pathology_label".

    The manifest sits next to the split file. Returns {} on any failure — a missing
    manifest must degrade to un-grouped metrics, never break the EF path.
    """
    if not split_file:
        return {}
    path = os.path.join(os.path.dirname(split_file), "manifest.csv")
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if rows and column not in rows[0]:
            logging.warning(f"manifest has no column '{column}'; skipping grouping")
            return {}
        return {r["id"]: r[column] for r in rows if r.get("id") and r.get(column)}
    except Exception as e:
        logging.warning(f"manifest column '{column}' unavailable ({e}); skipping grouping")
        return {}


# ── rendering ───────────────────────────────────────────────────────────────
def pick_planes(D, n=N_FILM_PLANES):
    """`n` evenly-spaced z-plane indices spanning the WHOLE stack [0, D).

    Replaces a `mid±2` window. Under native-z `D` varies 5-21 per subject and `D//2` is
    provably the reference slot, so that window rendered a subject-dependent fraction of
    the stack centred on the one plane the model gets for free — and at D=6 never showed
    apex plane 0. Evenly spaced always includes both apex and base.

    Length is always exactly `n` (the GIF tiler reshapes on it); indices may repeat when
    D < n, which keeps the panel geometry fixed.
    """
    if D <= 1 or n <= 1:
        return [0] * n
    return [min(int(round(i * (D - 1) / (n - 1))), D - 1) for i in range(n)]


def to_float(v):
    """Tensor/np scalar → float, or None. Keeps logged rows JSON/CSV-safe."""
    if v is None:
        return None
    try:
        return float(v.item() if hasattr(v, "item") else v)
    except Exception:
        return None


# ── respiratory damage ──────────────────────────────────────────────────────
def resp_offslab_stats(batch, b=0):
    """How much of subject `b`'s stack the simulated breathing pushed off the slab.

    docs/59 F16 is accepted-not-fixed: the shift is one-sided, so basal slots run off the
    end and `padding_mode="zeros"` blanks or dims them. The damaged FRACTION is
    `d/((D-1)*dz)`, which is far worse for short stacks — exactly the fine-pitch subjects
    native-z was added to support. Unrecoverable after the fact (applied on GPU, never
    persisted), which is why it is recorded per subject.

    Landing plane = `z_i + d_D/dz`. Returns {} when breathing is off.
    """
    disp = batch.get("resp_disp_mm")
    slice_idx = batch.get("slice_indices")
    dz_t = batch.get("dz_mm")
    phases = batch.get("phases")
    if disp is None or slice_idx is None or dz_t is None or phases is None:
        return {}
    D = int(phases.shape[2])
    dz = max(float(dz_t.reshape(-1)[b]), 1e-6)
    landing = slice_idx[b].float() + disp[b].float()[..., 0] / dz        # (S,) planes
    # `off` = ANY attenuation: outside [0, D-1] the reslicer's `padding_mode="zeros"`
    # mixes zero-padding in, so retained < 1. (Verified against the real reslicer: this is
    # exactly `retained < 1`, blanked and partially-attenuated slots together.)
    off = (landing < 0) | (landing > D - 1)
    # "Dimmed" = the PARTIAL subset of `off`: one bracketing plane is real and the other is
    # zero-padding, so retained is in (0, 1). Beyond [-1, D] both brackets are padding and
    # the slot is fully blank. `dimmed ⊂ off`, so blanked-fraction = offslab - dimmed.
    #
    # This band was WRONG until 2026-08-01 (docs/62 §5.3): it read
    # `((landing > 0) & (landing < 1)) | ((landing > D-2) & (landing < D-1))`, one plane too
    # low at BOTH ends. Those ranges interpolate between two REAL planes and retain 1.0000,
    # so the metric had ZERO true positives while reporting a 7-35% rate. The old comment
    # justified it as "inside the slab but not on an exact end plane" — trilinear only mixes
    # in padding once a bracket falls OUTSIDE [0, D-1], which is `off`, not "not exact".
    dimmed = ((landing > -1) & (landing < 0)) | ((landing > D - 1) & (landing < D))
    extent_mm = max((D - 1) * dz, 1e-6)
    return {
        "frac_slots_offslab": float(off.float().mean()),
        "frac_slots_dimmed": float(dimmed.float().mean()),
        "disp_frac_of_extent": float(disp[b].float()[..., 0].abs().mean()) / extent_mm,
    }
