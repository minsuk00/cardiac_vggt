"""Regression tests for docs/59 F21 — the ef_val_sweep ES half must be reachable.

`ef_val_sweep` builds `val_targets = [(i, ED) for every subject] + [(i, ES) for every subject]`,
i.e. **2N entries**, and `get_data` indexes it by `seq_index % len(val_targets)`. But `__len__`
returns `len_train`, and a dataloader cannot yield more samples than the dataset declares — so if
`len_train` is left at the SUBJECT count `N`, `seq_index` only ever reaches `0..N-1` and **every ES
entry is silently unreachable**. `EF = (EDV - ESV)/EDV`, so that kills the predicted-EF metric with
no error: the run just quietly evaluates ED twice as often as it should and writes half the volumes.

This actually shipped (docs/59 §10): `len_train = len(self.subjects)` was assigned *before*
`val_targets` was built. `pytest` was 251-green with the bug present, and the identity gate could
not see it either — it took an end-to-end run plus counting the NIfTIs on disk. Hence these tests.

The invariant, stated once: **`len(dataset)` must equal the number of things one epoch should
visit** — which is the sweep length when the sweep is on, and the subject count when it is off.
"""
from __future__ import annotations

import csv
import os

import pytest

from data.datasets.mri_dataset import MRIDataset

NUM_PHASES = 12
# Deliberately != ED and != each other, and both inside [0, NUM_PHASES) so
# `_build_val_targets`' range guard does not fire for an unrelated reason.
ED_ES = {"Train_P001": (0, 5), "Val_P001": (1, 7)}


@pytest.fixture(scope="module")
def sweep_split_file(synthetic_root, tmp_path_factory):
    """A split whose [val] section holds BOTH synthetic subjects, so the sweep is 4 entries.

    Two subjects (not one) matters: with N=1 the buggy `len_train = N` gives len 1 vs 2, but with
    N=2 it gives 2 vs 4 — the same factor-of-2 truncation the real 133-vs-266 case had, so the
    test fails for the true reason rather than an off-by-one.
    """
    p = tmp_path_factory.mktemp("sweep_split") / "sweep.txt"
    p.write_text("[train]\nTrain_P001\n\n[val]\nTrain_P001\nVal_P001\n\n[test]\n")
    return str(p)


@pytest.fixture(scope="module")
def cardiac_phase_csv(tmp_path_factory):
    p = tmp_path_factory.mktemp("whs") / "cardiac_phase.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["subject", "ED", "ES"])
        w.writeheader()
        for sid, (ed, es) in ED_ES.items():
            w.writerow({"subject": sid, "ED": ed, "ES": es})
    return str(p)


def _sweep_ds(synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv):
    return MRIDataset(
        common_conf, synthetic_root,
        split="val", split_file=sweep_split_file,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
        cache_dir=monai_cache_dir,
        ef_val_sweep=True, cardiac_phase_csv=cardiac_phase_csv,
    )


def test_sweep_length_is_targets_not_subjects(
        synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv):
    """THE regression assertion: an epoch enumerates sweep entries, not subjects."""
    ds = _sweep_ds(synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv)
    n = len(ds.subjects)
    assert n == 2, f"fixture should give 2 val subjects, got {n}"
    assert len(ds.val_targets) == 2 * n, "sweep must be ED-for-all then ES-for-all"
    assert len(ds) == len(ds.val_targets), (
        f"len(dataset)={len(ds)} but the sweep has {len(ds.val_targets)} entries — the dataloader "
        f"would stop early and the ES half would never be evaluated (docs/59 F21)"
    )


def test_every_sweep_entry_including_ES_is_reachable(
        synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv):
    """Walk the seq_index range an epoch actually issues and confirm full coverage."""
    ds = _sweep_ds(synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv)
    vt = len(ds.val_targets)
    n = len(ds.subjects)
    reached = {i % vt for i in range(len(ds))}
    assert reached == set(range(vt)), f"unreachable sweep entries: {sorted(set(range(vt)) - reached)}"
    # The ES half is the back half by construction; it is what EF needs.
    es_reached = {i for i in reached if i >= n}
    assert len(es_reached) == n, f"ES half reached {len(es_reached)}/{n} — EF cannot be computed"


def test_sweep_visits_both_ED_and_ES_phases(
        synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv):
    """End-to-end tell: the t_target values an epoch yields must include every ES phase.

    This is the check that would have caught the shipped bug from the logs alone — the real run's
    per-phase panel showed only ED phases.
    """
    ds = _sweep_ds(synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv)
    vt = len(ds.val_targets)
    phases = {ds.val_targets[i % vt][1] for i in range(len(ds))}
    for sid, (ed, es) in ED_ES.items():
        assert ed in phases, f"{sid}: ED phase {ed} never visited"
        assert es in phases, f"{sid}: ES phase {es} never visited (docs/59 F21)"


def test_fault_injection_the_assertions_have_teeth(
        synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv):
    """PROOF the tests above can fail: restore the buggy `len_train = len(subjects)` and confirm
    both the length invariant and ES reachability break. A check never shown to fire is not
    evidence (standing fault-injection rule)."""
    ds = _sweep_ds(synthetic_root, sweep_split_file, common_conf, monai_cache_dir, cardiac_phase_csv)
    n = len(ds.subjects)
    vt = len(ds.val_targets)

    ds.len_train = n                                    # <-- the docs/59 F21 bug, re-injected
    assert len(ds) != vt, "fault injection did not change the observed length"
    reached = {i % vt for i in range(len(ds))}
    assert len(reached) < vt, "fault injection should truncate the sweep"
    assert not [i for i in reached if i >= n], (
        "with the bug re-injected the ES half must be UNREACHABLE — if it is still reached, "
        "these tests are not actually testing the truncation"
    )

    ds.len_train = vt                                   # restore; confirm recovery
    assert {i % vt for i in range(len(ds))} == set(range(vt))


def test_no_sweep_length_is_subject_count(synthetic_root, split_file, common_conf, monai_cache_dir):
    """Guard the OTHER direction (docs/59 F6): with the sweep off, an epoch is exactly one pass
    over the subjects — not the old `max(1000, N)`, which double-sampled a seed-invariant residual.
    """
    ds = MRIDataset(
        common_conf, synthetic_root,
        split="val", split_file=split_file,
        mode="dynamic", mri_mode="axial",
        num_slices=12, target_size=518,
        cache_dir=monai_cache_dir,
        ef_val_sweep=False,
    )
    assert ds.val_targets is None
    assert len(ds) == len(ds.subjects)
