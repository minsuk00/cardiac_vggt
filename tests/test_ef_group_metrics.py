"""Tests for the pathology-split EF metrics (docs/60 item 10).

WHY THESE ARE UNIT TESTS. The end-to-end EF path runs nnU-Net in a separate conda env as
a subprocess, and it is flaky here — a real smoke run hit `rc=1` on all 5 retries, so
`compute_ef_metrics` was never reached and the grouping change went unexercised. The
segmentation is not what changed; the metric aggregation is. So we synthesise segs
(label 1 = LV cavity, the only label `_lv_ml` reads) and test the aggregation directly.
"""

import numpy as np
import nibabel as nib
import pytest

from ef_eval import compute_ef_metrics


def _write_seg(path, n_lv_voxels, zooms=(1.4, 1.4, 10.0)):
    """A seg whose LV cavity (label 1) has exactly `n_lv_voxels` voxels."""
    arr = np.zeros((32, 32, 8), dtype=np.uint8)
    flat = arr.reshape(-1)
    flat[:n_lv_voxels] = 1
    nib.save(nib.Nifti1Image(arr, np.diag([*zooms, 1.0])), str(path))


@pytest.fixture
def ef_setup(tmp_path):
    """Build segs + a GT csv for 8 subjects with a clean EF spread.

    Predicted EF = (v_ed - v_es)/v_ed. We choose voxel counts so predicted EF tracks GT
    exactly (slope 1), then group the subjects into two pathologies.
    """
    seg_dir = tmp_path / "seg"
    seg_dir.mkdir()
    csv_path = tmp_path / "cardiac_phase.csv"

    # Column names must match the REAL cardiac_phase.csv schema that load_gt_ef reads.
    subjects, rows = [], ["subject,ED,ES,EF_pct,seg_flag"]
    for i in range(8):
        sid, ef = f"S{i:02d}", 30.0 + 5.0 * i          # GT EF 30..65 %
        v_ed = 1000
        v_es = int(round(v_ed * (1.0 - ef / 100.0)))
        _write_seg(seg_dir / f"{sid}_t00.nii.gz", v_ed)
        _write_seg(seg_dir / f"{sid}_t06.nii.gz", v_es)
        subjects.append((sid, 0, 6))
        rows.append(f"{sid},0,6,{ef},ok")
    csv_path.write_text("\n".join(rows) + "\n")
    return str(seg_dir), subjects, str(csv_path)


def test_overall_metrics_unchanged_without_groups(ef_setup):
    """Backward compatibility: the 3-arg call must behave exactly as before."""
    seg_dir, subjects, csv_path = ef_setup
    m = compute_ef_metrics(seg_dir, subjects, csv_path)
    assert m is not None
    assert m["n"] == 8 and m["n_skipped"] == 0
    assert m["slope"] == pytest.approx(1.0, abs=0.02)
    assert "by_group" not in m, "no groups passed ⇒ no by_group key"


def test_by_group_splits_and_keeps_overall(ef_setup):
    seg_dir, subjects, csv_path = ef_setup
    groups = {f"S{i:02d}": ("diseased" if i < 4 else "healthy") for i in range(8)}
    m = compute_ef_metrics(seg_dir, subjects, csv_path, groups=groups)

    assert m["n"] == 8                       # overall unchanged by grouping
    assert set(m["by_group"]) == {"diseased", "healthy"}
    assert m["by_group"]["diseased"]["n"] == 4
    assert m["by_group"]["healthy"]["n"] == 4
    for g in m["by_group"].values():
        assert g["slope"] == pytest.approx(1.0, abs=0.05)


def test_group_too_small_is_dropped_not_crashed(ef_setup):
    """A regression over <3 points is meaningless; that group must be omitted, and the
    overall number must still be reported."""
    seg_dir, subjects, csv_path = ef_setup
    groups = {f"S{i:02d}": ("rare" if i < 2 else "common") for i in range(8)}
    m = compute_ef_metrics(seg_dir, subjects, csv_path, groups=groups)
    assert "rare" not in m["by_group"]
    assert m["by_group"]["common"]["n"] == 6
    assert m["n"] == 8


def test_subjects_missing_from_groups_are_ungrouped_only(ef_setup):
    """A manifest that lacks a pathology label for some subjects must not drop them from
    the overall metric — `manifest.csv` has 1342/1343 populated, so this happens."""
    seg_dir, subjects, csv_path = ef_setup
    groups = {f"S{i:02d}": "diseased" for i in range(5)}   # 3 subjects unlabelled
    m = compute_ef_metrics(seg_dir, subjects, csv_path, groups=groups)
    assert m["n"] == 8
    assert set(m["by_group"]) == {"diseased"}
    assert m["by_group"]["diseased"]["n"] == 5


def test_group_slope_is_attenuated_by_a_narrow_gt_spread(ef_setup, tmp_path):
    """The REASON this split exists (docs/60 item 10): slope is a regression over the GT
    spread, so a group with a narrow spread is attenuated even with a good model. Val
    measures sigma_EF 16.2 diseased vs 6.2 healthy — pooling them lets the val health mix
    move the headline slope. Here: identical *relative* prediction error, different spread
    ⇒ different slope, which is exactly what a pooled number would hide."""
    seg_dir = tmp_path / "seg2"
    seg_dir.mkdir()
    csv_path = tmp_path / "cp2.csv"
    rows = ["subject,ED,ES,EF_pct,seg_flag"]
    subjects, groups = [], {}
    # wide: GT EF 20..70 (spread 50). narrow: GT EF 48..52 (spread 4).
    for i, ef in enumerate([20.0, 32.5, 45.0, 57.5, 70.0]):
        sid = f"W{i}"
        _mk_pair(seg_dir, sid, ef, shrink=0.5)     # predicted contraction halved
        subjects.append((sid, 0, 6)); rows.append(f"{sid},0,6,{ef},ok"); groups[sid] = "wide"
    for i, ef in enumerate([48.0, 49.0, 50.0, 51.0, 52.0]):
        sid = f"N{i}"
        _mk_pair(seg_dir, sid, ef, shrink=0.5)
        subjects.append((sid, 0, 6)); rows.append(f"{sid},0,6,{ef},ok"); groups[sid] = "narrow"
    csv_path.write_text("\n".join(rows) + "\n")

    m = compute_ef_metrics(str(seg_dir), subjects, str(csv_path), groups=groups)
    # Both groups have the same underlying model behaviour (contraction halved), so both
    # slopes are ~0.5 — the point is that they are reported SEPARATELY, so a shifting
    # val health mix cannot move a single pooled slope without it being visible.
    assert m["by_group"]["wide"]["slope"] == pytest.approx(0.5, abs=0.05)
    assert m["by_group"]["narrow"]["slope"] == pytest.approx(0.5, abs=0.05)
    assert m["by_group"]["wide"]["n"] == m["by_group"]["narrow"]["n"] == 5


def _mk_pair(seg_dir, sid, gt_ef, shrink=1.0):
    v_ed = 1000
    v_es = int(round(v_ed * (1.0 - (gt_ef * shrink) / 100.0)))
    _write_seg(seg_dir / f"{sid}_t00.nii.gz", v_ed)
    _write_seg(seg_dir / f"{sid}_t06.nii.gz", v_es)


def test_voxel_volume_comes_from_the_seg_header(tmp_path):
    """docs/59 F14: `_lv_ml` must read zooms from the seg, not a 12 mm constant — under
    native-z each subject has its own pitch. EF is a ratio so it cancels, but this pins
    that the reader is header-driven."""
    from ef_eval import _lv_ml
    p = tmp_path / "s.nii.gz"
    _write_seg(p, 1000, zooms=(1.4, 1.4, 10.0))
    expected = 1000 * (1.4 * 1.4 * 10.0) / 1000.0
    assert _lv_ml(str(p)) == pytest.approx(expected, rel=1e-4)
