"""Tests for the val-logging helpers added in docs/60.

WHY THIS FILE EXISTS. The first end-to-end run of this feature wrote `subject="sax"` and
`source="sax"` for all 266 val rows, and picked ONE visual subject instead of one per
source. Cause: `MRIDataset._find_subjects` builds `<data_root>/<split-file line>/sax`, so
`os.path.basename(path)` is always the literal "sax" — the id is the PARENT directory.
The existing code knew this (`_save_ef_volume` uses `basename(dirname(...))`); the new
helpers did not, and no test encoded the real layout, so nothing failed.

Every test here therefore uses REAL-SHAPED paths, including the trailing "/sax".
"""

import pytest

from train_utils.val_logging import (
    pick_one_index_per_source,
    pick_planes,
    resp_offslab_stats,
    seq_index_to_subject,
    subject_id,
    subject_source,
)

# Real layout: <data_root>/<split-file line>/sax
ACDC = "/root/ACDC_sax/ACDC_patient006/sax"
MNMS = "/root/MNMs_sax/MNMs_A9C5P4/sax"
CMRX23 = "/root/CMRxRecon2023/Cine_combined/CMRx23_Train_P069/sax"
CMRX24 = "/root/CMRxRecon2024/Cine_combined/CMRx24_Train_P011/sax"
CMRX25 = "/root/CMRxRecon2025/Cine_combined/CMRx25_train_Center001_UIH_30T_umr780_P005/sax"


@pytest.mark.parametrize("path,expected", [
    (ACDC, "ACDC_patient006"),
    (MNMS, "MNMs_A9C5P4"),
    (CMRX23, "CMRx23_Train_P069"),
    (CMRX25, "CMRx25_train_Center001_UIH_30T_umr780_P005"),
])
def test_subject_id_strips_the_sax_leaf(path, expected):
    """The id must be the parent dir, never the literal "sax" leaf."""
    assert subject_id(path) == expected
    assert subject_id(path) != "sax"


def test_subject_id_matches_manifest_ids():
    """The id is used to join `manifest.csv` (column `id`), so it must equal that format —
    otherwise pathology grouping and the z-coverage summary silently find nothing."""
    assert subject_id(ACDC) == "ACDC_patient006"      # manifest id format
    assert subject_id(CMRX24) == "CMRx24_Train_P011"


@pytest.mark.parametrize("path,expected", [
    (ACDC, "ACDC"), (MNMS, "MNMs"), (CMRX23, "CMRx23"),
    (CMRX24, "CMRx24"), (CMRX25, "CMRx25"),
])
def test_subject_source_distinguishes_all_five_sources(path, expected):
    assert subject_source(path) == expected


def test_pick_one_index_per_source_spreads_over_sources():
    """THE REGRESSION: the pooled val split is sorted, so the old fixed indices
    (0, 7, 14, 21) were 3 ACDC + 1 CMRx2023 — zero M&Ms, the largest val source."""
    subjects = ([ACDC] * 15) + ([CMRX23] * 19) + ([MNMS] * 33) + ([CMRX24] * 29)
    picks = pick_one_index_per_source(subjects)
    assert picks == (0, 15, 34, 67)
    assert [subject_source(subjects[i]) for i in picks] == ["ACDC", "CMRx23", "MNMs", "CMRx24"]


def test_pick_one_index_per_source_is_deterministic():
    """Panels must track the SAME subjects across epochs and runs to be comparable."""
    subjects = [ACDC, MNMS, CMRX23, MNMS, ACDC]
    assert pick_one_index_per_source(subjects) == pick_one_index_per_source(subjects)


def test_pick_one_index_per_source_respects_max():
    subjects = [ACDC, MNMS, CMRX23, CMRX24, CMRX25]
    assert len(pick_one_index_per_source(subjects, max_picks=3)) == 3


@pytest.mark.parametrize("D", [5, 6, 8, 11, 12, 18, 21])
def test_pick_planes_always_includes_apex_and_base(D):
    """The old `mid±2` window never showed apex plane 0 at D=6 — the exact plane docs/59
    F1 was about — and covered a D-dependent 83%->28% of the stack."""
    planes = pick_planes(D, 5)
    assert planes[0] == 0, f"apex plane missing at D={D}"
    assert planes[-1] == D - 1, f"base plane missing at D={D}"
    assert all(0 <= p < D for p in planes)
    assert planes == sorted(planes)


def test_pick_planes_is_not_centred_on_the_reference_slot():
    """`D//2` IS the reference slot (mri_dataset sets z_mid=(bbox_z0+bbox_z1)//2 and z is
    never padded, so bbox=[0,D)). A window centred there shows the free plane."""
    for D in (12, 18, 21):
        planes = pick_planes(D, 5)
        assert planes != [D // 2 + off for off in (-2, -1, 0, 1, 2)]


def test_pick_planes_fixed_length_even_for_tiny_stacks():
    """Panel geometry is a fixed n columns; duplicates are acceptable, a short list is not
    (it would misalign the GIF tiler, which reshapes on n)."""
    for D in (1, 2, 3, 5, 21):
        assert len(pick_planes(D, 5)) == 5
    assert len(pick_planes(12, 3)) == 3


def test_seq_index_to_subject_uses_val_targets_when_sweeping():
    """Under ef_val_sweep the sweep is 2N long and blocked [all ED]+[all ES], so seq_index
    N maps back to subject 0 — not to subject N."""
    class DS:
        subjects = [ACDC, MNMS, CMRX23]
        val_targets = [(0, 0), (1, 0), (2, 0), (0, 7), (1, 7), (2, 7)]

    assert seq_index_to_subject(DS(), 0) == ("ACDC_patient006", "ACDC")
    assert seq_index_to_subject(DS(), 3) == ("ACDC_patient006", "ACDC")   # ES half, subj 0
    assert seq_index_to_subject(DS(), 4) == ("MNMs_A9C5P4", "MNMs")


def test_seq_index_to_subject_without_sweep_wraps_on_subjects():
    class DS:
        subjects = [ACDC, MNMS]
        val_targets = None

    assert seq_index_to_subject(DS(), 0)[1] == "ACDC"
    assert seq_index_to_subject(DS(), 3)[1] == "MNMs"      # wraps


def test_seq_index_to_subject_degrades_quietly():
    """A malformed dataset must yield (None, None), never raise into the val loop."""
    class Broken:
        subjects = []
        val_targets = None

    assert seq_index_to_subject(Broken(), 0) == (None, None)
    assert seq_index_to_subject(None, 0) == (None, None)


def test_pitch_bucket_splits_at_10mm():
    from trainer import Trainer
    assert Trainer._pitch_bucket(12.0) == "coarse_ge10mm"
    assert Trainer._pitch_bucket(10.0) == "coarse_ge10mm"
    assert Trainer._pitch_bucket(9.6) == "fine_lt10mm"
    assert Trainer._pitch_bucket(5.0) == "fine_lt10mm"
    assert Trainer._pitch_bucket(None) is None


# ── Non-finite guard (docs/60 item 4) ────────────────────────────────────────
# The guard must suppress LOGGING ONLY. Training control flow (backward / skip) lives in
# _run_steps_on_batch_chunks and must be unaffected, so these tests pin both halves:
# a finite batch behaves exactly as before, and a NaN batch is counted, named, and kept
# out of the AverageMeters — but the step counter still advances either way.

def _finite_stub():
    from types import SimpleNamespace
    from collections import defaultdict
    from trainer import Trainer
    stub = SimpleNamespace()
    stub.steps = {"train": 7, "val": 0}
    stub._nonfinite_logged = defaultdict(int)
    stub._logged = []
    stub._log_scalar = lambda k, v, s: stub._logged.append((k, v, s))
    stub._log_if_finite = Trainer._log_if_finite.__get__(stub)
    return stub


def test_finite_objective_passes_through_untouched():
    import torch
    stub = _finite_stub()
    assert stub._log_if_finite({"objective": torch.tensor(0.02)}, "train") is True
    assert stub._logged == [], "a healthy batch must not emit a nonfinite scalar"
    assert stub._nonfinite_logged["train"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_objective_is_blocked_and_counted(bad):
    import torch
    stub = _finite_stub()
    ok = stub._log_if_finite(
        {"objective": torch.tensor(bad), "seq_name": ["CMRx24_Train_P011"]}, "train")
    assert ok is False
    assert stub._nonfinite_logged["train"] == 1
    assert stub._logged and stub._logged[0][0] == "train/optim/nonfinite_logged_cumulative"


def test_nonfinite_guard_covers_val_too():
    """Val had no finiteness check at all before docs/60, and NaN is a value rather than
    an exception, so it flowed through every surrounding try/except silently."""
    import torch
    stub = _finite_stub()
    assert stub._log_if_finite({"objective": torch.tensor(float("nan"))}, "val") is False
    assert stub._nonfinite_logged["val"] == 1


def test_guard_never_raises_on_a_malformed_batch():
    """If the guard itself breaks it must fail OPEN (log anyway), never take down a run."""
    stub = _finite_stub()
    assert stub._log_if_finite({"objective": object()}, "train") is True
    assert stub._log_if_finite({}, "train") is True          # no objective key at all


def test_resp_scalars_are_written_without_wandb():
    """docs/60: the off-slab family is the ONE thing unrecoverable after a run (applied on
    GPU, never persisted), so it must reach disk even with no wandb writer. It used to sit
    behind `if not self.wandb_writer: return`. A WANDB_MODE=offline run still builds a
    writer, so only this test covers the no-writer path."""
    import torch
    from types import SimpleNamespace
    from trainer import Trainer

    stub = SimpleNamespace()
    stub.wandb_writer = None                       # the case that used to silently no-op
    stub.written = []
    stub._log_scalar = lambda k, v, s: stub.written.append(k)
    stub._log_resp_disp_scalar = Trainer._log_resp_disp_scalar.__get__(stub)

    S, D = 4, 8
    batch = {
        "resp_disp_mm": torch.zeros(1, S, 3),
        "slice_indices": torch.arange(S, dtype=torch.float32).reshape(1, S),
        "dz_mm": torch.tensor([[10.0]]),
        "phases": torch.zeros(1, 12, D, 8, 8),
    }
    batch["resp_disp_mm"][0, 0, 0] = 100.0          # slot 0 driven far off the slab
    stub._log_resp_disp_scalar(batch, step=0, prefix="train")

    assert any("frac_slots_offslab" in k for k in stub.written), stub.written
    assert any("disp_frac_of_extent" in k for k in stub.written)


def test_resp_offslab_counts_slots_that_leave_the_slab():
    """Landing plane = z_i + d/dz; outside [0, D-1] is zero-padded. With 1 of 4 slots
    driven off, the fraction must be 0.25 — not 0, which is what a broken sign or a
    wrong dz would give."""
    import torch
    from types import SimpleNamespace
    from trainer import Trainer

    stub = SimpleNamespace()
    stub.wandb_writer = None
    stub.vals = {}
    stub._log_scalar = lambda k, v, s: stub.vals.__setitem__(k.split("/")[-1], v)
    stub._log_resp_disp_scalar = Trainer._log_resp_disp_scalar.__get__(stub)

    S, D = 4, 8
    batch = {
        "resp_disp_mm": torch.zeros(1, S, 3),
        "slice_indices": torch.arange(S, dtype=torch.float32).reshape(1, S),
        "dz_mm": torch.tensor([[10.0]]),
        "phases": torch.zeros(1, 12, D, 8, 8),
    }
    batch["resp_disp_mm"][0, 3, 0] = 100.0          # slot 3 (z=3) -> plane 13 > D-1=7
    stub._log_resp_disp_scalar(batch, step=0, prefix="val")
    assert stub.vals["frac_slots_offslab"] == pytest.approx(0.25)

    batch["resp_disp_mm"].zero_()                   # nobody leaves
    stub._log_resp_disp_scalar(batch, step=0, prefix="val")
    assert stub.vals["frac_slots_offslab"] == pytest.approx(0.0)


def test_resp_offslab_dimmed_has_no_false_floor_at_zero_displacement():
    """Slots landing inside [0, D-1] interpolate between two REAL planes — trilinear mixes
    in no zero-padding — so they must not count as dimmed. The pre-2026-08-01 band counted
    them and gave every subject a spurious floor even with breathing effectively off."""
    import torch
    S, D = 8, 8
    batch = {
        "resp_disp_mm": torch.zeros(1, S, 3),
        "slice_indices": torch.arange(S, dtype=torch.float32).reshape(1, S),
        "dz_mm": torch.tensor([[10.0]]),
        "phases": torch.zeros(1, 12, D, 4, 4),
    }
    st = resp_offslab_stats(batch, 0)
    assert st["frac_slots_offslab"] == pytest.approx(0.0)
    assert st["frac_slots_dimmed"] == pytest.approx(0.0), "exact end planes are not dimmed"


def test_resp_offslab_counts_a_partial_plane_as_dimmed():
    """Only a landing OUTSIDE [0, D-1] but within one plane of it is partially attenuated:
    one bracket is real, the other is zero-padding. A landing of 0.5 is NOT — it sits
    between planes 0 and 1, both real, and measurably retains 1.0000 (docs/62 §5.3)."""
    import torch
    S, D = 4, 8
    batch = {
        "resp_disp_mm": torch.zeros(1, S, 3),
        "slice_indices": torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
        "dz_mm": torch.tensor([[10.0]]),
        "phases": torch.zeros(1, 12, D, 4, 4),
    }
    batch["resp_disp_mm"][0, 0, 0] = 5.0        # plane 0 -> 0.5: between two REAL planes
    st = resp_offslab_stats(batch, 0)
    assert st["frac_slots_dimmed"] == pytest.approx(0.0)
    assert st["frac_slots_offslab"] == pytest.approx(0.0)

    batch["resp_disp_mm"][0, 0, 0] = -5.0       # plane 0 -> -0.5: half off the near edge
    st = resp_offslab_stats(batch, 0)
    assert st["frac_slots_dimmed"] == pytest.approx(0.25)
    assert st["frac_slots_offslab"] == pytest.approx(0.25), "dimmed is a subset of offslab"

    batch["resp_disp_mm"][0, 0, 0] = -15.0      # plane 0 -> -1.5: fully blank, not dimmed
    st = resp_offslab_stats(batch, 0)
    assert st["frac_slots_dimmed"] == pytest.approx(0.0)
    assert st["frac_slots_offslab"] == pytest.approx(0.25)


def test_resp_offslab_frac_of_extent_uses_this_subjects_geometry():
    """The damaged fraction is d/((D-1)*dz) — the whole point is that it is far larger for
    a short stack than a long one at the same absolute displacement."""
    import torch

    def stats(D, dz):
        batch = {
            "resp_disp_mm": torch.full((1, 4, 3), 0.0),
            "slice_indices": torch.zeros(1, 4),
            "dz_mm": torch.tensor([[float(dz)]]),
            "phases": torch.zeros(1, 12, D, 4, 4),
        }
        batch["resp_disp_mm"][0, :, 0] = 6.0
        return resp_offslab_stats(batch, 0)["disp_frac_of_extent"]

    short = stats(D=5, dz=8.0)      # 32 mm stack
    long_ = stats(D=12, dz=12.0)    # 132 mm stack
    assert short == pytest.approx(6.0 / 32.0)
    assert long_ == pytest.approx(6.0 / 132.0)
    assert short > 4 * long_


def test_resp_offslab_returns_empty_when_breathing_is_off():
    assert resp_offslab_stats({}, 0) == {}
