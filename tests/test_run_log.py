"""Tests for the on-disk run log (`training/train_utils/run_log.py`).

These files are the ONLY numeric record of a run that does not require wandb, so the
properties that matter are: they append (never truncate), they survive a SLURM requeue
that replays steps, they survive being killed mid-write, and a resumed process cannot
silently change the CSV's shape. Each test below is written to FAIL if one of those
regresses — see the per-test docstrings.
"""

import csv
import json
import os

import pytest

from train_utils.run_log import RunLog, file_md5


def _lines(path):
    with open(path) as f:
        return [l for l in f.read().splitlines() if l.strip()]


def test_scalar_appends_and_survives_reopen(tmp_path):
    """A second RunLog on the same dir must APPEND, not truncate.

    This is the requeue case: the process dies, SLURM restarts it, a fresh RunLog is
    constructed against the same log_dir. If this ever opened with "w" the entire
    pre-requeue history would vanish silently.
    """
    rl = RunLog(str(tmp_path))
    rl.scalar("val/psnr/bbox_mean", 28.6, step=100)
    rl.scalar("val/psnr/bbox_mean", 29.1, step=200)

    rl2 = RunLog(str(tmp_path))          # simulates the post-requeue process
    rl2.scalar("val/psnr/bbox_mean", 29.4, step=300)

    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.SCALAR_FILE)]
    assert [r["step"] for r in recs] == [100, 200, 300]
    assert [r["value"] for r in recs] == [28.6, 29.1, 29.4]


def test_scalar_replayed_steps_are_both_kept(tmp_path):
    """Requeue replays steps between the last checkpoint and the kill.

    The writer must keep BOTH rows (the file is the evidence that a requeue happened);
    deduping is the reader's job. If the writer ever de-duplicated, a post-mortem could
    not tell a requeue from a clean run.
    """
    rl = RunLog(str(tmp_path))
    rl.scalar("train/loss/objective", 0.0201, step=50)
    RunLog(str(tmp_path)).scalar("train/loss/objective", 0.0198, step=50)   # replay

    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.SCALAR_FILE)]
    assert len(recs) == 2, "replayed step must not be silently dropped by the writer"
    assert recs[0]["value"] != recs[1]["value"]


def test_non_numeric_scalar_is_skipped_not_crashed(tmp_path):
    """`_log_scalar` is also reached with wandb.Image-like objects; those must be
    ignored without raising and without writing a junk row."""
    rl = RunLog(str(tmp_path))
    rl.scalar("media/panel", object(), step=1)
    rl.scalar("metric/ok", 1.5, step=1)

    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.SCALAR_FILE)]
    assert [r["name"] for r in recs] == ["metric/ok"]


def test_truncated_final_line_does_not_break_later_appends(tmp_path):
    """A SIGUSR1 landing mid-write leaves a partial final line.

    JSONL is chosen precisely so this costs one row rather than the whole file. The
    writer must still append cleanly afterwards, and every OTHER line must remain valid
    JSON. (The reader is what skips the bad line — see tools/load_run.py.)
    """
    rl = RunLog(str(tmp_path))
    rl.scalar("a", 1.0, step=1)
    with open(tmp_path / RunLog.SCALAR_FILE, "a") as f:
        f.write('{"step": 2, "name": "b", "val')        # killed mid-write

    RunLog(str(tmp_path)).scalar("c", 3.0, step=3)

    lines = _lines(tmp_path / RunLog.SCALAR_FILE)
    good = [l for l in lines if _is_json(l)]
    assert len(lines) == 3 and len(good) == 2
    assert json.loads(good[-1])["name"] == "c"


def _is_json(line):
    try:
        json.loads(line)
        return True
    except Exception:
        return False


def test_subject_csv_writes_header_once_and_appends(tmp_path):
    rl = RunLog(str(tmp_path))
    rl.subject_row({"epoch": 0, "seq_name": "s1", "D": 12, "dz_mm": 12.0, "metric_psnr": 30.0})
    rl.subject_row({"epoch": 0, "seq_name": "s2", "D": 8, "dz_mm": 10.0, "metric_psnr": 28.0})

    with open(tmp_path / RunLog.SUBJECT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["seq_name"] == "s1" and rows[1]["D"] == "8"
    assert _lines(tmp_path / RunLog.SUBJECT_FILE)[0].startswith("epoch,seq_name")


def test_subject_csv_identifying_columns_come_first(tmp_path):
    """Column order must be stable and readable, not dict-insertion order, so a human
    or agent opening the CSV sees who/what before the metric soup."""
    rl = RunLog(str(tmp_path))
    rl.subject_row({"metric_z": 1.0, "metric_a": 2.0, "D": 12, "seq_name": "s",
                    "epoch": 3, "t_target": 0, "dz_mm": 12.0, "S": 12, "step": 9,
                    "source": "ACDC"})
    header = _lines(tmp_path / RunLog.SUBJECT_FILE)[0].split(",")
    assert header[:8] == ["epoch", "step", "seq_name", "source", "t_target", "dz_mm", "D", "S"]
    assert header[8:] == ["metric_a", "metric_z"], "trailing metrics should be sorted"


def test_subject_csv_widens_for_a_new_metric(tmp_path):
    """THE REGRESSION THIS FILE EXISTS FOR.

    Several metrics are per-sample CONDITIONAL (the heart-ROI family needs a valid
    `heart_roi_canonical`; `recov_frac_heart` needs a non-degenerate oracle span). Freezing
    the header on the first row meant that if val subject 0 happened to lack one, that
    column was silently dropped for the ENTIRE run — including the headline heartseg
    metrics. The header must GROW instead, and existing rows must keep their values
    aligned (blank in the new column), never shift.
    """
    rl = RunLog(str(tmp_path))
    rl.subject_row({"epoch": 0, "seq_name": "s1", "metric_a": 1.0})
    rl.subject_row({"epoch": 0, "seq_name": "s2", "metric_a": 2.0, "metric_NEW": 9.9})

    with open(tmp_path / RunLog.SUBJECT_FILE, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["epoch", "seq_name", "metric_a", "metric_NEW"]
        rows = list(reader)
    assert len(rows) == 2
    assert rows[0]["metric_a"] == "1.0" and rows[0]["metric_NEW"] == ""   # widened, blank
    assert rows[1]["metric_a"] == "2.0" and rows[1]["metric_NEW"] == "9.9"


def test_subject_csv_widens_across_a_resume(tmp_path):
    """Same, but the new metric first appears in a LATER process (requeue)."""
    RunLog(str(tmp_path)).subject_row({"epoch": 0, "seq_name": "s1", "metric_a": 1.0})
    RunLog(str(tmp_path)).subject_row(
        {"epoch": 1, "seq_name": "s2", "metric_a": 2.0, "metric_heartseg": 30.0})

    with open(tmp_path / RunLog.SUBJECT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["metric_heartseg"] == ""
    assert rows[1]["metric_heartseg"] == "30.0"
    assert rows[1]["metric_a"] == "2.0", "widening must not shift existing values"


def test_subject_csv_missing_key_on_resume_leaves_blank_not_shifted(tmp_path):
    """The mirror image: a resumed process computing FEWER metrics must leave the
    column blank rather than shifting subsequent values left."""
    RunLog(str(tmp_path)).subject_row({"epoch": 0, "seq_name": "s1",
                                       "metric_a": 1.0, "metric_b": 2.0})
    RunLog(str(tmp_path)).subject_row({"epoch": 1, "seq_name": "s2", "metric_b": 5.0})

    with open(tmp_path / RunLog.SUBJECT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[1]["metric_a"] == ""
    assert rows[1]["metric_b"] == "5.0"


def test_meta_appends_one_line_per_launch_with_git(tmp_path):
    """A requeued run must leave one meta line per SEGMENT, so a mid-run code edit is
    visible as a changed sha rather than being overwritten."""
    RunLog(str(tmp_path)).meta({"launch": 0, "resumed_from_epoch": None})
    RunLog(str(tmp_path)).meta({"launch": 1, "resumed_from_epoch": 42})

    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.META_FILE)]
    assert [r["launch"] for r in recs] == [0, 1]
    assert recs[1]["resumed_from_epoch"] == 42
    assert "git" in recs[0] and set(recs[0]["git"]) == {"sha", "dirty"}


def test_meta_survives_unserialisable_payload(tmp_path):
    """Hydra configs carry objects json cannot encode; `default=str` must absorb that
    rather than losing the whole meta record."""
    RunLog(str(tmp_path)).meta({"cfg": {"obj": object()}, "n": 1})
    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.META_FILE)]
    assert recs[0]["n"] == 1


def test_disabled_when_log_dir_unusable(tmp_path):
    """A bad log_dir must disable the logger, not raise — training outranks logging."""
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    rl = RunLog(str(blocker / "sub"))       # parent is a file → mkdir fails
    assert rl.enabled is False
    rl.scalar("a", 1.0, 1)                  # must not raise
    rl.subject_row({"epoch": 0})
    rl.meta({"a": 1})


def test_file_md5_matches_hashlib(tmp_path):
    p = tmp_path / "split.txt"
    p.write_text("[train]\nsubjA\nsubjB\n")
    import hashlib
    assert file_md5(str(p)) == hashlib.md5(p.read_bytes()).hexdigest()
    assert file_md5(str(tmp_path / "nope.txt")) is None


def test_subject_csv_truncated_row_does_not_corrupt_next(tmp_path):
    """Same mid-write-kill hazard as the JSONL, for the CSV.

    Without the newline repair, the resumed process's first row concatenates onto the
    partial one, so DictReader silently mis-parses a row that LOOKS fine — the worst
    possible failure for a file whose whole purpose is post-hoc analysis.
    """
    RunLog(str(tmp_path)).subject_row({"epoch": 0, "seq_name": "s1", "metric_a": 1.0})
    with open(tmp_path / RunLog.SUBJECT_FILE, "a") as f:
        f.write("1,s2,2.")                       # killed mid-row

    RunLog(str(tmp_path)).subject_row({"epoch": 2, "seq_name": "s3", "metric_a": 3.0})

    with open(tmp_path / RunLog.SUBJECT_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["seq_name"] == "s3", "the good row must survive the truncated one"
    assert rows[-1]["metric_a"] == "3.0"


def test_nonfinite_scalar_is_written_as_json_null(tmp_path):
    """`json.dumps(nan)` emits a bare `NaN` token — valid for Python, rejected by jq, JS,
    Go and pandas.read_json. One diverged metric would make the whole file unparseable by
    everything except this repo, so non-finite values are stored as JSON null."""
    rl = RunLog(str(tmp_path))
    rl.scalar("a", float("nan"), 1)
    rl.scalar("b", float("inf"), 2)
    rl.scalar("c", 3.5, 3)

    text = (tmp_path / RunLog.SCALAR_FILE).read_text()
    assert "NaN" not in text and "Infinity" not in text
    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.SCALAR_FILE)]   # must not raise
    assert [r["value"] for r in recs] == [None, None, 3.5]
    assert [r["name"] for r in recs] == ["a", "b", "c"]   # rows are kept, not dropped


def test_scalar_records_carry_epoch_and_wall_clock(tmp_path):
    """`step` spans two counters (val scalars use steps["val"], val panels the train step),
    so `epoch` is the join key; `t` gives the wall-clock axis for post-mortems."""
    RunLog(str(tmp_path)).scalar("m", 1.0, step=7, epoch=3)
    rec = json.loads(_lines(tmp_path / RunLog.SCALAR_FILE)[0])
    assert rec["epoch"] == 3 and rec["step"] == 7 and rec["t"] > 0


def test_one_transient_failure_does_not_disable_logging(tmp_path, monkeypatch):
    """A single GPFS hiccup must not silence the run's whole log; only a sustained
    failure streak should, and a success must re-arm."""
    rl = RunLog(str(tmp_path))
    real = rl._append_line
    calls = {"n": 0}

    def flaky(filename, text):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("transient EIO")
        return real(filename, text)

    monkeypatch.setattr(rl, "_append_line", flaky)
    rl.scalar("a", 1.0, 1)                 # fails
    assert rl.enabled is True, "one failure must not disable logging"
    rl.scalar("b", 2.0, 2)                 # succeeds, re-arms
    assert rl._fail_streak == 0
    assert json.loads(_lines(tmp_path / RunLog.SCALAR_FILE)[0])["name"] == "b"


def test_sustained_failures_eventually_disable(tmp_path, monkeypatch):
    rl = RunLog(str(tmp_path))
    monkeypatch.setattr(rl, "_append_line",
                        lambda *a: (_ for _ in ()).throw(OSError("disk gone")))
    for i in range(RunLog.MAX_CONSECUTIVE_FAILURES):
        rl.scalar("a", 1.0, i)
    assert rl.enabled is False


def test_exit_record_is_distinguishable_from_a_launch(tmp_path):
    """Without an exit line, a crashed run and a completed one look identical on disk —
    you have to guess from whether a `.tmp` checkpoint was left behind. Readers must be
    able to tell the two record kinds apart, since an exit line carries no provenance."""
    rl = RunLog(str(tmp_path))
    rl.meta({"event": "launch", "split_md5": "abc", "resumed_from_epoch": None})
    rl.meta({"event": "exit", "status": "error", "error": "CUDA OOM", "final_epoch": 12})

    recs = [json.loads(l) for l in _lines(tmp_path / RunLog.META_FILE)]
    launches = [r for r in recs if r.get("event") != "exit"]
    exits = [r for r in recs if r.get("event") == "exit"]
    assert len(launches) == 1 and len(exits) == 1
    assert launches[-1]["split_md5"] == "abc", "provenance must come from the LAUNCH line"
    assert exits[-1]["status"] == "error" and exits[-1]["final_epoch"] == 12
