"""On-disk run log — a structured mirror of everything we send to wandb.

WHY THIS EXISTS. Until now the only machine-readable numeric artifact a finished run
left behind was `baseline_identity.json` (822 bytes). Every other number lived either
in wandb (needs network + auth + run-id discovery, and `run.history()` downsamples to
500 points by default) or in `log.txt` as prose — `Loss/train_metric_mae_3d_full:
0.0118 (0.0118)` — parseable only by regex, and mixing instantaneous with
running-average values. Analysing a finished run therefore meant hitting the wandb API
or re-running it.

This module writes three append-only files into `log_dir` so that ALL numeric analysis
of a run can be done from disk alone:

    run_meta.jsonl        one line per PROCESS LAUNCH (not per run) — git sha, config,
                          split/manifest hashes, cohort sizes, wandb id, SLURM job/node.
                          A requeued run appends a line per segment, so a mid-run code
                          edit shows up as a changed sha between segments.
    metrics.jsonl         one line per scalar, mirroring `Trainer._log_scalar` — which is
                          the single chokepoint for scalars, so nothing scalar escapes.
    val_per_subject.csv   one row per val sample: subject id, D, dz, t_target + every
                          metric. This is NEW information — it exists nowhere else, not
                          even in wandb, because batch_size is pinned to 1 and the
                          per-subject value is averaged away by the AverageMeter.

NOT covered (deliberately): the 8 `wandb.Image`/`wandb.Video` panels. Those are figures,
not numbers; they stay wandb-only and are regenerable from a checkpoint via `tools/render_*`.

RESUME SEMANTICS. `steps` is checkpointed and restored, and checkpoints are written at
epoch boundaries — so on SLURM requeue every step between the last checkpoint and the
kill is REPLAYED. Those rows are appended a second time with the same
`(phase, step, name)` but a different value (different data order / aug draw). We keep
the raw file faithful and dedupe in the READER (`tools/load_run.py`), because truncating
the file back to the resume point would destroy the evidence that a requeue happened.

A SIGUSR1 can also land mid-`write()`, leaving a partial final line. That is why these
are JSONL and not one big JSON array: a truncated array is wholly unparseable, a
truncated JSONL loses exactly one row. Every line is flushed as it is written so at most
one row is ever lost.

FAILURE ISOLATION. Every public method swallows its own exceptions — diagnostics must
never raise into training. A single transient I/O error is survivable; logging is only
disabled after MAX_CONSECUTIVE_FAILURES in a row, and re-arms on the first success.
"""

import csv
import json
import logging
import math
import os
import subprocess
import time


class RunLog:
    """Append-only structured logger for one training run. See module docstring."""

    META_FILE = "run_meta.jsonl"
    SCALAR_FILE = "metrics.jsonl"
    SUBJECT_FILE = "val_per_subject.csv"
    MAX_CONSECUTIVE_FAILURES = 10

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.enabled = True
        # Column order for the per-subject CSV: seeded from the first row (or the existing
        # header on resume) and WIDENED as new metrics appear — see `subject_row`.
        self._subject_fields = None
        self._fail_streak = 0
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception as e:
            logging.warning(f"RunLog: cannot create {log_dir} ({e}); disabling.")
            self.enabled = False

    # ── paths ────────────────────────────────────────────────────────────
    def _path(self, name):
        return os.path.join(self.log_dir, name)

    def _write(self, filename, text):
        """`_append_line` with failure accounting.

        A single transient GPFS hiccup must not silence logging for the remaining days of
        a run, so one failure is not fatal; but a genuinely broken filesystem must not
        raise (and re-warn) on every step either. Disable only after several CONSECUTIVE
        failures, and re-arm on the first success.
        """
        try:
            self._append_line(filename, text)
            self._fail_streak = 0
        except Exception as e:
            self._fail_streak += 1
            logging.warning(f"RunLog write to {filename} failed (ignored): {e}")
            if self._fail_streak >= self.MAX_CONSECUTIVE_FAILURES:
                logging.error(f"RunLog: {self._fail_streak} consecutive write failures; "
                              "disabling on-disk logging for the rest of this run.")
                self.enabled = False

    def _append_line(self, filename, text):
        """Append one already-serialised line, flushing so a kill loses at most this row.

        Repairs a missing trailing newline first. A process killed mid-`write()` leaves a
        partial line with no "\\n", and a naive append would CONCATENATE onto it —
        corrupting the next row as well as the truncated one, i.e. losing two rows where
        the format is supposed to cost one. Writing a separator first confines the damage
        to the single partial line.
        """
        path = self._path(filename)
        with open(path, "a") as f:
            if _needs_newline(path):
                f.write("\n")
            f.write(text + "\n")
            f.flush()

    # ── public API ───────────────────────────────────────────────────────
    def meta(self, payload):
        """Append one line describing THIS process launch."""
        if not self.enabled:
            return
        try:
            record = {"git": _git_state(), **payload}
            self._append_line(self.META_FILE, json.dumps(record, default=str))
        except Exception as e:
            logging.warning(f"RunLog.meta failed (ignored): {e}")

    def scalar(self, name, value, step, epoch=None):
        """Mirror one wandb scalar. Non-numeric values (images, tables) are skipped.

        `epoch` and `t` matter for reading this back. The trainer logs in TWO step spaces
        — val scalars use `steps["val"]` while the val-epoch panels use the train step —
        so `step` alone does not align train and val series. Epoch does. `t` gives a
        wall-clock axis, which is how you see where a run slowed down or died.
        """
        if not self.enabled:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return          # not a scalar — wandb.Image and friends land here
        # NaN/Inf → JSON null. `json.dumps` would emit a bare `NaN`/`Infinity` token, which
        # Python reads back but jq, JS, Go and pandas.read_json all reject — one diverged
        # metric would make the whole file unparseable by everything except this repo.
        # Keeping the row (rather than dropping it) preserves the fact that it was logged.
        rec = {"t": round(time.time(), 3), "step": int(step), "name": name,
               "value": v if math.isfinite(v) else None}
        if epoch is not None:
            rec["epoch"] = int(epoch)
        self._write(self.SCALAR_FILE, json.dumps(rec))

    def subject_row(self, row):
        """Append one val sample's metrics. `row` is a flat dict of scalars.

        Columns GROW to fit: several metrics are per-sample conditional (the heart-ROI
        family needs a valid `heart_roi_canonical`; `recov_frac_heart` needs a non-degenerate
        oracle span), so the first row is NOT a reliable schema. Freezing on it meant that if
        val subject 0 happened to lack one, that column was dropped for the entire run —
        silently, including the headline heartseg metrics. Widening rewrites the file with
        the union header; it is rare (the set stabilises within one val epoch) and the file
        is small.
        """
        if not self.enabled:
            return
        try:
            if self._subject_fields is None:
                self._subject_fields = self._resolve_fields(row)
            new_keys = [k for k in row if k not in self._subject_fields]
            if new_keys:
                self._widen(new_keys)
            path = self._path(self.SUBJECT_FILE)
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                if _needs_newline(path):
                    f.write("\n")       # same mid-write-kill repair as _append_line
                w = csv.DictWriter(f, fieldnames=self._subject_fields, extrasaction="ignore")
                if new_file:
                    w.writeheader()
                w.writerow(row)
                f.flush()
            self._fail_streak = 0
        except Exception as e:
            self._fail_streak += 1
            logging.warning(f"RunLog.subject_row failed (ignored): {e}")
            if self._fail_streak >= self.MAX_CONSECUTIVE_FAILURES:
                self.enabled = False

    def _widen(self, new_keys):
        """Add columns, rewriting existing rows under the union header."""
        path = self._path(self.SUBJECT_FILE)
        self._subject_fields = self._subject_fields + sorted(new_keys)
        logging.info(f"RunLog: {self.SUBJECT_FILE} gained columns {sorted(new_keys)}")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            return
        with open(path, newline="") as f:
            old_rows = list(csv.DictReader(f))
        tmp = path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._subject_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(old_rows)
        os.replace(tmp, path)           # atomic: a crash mid-rewrite leaves the old file

    # ── helpers ──────────────────────────────────────────────────────────
    def _resolve_fields(self, row):
        """Column order: the existing header if the file already has one (resume), else
        the identifying columns first and the rest sorted (stable across launches)."""
        path = self._path(self.SUBJECT_FILE)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, newline="") as f:
                    header = next(csv.reader(f))
                if header:
                    return header
            except Exception:
                pass        # unreadable header → fall through and define a fresh one
        lead = [k for k in ("epoch", "step", "seq_name", "source", "t_target",
                            "dz_mm", "D", "S") if k in row]
        return lead + sorted(k for k in row if k not in lead)


def _needs_newline(path):
    """True when `path` exists, is non-empty, and does NOT end in a newline — i.e. a
    previous process was killed mid-write. See `RunLog._append_line`."""
    try:
        if os.path.getsize(path) == 0:
            return False
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            return f.read(1) != b"\n"
    except (OSError, ValueError):
        return False


def _git_state():
    """Repo sha + dirty flag. Returns a dict with nulls if git is unavailable."""
    def _run(args):
        return subprocess.run(
            args, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            capture_output=True, text=True, timeout=10,
        )
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        dirty = _run(["git", "status", "--porcelain"])
        return {
            "sha": sha.stdout.strip() or None,
            "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        }
    except Exception:
        return {"sha": None, "dirty": None}


def file_md5(path):
    """md5 of a file, for recording WHICH split/manifest a run used. None on any failure.

    Hashes the whole file — the callers are the split (~50 KB) and manifest (~1.3 MB), and
    a partial hash would silently claim two different cohorts were the same.
    """
    import hashlib
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None
