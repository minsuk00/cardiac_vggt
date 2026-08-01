#!/usr/bin/env python
"""Load a training run's on-disk logs (docs/60) — the wandb-free way to analyse a run.

    from tools.load_run import load_run
    meta, scalars, subjects = load_run("scratch/logs/<exp_dir>")

    scalars                      # tidy: step, name, value  (deduped)
    scalars.pivot_table(index="step", columns="name", values="value")
    subjects.groupby("source")["metric_psnr_3d_bbox"].mean()

Or from the shell, for a quick look:

    python tools/load_run.py scratch/logs/<exp_dir>

WHY DEDUPE. `steps` is checkpointed and restored at epoch boundaries, so a SLURM requeue
REPLAYS every step between the last checkpoint and the kill. The writer keeps both rows on
purpose (the duplicates are the evidence a requeue happened); this reader keeps the LAST
occurrence of each (name, step), which is the one that actually continued the run.
Pass `dedupe=False` to see the raw record.
"""

import argparse
import json
import os
import sys

import pandas as pd

SCALAR_FILE = "metrics.jsonl"
SUBJECT_FILE = "val_per_subject.csv"
META_FILE = "run_meta.jsonl"
BASELINE_FILE = "baseline_identity.json"


def _read_jsonl(path):
    """Parse a JSONL file, skipping unparseable lines (a kill mid-write leaves one)."""
    rows, bad = [], 0
    if not os.path.exists(path):
        return rows, bad
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def load_run(log_dir, dedupe=True):
    """Return (meta, scalars_df, subjects_df) for one run directory.

    meta      list of dicts, one per process launch (oldest first)
    scalars   DataFrame [step, name, value] — empty DataFrame if the file is absent
    subjects  DataFrame of per-subject val rows — empty DataFrame if absent
    """
    meta, bad_meta = _read_jsonl(os.path.join(log_dir, META_FILE))
    scalar_rows, bad_scalars = _read_jsonl(os.path.join(log_dir, SCALAR_FILE))
    if bad_meta or bad_scalars:
        print(f"[load_run] skipped {bad_meta + bad_scalars} unparseable line(s) "
              "(expected after a mid-write kill)", file=sys.stderr)

    # NOT `columns=[...]` — on a list of dicts that SELECTS rather than extends, which
    # silently dropped `epoch` and `t`, the two fields the writer records precisely because
    # `step` spans two counters (val scalars use steps["val"], val panels use the train step).
    scalars = pd.DataFrame(scalar_rows)
    if scalars.empty:
        scalars = pd.DataFrame(columns=["t", "step", "epoch", "name", "value"])
    if dedupe and not scalars.empty:
        scalars = scalars.drop_duplicates(subset=["name", "step"], keep="last")
        scalars = scalars.sort_values(["name", "step"]).reset_index(drop=True)

    subj_path = os.path.join(log_dir, SUBJECT_FILE)
    subjects = pd.DataFrame()
    if os.path.exists(subj_path):
        # A truncated final row can have the wrong field count; skip just that row.
        subjects = pd.read_csv(subj_path, on_bad_lines="skip")
        if dedupe and {"epoch", "seq_name", "t_target"} <= set(subjects.columns):
            subjects = subjects.drop_duplicates(
                subset=["epoch", "seq_name", "t_target"], keep="last").reset_index(drop=True)
    return meta, scalars, subjects


def load_identity_baseline(log_dir):
    """Per-subject identity floors as a DataFrame, for normalising per-subject metrics.

    Raw per-subject PSNR is NOT comparable across this cohort — the achievable ceiling
    moves with D, dz and FOV — so ranking subjects means dividing by their own floor.
    """
    path = os.path.join(log_dir, BASELINE_FILE)
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        return pd.DataFrame(json.load(f).get("per_subject", []))


COMMENSURABILITY_KEYS = ("split_md5", "manifest_md5", "cardiac_phase_md5",
                         "data_cache_signature", "n_val_subjects")


def compare_runs(log_dirs, metric="metric_psnr_3d_heartseg", epoch="last"):
    """Compare runs on PAIRED per-subject values, and check they are comparable at all.

    Paired beats comparing cohort means: the same 133 subjects appear in every run, and
    per-subject achievable quality varies enormously across this cohort, so a paired
    delta cancels the subject effect that dominates an unpaired difference.

    Returns (wide_df, commensurability_df). The second is the important one — it flags
    when two runs used different data or a different val protocol, in which case the
    numbers are not comparable no matter how good the plot looks. `split_md5` alone is not
    enough: it hashes subject NAMES, so it does not move when the voxels change (this repo
    flipped 893 subjects' arrays in one day). `data_cache_signature` is what catches that.
    """
    frames, meta_rows = [], []
    for d in log_dirs:
        meta, _, subjects = load_run(d)
        name = os.path.basename(os.path.normpath(d))
        # The last LAUNCH line, not meta[-1] — that may be an "exit" record, which
        # carries no provenance fields.
        launches = [m for m in meta if m.get("event") != "exit"]
        last = dict(launches[-1]) if launches else {}
        meta_rows.append({"run": name,
                          "git_sha": ((last.get("git") or {}).get("sha") or "")[:8],
                          **{k: last.get(k) for k in COMMENSURABILITY_KEYS}})
        if subjects.empty or metric not in subjects.columns:
            print(f"[compare_runs] {name}: no '{metric}' — skipped", file=sys.stderr)
            continue
        sel = subjects[subjects["epoch"] == subjects["epoch"].max()] if epoch == "last" \
            else subjects[subjects["epoch"] == epoch]
        frames.append(sel.set_index(["seq_name", "t_target"])[metric].rename(name))

    wide = pd.concat(frames, axis=1, join="inner") if frames else pd.DataFrame()
    comm = pd.DataFrame(meta_rows)
    if len(comm) > 1:
        differing = [k for k in COMMENSURABILITY_KEYS if comm[k].nunique(dropna=False) > 1]
        if differing:
            print(f"[compare_runs] ⚠️  runs differ on {differing} — NOT commensurable",
                  file=sys.stderr)
    return wide, comm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir", nargs="+", help="one run dir, or several to compare")
    ap.add_argument("--metric", default="val/psnr/bbox_mean",
                    help="scalar name to print the tail of")
    ap.add_argument("--compare-metric", default="metric_psnr_3d_heartseg",
                    help="per-subject metric used when several log_dirs are given")
    args = ap.parse_args()

    if len(args.log_dir) > 1:
        wide, comm = compare_runs(args.log_dir, metric=args.compare_metric)
        print("commensurability:"); print(comm.to_string(index=False))
        if not wide.empty:
            print(f"\npaired {args.compare_metric} (n={len(wide)} subject-phases):")
            print(wide.mean().to_string())
            base = wide.columns[0]
            # Win direction depends on the metric: psnr/ssim/recov are higher-is-better,
            # mae/mse/hole_frac/loss are lower-is-better. Getting this wrong inverts the
            # reported win-rate silently, and STRATA_METRICS contains both kinds.
            higher_better = not any(k in args.compare_metric
                                    for k in ("mae", "mse", "hole", "loss", "epe"))
            print(f"\ndelta vs first run ({'higher' if higher_better else 'lower'} is better):")
            for c in wide.columns[1:]:
                d = wide[c] - wide[base]
                wins = int((d > 0).sum() if higher_better else (d < 0).sum())
                print(f"  {c}: {d.mean():+.4f}  wins {wins}/{len(d)}")
        return

    meta, scalars, subjects = load_run(args.log_dir[0])
    exits = [m for m in meta if m.get("event") == "exit"]
    launches = [m for m in meta if m.get("event") != "exit"]
    print(f"launches: {len(launches)}")
    if exits:
        e = exits[-1]
        print(f"  last exit: {e.get('status')} at epoch {e.get('final_epoch')}"
              + (f" — {e.get('error')}" if e.get("error") else ""))
    else:
        print("  no exit record — still running, or killed/requeued (SIGUSR1 bypasses it)")
    for m in launches:
        print(f"  sha={((m.get('git') or {}).get('sha') or '?')[:8]} "
              f"dirty={(m.get('git') or {}).get('dirty')} "
              f"resumed_from_epoch={m.get('resumed_from_epoch')} "
              f"n_train={m.get('n_train_subjects')} n_val={m.get('n_val_subjects')} "
              f"job={m.get('slurm_job_id')}")
    print(f"\nscalars: {len(scalars)} rows, {scalars['name'].nunique() if len(scalars) else 0} names")
    sel = scalars[scalars["name"] == args.metric] if len(scalars) else scalars
    if len(sel):
        print(f"\n{args.metric} (last 10):")
        print(sel.tail(10).to_string(index=False))
    else:
        print(f"\n(no scalar named {args.metric})")

    print(f"\nper-subject val rows: {len(subjects)}")
    if len(subjects) and "source" in subjects:
        last = subjects[subjects["epoch"] == subjects["epoch"].max()]
        cols = [c for c in ("metric_psnr_3d_bbox", "metric_recov_frac_heart") if c in last]
        if cols:
            print(f"\nlast epoch ({int(last['epoch'].iloc[0])}) by source:")
            print(last.groupby("source")[cols].agg(["mean", "count"]).to_string())


if __name__ == "__main__":
    main()
