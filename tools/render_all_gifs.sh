#!/bin/bash
# Batch-render the per-arm GT-vs-recon GIFs (gif_{clean,breath}.gif) for every
# subject-arm in a worklist, via evaluation/src/score/image_metrics.py + analysis/viz.py. CPU-only,
# no model: it just reads the already-saved recon volumes. Regenerates metrics.json too
# (bit-identical values + additive provenance fields), so it is safe to re-run.
#
# Usage:  micromamba run -n svr bash tools/render_all_gifs.sh <worklist.tsv> [jobs] [logdir]
#   worklist.tsv : one "<dataset>\t<subject>\t<arm>" per line
#   jobs         : parallel workers (default 4)
#   logdir       : per-task logs + progress (default alongside the worklist)
set -u
# THIS tree's root, never a hardcoded one (same rule as sbatch/eval_pooled_val.sh): a hardcoded
# /home/minsukc/vggt made the worktree run the MAIN tree's copy of the scoring scripts.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

WL="${1:?worklist tsv required}"
JOBS="${2:-4}"
LOGDIR="${3:-$(dirname "$WL")/gif_logs}"
mkdir -p "$LOGDIR"
PROGRESS="$LOGDIR/progress.log"
TOTAL=$(wc -l < "$WL")
: > "$PROGRESS"

echo "rendering $TOTAL subject-arm GIFs with $JOBS workers -> $LOGDIR" | tee -a "$PROGRESS"

render_one() {
  local ds="$1" subj="$2" arm="$3"
  local tlog="$LOGDIR/${ds}_${subj}_${arm}.log"
  if EVAL_DATASET="$ds" python evaluation/src/score/image_metrics.py "$subj" "$arm" > "$tlog" 2>&1 \
     && EVAL_DATASET="$ds" python evaluation/src/analysis/viz.py "$subj" "$arm" >> "$tlog" 2>&1; then
    echo "OK   $ds $subj $arm" >> "$PROGRESS"
  else
    echo "FAIL $ds $subj $arm  (see $tlog)" >> "$PROGRESS"
  fi
}
export -f render_one
export LOGDIR PROGRESS

# xargs -L1 feeds one worklist line (its whitespace/tab-split fields) as args to bash -c;
# $0=_ placeholder, $1=dataset $2=subject $3=arm.
< "$WL" xargs -P "$JOBS" -L1 bash -c 'render_one "$1" "$2" "$3"' _

done_ok=$(grep -c '^OK'   "$PROGRESS")
done_fail=$(grep -c '^FAIL' "$PROGRESS")
echo "DONE: $done_ok ok, $done_fail failed (of $TOTAL)" | tee -a "$PROGRESS"
# a batch with failures must not exit 0 — callers (sbatch mail-on-fail, CI) would read success
[ "$done_fail" -eq 0 ]
