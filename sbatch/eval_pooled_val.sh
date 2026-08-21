#!/bin/bash
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --job-name=eval_pooled_val
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/eval_pooled_val_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# Score one VGGT checkpoint on the frozen gated + breathing-simulated bundles, per source.
#
#   build_inputs/pooled.py   -> the frozen bundle (gt / clean / breath + mask + manifest)
#   run_vggt.py              -> per-subject recons for both arms
#   score/image_metrics.py   -> metrics.json (+ analysis/viz.py GIFs unless SKIP_GIF=1)
#   aggregate.py             -> the per-source roll-up
#
# The BUILD step is IDEMPOTENT and INCREMENTAL: a subject that already has a manifest.json is
# skipped, and bundles are keyed on the subject NAME (breathing seed and slot draw both hash the
# name), so appending subjects to the split file leaves every existing bundle and recon valid.
# The RECON + SCORE steps are NOT incremental: run_vggt re-reconstructs and assemble re-scores
# every subject on a re-run (deterministic, so results are identical — but the full GPU sweep
# is repeated). To add a few subjects cheaply, run the per-subject commands by hand instead.
#
# Covers all SEVEN sources by default, each with its own split file (see split_file_for below).
# Narrow it with SOURCES="cmrx2024 ocmr"; force one split file for all with SPLIT_FILE=<path>.
#
# NOT self-submitting on purpose — submit it yourself with `sbatch sbatch/eval_pooled_val.sh`.
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`)
# or it runs INLINE on the interactive node instead of queueing.
# ============================================================================================

set -euo pipefail

# Derived from THIS script's location, so running the copy in a worktree evaluates the worktree's
# code. It used to hardcode /home/minsukc/vggt, which silently ran main-tree code from anywhere.
# Under sbatch the script runs from SLURM's spool copy, so BASH_SOURCE points at
# /var/spool/... — use the submit dir there; the dirname fallback covers `bash <path>`.
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
CKPT=${CKPT:-$REPO/scratch/logs/213338187_augaggr224hw2_pooled1337/ckpts/checkpoint_last.pt}
MODEL_NAME=${MODEL_NAME:-augaggr224hw2_ep300}
SPLIT=${SPLIT:-val}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
SKIP_GIF=${SKIP_GIF:-0}          # 1 = metrics only (GIF rendering dominates wall-clock)
ARMS=${ARMS:-breath}             # `breath` = the deliverable. Add `clean` ("clean breath") only
                                 # for the no-breathing PSNR ceiling; it ~doubles scoring time.

# Each source names its own split file — there is NO single file listing all seven, and forcing one
# was why this driver could reach only 5 of the 7 committed results (miitt/ocmr had to be built by
# hand). pooled.txt carries the five trained-on sources; MIITT rides in pooled_miitt.txt (pooled.txt
# VERBATIM + 13 MIITT lines, 5 train / 3 val / 5 test); OCMR is eval-only so its split lives under
# evaluation/splits/ precisely so it can never be pulled into a training pool.
# Setting SPLIT_FILE overrides the lookup for EVERY source (the old single-file behaviour).
split_file_for() {
  if [ -n "${SPLIT_FILE:-}" ]; then echo "$SPLIT_FILE"; return; fi
  case "$1" in
    miitt) echo "$REPO/training/splits/pooled_miitt.txt" ;;
    ocmr)  echo "$REPO/evaluation/splits/ocmr_eval.txt" ;;
    *)     echo "$REPO/training/splits/pooled.txt" ;;
  esac
}

cd "$REPO"
export PYTHONPATH=training:.
export SPLIT                     # aggregate.py summarizes only subjects whose manifest split matches
# Direct interpreter, NOT `micromamba run`: the assemble loop below is ~144 short invocations and
# micromamba's lockfile deadlocks under exactly that pattern.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}

echo "repo        : $REPO"
echo "ckpt        : $CKPT"
echo "arm         : vggt_$MODEL_NAME"
echo "split       : [$SPLIT] ${SPLIT_FILE:-per-source (see split_file_for)}"
echo "sources     : $SOURCES"
echo "arms        : $ARMS"
# The MODEL protocol (img_size, backbone, sampling knobs) is read from the ckpt's own
# run_meta.jsonl, never from the live default.yaml — see inference/load_run.py for why.

for S in $SOURCES; do
  SF=$(split_file_for "$S")
  echo "=== [$S] build bundles  (split file: ${SF#$REPO/}) ===================="
  $PY evaluation/src/engine/build_inputs/pooled.py \
      --source "$S" --split-file "$SF" --split "$SPLIT"

  echo "=== [$S] score ======================================================="
  $PY evaluation/src/engine/run_vggt.py \
      --dataset "$S" --ckpt "$CKPT" --model-name "$MODEL_NAME" --split "$SPLIT" --arms $ARMS

  echo "=== [$S] assemble + metrics =========================================="
  # Scores every built subject; aggregate.py is the step that enforces $SPLIT, so an off-split
  # bundle sharing this tree costs scoring time but never enters the summary.
  # Glob, not $(ls ...): a command substitution that fails does NOT trip `set -e` in a for-list, so a
  # missing/empty out/ dir would silently score nothing and only surface later at aggregate.py.
  # `nullglob` off by default means an empty dir yields the literal pattern -> the -d test skips it
  # and the counter below turns "built nothing" into a loud failure here, where it happened.
  N_SCORED=0
  N_FAILED=${N_FAILED:-0}
  for SUBJ_DIR in "evaluation/volumes/$S/out"/*/; do
      [ -d "$SUBJ_DIR" ] || continue
      SUBJ="$(basename "$SUBJ_DIR")"
      # off-split bundles share this tree; run_vggt skips them (no arm dir), so scoring them can
      # only fail — skip here too, keeping failures on reconned subjects loud.
      [ -d "$SUBJ_DIR/vggt_$MODEL_NAME" ] || { echo "  skip $SUBJ (no recon for this arm — off-split)"; continue; }
      N_SCORED=$((N_SCORED + 1))
      EVAL_DATASET="$S" \
        $PY evaluation/src/score/image_metrics.py "$SUBJ" "vggt_$MODEL_NAME" || \
        { echo "  [warn] scoring FAILED for $SUBJ — continuing (job will exit nonzero; the"
          echo "         aggregate may carry a stale earlier metrics.json for this subject)"
          N_FAILED=$((N_FAILED + 1)); continue; }
      # rendering is decoupled from scoring now; SKIP_GIF=1 -> metrics only
      [ "$SKIP_GIF" = "1" ] || EVAL_DATASET="$S" \
        $PY evaluation/src/analysis/viz.py "$SUBJ" "vggt_$MODEL_NAME" || \
        echo "  [warn] gif render failed for $SUBJ — continuing"
  done
  [ "$N_SCORED" -gt 0 ] || { echo "  [fatal] no built subjects under evaluation/volumes/$S/out"; exit 1; }

  echo "=== [$S] aggregate ==================================================="
  $PY evaluation/src/score/aggregate.py "$S" "vggt_$MODEL_NAME"
done

echo "DONE — per-source summaries under evaluation/metric_results/"
# scoring failures must not read as job success (and their subjects' summaries may hold stale rows)
[ "${N_FAILED:-0}" -eq 0 ] || { echo "EXIT 1: $N_FAILED subject(s) failed scoring"; exit 1; }
