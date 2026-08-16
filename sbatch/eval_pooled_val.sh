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
#   assemble_and_gif.py      -> metrics.json (+ GIFs and diagnostic panels unless SKIP_GIF=1)
#   aggregate.py             -> the per-source roll-up
#
# The build step is IDEMPOTENT and INCREMENTAL: a subject that already has a manifest.json is
# skipped, and bundles are keyed on the subject NAME (breathing seed and slot draw both hash the
# name), so appending subjects to the split file leaves every existing bundle and recon valid.
# Re-running this whole script after adding val subjects therefore only does the new work.
#
# NOT self-submitting on purpose — submit it yourself with `sbatch sbatch/eval_pooled_val.sh`.
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`)
# or it runs INLINE on the interactive node instead of queueing.
# ============================================================================================

set -euo pipefail

REPO=${REPO:-/home/minsukc/vggt}
CKPT=${CKPT:-$REPO/scratch/logs/213338187_augaggr224hw2_pooled1337/ckpts/checkpoint_last.pt}
MODEL_NAME=${MODEL_NAME:-augaggr224hw2_ep300}
SPLIT_FILE=${SPLIT_FILE:-$REPO/training/splits/pooled.txt}
SPLIT=${SPLIT:-val}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms"}
SKIP_GIF=${SKIP_GIF:-0}          # 1 = metrics only (GIF rendering dominates wall-clock)
ARMS=${ARMS:-breath}             # `breath` = the deliverable. Add `clean` ("clean breath") only
                                 # for the no-breathing PSNR ceiling; it ~doubles scoring time.

cd "$REPO"
export PYTHONPATH=training:.
PY="micromamba run -n svr python"

echo "ckpt        : $CKPT"
echo "arm         : vggt_$MODEL_NAME"
echo "split       : $SPLIT_FILE [$SPLIT]"
echo "sources     : $SOURCES"
echo "arms        : $ARMS"
# The MODEL protocol (img_size, backbone, sampling knobs) is read from the ckpt's own
# run_meta.jsonl, never from the live default.yaml — see inference/load_run.py for why.

for S in $SOURCES; do
  echo "=== [$S] build bundles ==============================================="
  $PY evaluation/engine/build_inputs/pooled.py \
      --source "$S" --split-file "$SPLIT_FILE" --split "$SPLIT"

  echo "=== [$S] score ======================================================="
  $PY evaluation/engine/run_vggt.py \
      --dataset "$S" --ckpt "$CKPT" --model-name "$MODEL_NAME" --split "$SPLIT" --arms $ARMS

  echo "=== [$S] assemble + metrics =========================================="
  for SUBJ in $(ls "evaluation/volumes/$S/out"); do
      EVAL_DATASET="$S" SKIP_GIF="$SKIP_GIF" \
        $PY evaluation/engine/assemble_and_gif.py "$SUBJ" "vggt_$MODEL_NAME" || \
        echo "  [warn] scoring failed for $SUBJ — continuing"
  done

  echo "=== [$S] aggregate ==================================================="
  $PY evaluation/engine/aggregate.py "$S" "vggt_$MODEL_NAME"
done

echo "DONE — per-source summaries under evaluation/results/"
