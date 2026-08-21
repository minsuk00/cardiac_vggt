#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --job-name=ef_dice
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/ef_dice_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# EF/Dice chain for ONE arm across cohorts, then fold into the per-source aggregates.
#
#   ef_dice.py dump   (svr env)    -> per-phase NIfTIs sliced from the SCORED cines
#   run_seg.sh        (nnunet env) -> nnU-Net v1 Task114 segmentations
#   ef_dice.py score  (svr env)    -> metric_results/_ef/vggt_<MODEL_NAME>.json (per-cohort merge)
#   aggregate.py      (svr env)    -> re-roll each source summary so it carries the ef block
#
# Dump/seg live on node-local /tmp (sidx-keyed filenames REQUIRE a fresh dir every run — see
# ef_dice.py). Submit with: MODEL_NAME=<arm slug> sbatch sbatch/eval_ef_dice.sh
# ============================================================================================

set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}}
MODEL_NAME=${MODEL_NAME:?arm slug, e.g. augaggr224hw2_ep300 (method = vggt_$MODEL_NAME)}
SOURCES=${SOURCES:-"cmrx2023 cmrx2024 cmrx2025 acdc mnms miitt ocmr"}
METHOD="vggt_$MODEL_NAME"
WORK=/tmp/ef_dice_${USER}_${SLURM_JOB_ID}
PY=/home/minsukc/micromamba/envs/svr/bin/python

cd "$REPO"
echo "arm     : $METHOD"
echo "sources : $SOURCES"
echo "work    : $WORK"

mkdir -p "$WORK/in" "$WORK/seg"
t0=$(date +%s)
PYTHONPATH=training:. $PY evaluation/src/score/ef_dice.py dump "$WORK/in" \
    --method "$METHOD" --cohorts $SOURCES
t1=$(date +%s); echo "[timing] dump: $((t1-t0))s  ($(ls "$WORK/in" | wc -l) volumes)"

bash evaluation/src/engine/run_seg.sh "$WORK/in" "$WORK/seg"
t2=$(date +%s); echo "[timing] seg: $((t2-t1))s"

PYTHONPATH=training:. $PY evaluation/src/score/ef_dice.py score "$WORK/seg" --input "$WORK/in"
t3=$(date +%s); echo "[timing] score: $((t3-t2))s"

for S in $SOURCES; do
  echo "=== [$S] aggregate ==="
  PYTHONPATH=training:. $PY evaluation/src/score/aggregate.py "$S" "$METHOD"
done
t4=$(date +%s)
rm -rf "$WORK"
echo "[timing] total: $((t4-t0))s"
echo "DONE — ef folded into metric_results/<source>/$METHOD.json"
