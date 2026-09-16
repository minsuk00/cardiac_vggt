#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=08:00:00
#SBATCH --job-name=score_baseline
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/score_baseline_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# Image-metric scoring (evaluation/src/score/run.py) for one or more METHODS x SPLITS, run in
# PARALLEL on one shared A40 (--gpu_cmode=shared is required: pose_psf.py's 6-DOF NCC
# registration uses CUDA when present, and >1 process per GPU fails "device busy" without it —
# see feedback_gpu_cmode_shared.md). Mirrors the score_svrtk_v2b job from 2026-09-14.
#
# Submit with: METHODS="nesvor nesvor_scatter" SPLITS="val test" sbatch sbatch/eval_score_baseline.sh
# ============================================================================================
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-/home/minsukc/vggt}}
cd "$REPO"
export PYTHONPATH=training:.
PY=/home/minsukc/micromamba/envs/svr/bin/python
METHODS=${METHODS:?set METHODS="nesvor nesvor_scatter"}
SPLITS=${SPLITS:-"val test"}

echo "methods : $METHODS"
echo "splits  : $SPLITS"

pids=()
for METHOD in $METHODS; do
  for SPLIT in $SPLITS; do
    echo "=== launch $METHOD split=$SPLIT ==="
    $PY evaluation/src/score/run.py --method "$METHOD" --split "$SPLIT" \
        > "slurm_logs/score_baseline_${SLURM_JOB_ID}_${METHOD}_${SPLIT}.out" 2>&1 &
    pids+=($!)
  done
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit $rc
