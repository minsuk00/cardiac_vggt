#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=06:00:00
#SBATCH --array=0-19%10
#SBATCH --job-name=rhythm24_vggt
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/rhythm24_vggt_%A_%a.out
#SBATCH --open-mode=append

# ============================================================================================
# VGGT recon on a 24-frame rhythm arm (docs/115): 4 model arms x 5 source cohorts = 20 tasks,
# at most 10 at once. ARM=af24 (default) | hrv24 | regular24.
#
# TIMING FAIRNESS: the main benchmark's GPU config (eval_pooled_val.sh: spgpu = NVIDIA A40,
# 1 GPU, 5 CPUs, 48 GB). run_vggt.py stamps the GPU name into timing.json + provenance.txt; this
# script refuses to run on anything that is not an A40.
#
# `!! N companion slot phase(s) re-pinned to the bundle's scatter draw` on every subject is the
# frame-index remap working (docs/110 s11.8) -- expected, not an error.
#
# NOT self-submitting:  sbatch --export=ALL,ARM=af24 sbatch/rhythm24_vggt.sh
# ============================================================================================
set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
[ -d "$REPO/evaluation/src/engine" ] || REPO=${SLURM_SUBMIT_DIR:-/home/minsukc/vggt-afsim}
cd "$REPO" || exit 1
export PYTHONPATH=training:.
PY=${PY:-/home/minsukc/micromamba/envs/svr/bin/python}
LOGS=/home/minsukc/vggt/scratch/logs

ARM=${ARM:-af24}
MODELS=(
  "final518_base_ep300      $LOGS/210823094_final518_base_curated898/ckpts/checkpoint_last.pt"
  "final518_diff1000_ep300  $LOGS/210823094_final518_diff1000_curated898/ckpts/checkpoint_last.pt"
  "final518_nogather_ep300  $LOGS/210498476_final518_nogather_curated898/ckpts/checkpoint_last.pt"
  "final518_hw0_ep300       $LOGS/210484679_final518_hw0_curated898/ckpts/checkpoint_last.pt"
)
# The five sources a rhythm arm was built for -- NOT paths.DATASETS, which also lists miitt/ocmr:
# the task index is (model, source), so a phantom source silently drops real (model, source) pairs.
SOURCES=(cmrx2023 cmrx2024 cmrx2025 acdc mnms)
IDX=${SLURM_ARRAY_TASK_ID:-0}
NS=${#SOURCES[@]}
# PER_MODEL=1: task index = model index and one job runs all 5 cohorts in sequence (generation is
# fast; 1 A40 job per model).  Default: task index = (model, source), 20 tasks.
if [ -n "${PER_MODEL:-}" ]; then
    [ "$IDX" -lt "${#MODELS[@]}" ] || { echo "task $IDX: nothing to do"; exit 0; }
    read -r NAME CKPT <<< "${MODELS[$IDX]}"
    DSS=("${SOURCES[@]/%/_${ARM}}")
else
    [ "$IDX" -lt $(( ${#MODELS[@]} * NS )) ] || { echo "task $IDX: nothing to do"; exit 0; }
    read -r NAME CKPT <<< "${MODELS[$((IDX / NS))]}"
    DSS=("${SOURCES[$((IDX % NS))]}_${ARM}")
fi
[ -f "$CKPT" ] || { echo "missing checkpoint $CKPT"; exit 3; }

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "=== task $IDX  model=$NAME  datasets=${DSS[*]}  gpu=$GPU ==="
if [ "${1:-}" = "--dry-run" ]; then echo "dry-run: nothing run"; exit 0; fi
[[ "$GPU" == *A40* ]] || [ -n "${ALLOW_ANY_GPU:-}" ] || { echo "not an A40 ($GPU) -- timing would not be comparable"; exit 4; }

rc=0
for DS in "${DSS[@]}"; do
    $PY evaluation/src/engine/run_vggt.py --dataset "$DS" --ckpt "$CKPT" --model-name "$NAME" \
        --split test --arms breath --input scatter ${SUBJECTS:+--subjects $SUBJECTS} || rc=$?
done
echo "=== task $IDX done (rc=$rc) ==="
exit $rc
