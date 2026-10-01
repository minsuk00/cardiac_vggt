#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --array=0-1
#SBATCH --job-name=framesel_train
#SBATCH --output=/home/minsukc/vggt/slurm_logs/framesel_train_%A_%a.out

# Frame selector training (docs/128): task 0 = variant A (concat -> MLP), task 1 = variant B
# (cross-attention -> MLP). Both: per-plane centering, per-plane RMS scale, attention pooling.
# Gated cine + continuous breathing, pooled_curated_v2 train/val.
# Then predicts selection tables + offline phase error on the af12/hrv12 test bundles.
#   sbatch sbatch/framesel_train.sh
set -euo pipefail
REPO=${REPO:-/home/minsukc/vggt-framesel}
cd "$REPO"
export PYTHONPATH=training:.
PY=/home/minsukc/micromamba/envs/svr/bin/python
VARIANTS=(A B)
V=${VARIANTS[${SLURM_ARRAY_TASK_ID:-1}]}
RUN=scratch/framesel/runs/$V
echo "=== variant $V  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
$PY tools/framesel_train.py --variant "$V" --out "$RUN" --steps ${STEPS:-2000} --accum 4 --workers 5 \
    --center --norm slice --attn-pool
COHORTS=""
for a in af12 hrv12; do for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do COHORTS="$COHORTS ${s}_$a"; done; done
$PY tools/framesel_predict.py --run "$RUN" --cohorts $COHORTS --out "scratch/framesel/sel_tables/$V"
echo "DONE $V"
