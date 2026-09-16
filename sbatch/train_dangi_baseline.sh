#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --job-name=train_dangi
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/train_dangi_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# Dangi et al. (2018) Stage A (LV-centre regression CNN) trained on the VGGT pooled cohort:
# pooled_curated_v2 [train]/[val], all 12 phases, nnU-Net LV-cavity centroid targets, one random
# 189-grid transform per slice per epoch (the paper's epoch scheme), 100 epochs, best-val ckpt.
# Baseline for the reconstruction comparison — docs/NN_dangi_baseline.md.
#
# Run dir on GPFS (scratch/baselines/dangi/pool_v2); preprocessed-slice cache on node-local
# /tmp (rebuilt per node, ~5 min). Resumes from last.pt automatically on requeue/resubmit.
#
# Submit: sbatch sbatch/train_dangi_baseline.sh
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`).
# ============================================================================================

set -euo pipefail
REPO=${REPO:-/home/minsukc/vggt}
cd "$REPO"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN="$REPO/scratch/baselines/dangi/pool_v2"
CACHE="/tmp/dangi_pool_cache_${USER}"
mkdir -p "$RUN" "$CACHE"

resume=()
if [[ -f "$RUN/last.pt" ]]; then resume=(--resume "$RUN/last.pt"); fi

PYTHONPATH=baselines/dangi-cmr-alignment micromamba run -n svr python -u -m dangi.train \
    --data "$REPO/scratch/data" \
    --pool-split "$REPO/training/splits/pooled_curated_v2.txt" \
    --pool-cache "$CACHE" \
    --output "$RUN" --epochs 100 --batch-size 32 --workers 8 --device cuda \
    "${resume[@]}"
