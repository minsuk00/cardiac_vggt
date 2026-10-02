#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64g
#SBATCH --time=14:00:00
#SBATCH --job-name=frame_selector_train
#SBATCH --output=slurm_logs/frame_selector_train_%j.out

# ============================================================================================
# Frame selector training (docs/130), then selection tables + offline phase error on the 10
# af12/hrv12 test cohorts. Default = the best recipe, B+ (heart-ROI attention loss, 20k steps,
# ~12 h on an L40S); ROI_WEIGHT=0 trains plain B.
#
#   mkdir -p slurm_logs && sbatch sbatch/frame_selector_train.sh                 # from the repo root
#   RUN_NAME=B STEPS=2000 ROI_WEIGHT=0 sbatch sbatch/frame_selector_train.sh
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`).
# ============================================================================================
set -euo pipefail
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
cd "$REPO"
PY=${PY:-${MAMBA_ROOT_PREFIX:-$HOME/micromamba}/envs/svr/bin/python}
RUN_NAME=${RUN_NAME:-B_roi_20k}
RUN=scratch/framesel/runs/$RUN_NAME
echo "=== $RUN_NAME  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
$PY -m frame_selector.train --out "$RUN" --steps "${STEPS:-20000}" --roi-weight "${ROI_WEIGHT:-1.0}" \
    --accum 4 --workers 5
COHORTS=""
for a in af12 hrv12; do for s in cmrx2023 cmrx2024 cmrx2025 acdc mnms; do COHORTS="$COHORTS ${s}_$a"; done; done
$PY -m frame_selector.predict --run "$RUN" --cohorts $COHORTS --out "scratch/framesel/sel_tables/$RUN_NAME"
echo "DONE $RUN_NAME"
