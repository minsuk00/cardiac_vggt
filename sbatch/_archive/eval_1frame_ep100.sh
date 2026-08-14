#!/bin/bash
#SBATCH --job-name=eval_1f_ep100
#SBATCH --account=jjparkcv0
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/minsukc/vggt/slurm_logs/eval_1f_ep100_%A_%a.log
#
# GPU eval of the 6 ep100 (prev_epoch=99) one-frame ablation models on 4 cohorts
# (cmrxrecon/miitt/ocmr/acdc), ONE model per array task -> ~6 GPUs in parallel, ~40min wall
# (acdc x40 is the long pole per model). Frozen-bundle harness (run_vggt.py --regime onef).
#   sbatch --account=jjparkcv0 sbatch/eval_1frame_ep100.sh
set -uo pipefail
VARIANTS=(gather05 no_gather aug_moderate contz dino_ft lowdiff100)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr
cd /home/minsukc/vggt

sleep $((SLURM_ARRAY_TASK_ID * 15))   # stagger mamba/GPFS
echo "[$(date +%H:%M:%S)] eval variant=$V (task $SLURM_ARRAY_TASK_ID) on $(hostname)"
bash scratch/eval/engine/run_1frame_series_v3.sh "$V"
echo "[$(date +%H:%M:%S)] DONE variant=$V"
