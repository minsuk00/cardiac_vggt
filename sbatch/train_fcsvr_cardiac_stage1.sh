#!/bin/bash
#SBATCH --account=jjparkcv_owned1
#SBATCH --partition=spgpu2
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=5-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --gpu_cmode=shared
#SBATCH --job-name=fcsvr_cardiac_s1
#SBATCH --output=/home/minsukc/vggt/slurm_logs/%x_%j.log

set -eo pipefail

export MAMBA_EXE=/home/minsukc/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/home/minsukc/micromamba
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate svr

cd /home/minsukc/vggt
mkdir -p /home/minsukc/vggt/slurm_logs

export PYTHONPATH=baselines/fcsvr_cardiac:training:.
export WANDB_MODE=online

python baselines/fcsvr_cardiac/train.py
