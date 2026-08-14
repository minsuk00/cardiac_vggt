#!/bin/bash
#SBATCH --job-name=s20contz7row
#SBATCH --partition=spgpu
#SBATCH --account=jjparkcv0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/s20contz7row_%j.log

# s20contz 1-frame-per-slice CONTINUOUS-Z gated breathing-sim sweep (MIITT/OCMR/ACDC x clean/normal).
# Runs on a fresh spgpu node (the interactive GPU is occupied by a training run). GPU forwards +
# inline hi-res (dpi 130) GIF render + npz, all in one self-contained job. Resumable (skip-if-npz).
set -e
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr
export PYTHONPATH=training:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u tools/miitt_viz/gated_s20contz_sweep.py
echo "S20CONTZ SWEEP DONE"
