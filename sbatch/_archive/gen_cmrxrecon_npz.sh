#!/bin/bash
#SBATCH --job-name=gen_cmrx7row
#SBATCH --partition=spgpu
#SBATCH --account=jjparkcv0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=0:40:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/gen_cmrx7row_%j.log

# GPU generation (npz only) for CMRxRecon val subjects + the corrupt-npz regen, on a fresh
# spgpu node (the interactive GPU is occupied by a training run). Rendering is a separate CPU job.
set -e
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr
export PYTHONPATH=training:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u tools/miitt_viz/gen_cmrxrecon_npz.py
echo "GEN DONE"
