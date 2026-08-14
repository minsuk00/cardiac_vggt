#!/bin/bash
#SBATCH --job-name=ctrl0contz7row
#SBATCH --partition=spgpu
#SBATCH --account=jjparkcv0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:30:00
#SBATCH --output=/home/minsukc/vggt/slurm_logs/ctrl0contz7row_%j.log

# control0 (the gather-aux=0 "gather0" 1-frame model) run through the 1frame_contz regime — a
# GENERALIZATION probe: this model was trained on SNAPPED /12 integer z, so feeding it true
# fractional physical z tests whether its Fourier z-embedding extrapolates off-grid. Same
# datasets/conditions as the s20contz sweep. Output -> result/gated_model_sweep/control0/...
set -e
cd /home/minsukc/vggt
eval "$(micromamba shell hook --shell bash)"
micromamba activate svr
export PYTHONPATH=training:.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u tools/miitt_viz/gated_s20contz_sweep.py control0
echo "CONTROL0 CONTZ SWEEP DONE"
