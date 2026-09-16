#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --job-name=time_dangi_a40
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/home/minsukc/vggt/slurm_logs/time_dangi_a40_%j.out
#SBATCH --open-mode=append

# ============================================================================================
# Re-time the Dangi baseline (docs/100) on an A40 for the paper's compute-cost comparison.
# Generation was first run on CPU (a deliberate but under-communicated choice) — this regenerates
# every val+test x scatter+gated volume on GPU with --overwrite so recon_wall_sec / timing.json
# reflect real A40 wall-clock. Output is numerically the same model on the same inputs (fp32
# forward pass), so metrics.json / EF are NOT rescored here — only volumes + timing change.
#
# Submit: sbatch sbatch/time_dangi_baseline_gpu.sh
# ⚠️ From inside an interactive GPU allocation, clear the SLURM env first (`unset ${!SLURM_@}`).
# ============================================================================================

set -euo pipefail
REPO=${REPO:-/home/minsukc/vggt}
cd "$REPO"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=training:.

for job in "val scatter" "test scatter" "val gated" "test gated"; do
  set -- $job
  micromamba run -n svr python evaluation/src/engine/run_dangi.py \
    --split "$1" --input "$2" --device cuda --overwrite
done
echo GPU_RETIME_DONE
