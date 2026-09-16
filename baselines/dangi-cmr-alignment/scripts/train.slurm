#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=dangi-stage-a
#SBATCH --output=dangi-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
resume=()
if [[ -f runs/dangi/last.pt ]]; then
    resume=(--resume runs/dangi/last.pt)
fi
srun .venv/bin/python -u -m dangi.train --workers 4 "${resume[@]}" "$@"
